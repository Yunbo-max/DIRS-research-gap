#!/usr/bin/env python3
"""Dispatch GPU-release professional gate rechecks.

This is Loop 1/Loop 2 plumbing, not convergence evidence. It reruns existing
paper-specific professional gates only after the selected GPU has enough free
memory. Gate outputs can clear stale "no GPU slot" blockers, but a paper still
needs verifier-comparable paper-shaped results before acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent
STALENESS_AUDIT = RUN_ROOT / "explicit_blocker_staleness_audit_20260723.json"
REFRESH_SCRIPT = RUN_ROOT / "refresh_longgoal_status.py"
STATE_PATH = RUN_ROOT / "gpu_recheck_dispatcher_state.json"
REPORT_PATH = RUN_ROOT / "gpu_recheck_dispatcher_report.json"
STATUS_MD = RUN_ROOT / "GPU_RECHECK_DISPATCHER_STATUS.md"

RUNNERS = RUN_ROOT / "specialized_runners"
RESOLVED_BATCH = RUNNERS / "resolved_repo_batch" / "run_resolved_repo_professional_gates.py"

RECHECK_COMMANDS: dict[str, dict[str, Any]] = {
    "CVPR2026_016_nuwa_class_specific_vit_pruning": {
        "short": "nuwa",
        "cmd": [sys.executable, str(RUNNERS / "nuwa" / "run_nuwa_professional_gate.py")],
        "cwd": str(RUNNERS / "nuwa"),
    },
    "CVPR2026_052_seacache_spectral_evolution_cache": {
        "short": "seacache",
        "cmd": [sys.executable, str(RUNNERS / "seacache" / "run_seacache_professional_gate.py")],
        "cwd": str(RUNNERS / "seacache"),
    },
    "CVPR2026_053_sencache_sensitivity_aware_caching": {
        "short": "sencache",
        "cmd": [sys.executable, str(RUNNERS / "sencache" / "run_sencache_professional_gate.py")],
        "cwd": str(RUNNERS / "sencache"),
    },
    "CVPR2026_067_rdvq_differentiable_vq_rate_distortion": {
        "short": "rdvq",
        "cmd": [sys.executable, str(RUNNERS / "rdvq" / "run_rdvq_professional_gate.py")],
        "cwd": str(RUNNERS / "rdvq"),
    },
    "ICLR2026_88ZLp7xYxw_prism_fmri_structured_text": {
        "short": "prism",
        "cmd": [sys.executable, str(RUNNERS / "prism" / "run_prism_professional_gate.py")],
        "cwd": str(RUNNERS / "prism"),
    },
    "ICLR2026_JEYWpFGzvn_infotok_adaptive_video_tokenizer": {
        "short": "infotok",
        "cmd": [sys.executable, str(RUNNERS / "infotok" / "run_infotok_professional_gate.py")],
        "cwd": str(RUNNERS / "infotok"),
    },
    "ICLR2026_LaVrNaBNwM_hsd_lossless_speculative_decoding": {
        "short": "hsd",
        "cmd": [sys.executable, str(RESOLVED_BATCH), "--only", "hsd"],
        "cwd": str(RESOLVED_BATCH.parent),
    },
    "ICLR2026_h06l9w1clt_locality_parallel_decoding_ar_image": {
        "short": "lpd",
        "cmd": [sys.executable, str(RESOLVED_BATCH), "--only", "lpd"],
        "cwd": str(RESOLVED_BATCH.parent),
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def gpu_inventory() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        idx, name, used, total, util = parts
        used_i = int(float(used))
        total_i = int(float(total))
        gpus.append(
            {
                "index": idx,
                "name": name,
                "memory_used_mib": used_i,
                "memory_total_mib": total_i,
                "memory_free_mib": total_i - used_i,
                "utilization_gpu_pct": int(float(util)),
            }
        )
    return gpus


def active_llada_processes() -> list[str]:
    result = subprocess.run(
        ["ps", "aux"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return []
    markers = [
        "prophet_custom_full_gsm8k_runner.py",
        "prophet_generation_worker.py",
    ]
    return [
        line
        for line in result.stdout.splitlines()
        if any(marker in line for marker in markers)
        and "gpu_recheck_dispatcher.py" not in line
        and "strict_dirs_goal_supervisor.py" not in line
    ]


def selected_gpu_busy(processes: list[str], gpu: str) -> bool:
    selected = str(gpu)
    for line in processes:
        if f"--gpu {selected}" in line or f"--gpu={selected}" in line:
            return True
        if f"CUDA_VISIBLE_DEVICES={selected}" in line:
            return True
    return False


def selected_gpu(gpu: str) -> dict[str, Any] | None:
    return next((item for item in gpu_inventory() if item["index"] == str(gpu)), None)


def pending_recheck_papers(state: dict[str, Any]) -> list[dict[str, Any]]:
    audit = read_json(STALENESS_AUDIT, {})
    attempted_fingerprints = state.get("attempted_recheck_fingerprints", {})
    papers = []
    for paper in audit.get("papers", []):
        paper_id = paper.get("paper_id")
        if paper.get("status") != "explicit_blocker_valid_recheck_after_gpu_release":
            continue
        command = RECHECK_COMMANDS.get(paper_id)
        if not command:
            continue
        fingerprint_payload = {
            "blocker_tags": paper.get("blocker_tags"),
            "operational_blocker_count": paper.get("operational_blocker_count"),
            "required_update_nodes": paper.get("required_update_nodes"),
            "verifier_required_update_count": paper.get("verifier_required_update_count"),
            "weak_evidence": paper.get("weak_evidence"),
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        if attempted_fingerprints.get(paper_id) == fingerprint:
            continue
        paper["blocker_fingerprint"] = fingerprint
        papers.append({**paper, "command": command})
    return papers


def run_cmd(cmd: list[str], cwd: str, timeout: int) -> dict[str, Any]:
    started = time.time()
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        "cmd": cmd,
        "cwd": cwd,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def refresh_status() -> dict[str, Any]:
    if not REFRESH_SCRIPT.exists():
        return {"attempted": False, "reason": "missing_refresh_script"}
    return run_cmd([sys.executable, str(REFRESH_SCRIPT)], str(RUN_ROOT), 120)


def run_rechecks(papers: list[dict[str, Any]], timeout: int) -> list[dict[str, Any]]:
    results = []
    for paper in papers:
        command = paper["command"]
        cmd = command["cmd"]
        missing = [part for part in cmd if part.endswith(".py") and not Path(part).exists()]
        if missing:
            results.append(
                {
                    "paper_id": paper["paper_id"],
                    "blocker_fingerprint": paper.get("blocker_fingerprint"),
                    "short": command["short"],
                    "attempted": False,
                    "status": "missing_professional_gate_script",
                    "missing": missing,
                }
            )
            continue
        result = run_cmd(cmd, command["cwd"], timeout)
        results.append(
            {
                "paper_id": paper["paper_id"],
                "blocker_fingerprint": paper.get("blocker_fingerprint"),
                "short": command["short"],
                "attempted": True,
                "status": "gate_recheck_executed",
                **result,
            }
        )
    return results


def render_status(report: dict[str, Any]) -> None:
    lines = [
        "# GPU Recheck Dispatcher Status",
        "",
        f"- Updated: `{report['created_at_utc']}`",
        f"- Status: `{report['status']}`",
        "- Policy: professional gate rechecks are support only; they cannot converge a paper without full verifier-comparable outputs.",
        f"- Selected GPU: `{report.get('gpu')}`",
        f"- Pending before dispatch: `{report.get('pending_before_count')}`",
        f"- Rechecks executed: `{len(report.get('recheck_results', []))}`",
        f"- State: `{STATE_PATH}`",
        f"- Report: `{REPORT_PATH}`",
        "",
        "## GPU Inventory",
        "",
    ]
    for gpu in report.get("gpu_inventory", []):
        lines.append(
            f"- GPU `{gpu['index']}` {gpu['name']} free=`{gpu['memory_free_mib']}` MiB "
            f"used=`{gpu['memory_used_mib']}` MiB util=`{gpu['utilization_gpu_pct']}`%"
        )
    if report.get("active_llada_processes"):
        lines += ["", "## Active Prophet Processes", ""]
        for line in report["active_llada_processes"]:
            lines.append(f"- `{line}`")
    lines += ["", "## Recheck Results", ""]
    for item in report.get("recheck_results", []):
        lines.append(
            f"- `{item.get('paper_id')}` short=`{item.get('short')}` "
            f"attempted=`{item.get('attempted')}` status=`{item.get('status')}` "
            f"returncode=`{item.get('returncode')}`"
        )
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--min-free-mib", type=int, default=21000)
    parser.add_argument("--launch-next", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    state = {} if args.reset else read_json(STATE_PATH, {})
    state.setdefault("artifact_kind", "gpu_recheck_dispatcher_state")
    state.setdefault("attempted_paper_ids", [])
    state.setdefault("attempted_recheck_fingerprints", {})
    state.setdefault("attempts", [])
    pending = pending_recheck_papers(state)
    gpu = selected_gpu(str(args.gpu))
    active = active_llada_processes()
    report: dict[str, Any] = {
        "artifact_kind": "gpu_recheck_dispatcher_report",
        "created_at_utc": utc_now(),
        "gpu": str(args.gpu),
        "gpu_inventory": gpu_inventory(),
        "active_llada_processes": active,
        "pending_before_count": len(pending),
        "policy": {
            "can_converge_papers": False,
            "reduced_or_proxy_evidence_allowed": False,
            "purpose": "rerun exact paper-specific professional gates after GPU-release blockers may be stale",
        },
        "recheck_results": [],
    }
    if not args.launch_next and not args.run_all:
        report["status"] = "manifest_only_no_dispatch_requested"
    elif not gpu:
        report["status"] = "deferred_selected_gpu_missing"
    elif selected_gpu_busy(active, str(args.gpu)):
        report["status"] = "deferred_selected_gpu_has_active_prophet_runner"
    elif int(gpu["memory_free_mib"]) < args.min_free_mib:
        report["status"] = "deferred_insufficient_selected_gpu_memory"
        report["selected_gpu_state"] = gpu
        report["min_free_mib"] = args.min_free_mib
    elif not pending:
        report["status"] = "no_pending_gpu_recheck_papers"
    else:
        selected = pending[:1] if args.launch_next else pending
        report["status"] = "executed_gpu_release_rechecks"
        report["recheck_results"] = run_rechecks(selected, args.timeout_seconds)
        for item in report["recheck_results"]:
            if item.get("attempted"):
                paper_id = item.get("paper_id")
                if paper_id not in state["attempted_paper_ids"]:
                    state["attempted_paper_ids"].append(paper_id)
                if item.get("blocker_fingerprint"):
                    state["attempted_recheck_fingerprints"][paper_id] = item["blocker_fingerprint"]
        state["attempts"].extend(report["recheck_results"])
        state["last_dispatch_at_utc"] = report["created_at_utc"]
        report["refresh_result"] = refresh_status()

    state["updated_at_utc"] = report["created_at_utc"]
    state["last_report_status"] = report["status"]
    state["last_pending_before_count"] = report["pending_before_count"]
    write_json(STATE_PATH, state)
    write_json(REPORT_PATH, report)
    render_status(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
