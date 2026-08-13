#!/usr/bin/env python3
"""Sweep paper-specific professional gates for stale operational blockers.

This wrapper reruns existing gate scripts and records what changed. It is a
Loop 1 hygiene/action artifact only: a professional gate can update blockers,
but it cannot accept a paper without full paper-shaped outputs and verifier
comparison against the paper evidence channels.
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
RUNNERS = RUN_ROOT / "specialized_runners"
STALENESS_AUDIT = RUN_ROOT / "explicit_blocker_staleness_audit_20260723.json"
REFRESH_SCRIPT = RUN_ROOT / "refresh_longgoal_status.py"
STATE_PATH = RUN_ROOT / "professional_gate_sweep_state.json"
REPORT_PATH = RUN_ROOT / "professional_gate_sweep_report.json"
STATUS_MD = RUN_ROOT / "PROFESSIONAL_GATE_SWEEP_STATUS.md"

RESOLVED_BATCH = RUNNERS / "resolved_repo_batch" / "run_resolved_repo_professional_gates.py"

GATE_COMMANDS: dict[str, dict[str, Any]] = {
    "CVPR2026_016_nuwa_class_specific_vit_pruning": {
        "short": "nuwa",
        "cmd": [sys.executable, str(RUNNERS / "nuwa" / "run_nuwa_professional_gate.py")],
        "cwd": str(RUNNERS / "nuwa"),
    },
    "CVPR2026_030_spark_vlm_articulated_reconstruction": {
        "short": "spark",
        "cmd": [sys.executable, str(RUNNERS / "spark" / "run_spark_professional_gate.py")],
        "cwd": str(RUNNERS / "spark"),
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
    "CVPR2026_065_trellis2_native_compact_structured_latents": {
        "short": "trellis2",
        "cmd": [sys.executable, str(RUNNERS / "trellis2" / "run_trellis2_professional_gate.py")],
        "cwd": str(RUNNERS / "trellis2"),
    },
    "CVPR2026_067_rdvq_differentiable_vq_rate_distortion": {
        "short": "rdvq",
        "cmd": [sys.executable, str(RUNNERS / "rdvq" / "run_rdvq_professional_gate.py")],
        "cwd": str(RUNNERS / "rdvq"),
    },
    "CVPR2026_103_atoken_unified_visual_tokenizer": {
        "short": "atoken",
        "cmd": [sys.executable, str(RUNNERS / "atoken" / "run_atoken_professional_gate.py")],
        "cwd": str(RUNNERS / "atoken"),
    },
    "ICLR2026_1J63FJYJKg_mrrope_mixed_radix_rope": {
        "short": "mrrope",
        "cmd": [sys.executable, str(RUNNERS / "mrrope" / "run_mrrope_professional_gate.py")],
        "cwd": str(RUNNERS / "mrrope"),
    },
    "ICLR2026_88ZLp7xYxw_prism_fmri_structured_text": {
        "short": "prism",
        "cmd": [sys.executable, str(RUNNERS / "prism" / "run_prism_professional_gate.py")],
        "cwd": str(RUNNERS / "prism"),
    },
    "ICLR2026_EQhUvWH78U_rational_information_seeking_agents": {
        "short": "battleship",
        "cmd": [sys.executable, str(RUNNERS / "battleship" / "run_battleship_professional_gate.py")],
        "cwd": str(RUNNERS / "battleship"),
    },
    "ICLR2026_H6rDX4w6Al_flashvid_vllm_token_merging": {
        "short": "flashvid",
        "cmd": [sys.executable, str(RUNNERS / "flashvid" / "run_flashvid_professional_gate.py")],
        "cwd": str(RUNNERS / "flashvid"),
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
    "ICLR2026_o29E01Q6bv_loongrl_long_context_reasoning": {
        "short": "loongrl",
        "cmd": [sys.executable, str(RUNNERS / "loongrl" / "run_loongrl_professional_gate.py")],
        "cwd": str(RUNNERS / "loongrl"),
    },
    "ICLR2026_h06l9w1clt_locality_parallel_decoding_ar_image": {
        "short": "lpd",
        "cmd": [sys.executable, str(RESOLVED_BATCH), "--only", "lpd"],
        "cwd": str(RESOLVED_BATCH.parent),
    },
    "ICLR2026_P5B97gZwRb_hyperparameter_trajectory_inference_clot": {
        "short": "clot",
        "cmd": [sys.executable, str(RESOLVED_BATCH), "--only", "clot"],
        "cwd": str(RESOLVED_BATCH.parent),
    },
    "ICLR2026_QMItTyQW92_dto_kd_dynamic_tradeoff_distillation": {
        "short": "dto_kd",
        "cmd": [sys.executable, str(RUNNERS / "dto_kd" / "run_dto_kd_external_professional_gate.py")],
        "cwd": str(RUNNERS / "dto_kd"),
    },
    "ICLR2026_VdLEaGPYWT_sparserl_sparse_cuda_rl": {
        "short": "sparserl",
        "cmd": [sys.executable, str(RUNNERS / "sparserl" / "run_sparserl_professional_gate.py")],
        "cwd": str(RUNNERS / "sparserl"),
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


def blocker_fingerprint(paper: dict[str, Any]) -> str:
    payload = {
        "status": paper.get("status"),
        "blocker_tags": paper.get("blocker_tags"),
        "required_update_nodes": paper.get("required_update_nodes"),
        "verifier_required_update_count": paper.get("verifier_required_update_count"),
        "blocked_verifier_check_count": paper.get("blocked_verifier_check_count"),
        "operational_blocker_count": paper.get("operational_blocker_count"),
        "weak_evidence": paper.get("weak_evidence"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def selected_papers(scope: str, state: dict[str, Any], force: bool) -> list[dict[str, Any]]:
    audit = read_json(STALENESS_AUDIT, {})
    seen_fingerprints = state.get("attempted_gate_fingerprints", {})
    papers = []
    for paper in audit.get("papers", []):
        paper_id = paper.get("paper_id")
        if paper_id not in GATE_COMMANDS:
            continue
        if paper.get("status") == "running_not_explicit_blocker":
            continue
        if scope == "evidence-bound" and paper.get("status") != "explicit_blocker_evidence_bound":
            continue
        if scope == "gpu-recheck" and paper.get("status") != "explicit_blocker_valid_recheck_after_gpu_release":
            continue
        fp = blocker_fingerprint(paper)
        if not force and seen_fingerprints.get(paper_id) == fp:
            continue
        paper["blocker_fingerprint"] = fp
        papers.append(paper)
    return papers


def run_cmd(cmd: list[str], cwd: str, timeout: int) -> dict[str, Any]:
    started = time.time()
    try:
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
            "timeout": False,
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "cwd": cwd,
            "returncode": None,
            "timeout": True,
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }


def run_gate(paper: dict[str, Any], timeout: int) -> dict[str, Any]:
    paper_id = paper["paper_id"]
    command = GATE_COMMANDS[paper_id]
    missing = [part for part in command["cmd"] if part.endswith(".py") and not Path(part).exists()]
    if missing:
        return {
            "paper_id": paper_id,
            "short": command["short"],
            "blocker_fingerprint": paper.get("blocker_fingerprint"),
            "attempted": False,
            "status": "missing_professional_gate_script",
            "missing": missing,
        }
    result = run_cmd(command["cmd"], command["cwd"], timeout)
    return {
        "paper_id": paper_id,
        "short": command["short"],
        "blocker_fingerprint": paper.get("blocker_fingerprint"),
        "attempted": True,
        "status": "professional_gate_executed",
        **result,
    }


def refresh_status() -> dict[str, Any]:
    if not REFRESH_SCRIPT.exists():
        return {"attempted": False, "reason": "missing_refresh_script"}
    return run_cmd([sys.executable, str(REFRESH_SCRIPT)], str(RUN_ROOT), 120)


def render_status(report: dict[str, Any]) -> None:
    lines = [
        "# Professional Gate Sweep Status",
        "",
        f"- Updated: `{report['created_at_utc']}`",
        f"- Status: `{report['status']}`",
        f"- Scope: `{report['scope']}`",
        "- Policy: gate sweeps refresh blockers only; they cannot converge a paper without paper-shaped outputs.",
        f"- Selected papers before cap: `{report['selected_before_cap_count']}`",
        f"- Executed: `{len(report.get('results', []) or [])}`",
        f"- State: `{STATE_PATH}`",
        f"- Report: `{REPORT_PATH}`",
        "",
        "## Results",
        "",
    ]
    for item in report.get("results", []):
        lines.append(
            f"- `{item.get('paper_id')}` short=`{item.get('short')}` "
            f"status=`{item.get('status')}` returncode=`{item.get('returncode')}` "
            f"timeout=`{item.get('timeout')}`"
        )
    if report.get("selected_papers"):
        lines += ["", "## Selected Papers", ""]
        for paper in report["selected_papers"]:
            lines.append(
                f"- `{paper.get('paper_id')}` status=`{paper.get('status')}` "
                f"tags=`{paper.get('blocker_tags')}` fingerprint=`{paper.get('blocker_fingerprint')}`"
            )
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["all", "evidence-bound", "gpu-recheck"], default="evidence-bound")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-count", type=int, default=0, help="0 means no cap")
    parser.add_argument("--timeout-seconds", type=int, default=420)
    args = parser.parse_args()

    state = read_json(STATE_PATH, {})
    state.setdefault("artifact_kind", "professional_gate_sweep_state")
    state.setdefault("attempted_gate_fingerprints", {})
    state.setdefault("attempted_paper_ids", [])
    state.setdefault("attempts", [])

    selected = selected_papers(args.scope, state, args.force)
    selected_before_cap = len(selected)
    if args.max_count:
        selected = selected[: args.max_count]

    report: dict[str, Any] = {
        "artifact_kind": "professional_gate_sweep_report",
        "created_at_utc": utc_now(),
        "status": "manifest_only_no_run_requested",
        "scope": args.scope,
        "selected_before_cap_count": selected_before_cap,
        "selected_papers": selected,
        "policy": {
            "can_converge_papers": False,
            "reduced_or_proxy_evidence_allowed": False,
            "purpose": "periodic blocker refresh from existing paper-specific professional gates",
        },
        "results": [],
    }

    if args.run:
        report["status"] = "executed_professional_gate_sweep"
        report["results"] = [run_gate(paper, args.timeout_seconds) for paper in selected]
        for item in report["results"]:
            if item.get("attempted"):
                paper_id = item["paper_id"]
                if paper_id not in state["attempted_paper_ids"]:
                    state["attempted_paper_ids"].append(paper_id)
                if item.get("blocker_fingerprint"):
                    state["attempted_gate_fingerprints"][paper_id] = item["blocker_fingerprint"]
        state["attempts"].extend(report["results"])
        report["refresh_result"] = refresh_status()

    state["updated_at_utc"] = report["created_at_utc"]
    state["last_report_status"] = report["status"]
    state["last_scope"] = args.scope
    state["last_selected_before_cap_count"] = selected_before_cap
    write_json(STATE_PATH, state)
    write_json(REPORT_PATH, report)
    render_status(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
