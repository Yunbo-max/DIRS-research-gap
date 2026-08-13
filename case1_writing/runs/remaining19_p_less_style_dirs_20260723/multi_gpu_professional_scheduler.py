#!/usr/bin/env python3
"""One-paper-first multi-GPU scheduler for the strict DIRS long goal.

The scheduler does not create reduced evidence. It fills idle GPUs with
full-paper Prophet operational nodes before spilling to any other paper. Each
launched job is a distinct paper-evidence target and must still pass verifier
comparison before it can count toward convergence.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent
PROPHET_DIR = RUN_ROOT / "specialized_runners/prophet"
CUSTOM_FULL_STATUS = PROPHET_DIR / "custom_full_gsm8k_llada8b/status.json"
ABLATION_LAUNCHER = PROPHET_DIR / "prophet_ablation_grid_launcher.py"
PROTOCOL_REPAIR_LAUNCHER = PROPHET_DIR / "prophet_gsm8k_protocol_repair_launcher.py"
TABLE1_THRESHOLD_REPAIR_LAUNCHER = PROPHET_DIR / "prophet_table1_threshold_repair_launcher.py"
MULTIBENCHMARK_LAUNCHER = PROPHET_DIR / "prophet_multibenchmark_grid_launcher.py"
REPORT_PATH = RUN_ROOT / "multi_gpu_professional_scheduler_report.json"
STATE_PATH = RUN_ROOT / "multi_gpu_professional_scheduler_state.json"
STATUS_MD = RUN_ROOT / "MULTI_GPU_PROFESSIONAL_SCHEDULER_STATUS.md"
LOCK_PATH = RUN_ROOT / "multi_gpu_professional_scheduler.lock"

PAPER_ID = "ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding"
PAPER_TITLE = "Diffusion Language Models Know the Answer Before Decoding"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def process_alive(pid: Any) -> bool:
    if pid in (None, "", 0):
        return False
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def acquire_lock() -> dict[str, Any]:
    if LOCK_PATH.exists():
        existing = read_json(LOCK_PATH, {})
        if process_alive(existing.get("pid")):
            return {"acquired": False, "reason": "scheduler_lock_active", "lock": existing}
        try:
            LOCK_PATH.unlink()
        except OSError:
            return {"acquired": False, "reason": "stale_lock_unlink_failed", "lock": existing}
    payload = {"pid": os.getpid(), "created_at_utc": utc_now()}
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return {"acquired": False, "reason": "scheduler_lock_race", "lock_path": str(LOCK_PATH)}
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return {"acquired": True, "lock": payload, "lock_path": str(LOCK_PATH)}


def release_lock() -> None:
    if not LOCK_PATH.exists():
        return
    existing = read_json(LOCK_PATH, {})
    if existing.get("pid") == os.getpid():
        LOCK_PATH.unlink()


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
    if result.returncode != 0:
        return gpus
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


def iter_status_files() -> list[Path]:
    paths = [CUSTOM_FULL_STATUS]
    paths.extend((PROPHET_DIR / "protocol_repair_full_gsm8k").glob("*/status.json"))
    paths.extend((PROPHET_DIR / "ablation_grid_full_gsm8k").glob("*/status.json"))
    paths.extend((PROPHET_DIR / "table1_threshold_repair_full_gsm8k").glob("*/status.json"))
    paths.extend((PROPHET_DIR / "multibenchmark_table1_full").glob("*/status.json"))
    return [path for path in paths if path.exists()]


def active_gpu_claims() -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for path in iter_status_files():
        status = read_json(path, {})
        pid = status.get("pid")
        if not process_alive(pid):
            continue
        gpu = status.get("gpu") or status.get("cuda_visible_devices")
        claims.append(
            {
                "pid": pid,
                "gpu": str(gpu) if gpu is not None else None,
                "status": status.get("status"),
                "config_id": status.get("config_id") or status.get("run_label"),
                "status_path": str(path),
            }
        )
    return claims


def run_launcher(script: Path, gpu: str, min_free_mib: int, launch: bool, timeout_seconds: int) -> dict[str, Any]:
    if not script.exists():
        return {"attempted": False, "reason": "missing_launcher", "script": str(script)}
    cmd = [sys.executable, str(script)]
    if launch:
        cmd.extend(["--launch-next", "--gpu", str(gpu), "--min-free-mib", str(min_free_mib)])
    started = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(PROPHET_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        parsed = {}
    launch_result = parsed.get("launch_result")
    return {
        "attempted": True,
        "cmd": cmd,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
        "parsed": parsed,
        "launch_result": launch_result,
        "launched": bool(isinstance(launch_result, dict) and launch_result.get("launched")),
    }


def refresh_manifests(timeout_seconds: int) -> list[dict[str, Any]]:
    return [
        run_launcher(PROTOCOL_REPAIR_LAUNCHER, gpu="", min_free_mib=0, launch=False, timeout_seconds=timeout_seconds),
        run_launcher(ABLATION_LAUNCHER, gpu="", min_free_mib=0, launch=False, timeout_seconds=timeout_seconds),
        run_launcher(TABLE1_THRESHOLD_REPAIR_LAUNCHER, gpu="", min_free_mib=0, launch=False, timeout_seconds=timeout_seconds),
        run_launcher(MULTIBENCHMARK_LAUNCHER, gpu="", min_free_mib=0, launch=False, timeout_seconds=timeout_seconds),
    ]


def schedule(args: argparse.Namespace) -> dict[str, Any]:
    lock = acquire_lock()
    if not lock.get("acquired"):
        report = {
            "artifact_kind": "multi_gpu_professional_scheduler_report",
            "created_at_utc": utc_now(),
            "status": "skipped_lock_not_acquired",
            "lock": lock,
        }
        write_json(REPORT_PATH, report)
        render_status(report)
        return report

    try:
        refresh_results = refresh_manifests(args.launcher_timeout_seconds)
        gpus = gpu_inventory()
        gpu_by_id = {gpu["index"]: gpu for gpu in gpus}
        pool = [item.strip() for item in args.gpu_pool.split(",") if item.strip()]
        initial_claims = active_gpu_claims()
        busy_gpus = {claim["gpu"] for claim in initial_claims if claim.get("gpu")}
        launches: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        launched_count = 0

        for gpu_id in pool:
            gpu = gpu_by_id.get(gpu_id)
            if not gpu:
                skipped.append({"gpu": gpu_id, "reason": "gpu_not_found"})
                continue
            if gpu_id in busy_gpus:
                skipped.append({"gpu": gpu_id, "reason": "gpu_already_claimed_by_active_prophet_node"})
                continue
            if int(gpu["memory_free_mib"]) < args.min_free_mib:
                skipped.append(
                    {
                        "gpu": gpu_id,
                        "reason": "insufficient_free_memory",
                        "memory_free_mib": gpu["memory_free_mib"],
                        "min_free_mib": args.min_free_mib,
                    }
                )
                continue
            if launched_count >= args.max_launches:
                skipped.append({"gpu": gpu_id, "reason": "max_launches_reached"})
                continue
            if args.dry_run:
                skipped.append({"gpu": gpu_id, "reason": "dry_run_would_launch_one_paper_node"})
                continue

            repair = run_launcher(
                PROTOCOL_REPAIR_LAUNCHER,
                gpu_id,
                args.min_free_mib,
                launch=True,
                timeout_seconds=args.launcher_timeout_seconds,
            )
            repair["campaign"] = "prophet_gsm8k_protocol_repair_full"
            launches.append(repair)
            if repair.get("launched"):
                launched_count += 1
                launch_result = repair.get("launch_result", {})
                busy_gpus.add(str(launch_result.get("gpu") or gpu_id))
                continue

            repair_reason = (repair.get("launch_result") or {}).get("reason")
            if repair_reason != "no_pending_runnable_configs":
                skipped.append({"gpu": gpu_id, "reason": "protocol_repair_launcher_blocked", "launch_result": repair.get("launch_result")})
                continue

            table1_threshold = run_launcher(
                TABLE1_THRESHOLD_REPAIR_LAUNCHER,
                gpu_id,
                args.min_free_mib,
                launch=True,
                timeout_seconds=args.launcher_timeout_seconds,
            )
            table1_threshold["campaign"] = "prophet_table1_threshold_repair_full_gsm8k"
            launches.append(table1_threshold)
            if table1_threshold.get("launched"):
                launched_count += 1
                launch_result = table1_threshold.get("launch_result", {})
                busy_gpus.add(str(launch_result.get("gpu") or gpu_id))
                continue

            reason = (table1_threshold.get("launch_result") or {}).get("reason")
            if reason != "no_pending_runnable_configs":
                skipped.append(
                    {
                        "gpu": gpu_id,
                        "reason": "table1_threshold_repair_launcher_blocked",
                        "launch_result": table1_threshold.get("launch_result"),
                    }
                )
                continue

            ablation = run_launcher(
                ABLATION_LAUNCHER,
                gpu_id,
                args.min_free_mib,
                launch=True,
                timeout_seconds=args.launcher_timeout_seconds,
            )
            ablation["campaign"] = "prophet_ablation_grid_full_gsm8k"
            launches.append(ablation)
            if ablation.get("launched"):
                launched_count += 1
                launch_result = ablation.get("launch_result", {})
                busy_gpus.add(str(launch_result.get("gpu") or gpu_id))
                continue

            reason = (ablation.get("launch_result") or {}).get("reason")
            if reason != "no_pending_runnable_configs":
                skipped.append({"gpu": gpu_id, "reason": "ablation_launcher_blocked", "launch_result": ablation.get("launch_result")})
                continue

            multibench = run_launcher(
                MULTIBENCHMARK_LAUNCHER,
                gpu_id,
                args.min_free_mib,
                launch=True,
                timeout_seconds=args.launcher_timeout_seconds,
            )
            multibench["campaign"] = "prophet_multibenchmark_table1_full"
            launches.append(multibench)
            if multibench.get("launched"):
                launched_count += 1
                launch_result = multibench.get("launch_result", {})
                busy_gpus.add(str(launch_result.get("gpu") or gpu_id))
            else:
                skipped.append({"gpu": gpu_id, "reason": "no_one_paper_node_launched", "launch_result": multibench.get("launch_result")})

        refresh_after = refresh_manifests(args.launcher_timeout_seconds)
        final_claims = active_gpu_claims()
        previous_state = read_json(STATE_PATH, {})
        launch_history = list(previous_state.get("launch_history", []))
        seen_history_keys = {
            (
                item.get("config_id"),
                str(item.get("gpu")),
                item.get("pid"),
                item.get("status_path"),
            )
            for item in launch_history
        }
        for item in launches:
            launch_result = item.get("launch_result") or {}
            if not launch_result.get("launched"):
                continue
            event = {
                "created_at_utc": launch_result.get("updated_at_utc") or utc_now(),
                "campaign": item.get("campaign"),
                "config_id": launch_result.get("config_id"),
                "gpu": launch_result.get("gpu"),
                "pid": launch_result.get("pid"),
                "cmd": launch_result.get("cmd"),
                "log_path": launch_result.get("log_path"),
                "out_dir": launch_result.get("out_dir"),
                "status_path": str(Path(launch_result.get("out_dir", "")) / "status.json")
                if launch_result.get("out_dir")
                else None,
                "source": "scheduler_launch_result",
            }
            key = (event.get("config_id"), str(event.get("gpu")), event.get("pid"), event.get("status_path"))
            if key not in seen_history_keys:
                launch_history.append(event)
                seen_history_keys.add(key)
        for claim in final_claims:
            status_path = claim.get("status_path") or ""
            if "custom_full_gsm8k_llada8b" in status_path:
                continue
            event = {
                "created_at_utc": utc_now(),
                "campaign": (
                    "prophet_gsm8k_protocol_repair_full"
                    if "protocol_repair_full_gsm8k" in status_path
                    else (
                        "prophet_ablation_grid_full_gsm8k"
                        if "ablation_grid_full_gsm8k" in status_path
                        else (
                            "prophet_table1_threshold_repair_full_gsm8k"
                            if "table1_threshold_repair_full_gsm8k" in status_path
                            else "prophet_multibenchmark_table1_full"
                        )
                    )
                ),
                "config_id": claim.get("config_id"),
                "gpu": claim.get("gpu"),
                "pid": claim.get("pid"),
                "status_path": status_path,
                "source": "reconstructed_from_active_status",
            }
            key = (event.get("config_id"), str(event.get("gpu")), event.get("pid"), event.get("status_path"))
            if key not in seen_history_keys:
                launch_history.append(event)
                seen_history_keys.add(key)
        if launched_count:
            report_status = "launched_one_paper_nodes"
        elif skipped and all(item.get("reason") == "gpu_already_claimed_by_active_prophet_node" for item in skipped):
            report_status = "all_pool_gpus_claimed_by_active_one_paper_nodes"
        else:
            report_status = "no_launches"
        report = {
            "artifact_kind": "multi_gpu_professional_scheduler_report",
            "created_at_utc": utc_now(),
            "status": report_status,
            "paper_id": PAPER_ID,
            "paper_title": PAPER_TITLE,
            "policy": {
                "paper_scheduling": "one_paper_first",
                "gpu_pool": pool,
                "reduced_or_proxy_convergence_allowed": False,
                "loop2_input": "paper_author_gap_dag.json only",
                "different_paper_spillover_rule": "only when active paper has no runnable GPU-parallel nodes left",
            },
            "min_free_mib": args.min_free_mib,
            "max_launches": args.max_launches,
            "dry_run": args.dry_run,
            "gpu_inventory": gpus,
            "initial_active_gpu_claims": initial_claims,
            "final_active_gpu_claims": final_claims,
            "refresh_before": refresh_results,
            "launches": launches,
            "skipped_gpus": skipped,
            "refresh_after": refresh_after,
            "launch_history": launch_history,
            "launch_history_count": len(launch_history),
            "recent_launch_history": launch_history[-12:],
            "report_path": str(REPORT_PATH),
            "state_path": str(STATE_PATH),
            "status_path": str(STATUS_MD),
        }
        write_json(REPORT_PATH, report)
        write_json(STATE_PATH, report)
        render_status(report)
        return report
    finally:
        release_lock()


def render_status(report: dict[str, Any]) -> None:
    lines = [
        "# Multi-GPU Professional Scheduler",
        "",
        f"- Updated: `{report.get('created_at_utc')}`",
        f"- Status: `{report.get('status')}`",
        f"- Paper: `{report.get('paper_title', PAPER_TITLE)}`",
        "- Policy: one paper first; full-paper nodes only; no reduced/proxy convergence.",
        f"- GPU pool: `{','.join(report.get('policy', {}).get('gpu_pool', []))}`",
        f"- Launches this tick: `{sum(1 for item in report.get('launches', []) if item.get('launched'))}`",
        f"- Cumulative launch history: `{report.get('launch_history_count')}`",
        "",
        "## Active GPU Claims",
        "",
    ]
    for claim in report.get("final_active_gpu_claims", []) or report.get("initial_active_gpu_claims", []):
        lines.append(
            f"- GPU `{claim.get('gpu')}` pid=`{claim.get('pid')}` config=`{claim.get('config_id')}` status=`{claim.get('status')}`"
        )
    lines += ["", "## Launch Attempts", ""]
    for item in report.get("launches", []):
        launch_result = item.get("launch_result") or {}
        lines.append(
            f"- `{item.get('campaign')}` launched=`{item.get('launched')}` "
            f"config=`{launch_result.get('config_id')}` gpu=`{launch_result.get('gpu')}` "
            f"reason=`{launch_result.get('reason')}` returncode=`{item.get('returncode')}`"
        )
    lines += ["", "## Recent Launch History", ""]
    for item in report.get("recent_launch_history", []):
        lines.append(
            f"- `{item.get('campaign')}` config=`{item.get('config_id')}` gpu=`{item.get('gpu')}` pid=`{item.get('pid')}` source=`{item.get('source')}`"
        )
    lines += ["", "## Skipped GPUs", ""]
    for item in report.get("skipped_gpus", []):
        lines.append(f"- GPU `{item.get('gpu')}` reason=`{item.get('reason')}`")
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-pool", default="0,1,2,3")
    parser.add_argument("--min-free-mib", type=int, default=21000)
    parser.add_argument("--max-launches", type=int, default=4)
    parser.add_argument("--launcher-timeout-seconds", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = schedule(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
