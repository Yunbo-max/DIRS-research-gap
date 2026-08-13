#!/usr/bin/env python3
"""Lightweight supervisor for the strict remaining-19 DIRS long goal.

The supervisor does not run reduced experiments and does not alter convergence
criteria. It periodically refreshes live verifiers, runs the strict completion
auditor, and records whether the active non-reduced run is making progress.
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
REFRESH_SCRIPT = RUN_ROOT / "refresh_longgoal_status.py"
LOOP1_SYNC_SCRIPT = RUN_ROOT / "sync_verifier_blockers_to_dags.py"
LOOP1_SYNC_AUDIT = RUN_ROOT / "loop1_dag_repair_audit_20260723.json"
LIVE_VERIFIER = RUN_ROOT / "paper_runs/iclr2026_g88nt4ietg_prophet_dlm_early_commit_decoding/verifier_result_iter_07_live.json"
COMPLETION_AUDIT = RUN_ROOT / "strict_dirs_completion_audit_20260723.json"
STATE_PATH = RUN_ROOT / "strict_dirs_goal_supervisor_state.json"
STATUS_MD = RUN_ROOT / "STRICT_DIRS_GOAL_SUPERVISOR_STATUS.md"
LOG_PATH = RUN_ROOT / "strict_dirs_goal_supervisor.log"
PID_PATH = RUN_ROOT / "strict_dirs_goal_supervisor.pid"
PROPHET_RUNNER_DIR = RUN_ROOT / "specialized_runners/prophet"
CUSTOM_GSM8K_RUNNER = PROPHET_RUNNER_DIR / "prophet_custom_full_gsm8k_runner.py"
CUSTOM_GSM8K_OUT_DIR = PROPHET_RUNNER_DIR / "custom_full_gsm8k_llada8b"
PROTOCOL_REPAIR_LAUNCHER = PROPHET_RUNNER_DIR / "prophet_gsm8k_protocol_repair_launcher.py"
ABLATION_LAUNCHER = PROPHET_RUNNER_DIR / "prophet_ablation_grid_launcher.py"
TABLE1_THRESHOLD_REPAIR_LAUNCHER = PROPHET_RUNNER_DIR / "prophet_table1_threshold_repair_launcher.py"
MULTIBENCHMARK_LAUNCHER = PROPHET_RUNNER_DIR / "prophet_multibenchmark_grid_launcher.py"
GPU_RECHECK_DISPATCHER = RUN_ROOT / "gpu_recheck_dispatcher.py"
MULTI_GPU_SCHEDULER = RUN_ROOT / "multi_gpu_professional_scheduler.py"
POST_GSM8K_LAUNCH_LOG = RUN_ROOT / "strict_dirs_post_gsm8k_launch_attempts.jsonl"
DEFAULT_GPU = "3"
GPU_POOL = "0,1,2,3"
MIN_FULL_LLADA_FREE_MIB = 21000
PROTOCOL_REPAIR_DIR = PROPHET_RUNNER_DIR / "protocol_repair_full_gsm8k"
ABLATION_GRID_DIR = PROPHET_RUNNER_DIR / "ablation_grid_full_gsm8k"
TABLE1_THRESHOLD_REPAIR_DIR = PROPHET_RUNNER_DIR / "table1_threshold_repair_full_gsm8k"
MULTIBENCHMARK_GRID_DIR = PROPHET_RUNNER_DIR / "multibenchmark_table1_full"


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


def append_log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


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


def active_custom_runner_processes() -> list[str]:
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
    return [
        line
        for line in result.stdout.splitlines()
        if "prophet_custom_full_gsm8k_runner.py" in line and "strict_dirs_goal_supervisor.py" not in line
    ]


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
    if result.returncode != 0:
        return []
    gpus = []
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


def row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def campaign_rows(campaign_dir: Path) -> int:
    return sum(row_count(path) for path in campaign_dir.glob("*/per_sample_results.jsonl"))


def selected_gpu() -> dict[str, Any] | None:
    return next((gpu for gpu in gpu_inventory() if gpu["index"] == DEFAULT_GPU), None)


def run_refresh() -> dict[str, Any]:
    started = time.time()
    result = subprocess.run(
        ["python", str(REFRESH_SCRIPT)],
        cwd=str(RUN_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def attempt_post_gsm8k_launch(live: dict[str, Any]) -> dict[str, Any]:
    runner = live.get("live_full_gsm8k_runner", {})
    full_complete = bool(runner.get("full_gsm8k_complete"))
    if not full_complete:
        return {
            "attempted": False,
            "reason": "full_gsm8k_not_complete",
            "paired_completed_samples": runner.get("paired_completed_samples"),
            "total_samples": runner.get("total_samples"),
        }
    active = active_custom_runner_processes()
    if active:
        return {
            "attempted": False,
            "reason": "prophet_custom_runner_process_still_active",
            "active_processes": active,
        }
    launch_sequence = [
        {
            "campaign": "gsm8k_protocol_repair",
            "script": PROTOCOL_REPAIR_LAUNCHER,
            "blocked_statuses": set(),
        },
        {
            "campaign": "table1_threshold_repair",
            "script": TABLE1_THRESHOLD_REPAIR_LAUNCHER,
            "blocked_statuses": set(),
        },
        {
            "campaign": "ablation_grid",
            "script": ABLATION_LAUNCHER,
            "blocked_statuses": set(),
        },
        {
            "campaign": "multibenchmark_grid",
            "script": MULTIBENCHMARK_LAUNCHER,
            "blocked_statuses": {
                "ready_waiting_for_gpu_capacity_and_prompt_scorer_parity_resolution",
            },
        },
    ]
    attempts = []
    for item in launch_sequence:
        script = item["script"]
        if not script.exists():
            attempts.append({"campaign": item["campaign"], "attempted": False, "reason": "missing_launcher", "script": str(script)})
            continue
        started = time.time()
        result = subprocess.run(
            [
                "python",
                str(script),
                "--launch-next",
                "--gpu",
                DEFAULT_GPU,
                "--min-free-mib",
                str(MIN_FULL_LLADA_FREE_MIB),
            ],
            cwd=str(PROPHET_RUNNER_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
        payload = {
            "campaign": item["campaign"],
            "attempted": True,
            "returncode": result.returncode,
            "elapsed_seconds": round(time.time() - started, 3),
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
        attempts.append(payload)
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError:
            parsed = {}
        if parsed.get("launch_result", {}).get("launched"):
            payload["launched"] = True
            return {
                "attempted": True,
                "status": "launched_next_full_campaign",
                "selected_campaign": item["campaign"],
                "attempts": attempts,
            }
        payload["launched"] = False
        payload["launch_result"] = parsed.get("launch_result")
        if parsed.get("launch_result", {}).get("reason") != "no_pending_runnable_configs":
            return {
                "attempted": True,
                "status": "launch_blocked_or_waiting",
                "selected_campaign": item["campaign"],
                "attempts": attempts,
            }
    return {
        "attempted": True,
        "status": "no_post_gsm8k_campaign_launched",
        "attempts": attempts,
    }


def launch_full_gsm8k_resume() -> dict[str, Any]:
    gpu = selected_gpu()
    if not gpu:
        return {
            "attempted": False,
            "reason": f"gpu_{DEFAULT_GPU}_not_found_for_full_gsm8k_resume",
        }
    if int(gpu["memory_free_mib"]) < MIN_FULL_LLADA_FREE_MIB:
        return {
            "attempted": False,
            "reason": "insufficient_free_gpu_memory_for_full_gsm8k_resume",
            "gpu": gpu,
            "min_free_mib": MIN_FULL_LLADA_FREE_MIB,
        }
    CUSTOM_GSM8K_OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now().replace(":", "").replace("-", "")
    log_path = CUSTOM_GSM8K_OUT_DIR / f"stdout_stderr_autoresume_{timestamp}.log"
    cmd = [
        "python",
        str(CUSTOM_GSM8K_RUNNER),
        "--gpu",
        DEFAULT_GPU,
        "--out-dir",
        str(CUSTOM_GSM8K_OUT_DIR),
        "--run-label",
        "full_gsm8k_llada8b_autoresume",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = DEFAULT_GPU
    env.setdefault("HF_HOME", "/tf/notebooks/.cache/huggingface")
    with log_path.open("ab", buffering=0) as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(RUN_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return {
        "attempted": True,
        "status": "resumed_full_gsm8k_runner",
        "launched": True,
        "pid": proc.pid,
        "gpu": DEFAULT_GPU,
        "cmd": cmd,
        "log_path": str(log_path),
    }


def attempt_gpu_blocker_recheck() -> dict[str, Any]:
    if not GPU_RECHECK_DISPATCHER.exists():
        return {
            "attempted": False,
            "reason": "missing_gpu_recheck_dispatcher",
            "dispatcher": str(GPU_RECHECK_DISPATCHER),
        }
    started = time.time()
    result = subprocess.run(
        [
            "python",
            str(GPU_RECHECK_DISPATCHER),
            "--run-all",
            "--gpu",
            DEFAULT_GPU,
            "--min-free-mib",
            str(MIN_FULL_LLADA_FREE_MIB),
        ],
        cwd=str(RUN_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=1200,
        check=False,
    )
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        parsed = {}
    return {
        "attempted": True,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "status": parsed.get("status"),
        "pending_before_count": parsed.get("pending_before_count"),
        "recheck_executed_count": len(parsed.get("recheck_results", []) or []),
        "report_path": str(RUN_ROOT / "gpu_recheck_dispatcher_report.json"),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def attempt_one_paper_gpu_scheduler() -> dict[str, Any]:
    if not MULTI_GPU_SCHEDULER.exists():
        return {
            "attempted": False,
            "reason": "missing_multi_gpu_professional_scheduler",
            "scheduler": str(MULTI_GPU_SCHEDULER),
        }
    started = time.time()
    result = subprocess.run(
        [
            "python",
            str(MULTI_GPU_SCHEDULER),
            "--gpu-pool",
            GPU_POOL,
            "--min-free-mib",
            str(MIN_FULL_LLADA_FREE_MIB),
            "--max-launches",
            "4",
        ],
        cwd=str(RUN_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
        check=False,
    )
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        parsed = {}
    return {
        "attempted": True,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "status": parsed.get("status"),
        "paper_id": parsed.get("paper_id"),
        "launch_count": sum(1 for item in parsed.get("launches", []) if item.get("launched")),
        "active_gpu_claim_count": len(parsed.get("final_active_gpu_claims", []) or []),
        "report_path": str(RUN_ROOT / "multi_gpu_professional_scheduler_report.json"),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def campaign_fingerprint(campaign: dict[str, Any]) -> dict[str, Any]:
    blocked_configs = []
    for item in campaign.get("blocked_configs", []) or []:
        if isinstance(item, dict):
            blocked_configs.append(
                {
                    "id": item.get("id"),
                    "status": item.get("status"),
                }
            )
        else:
            blocked_configs.append(str(item))
    return {
        "status": campaign.get("status"),
        "completed_config_count": campaign.get("completed_config_count"),
        "pending_config_count": campaign.get("pending_config_count"),
        "running_config_count": campaign.get("running_config_count"),
        "runnable_config_count": campaign.get("runnable_config_count"),
        "blocked_configs": blocked_configs,
    }


def professional_campaign_active(*campaigns: dict[str, Any]) -> bool:
    running_markers = ("running", "ready", "waiting_for_gpu_capacity")
    terminal_blocked_markers = ("explicit", "blocked_by_missing", "no_runnable_configs")
    for campaign in campaigns:
        if not isinstance(campaign, dict):
            continue
        status = str(campaign.get("status") or "").lower()
        running_count = int(campaign.get("running_config_count") or 0)
        pending_count = int(campaign.get("pending_config_count") or 0)
        runnable_count = int(campaign.get("runnable_config_count") or 0)
        status_has_running_marker = any(marker in status for marker in running_markers)
        status_is_terminal_blocked = any(marker in status for marker in terminal_blocked_markers)
        if running_count > 0:
            return True
        if (pending_count > 0 or runnable_count > 0) and status_has_running_marker and not status_is_terminal_blocked:
            return True
    return False


def feedback_fingerprint(live: dict[str, Any]) -> dict[str, Any]:
    """Summarize verifier feedback shape without sample-level metric churn.

    This intentionally omits live per-sample counts and paper oracle values.
    The goal is to run Loop 1 sync when the required DAG shape changes, not
    every time a GPU job writes another row.
    """

    risk_audit = live.get("gsm8k_live_shape_risk_audit", {})
    protocol_audit = live.get("gsm8k_protocol_parity_audit", {})
    paper_result = live.get("paper_result_comparison", {})
    payload = {
        "status": live.get("status"),
        "convergence_decision": live.get("convergence_decision"),
        "converged": live.get("converged"),
        "professional_package_ready": live.get("professional_package_ready"),
        "required_updates": live.get("required_updates", []),
        "support_only_until": live.get("support_only_until", []),
        "checks": [
            {
                "name": check.get("name"),
                "status": check.get("status"),
            }
            for check in live.get("checks", [])
        ],
        "paper_result_comparison": {
            "status": paper_result.get("status"),
            "gate_status": paper_result.get("gate_status"),
            "primary_gsm8k_status": paper_result.get("primary_gsm8k_status"),
            "trajectory_status": paper_result.get("trajectory_status"),
            "explicit_debt_statuses": paper_result.get("explicit_debt_statuses"),
            "blockers": paper_result.get("blockers", []),
        },
        "unresolved_professional_debt": live.get("unresolved_professional_debt", []),
        "gsm8k_protocol_repair_campaign": campaign_fingerprint(live.get("gsm8k_protocol_repair_campaign", {})),
        "ablation_grid_campaign": campaign_fingerprint(live.get("ablation_grid_campaign", {})),
        "table1_threshold_repair_campaign": campaign_fingerprint(live.get("table1_threshold_repair_campaign", {})),
        "multibenchmark_grid_campaign": campaign_fingerprint(live.get("multibenchmark_grid_campaign", {})),
        "table2_acceleration_campaign": campaign_fingerprint(live.get("table2_acceleration_campaign", {})),
        "dream7b_axis_campaign": campaign_fingerprint(live.get("dream7b_axis_campaign", {})),
        "gsm8k_live_shape_risk_audit": {
            "status": risk_audit.get("status"),
            "failing_metrics": risk_audit.get("failing_metrics"),
            "repair_axis_ids": risk_audit.get("repair_axis_ids"),
            "can_converge_from_this_audit_alone": risk_audit.get("can_converge_from_this_audit_alone"),
            "do_not_stop_before_full_split": risk_audit.get("do_not_stop_before_full_split"),
        },
        "gsm8k_protocol_parity_audit": {
            "status": protocol_audit.get("status"),
            "finding_statuses": protocol_audit.get("finding_statuses"),
            "covered_repair_nodes": protocol_audit.get("covered_repair_nodes"),
            "can_converge_from_this_audit_alone": protocol_audit.get("can_converge_from_this_audit_alone"),
        },
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return {"signature": signature, "payload": payload}


def attempt_loop1_dag_sync(live: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    fingerprint = feedback_fingerprint(live)
    previous = previous or {}
    previous_signature = previous.get("loop1_dag_sync", {}).get("feedback_signature")
    if previous_signature == fingerprint["signature"] and LOOP1_SYNC_AUDIT.exists():
        return {
            "attempted": False,
            "reason": "verifier_feedback_signature_unchanged",
            "feedback_signature": fingerprint["signature"],
            "audit_path": str(LOOP1_SYNC_AUDIT),
        }
    if not LOOP1_SYNC_SCRIPT.exists():
        return {
            "attempted": False,
            "reason": "missing_loop1_sync_script",
            "script": str(LOOP1_SYNC_SCRIPT),
            "feedback_signature": fingerprint["signature"],
        }
    started = time.time()
    result = subprocess.run(
        ["python", str(LOOP1_SYNC_SCRIPT)],
        cwd=str(RUN_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    parsed_stdout: dict[str, Any]
    try:
        parsed_stdout = json.loads(result.stdout)
    except json.JSONDecodeError:
        parsed_stdout = {}
    audit = read_json(LOOP1_SYNC_AUDIT, {})
    active_paper = None
    for item in audit.get("papers", []) or []:
        if item.get("paper_id") == live.get("paper_id"):
            active_paper = item
            break
    return {
        "attempted": True,
        "returncode": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "feedback_signature": fingerprint["signature"],
        "status": "synced" if result.returncode == 0 else "sync_failed",
        "audit_path": str(LOOP1_SYNC_AUDIT),
        "active_paper": active_paper,
        "parsed_stdout": parsed_stdout,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def attempt_operational_transition(live: dict[str, Any]) -> dict[str, Any]:
    runner = live.get("live_full_gsm8k_runner", {})
    full_complete = bool(runner.get("full_gsm8k_complete"))
    pid_alive = bool(runner.get("pid_alive"))
    active = active_custom_runner_processes()
    if not full_complete:
        if pid_alive or active:
            scheduler = attempt_one_paper_gpu_scheduler()
            return {
                "attempted": False,
                "reason": "full_gsm8k_runner_still_active_and_one_paper_scheduler_checked",
                "paired_completed_samples": runner.get("paired_completed_samples"),
                "total_samples": runner.get("total_samples"),
                "pid": runner.get("pid"),
                "active_process_count": len(active),
                "one_paper_multi_gpu_scheduler": scheduler,
            }
        if not CUSTOM_GSM8K_RUNNER.exists():
            return {
                "attempted": False,
                "reason": "missing_custom_gsm8k_runner",
                "runner": str(CUSTOM_GSM8K_RUNNER),
            }
        return launch_full_gsm8k_resume()
    scheduler = attempt_one_paper_gpu_scheduler()
    gpu_recheck = attempt_gpu_blocker_recheck()
    return {
        "attempted": True,
        "status": "post_gsm8k_one_paper_scheduler_and_blocker_recheck_tick",
        "one_paper_multi_gpu_scheduler": scheduler,
        "gpu_blocker_recheck": gpu_recheck,
    }


def append_launch_log(payload: dict[str, Any]) -> None:
    with POST_GSM8K_LAUNCH_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"updated_at_utc": utc_now(), **payload}, sort_keys=True) + "\n")


def build_state(
    refresh_result: dict[str, Any],
    previous: dict[str, Any] | None,
    post_gsm8k_launch: dict[str, Any],
    loop1_dag_sync: dict[str, Any],
) -> dict[str, Any]:
    live = read_json(LIVE_VERIFIER, {})
    audit = read_json(COMPLETION_AUDIT, {})
    runner = live.get("live_full_gsm8k_runner", {})
    integrity = live.get("live_integrity", {})
    trajectory = live.get("trajectory_dataset_analysis", {})
    protocol_repair = live.get("gsm8k_protocol_repair_campaign", {})
    ablation = live.get("ablation_grid_campaign", {})
    ablation_integrity = live.get("ablation_grid_integrity", {})
    table1_threshold = live.get("table1_threshold_repair_campaign", {})
    multibench = live.get("multibenchmark_grid_campaign", {})
    table2 = live.get("table2_acceleration_campaign", {})
    dream = live.get("dream7b_axis_campaign", {})
    source_parity = live.get("source_parity_blocker_audit", {})
    risk_audit = live.get("gsm8k_live_shape_risk_audit", {})
    protocol_audit = live.get("gsm8k_protocol_parity_audit", {})
    scheduler = post_gsm8k_launch.get("one_paper_multi_gpu_scheduler") or read_json(
        RUN_ROOT / "multi_gpu_professional_scheduler_report.json",
        {},
    )
    paired = int(runner.get("paired_completed_samples") or 0)
    total = int(runner.get("total_samples") or 0)
    rows = int(runner.get("jsonl_rows") or 0)
    trajectory_rows = int(trajectory.get("rows_written") or 0)
    protocol_repair_rows = campaign_rows(PROTOCOL_REPAIR_DIR)
    ablation_grid_rows = campaign_rows(ABLATION_GRID_DIR)
    table1_threshold_repair_rows = campaign_rows(TABLE1_THRESHOLD_REPAIR_DIR)
    multibenchmark_grid_rows = campaign_rows(MULTIBENCHMARK_GRID_DIR)
    previous = previous or {}
    previous_paired = int(previous.get("prophet", {}).get("paired_completed_samples") or 0)
    previous_rows = int(previous.get("prophet", {}).get("jsonl_rows") or 0)
    previous_trajectory_rows = int(previous.get("prophet", {}).get("trajectory_rows_written") or 0)
    previous_protocol_repair_rows = int(previous.get("prophet", {}).get("protocol_repair_rows_written") or 0)
    previous_ablation_grid_rows = int(previous.get("prophet", {}).get("ablation_grid_rows_written") or 0)
    previous_table1_threshold_repair_rows = int(previous.get("prophet", {}).get("table1_threshold_repair_rows_written") or 0)
    previous_multibenchmark_grid_rows = int(previous.get("prophet", {}).get("multibenchmark_grid_rows_written") or 0)
    progress = (
        paired > previous_paired
        or rows > previous_rows
        or trajectory_rows > previous_trajectory_rows
        or protocol_repair_rows > previous_protocol_repair_rows
        or ablation_grid_rows > previous_ablation_grid_rows
        or table1_threshold_repair_rows > previous_table1_threshold_repair_rows
        or multibenchmark_grid_rows > previous_multibenchmark_grid_rows
    )
    pid = runner.get("pid")
    pid_alive = process_alive(pid)
    now = utc_now()
    if progress:
        last_progress_at = now
    else:
        last_progress_at = previous.get("last_progress_at") or now
    status = live.get("status") or "unknown"
    counts = audit.get("counts", {})
    active_campaign = professional_campaign_active(protocol_repair, ablation, table1_threshold, multibench, table2, dream)
    if audit.get("goal_complete"):
        supervisor_status = "goal_complete"
    elif integrity.get("gate_status") == "blocked":
        supervisor_status = "needs_attention_live_jsonl_integrity_failure"
    elif active_campaign:
        supervisor_status = "monitoring_active_professional_campaigns"
    elif "running" in status and pid_alive:
        supervisor_status = "monitoring_running_prophet"
    elif "blocked" in status:
        supervisor_status = "prophet_transitioned_to_explicit_blocker"
    else:
        supervisor_status = "needs_attention_no_running_or_blocked_status"
    return {
        "artifact_kind": "strict_dirs_goal_supervisor_state",
        "updated_at_utc": now,
        "supervisor_status": supervisor_status,
        "refresh_result": refresh_result,
        "post_gsm8k_launch": post_gsm8k_launch,
        "loop1_dag_sync": loop1_dag_sync,
        "last_progress_at": last_progress_at,
        "progress_since_previous_tick": progress,
        "prophet": {
            "status": status,
            "convergence_decision": live.get("convergence_decision"),
            "pid": pid,
            "pid_alive": pid_alive,
            "cuda_visible_devices": runner.get("cuda_visible_devices"),
            "paired_completed_samples": paired,
            "total_samples": total,
            "jsonl_rows": rows,
            "integrity_status": integrity.get("status"),
            "integrity_gate_status": integrity.get("gate_status"),
            "integrity_reasons": integrity.get("reasons"),
            "integrity_report_path": integrity.get("report_path"),
            "integrity_row_count": integrity.get("row_count"),
            "integrity_paired_completed_samples_from_rows": integrity.get("paired_completed_samples_from_rows"),
            "integrity_incomplete_sample_count": integrity.get("incomplete_sample_count"),
            "integrity_duplicate_pair_count": integrity.get("duplicate_pair_count"),
            "integrity_json_parse_error_count": integrity.get("json_parse_error_count"),
            "integrity_summary_consistency": integrity.get("summary_consistency"),
            "estimated_remaining_seconds": runner.get("estimated_remaining_seconds"),
            "full_gsm8k_complete": runner.get("full_gsm8k_complete"),
            "trajectory_status": trajectory.get("status"),
            "trajectory_settings_completed": trajectory.get("settings_completed"),
            "trajectory_total_settings": trajectory.get("total_settings"),
            "trajectory_rows_written": trajectory_rows,
            "protocol_repair_rows_written": protocol_repair_rows,
            "trajectory_status_note": trajectory.get("status_note"),
            "trajectory_complete": trajectory.get("trajectory_complete"),
            "gsm8k_protocol_repair_status": protocol_repair.get("status"),
            "gsm8k_protocol_repair_manifest_path": protocol_repair.get("manifest_path"),
            "gsm8k_protocol_repair_completed_config_count": protocol_repair.get("completed_config_count"),
            "gsm8k_protocol_repair_running_config_count": protocol_repair.get("running_config_count"),
            "gsm8k_protocol_repair_pending_config_count": protocol_repair.get("pending_config_count"),
            "ablation_grid_status": ablation.get("status"),
            "ablation_grid_manifest_path": ablation.get("manifest_path"),
            "ablation_grid_completed_config_count": ablation.get("completed_config_count"),
            "ablation_grid_pending_config_count": ablation.get("pending_config_count"),
            "ablation_grid_rows_written": ablation_grid_rows,
            "table1_threshold_repair_status": table1_threshold.get("status"),
            "table1_threshold_repair_manifest_path": table1_threshold.get("manifest_path"),
            "table1_threshold_repair_completed_config_count": table1_threshold.get("completed_config_count"),
            "table1_threshold_repair_running_config_count": table1_threshold.get("running_config_count"),
            "table1_threshold_repair_pending_config_count": table1_threshold.get("pending_config_count"),
            "table1_threshold_repair_rows_written": table1_threshold_repair_rows,
            "ablation_grid_integrity_status": ablation_integrity.get("status"),
            "ablation_grid_integrity_report_path": ablation_integrity.get("report_path"),
            "ablation_grid_integrity_running_config_count": len(ablation_integrity.get("running_config_ids", []) or []),
            "ablation_grid_integrity_complete_config_count": len(ablation_integrity.get("complete_config_ids", []) or []),
            "ablation_grid_integrity_blocked_config_count": len(ablation_integrity.get("blocked_config_ids", []) or []),
            "multibenchmark_grid_status": multibench.get("status"),
            "multibenchmark_grid_manifest_path": multibench.get("manifest_path"),
            "multibenchmark_grid_completed_config_count": multibench.get("completed_config_count"),
            "multibenchmark_grid_pending_config_count": multibench.get("pending_config_count"),
            "multibenchmark_grid_blocked_config_count": len(multibench.get("blocked_configs", []) or []),
            "multibenchmark_grid_rows_written": multibenchmark_grid_rows,
            "table2_acceleration_status": table2.get("status"),
            "table2_acceleration_manifest_path": table2.get("manifest_path"),
            "table2_linked_existing_complete_count": table2.get("linked_existing_complete_count"),
            "table2_blocked_config_count": len(table2.get("blocked_configs", []) or []),
            "dream7b_axis_status": dream.get("status"),
            "dream7b_axis_manifest_path": dream.get("manifest_path"),
            "dream7b_axis_runnable_config_count": dream.get("runnable_config_count"),
            "dream7b_axis_blocked_config_count": len(dream.get("blocked_configs", []) or []),
            "source_parity_audit_status": source_parity.get("status"),
            "source_parity_audit_report_path": source_parity.get("report_path"),
            "source_parity_audit_blocker_count": len(source_parity.get("blocker_ids", []) or []),
            "gsm8k_live_shape_risk_audit_status": risk_audit.get("status"),
            "gsm8k_live_shape_risk_audit_report_path": risk_audit.get("report_path"),
            "gsm8k_live_shape_risk_audit_loop2_author_can_read": risk_audit.get("loop2_author_can_read"),
            "gsm8k_live_shape_risk_audit_can_converge": risk_audit.get("can_converge_from_this_audit_alone"),
            "gsm8k_live_shape_risk_audit_do_not_stop_before_full_split": risk_audit.get(
                "do_not_stop_before_full_split"
            ),
            "gsm8k_live_shape_risk_audit_failing_metrics": risk_audit.get("failing_metrics"),
            "gsm8k_live_shape_risk_audit_repair_axis_ids": risk_audit.get("repair_axis_ids"),
            "gsm8k_protocol_parity_audit_status": protocol_audit.get("status"),
            "gsm8k_protocol_parity_audit_report_path": protocol_audit.get("report_path"),
            "gsm8k_protocol_parity_audit_loop2_author_can_read": protocol_audit.get("loop2_author_can_read"),
            "gsm8k_protocol_parity_audit_can_converge": protocol_audit.get("can_converge_from_this_audit_alone"),
            "gsm8k_protocol_parity_audit_finding_statuses": protocol_audit.get("finding_statuses"),
            "gsm8k_protocol_parity_audit_covered_repair_nodes": protocol_audit.get("covered_repair_nodes"),
        },
        "multi_gpu_professional_scheduler": {
            "status": scheduler.get("status"),
            "paper_id": scheduler.get("paper_id"),
            "launch_count": scheduler.get("launch_count")
            if "launch_count" in scheduler
            else sum(1 for item in scheduler.get("launches", []) if item.get("launched")),
            "active_gpu_claim_count": scheduler.get("active_gpu_claim_count")
            if "active_gpu_claim_count" in scheduler
            else len(scheduler.get("final_active_gpu_claims", []) or []),
            "report_path": scheduler.get("report_path")
            or str(RUN_ROOT / "multi_gpu_professional_scheduler_report.json"),
        },
        "completion_audit": {
            "goal_complete": audit.get("goal_complete"),
            "final_status": audit.get("final_status"),
            "counts": counts,
            "path": str(COMPLETION_AUDIT),
        },
        "policy": {
            "reduced_or_proxy_convergence_allowed": False,
            "loop2_input": "paper_author_gap_dag.json only",
            "completion_requires": "all 19 accepted or explicitly blocked under professional non-reduced gates",
        },
    }


def write_status_md(state: dict[str, Any]) -> None:
    prophet = state["prophet"]
    counts = state["completion_audit"].get("counts", {})
    lines = [
        "# Strict DIRS Goal Supervisor Status",
        "",
        f"- Updated: `{state['updated_at_utc']}`",
        f"- Supervisor status: `{state['supervisor_status']}`",
        f"- Progress since previous tick: `{state['progress_since_previous_tick']}`",
        f"- Last progress: `{state['last_progress_at']}`",
        "",
        "## Prophet",
        "",
        f"- Status: `{prophet['status']}`",
        f"- PID: `{prophet['pid']}` alive=`{prophet['pid_alive']}`",
        f"- GPU: `{prophet['cuda_visible_devices']}`",
        f"- Samples: `{prophet['paired_completed_samples']}/{prophet['total_samples']}`",
        f"- JSONL rows: `{prophet['jsonl_rows']}`",
        f"- JSONL integrity: `{prophet.get('integrity_status')}` gate=`{prophet.get('integrity_gate_status')}`",
        f"- Integrity rows: `{prophet.get('integrity_row_count')}` paired=`{prophet.get('integrity_paired_completed_samples_from_rows')}` duplicates=`{prophet.get('integrity_duplicate_pair_count')}` parse_errors=`{prophet.get('integrity_json_parse_error_count')}`",
        f"- Integrity report: `{prophet.get('integrity_report_path')}`",
        f"- ETA seconds: `{prophet['estimated_remaining_seconds']}`",
        f"- Trajectory: `{prophet['trajectory_settings_completed']}/{prophet['trajectory_total_settings']}` status=`{prophet['trajectory_status']}`",
        f"- Trajectory rows: `{prophet.get('trajectory_rows_written')}`",
        f"- Protocol repair rows: `{prophet.get('protocol_repair_rows_written')}`",
        f"- Multi-benchmark grid: `{prophet.get('multibenchmark_grid_status')}` completed=`{prophet.get('multibenchmark_grid_completed_config_count')}` pending=`{prophet.get('multibenchmark_grid_pending_config_count')}` blockers=`{prophet.get('multibenchmark_grid_blocked_config_count')}`",
        f"- Multi-benchmark rows: `{prophet.get('multibenchmark_grid_rows_written')}`",
        f"- Table 2 acceleration: `{prophet.get('table2_acceleration_status')}` linked_complete=`{prophet.get('table2_linked_existing_complete_count')}` blockers=`{prophet.get('table2_blocked_config_count')}`",
        f"- Dream-7B axis: `{prophet.get('dream7b_axis_status')}` runnable=`{prophet.get('dream7b_axis_runnable_config_count')}` blockers=`{prophet.get('dream7b_axis_blocked_config_count')}`",
        f"- Source parity audit: `{prophet.get('source_parity_audit_status')}` blockers=`{prophet.get('source_parity_audit_blocker_count')}`",
        f"- Source parity report: `{prophet.get('source_parity_audit_report_path')}`",
        f"- GSM8K live shape risk audit: `{prophet.get('gsm8k_live_shape_risk_audit_status')}` failing_metrics=`{prophet.get('gsm8k_live_shape_risk_audit_failing_metrics')}` loop2_visible=`{prophet.get('gsm8k_live_shape_risk_audit_loop2_author_can_read')}` can_converge=`{prophet.get('gsm8k_live_shape_risk_audit_can_converge')}`",
        f"- GSM8K risk audit report: `{prophet.get('gsm8k_live_shape_risk_audit_report_path')}`",
        f"- GSM8K protocol parity audit: `{prophet.get('gsm8k_protocol_parity_audit_status')}` findings=`{prophet.get('gsm8k_protocol_parity_audit_finding_statuses')}` loop2_visible=`{prophet.get('gsm8k_protocol_parity_audit_loop2_author_can_read')}` can_converge=`{prophet.get('gsm8k_protocol_parity_audit_can_converge')}`",
        f"- GSM8K protocol audit report: `{prophet.get('gsm8k_protocol_parity_audit_report_path')}`",
        f"- GSM8K protocol repair rerun: `{prophet.get('gsm8k_protocol_repair_status')}` completed=`{prophet.get('gsm8k_protocol_repair_completed_config_count')}` running=`{prophet.get('gsm8k_protocol_repair_running_config_count')}` pending=`{prophet.get('gsm8k_protocol_repair_pending_config_count')}`",
        f"- Ablation grid: `{prophet.get('ablation_grid_status')}` completed=`{prophet.get('ablation_grid_completed_config_count')}` pending=`{prophet.get('ablation_grid_pending_config_count')}`",
        f"- Ablation rows: `{prophet.get('ablation_grid_rows_written')}`",
        f"- Table 1 threshold repair: `{prophet.get('table1_threshold_repair_status')}` completed=`{prophet.get('table1_threshold_repair_completed_config_count')}` running=`{prophet.get('table1_threshold_repair_running_config_count')}` pending=`{prophet.get('table1_threshold_repair_pending_config_count')}`",
        f"- Table 1 threshold rows: `{prophet.get('table1_threshold_repair_rows_written')}`",
        f"- Ablation integrity: `{prophet.get('ablation_grid_integrity_status')}` running=`{prophet.get('ablation_grid_integrity_running_config_count')}` complete=`{prophet.get('ablation_grid_integrity_complete_config_count')}` blocked=`{prophet.get('ablation_grid_integrity_blocked_config_count')}`",
        f"- Post-GSM8K launcher: `{state.get('post_gsm8k_launch', {}).get('status') or state.get('post_gsm8k_launch', {}).get('reason')}`",
        f"- Loop1 DAG sync: `{state.get('loop1_dag_sync', {}).get('status') or state.get('loop1_dag_sync', {}).get('reason')}` signature=`{state.get('loop1_dag_sync', {}).get('feedback_signature')}`",
        f"- Loop1 DAG sync audit: `{state.get('loop1_dag_sync', {}).get('audit_path')}`",
        f"- Multi-GPU scheduler: `{state.get('multi_gpu_professional_scheduler', {}).get('status')}` launches=`{state.get('multi_gpu_professional_scheduler', {}).get('launch_count')}` active_claims=`{state.get('multi_gpu_professional_scheduler', {}).get('active_gpu_claim_count')}`",
        f"- GPU blocker recheck: `{state.get('post_gsm8k_launch', {}).get('gpu_blocker_recheck', {}).get('status') or state.get('post_gsm8k_launch', {}).get('gpu_blocker_recheck', {}).get('reason')}`",
        "",
        "## Completion Audit",
        "",
        f"- Goal complete: `{state['completion_audit'].get('goal_complete')}`",
        f"- Accepted: `{counts.get('accepted')}`",
        f"- Explicitly blocked: `{counts.get('explicitly_blocked')}`",
        f"- Running: `{counts.get('running')}`",
        f"- Unresolved: `{counts.get('unresolved')}`",
    ]
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tick() -> dict[str, Any]:
    previous = read_json(STATE_PATH, {})
    refresh = run_refresh()
    live_after_refresh = read_json(LIVE_VERIFIER, {})
    post_gsm8k_launch = attempt_operational_transition(live_after_refresh)
    if post_gsm8k_launch.get("attempted"):
        append_launch_log(post_gsm8k_launch)
        refresh = run_refresh()
        live_after_refresh = read_json(LIVE_VERIFIER, {})
    loop1_dag_sync = attempt_loop1_dag_sync(live_after_refresh, previous)
    if loop1_dag_sync.get("attempted") and loop1_dag_sync.get("returncode") == 0:
        refresh = run_refresh()
    state = build_state(refresh, previous, post_gsm8k_launch, loop1_dag_sync)
    write_json(STATE_PATH, state)
    write_status_md(state)
    append_log(
        "status={status} samples={paired}/{total} rows={rows} audit={counts}".format(
            status=state["supervisor_status"],
            paired=state["prophet"]["paired_completed_samples"],
            total=state["prophet"]["total_samples"],
            rows=state["prophet"]["jsonl_rows"],
            counts=state["completion_audit"].get("counts"),
        )
    )
    return state


def pid_is_running(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return False
    return process_alive(pid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--max-ticks", type=int, default=0, help="0 means run until goal_complete")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if not args.once:
        if pid_is_running(PID_PATH):
            print(json.dumps({"status": "already_running", "pid_path": str(PID_PATH)}, indent=2))
            return
        PID_PATH.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    tick_count = 0
    try:
        while True:
            state = tick()
            tick_count += 1
            print(json.dumps({"tick": tick_count, "state": str(STATE_PATH), "status": state["supervisor_status"]}, indent=2, sort_keys=True))
            if args.once or state["completion_audit"].get("goal_complete"):
                break
            if args.max_ticks and tick_count >= args.max_ticks:
                break
            time.sleep(max(30, args.interval_seconds))
    finally:
        if not args.once and PID_PATH.exists():
            try:
                if PID_PATH.read_text(encoding="utf-8").strip() == str(os.getpid()):
                    PID_PATH.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
