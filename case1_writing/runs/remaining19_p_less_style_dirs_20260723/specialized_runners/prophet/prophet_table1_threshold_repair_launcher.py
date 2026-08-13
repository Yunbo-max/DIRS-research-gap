#!/usr/bin/env python3
"""Prepare or launch full GSM8K Table 1 threshold-repair candidates.

The completed Table 1-compatible GSM8K run is full-split but still has the
wrong result shape: Prophet exits too late and loses accuracy relative to the
full-step baseline. This campaign gives Loop 2 a real author-style operational
move without exposing paper target values: keep the released Table 1 prompt and
constraint family fixed, vary only the early/mid/late confidence thresholds,
and run each Prophet candidate on the full GSM8K test split. The already
completed full-step baseline is reused as the paired baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNNER_DIR = Path(__file__).resolve().parent
CUSTOM_RUNNER = RUNNER_DIR / "prophet_custom_full_gsm8k_runner.py"
BASELINE_SUMMARY = RUNNER_DIR / "custom_full_gsm8k_llada8b/summary.json"
BASELINE_ROWS = RUNNER_DIR / "custom_full_gsm8k_llada8b/per_sample_results.jsonl"
CAMPAIGN_DIR = RUNNER_DIR / "table1_threshold_repair_full_gsm8k"
MANIFEST_PATH = CAMPAIGN_DIR / "table1_threshold_repair_campaign.json"
STATUS_MD = CAMPAIGN_DIR / "TABLE1_THRESHOLD_REPAIR_STATUS.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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


def baseline_status() -> dict[str, Any]:
    summary = read_json(BASELINE_SUMMARY, {})
    row_count = sum(1 for _ in BASELINE_ROWS.open("r", encoding="utf-8", errors="replace")) if BASELINE_ROWS.exists() else 0
    baseline = summary.get("aggregates", {}).get("baseline", {})
    return {
        "summary_path": str(BASELINE_SUMMARY),
        "rows_path": str(BASELINE_ROWS),
        "summary_status": summary.get("status"),
        "row_count": row_count,
        "baseline_completed_samples": baseline.get("completed_samples"),
        "baseline_mean_steps": baseline.get("mean_actual_steps"),
        "full_split_ready": summary.get("status") == "completed" and baseline.get("completed_samples") == 1319,
    }


def runnable_configs() -> list[dict[str, Any]]:
    return [
        {
            "id": "table1_threshold_relaxed_6_4_2",
            "paper_role": "Table 1 GSM8K confidence-threshold dynamics repair",
            "early_threshold": 6.0,
            "mid_threshold": 4.0,
            "late_threshold": 2.0,
        },
        {
            "id": "table1_threshold_relaxed_5_3_1",
            "paper_role": "Table 1 GSM8K confidence-threshold dynamics repair",
            "early_threshold": 5.0,
            "mid_threshold": 3.0,
            "late_threshold": 1.0,
        },
        {
            "id": "table1_threshold_relaxed_4_2_0p5",
            "paper_role": "Table 1 GSM8K confidence-threshold dynamics repair",
            "early_threshold": 4.0,
            "mid_threshold": 2.0,
            "late_threshold": 0.5,
        },
        {
            "id": "table1_threshold_relaxed_3_1p5_0p5",
            "paper_role": "Table 1 GSM8K confidence-threshold dynamics repair",
            "early_threshold": 3.0,
            "mid_threshold": 1.5,
            "late_threshold": 0.5,
        },
    ]


def config_status(config: dict[str, Any]) -> dict[str, Any]:
    out_dir = CAMPAIGN_DIR / config["id"]
    summary = read_json(out_dir / "summary.json", {})
    status = read_json(out_dir / "status.json", {})
    rows_path = out_dir / "per_sample_results.jsonl"
    row_count = sum(1 for _ in rows_path.open("r", encoding="utf-8", errors="replace")) if rows_path.exists() else 0
    pid_alive = process_alive(status.get("pid"))
    raw_status = status.get("status")
    if summary.get("status") == "completed" or raw_status == "completed":
        status_label = "completed"
    elif pid_alive:
        status_label = "running"
    elif raw_status in {"launched", "starting", "loading_model", "running", "running_or_partial"}:
        status_label = "stopped_partial_needs_resume" if row_count else "stopped_without_results"
    else:
        status_label = raw_status or "pending"
    prophet = summary.get("aggregates", {}).get("prophet", {})
    return {
        "id": config["id"],
        "status": status_label,
        "row_count": row_count,
        "summary_status": summary.get("status"),
        "out_dir": str(out_dir),
        "pid": status.get("pid"),
        "pid_alive": pid_alive,
        "prophet_completed_samples": prophet.get("completed_samples"),
        "prophet_mean_steps": prophet.get("mean_actual_steps"),
        "prophet_flexible_exact_match": prophet.get("flexible_exact_match"),
        "updated_at_utc": status.get("updated_at_utc") or summary.get("created_at_utc"),
    }


def build_manifest() -> dict[str, Any]:
    configs = runnable_configs()
    statuses = {config["id"]: config_status(config) for config in configs}
    return {
        "artifact_kind": "prophet_table1_threshold_repair_campaign",
        "created_at_utc": utc_now(),
        "paper_id": "ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding",
        "paper_title": "Diffusion Language Models Know the Answer Before Decoding",
        "convergence_role": (
            "full-split Table 1 GSM8K threshold-dynamics candidates after primary "
            "Table 1-compatible run was complete but result-shape rejected"
        ),
        "strict_policy": {
            "full_split_required": True,
            "reduced_or_small_runs_allowed_to_converge": False,
            "baseline_reuse_allowed_only_from_completed_table1_compatible_full_split": True,
            "paper_target_values_visible_to_loop2": False,
        },
        "fixed_protocol": {
            "variants": "prophet",
            "baseline_source": str(BASELINE_SUMMARY),
            "model_id": "GSAI-ML/LLaDA-8B-Instruct",
            "dataset": "openai/gsm8k main test",
            "prompt_profile": "official_zero_shot",
            "constraints_text": "200:The|201:answer|202:is",
            "answer_start_offset": 200,
            "gen_length": 256,
            "steps": 256,
            "block_length": 32,
            "remasking": "low_confidence",
        },
        "runner": str(CUSTOM_RUNNER),
        "campaign_dir": str(CAMPAIGN_DIR),
        "baseline_status": baseline_status(),
        "gpu_inventory": gpu_inventory(),
        "runnable_configs": configs,
        "blocked_configs": [],
        "config_statuses": statuses,
    }


def choose_next(manifest: dict[str, Any]) -> dict[str, Any] | None:
    if not manifest.get("baseline_status", {}).get("full_split_ready"):
        return None
    statuses = manifest["config_statuses"]
    for config in manifest["runnable_configs"]:
        if statuses.get(config["id"], {}).get("status") in {
            "pending",
            "stopped_without_results",
            "stopped_partial_needs_resume",
            None,
        }:
            return config
    return None


def launch_config(config: dict[str, Any], gpu: str) -> dict[str, Any]:
    out_dir = CAMPAIGN_DIR / config["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stdout_stderr.log"
    cmd = [
        "python",
        str(CUSTOM_RUNNER),
        "--gpu",
        gpu,
        "--out-dir",
        str(out_dir),
        "--run-label",
        config["id"],
        "--variants",
        "prophet",
        "--gen-length",
        "256",
        "--steps",
        "256",
        "--block-length",
        "32",
        "--remasking",
        "low_confidence",
        "--constraints-text",
        "200:The|201:answer|202:is",
        "--answer-start-offset",
        "200",
        "--prompt-profile",
        "official_zero_shot",
        "--early-threshold",
        str(config["early_threshold"]),
        "--mid-threshold",
        str(config["mid_threshold"]),
        "--late-threshold",
        str(config["late_threshold"]),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("HF_HOME", "/tf/notebooks/.cache/huggingface")
    with log_path.open("ab", buffering=0) as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(RUNNER_DIR),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    status = {
        "artifact_kind": "prophet_table1_threshold_repair_config_status",
        "status": "launched",
        "pid": proc.pid,
        "gpu": gpu,
        "config_id": config["id"],
        "cmd": cmd,
        "log_path": str(log_path),
        "out_dir": str(out_dir),
        "full_split_requested": True,
        "baseline_source": str(BASELINE_SUMMARY),
        "paper_target_values_visible_to_loop2": False,
        "updated_at_utc": utc_now(),
    }
    write_json(out_dir / "status.json", status)
    return status | {"launched": True}


def render_status(manifest: dict[str, Any], launch_result: dict[str, Any] | None) -> None:
    counts: dict[str, int] = {}
    for status in manifest["config_statuses"].values():
        counts[status["status"]] = counts.get(status["status"], 0) + 1
    lines = [
        "# Prophet Table 1 Threshold Repair Campaign",
        "",
        f"- Updated: `{manifest['created_at_utc']}`",
        "- Policy: full GSM8K split per Prophet candidate; no reduced/proxy convergence.",
        f"- Baseline full split ready: `{manifest['baseline_status']['full_split_ready']}`",
        f"- Runnable configs: `{len(manifest['runnable_configs'])}`",
        f"- Status counts: `{counts}`",
    ]
    if launch_result:
        lines.append(f"- Launch: `{launch_result}`")
    lines += ["", "## GPU Inventory", ""]
    for gpu in manifest["gpu_inventory"]:
        lines.append(
            f"- GPU `{gpu['index']}` free=`{gpu['memory_free_mib']}` MiB "
            f"used=`{gpu['memory_used_mib']}` MiB util=`{gpu['utilization_gpu_pct']}`%"
        )
    lines += ["", "## Config Statuses", ""]
    for status in manifest["config_statuses"].values():
        lines.append(
            f"- `{status['id']}` status=`{status['status']}` rows=`{status['row_count']}` "
            f"steps=`{status.get('prophet_mean_steps')}` acc=`{status.get('prophet_flexible_exact_match')}` "
            f"out=`{status['out_dir']}`"
        )
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-next", action="store_true")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--min-free-mib", type=int, default=21000)
    args = parser.parse_args()

    manifest = build_manifest()
    launch_result = None
    if args.launch_next:
        selected_gpu = next((gpu for gpu in manifest["gpu_inventory"] if gpu["index"] == str(args.gpu)), None)
        if not selected_gpu:
            launch_result = {"launched": False, "reason": f"gpu {args.gpu} not found"}
        elif not manifest["baseline_status"]["full_split_ready"]:
            launch_result = {
                "launched": False,
                "reason": "completed_table1_compatible_baseline_missing",
                "baseline_status": manifest["baseline_status"],
            }
        elif int(selected_gpu["memory_free_mib"]) < args.min_free_mib:
            launch_result = {
                "launched": False,
                "reason": "insufficient_free_gpu_memory_for_full_llada8b_job",
                "gpu": selected_gpu,
                "min_free_mib": args.min_free_mib,
            }
        else:
            next_config = choose_next(manifest)
            launch_result = (
                launch_config(next_config, str(args.gpu))
                if next_config
                else {"launched": False, "reason": "no_pending_runnable_configs"}
            )
            manifest = build_manifest()
    manifest["launch_result"] = launch_result
    write_json(MANIFEST_PATH, manifest)
    render_status(manifest, launch_result)
    print(json.dumps({"manifest": str(MANIFEST_PATH), "status": str(STATUS_MD), "launch_result": launch_result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
