#!/usr/bin/env python3
"""Launch full GSM8K protocol-repair reruns for Prophet.

These are not reduced/proxy runs. Each config reruns the full GSM8K test split
from DAG/released-artifact repair axes after the verifier found the completed
custom full run did not match the paper-shaped result direction.
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
CAMPAIGN_DIR = RUNNER_DIR / "protocol_repair_full_gsm8k"
MANIFEST_PATH = CAMPAIGN_DIR / "protocol_repair_campaign.json"
STATUS_MD = CAMPAIGN_DIR / "PROTOCOL_REPAIR_STATUS.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
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


def runnable_configs() -> list[dict[str, Any]]:
    common = {
        "paper_role": "GSM8K postcompletion protocol-shape repair rerun",
        "variants": "baseline,prophet",
        "gen_length": 256,
        "steps": 256,
        "block_length": 32,
        "remasking": "low_confidence",
        "constraints_text": "220:Answer",
        "answer_start_offset": 220,
        "prompt_profile": "trajectory_gsm8k_cot",
        "requires_full_split": True,
        "merge_group": "gsm8k_trajectory_prompt_constraint_L256_T256_full_split",
        "dag_repair_axes": [
            "prompt_template_parity",
            "suffix_constraint_semantics",
            "answer_region_start_and_length",
            "released_eval_harness_vs_custom_runner_semantics",
        ],
        "oracle_values_exposed_to_loop2": False,
    }
    configs = [
        {
            **common,
            "id": "gsm8k_trajectory_prompt_constraint_L256_T256",
            "execution_mode": "monolithic_full_split_prefix_and_fallback",
            "shard_start_index": 0,
            "shard_stop_index_exclusive": 1319,
            "start_index": 0,
            "max_samples": None,
        }
    ]
    for start, stop in [(220, 495), (495, 770), (770, 1045), (1045, 1319)]:
        configs.append(
            {
                **common,
                "id": f"gsm8k_trajectory_prompt_constraint_L256_T256_shard_{start:04d}_{stop - 1:04d}",
                "paper_role": "Supplemental full-split shard for GSM8K protocol repair merge",
                "execution_mode": "full_split_shard_for_merged_verifier_artifact",
                "shard_start_index": start,
                "shard_stop_index_exclusive": stop,
                "start_index": start,
                "max_samples": stop - start,
            }
        )
    return configs


def config_status(config: dict[str, Any]) -> dict[str, Any]:
    out_dir = CAMPAIGN_DIR / config["id"]
    status = read_json(out_dir / "status.json", {})
    summary = read_json(out_dir / "summary.json", {})
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
    return {
        "id": config["id"],
        "status": status_label,
        "row_count": row_count,
        "summary_status": summary.get("status"),
        "out_dir": str(out_dir),
        "pid": status.get("pid"),
        "pid_alive": pid_alive,
        "execution_mode": config.get("execution_mode"),
        "shard_start_index": config.get("shard_start_index"),
        "shard_stop_index_exclusive": config.get("shard_stop_index_exclusive"),
        "updated_at_utc": status.get("updated_at_utc") or summary.get("created_at_utc"),
    }


def summarize_merged_rows(rows: list[dict[str, Any]], total_samples: int, variants: list[str], rows_path: Path) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    for row in rows:
        variant = row.get("variant")
        if variant in by_variant:
            by_variant[variant].append(row)

    paired_indices = {
        int(row["sample_index"])
        for row in rows
        if all(
            any(other.get("sample_index") == row.get("sample_index") and other.get("variant") == variant for other in rows)
            for variant in variants
        )
    }
    summary: dict[str, Any] = {
        "artifact_kind": "prophet_protocol_repair_merged_gsm8k_summary",
        "created_at_utc": utc_now(),
        "status": "completed" if len(paired_indices) >= total_samples else "running_or_partial",
        "dataset": "openai/gsm8k main test",
        "total_samples": total_samples,
        "variants_requested": variants,
        "paired_completed_samples": len(paired_indices),
        "rows_path": str(rows_path),
        "aggregates": {},
    }
    for variant, variant_rows in by_variant.items():
        completed = len(variant_rows)
        strict_correct = sum(1 for row in variant_rows if row.get("strict_exact_match"))
        flexible_correct = sum(1 for row in variant_rows if row.get("flexible_exact_match"))
        seconds = [float(row.get("seconds", 0.0)) for row in variant_rows]
        actual_steps = [
            int(row.get("exit_info", {}).get("actual_steps", row.get("steps", 0)))
            for row in variant_rows
        ]
        early_exits = sum(
            1 for row in variant_rows if row.get("exit_info", {}).get("early_exit_triggered")
        )
        summary["aggregates"][variant] = {
            "completed_samples": completed,
            "strict_exact_match": strict_correct / completed if completed else None,
            "flexible_exact_match": flexible_correct / completed if completed else None,
            "strict_correct_count": strict_correct,
            "flexible_correct_count": flexible_correct,
            "mean_seconds": sum(seconds) / completed if completed else None,
            "mean_actual_steps": sum(actual_steps) / completed if completed else None,
            "early_exit_count": early_exits,
            "early_exit_rate": early_exits / completed if completed else None,
        }
    if all(v in summary["aggregates"] for v in ("baseline", "prophet")):
        base = summary["aggregates"]["baseline"]
        prop = summary["aggregates"]["prophet"]
        if base["mean_seconds"] and prop["mean_seconds"]:
            summary["paired_shape"] = {
                "speedup_mean_seconds": base["mean_seconds"] / prop["mean_seconds"],
                "flexible_accuracy_delta": (
                    prop["flexible_exact_match"] - base["flexible_exact_match"]
                    if prop["flexible_exact_match"] is not None
                    and base["flexible_exact_match"] is not None
                    else None
                ),
                "mean_step_reduction": (
                    base["mean_actual_steps"] - prop["mean_actual_steps"]
                    if prop["mean_actual_steps"] is not None
                    and base["mean_actual_steps"] is not None
                    else None
                ),
            }
    return summary


def merge_repair_artifacts(configs: list[dict[str, Any]], statuses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    variants = ["baseline", "prophet"]
    total_samples = 1319
    merged_rows_path = CAMPAIGN_DIR / "merged_per_sample_results.jsonl"
    merged_summary_path = CAMPAIGN_DIR / "merged_summary.json"
    merged_status_path = CAMPAIGN_DIR / "merged_status.json"
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    input_rows: dict[str, int] = {}
    malformed_rows: dict[str, int] = {}
    config_by_id = {config["id"]: config for config in configs}

    for config in configs:
        config_id = config["id"]
        rows_path = CAMPAIGN_DIR / config_id / "per_sample_results.jsonl"
        input_rows[config_id] = 0
        malformed_rows[config_id] = 0
        if not rows_path.exists():
            continue
        with rows_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                input_rows[config_id] += 1
                try:
                    row = json.loads(line)
                    sample_index = int(row.get("sample_index"))
                    variant = row.get("variant")
                except (TypeError, ValueError, json.JSONDecodeError):
                    malformed_rows[config_id] += 1
                    continue
                if variant not in variants or sample_index < 0 or sample_index >= total_samples:
                    malformed_rows[config_id] += 1
                    continue
                row = dict(row)
                row["merge_source_config_id"] = config_id
                row["merge_source_execution_mode"] = config_by_id.get(config_id, {}).get("execution_mode")
                by_key[(sample_index, variant)] = row

    merged_rows = [by_key[key] for key in sorted(by_key)]
    with merged_rows_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize_merged_rows(merged_rows, total_samples, variants, merged_rows_path)
    write_json(merged_summary_path, summary)
    completed_indices = {
        idx
        for idx in range(total_samples)
        if all((idx, variant) in by_key for variant in variants)
    }
    status = {
        "artifact_kind": "prophet_protocol_repair_merged_gsm8k_status",
        "status": summary["status"],
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "total_samples": total_samples,
        "variants": variants,
        "paired_completed_samples": len(completed_indices),
        "completed_sample_indices": len(completed_indices),
        "missing_sample_count": total_samples - len(completed_indices),
        "rows_path": str(merged_rows_path),
        "summary_path": str(merged_summary_path),
        "input_rows_by_config": input_rows,
        "malformed_rows_by_config": malformed_rows,
        "config_statuses": statuses,
        "full_split_requested": True,
        "oracle_values_exposed_to_loop2": False,
    }
    write_json(merged_status_path, status)
    return {
        "status": summary["status"],
        "paired_completed_samples": len(completed_indices),
        "total_samples": total_samples,
        "missing_sample_count": total_samples - len(completed_indices),
        "row_count": len(merged_rows),
        "rows_path": str(merged_rows_path),
        "summary_path": str(merged_summary_path),
        "status_path": str(merged_status_path),
        "input_rows_by_config": input_rows,
        "malformed_rows_by_config": malformed_rows,
        "oracle_values_exposed_to_loop2": False,
    }


def build_manifest() -> dict[str, Any]:
    configs = runnable_configs()
    statuses = {item["id"]: config_status(item) for item in configs}
    merged_artifact = merge_repair_artifacts(configs, statuses)
    return {
        "artifact_kind": "prophet_gsm8k_protocol_repair_campaign",
        "created_at_utc": utc_now(),
        "paper_id": "ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding",
        "paper_title": "Diffusion Language Models Know the Answer Before Decoding",
        "convergence_role": "full-split rerun for verifier-required GSM8K protocol shape repair",
        "strict_policy": {
            "full_split_required": True,
            "reduced_or_small_runs_allowed_to_converge": False,
            "oracle_values_exposed_to_loop2": False,
            "source": "DAG repair axes plus released README/trajectory artifacts",
        },
        "runner": str(CUSTOM_RUNNER),
        "campaign_dir": str(CAMPAIGN_DIR),
        "gpu_inventory": gpu_inventory(),
        "runnable_configs": configs,
        "blocked_configs": [],
        "config_statuses": statuses,
        "merged_artifact": merged_artifact,
    }


def choose_next(manifest: dict[str, Any]) -> dict[str, Any] | None:
    if manifest.get("merged_artifact", {}).get("status") == "completed":
        return None
    for config in manifest["runnable_configs"]:
        status = manifest["config_statuses"].get(config["id"], {}).get("status")
        if status in {"pending", "stopped_without_results", "stopped_partial_needs_resume", None}:
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
        config["variants"],
        "--gen-length",
        str(config["gen_length"]),
        "--steps",
        str(config["steps"]),
        "--block-length",
        str(config["block_length"]),
        "--remasking",
        config["remasking"],
        "--constraints-text",
        config["constraints_text"],
        "--answer-start-offset",
        str(config["answer_start_offset"]),
        "--prompt-profile",
        config["prompt_profile"],
    ]
    if config.get("start_index") is not None:
        cmd.extend(["--start-index", str(config["start_index"])])
    if config.get("max_samples") is not None:
        cmd.extend(["--max-samples", str(config["max_samples"])])
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
        "artifact_kind": "prophet_gsm8k_protocol_repair_config_status",
        "status": "launched",
        "pid": proc.pid,
        "gpu": gpu,
        "config_id": config["id"],
        "cmd": cmd,
        "log_path": str(log_path),
        "out_dir": str(out_dir),
        "full_split_requested": True,
        "execution_mode": config.get("execution_mode"),
        "shard_start_index": config.get("shard_start_index"),
        "shard_stop_index_exclusive": config.get("shard_stop_index_exclusive"),
        "oracle_values_exposed_to_loop2": False,
        "updated_at_utc": utc_now(),
    }
    write_json(out_dir / "status.json", status)
    return status | {"launched": True}


def render_status(manifest: dict[str, Any], launch_result: dict[str, Any] | None) -> None:
    lines = [
        "# Prophet GSM8K Protocol Repair Campaign",
        "",
        f"- Updated: `{manifest['created_at_utc']}`",
        "- Policy: full GSM8K split per repair config; no reduced/proxy convergence.",
        f"- Runnable configs: `{len(manifest['runnable_configs'])}`",
        (
            f"- Merged repair artifact: `{manifest.get('merged_artifact', {}).get('status')}` "
            f"paired=`{manifest.get('merged_artifact', {}).get('paired_completed_samples')}/"
            f"{manifest.get('merged_artifact', {}).get('total_samples')}`"
        ),
    ]
    if launch_result:
        lines.append(f"- Launch: `{launch_result}`")
    lines += ["", "## Config Statuses", ""]
    for status in manifest["config_statuses"].values():
        lines.append(
            f"- `{status['id']}` status=`{status['status']}` rows=`{status['row_count']}` out=`{status['out_dir']}`"
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
