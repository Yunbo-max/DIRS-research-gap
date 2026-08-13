#!/usr/bin/env python3
"""Prepare or launch full Prophet GSM8K ablation-grid jobs.

This is not a reduced proxy runner. Each runnable config points to the full
GSM8K test split through prophet_custom_full_gsm8k_runner.py with its own output
directory. The default mode writes the professional campaign manifest only; a
job is launched only with --launch-next and only when the selected GPU has
enough free memory for another LLaDA-8B process.
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
CAMPAIGN_DIR = RUNNER_DIR / "ablation_grid_full_gsm8k"
MANIFEST_PATH = CAMPAIGN_DIR / "ablation_grid_campaign.json"
STATUS_MD = CAMPAIGN_DIR / "ABLATION_GRID_STATUS.md"


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
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
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
    return {
        "id": config["id"],
        "status": status_label,
        "row_count": row_count,
        "summary_status": summary.get("status"),
        "out_dir": str(out_dir),
        "pid": status.get("pid"),
        "pid_alive": pid_alive,
        "updated_at_utc": status.get("updated_at_utc") or summary.get("created_at_utc"),
    }


def runnable_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for gen_length in (256, 128):
        for steps in (16, 32, 64, 128):
            if steps % (gen_length // 32) != 0:
                continue
            configs.append(
                {
                    "id": f"table3a_static_L{gen_length}_T{steps}",
                    "paper_role": "Table 3a static step-budget ablation",
                    "variants": "baseline",
                    "gen_length": gen_length,
                    "steps": steps,
                    "block_length": 32,
                    "remasking": "low_confidence",
                    "requires_full_split": True,
                }
            )
        full_steps = gen_length
        configs.append(
            {
                "id": f"table3a_prophet_L{gen_length}_T{full_steps}",
                "paper_role": "Table 3a Prophet adaptive-step row",
                "variants": "prophet",
                "gen_length": gen_length,
                "steps": full_steps,
                "block_length": 32,
                "remasking": "low_confidence",
                "requires_full_split": True,
            }
        )
    for remasking in ("random", "low_confidence"):
        configs.append(
            {
                "id": f"table3b_remasking_{remasking}",
                "paper_role": "Table 3b remasking strategy compatibility",
                "variants": "baseline,prophet",
                "gen_length": 128,
                "steps": 128,
                "block_length": 32,
                "remasking": remasking,
                "requires_full_split": True,
            }
        )
    for block_length in (8, 16, 32, 64, 128):
        configs.append(
            {
                "id": f"table4_block_length_{block_length}",
                "paper_role": "Table 4 block-length sensitivity",
                "variants": "baseline,prophet",
                "gen_length": 128,
                "steps": 128,
                "block_length": block_length,
                "remasking": "low_confidence",
                "requires_full_split": True,
            }
        )
    return configs


def blocked_configs() -> list[dict[str, Any]]:
    return [
        {
            "id": "table3b_remasking_top_k_margin",
            "paper_role": "Table 3b remasking strategy compatibility",
            "status": "blocked_by_missing_official_top_k_margin_remasking_implementation",
            "reason": (
                "The current released Prophet generate.py and generate_earlyexit.py support "
                "low_confidence and random remasking only. Hand-rolling top-k margin would not "
                "be exact professional reproduction evidence."
            ),
        },
        {
            "id": "dream7b_axis",
            "paper_role": "Dream-7B Table 1 axis",
            "status": "blocked_until_dream7b_exact_runner_and_memory_budget_are_resolved",
            "reason": (
                "The active code path and current custom runner are LLaDA-specific; adding Dream-7B "
                "needs exact model loading, prompt/config parity, and a free GPU memory window."
            ),
        },
    ]


def build_manifest() -> dict[str, Any]:
    configs = runnable_configs()
    statuses = {item["id"]: config_status(item) for item in configs}
    return {
        "artifact_kind": "prophet_full_ablation_grid_campaign",
        "created_at_utc": utc_now(),
        "paper_id": "ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding",
        "paper_title": "Diffusion Language Models Know the Answer Before Decoding",
        "convergence_role": (
            "professional artifact plan for Table 3a/3b/Table 4 debts; pending configs "
            "are not convergence evidence"
        ),
        "strict_policy": {
            "full_split_required": True,
            "reduced_or_small_runs_allowed_to_converge": False,
            "each_config_uses_distinct_output_dir": True,
        },
        "runner": str(CUSTOM_RUNNER),
        "campaign_dir": str(CAMPAIGN_DIR),
        "gpu_inventory": gpu_inventory(),
        "runnable_configs": configs,
        "blocked_configs": blocked_configs(),
        "config_statuses": statuses,
    }


def choose_next(manifest: dict[str, Any]) -> dict[str, Any] | None:
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
        config["variants"],
        "--gen-length",
        str(config["gen_length"]),
        "--steps",
        str(config["steps"]),
        "--block-length",
        str(config["block_length"]),
        "--remasking",
        config["remasking"],
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
        "artifact_kind": "prophet_full_ablation_grid_config_status",
        "status": "launched",
        "pid": proc.pid,
        "gpu": gpu,
        "config_id": config["id"],
        "cmd": cmd,
        "log_path": str(log_path),
        "out_dir": str(out_dir),
        "full_split_requested": True,
        "updated_at_utc": utc_now(),
    }
    write_json(out_dir / "status.json", status)
    return status | {
        "launched": True,
    }


def render_status(manifest: dict[str, Any], launch_result: dict[str, Any] | None) -> None:
    lines = [
        "# Prophet Full Ablation Grid Campaign",
        "",
        f"- Updated: `{manifest['created_at_utc']}`",
        "- Policy: full GSM8K split per runnable config; no reduced/proxy convergence.",
        f"- Runnable configs: `{len(manifest['runnable_configs'])}`",
        f"- Explicit blockers: `{len(manifest['blocked_configs'])}`",
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
            f"- `{status['id']}` status=`{status['status']}` rows=`{status['row_count']}` out=`{status['out_dir']}`"
        )
    lines += ["", "## Explicit Blockers", ""]
    for item in manifest["blocked_configs"]:
        lines.append(f"- `{item['id']}`: `{item['status']}`")
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
