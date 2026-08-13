#!/usr/bin/env python3
"""Prepare Prophet Table 2 acceleration-combination evidence.

Table 2 is not a generic sampler ablation. It asks for SDTT-distilled and
Fast-dLLM-integrated GSM8K rows. The released Prophet repo supplies the
baseline/Prophet LLaDA runner path, but not SDTT checkpoints/training artifacts
or a Fast-dLLM integration. This manifest records the exact operational debt
instead of treating the available Prophet row as the full Table 2 reproduction.
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
REPO = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/Prophet")
CAMPAIGN_DIR = RUNNER_DIR / "table2_acceleration_combinations"
MANIFEST_PATH = CAMPAIGN_DIR / "table2_acceleration_campaign.json"
STATUS_MD = CAMPAIGN_DIR / "TABLE2_ACCELERATION_STATUS.md"
CUSTOM_GSM8K_DIR = RUNNER_DIR / "custom_full_gsm8k_llada8b"


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


def gpu_inventory() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
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


def rows_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))


def linked_existing_artifacts() -> list[dict[str, Any]]:
    summary = read_json(CUSTOM_GSM8K_DIR / "summary.json", {})
    status = read_json(CUSTOM_GSM8K_DIR / "status.json", {})
    aggregates = summary.get("aggregates", {})
    return [
        {
            "id": "table2_llada_teacher_full_step",
            "paper_role": "Table 2a LLaDA teacher and Table 2b baseline row",
            "variant": "baseline",
            "status_path": str(CUSTOM_GSM8K_DIR / "status.json"),
            "summary_path": str(CUSTOM_GSM8K_DIR / "summary.json"),
            "rows_path": str(CUSTOM_GSM8K_DIR / "per_sample_results.jsonl"),
            "row_count": rows_count(CUSTOM_GSM8K_DIR / "per_sample_results.jsonl"),
            "completed_samples": aggregates.get("baseline", {}).get("completed_samples"),
            "full_split_complete": bool(
                aggregates.get("baseline", {}).get("completed_samples")
                and aggregates.get("baseline", {}).get("completed_samples")
                >= (status.get("total_samples") or summary.get("total_samples") or 1319)
            ),
            "note": "Live/full custom GSM8K baseline row; partial rows remain monitoring only.",
        },
        {
            "id": "table2_prophet_ours",
            "paper_role": "Table 2a/2b Prophet row",
            "variant": "prophet",
            "status_path": str(CUSTOM_GSM8K_DIR / "status.json"),
            "summary_path": str(CUSTOM_GSM8K_DIR / "summary.json"),
            "rows_path": str(CUSTOM_GSM8K_DIR / "per_sample_results.jsonl"),
            "row_count": rows_count(CUSTOM_GSM8K_DIR / "per_sample_results.jsonl"),
            "completed_samples": aggregates.get("prophet", {}).get("completed_samples"),
            "full_split_complete": bool(
                aggregates.get("prophet", {}).get("completed_samples")
                and aggregates.get("prophet", {}).get("completed_samples")
                >= (status.get("total_samples") or summary.get("total_samples") or 1319)
            ),
            "note": "Live/full custom GSM8K Prophet row; partial rows remain monitoring only.",
        },
    ]


def blocked_configs() -> list[dict[str, Any]]:
    return [
        {
            "id": "sdtt_distilled_student_row",
            "paper_role": "Table 2a SDTT distilled row",
            "status": "blocked_by_missing_sdtt_training_code_or_distilled_checkpoint",
            "reason": (
                "The Prophet release does not include the preliminary SDTT implementation, "
                "distilled 128-step LLaDA checkpoint, training data recipe, or verification script."
            ),
        },
        {
            "id": "sdtt_plus_prophet_row",
            "paper_role": "Table 2a SDTT + Prophet row",
            "status": "blocked_by_missing_sdtt_checkpoint_and_prophet_integration_path",
            "reason": (
                "Applying Prophet to the SDTT student requires the exact distilled model artifact "
                "and compatibility path for generate_earlyexit.py."
            ),
        },
        {
            "id": "fast_dllm_kv_cache_parallel_row",
            "paper_role": "Table 2b Fast-dLLM row",
            "status": "blocked_by_missing_fast_dllm_code_patch_and_speed_harness",
            "reason": (
                "The local Prophet repo does not contain Fast-dLLM KV-cache/parallel-decoding code, "
                "model patch, timing harness, or exact GSM8K evaluation script."
            ),
        },
        {
            "id": "fast_dllm_plus_prophet_row",
            "paper_role": "Table 2b Fast-dLLM + Prophet row",
            "status": "blocked_by_missing_fast_dllm_prophet_combined_runner",
            "reason": (
                "The paper's combined row needs a Fast-dLLM runner that exposes Prophet's answer-region "
                "confidence monitor and terminates without continuing cache updates."
            ),
        },
    ]


def build_manifest() -> dict[str, Any]:
    linked = linked_existing_artifacts()
    complete_linked = [item for item in linked if item.get("full_split_complete")]
    return {
        "artifact_kind": "prophet_table2_acceleration_combination_campaign",
        "created_at_utc": utc_now(),
        "paper_id": "ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding",
        "paper_title": "Diffusion Language Models Know the Answer Before Decoding",
        "convergence_role": (
            "professional artifact plan for Table 2 acceleration-combination debt; linked "
            "baseline/Prophet GSM8K rows are necessary but insufficient without SDTT and Fast-dLLM rows"
        ),
        "strict_policy": {
            "full_gsm8k_required_for_linked_rows": True,
            "reduced_or_small_runs_allowed_to_converge": False,
            "paper_target_scores_visible_to_loop2": False,
            "external_method_artifacts_required": True,
        },
        "repo": str(REPO),
        "campaign_dir": str(CAMPAIGN_DIR),
        "gpu_inventory": gpu_inventory(),
        "linked_existing_artifacts": linked,
        "linked_existing_complete_count": len(complete_linked),
        "runnable_configs": [],
        "config_statuses": {},
        "blocked_configs": blocked_configs(),
    }


def render_status(manifest: dict[str, Any], launch_result: dict[str, Any] | None) -> None:
    lines = [
        "# Prophet Table 2 Acceleration Campaign",
        "",
        f"- Updated: `{manifest['created_at_utc']}`",
        "- Policy: full GSM8K and exact external-method artifacts only; no reduced/proxy convergence.",
        f"- Linked existing artifacts: `{len(manifest['linked_existing_artifacts'])}`",
        f"- Linked existing complete: `{manifest['linked_existing_complete_count']}`",
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
    lines += ["", "## Linked Existing Artifacts", ""]
    for item in manifest["linked_existing_artifacts"]:
        lines.append(
            f"- `{item['id']}` complete=`{item.get('full_split_complete')}` "
            f"samples=`{item.get('completed_samples')}` rows=`{item.get('row_count')}`"
        )
    lines += ["", "## Explicit Blockers", ""]
    for item in manifest["blocked_configs"]:
        lines.append(f"- `{item['id']}`: `{item['status']}`")
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-next", action="store_true")
    args = parser.parse_args()
    launch_result = None
    if args.launch_next:
        launch_result = {
            "launched": False,
            "reason": "no_runnable_configs_without_sdtt_fastdllm_external_artifacts",
        }
    manifest = build_manifest()
    manifest["launch_result"] = launch_result
    write_json(MANIFEST_PATH, manifest)
    render_status(manifest, launch_result)
    print(json.dumps({"manifest": str(MANIFEST_PATH), "status": str(STATUS_MD), "launch_result": launch_result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
