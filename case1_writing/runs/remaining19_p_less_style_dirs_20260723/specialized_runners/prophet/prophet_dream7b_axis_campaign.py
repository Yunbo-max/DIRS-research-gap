#!/usr/bin/env python3
"""Prepare Prophet Dream-7B Table 1 axis evidence.

The paper reports a Dream-7B-Instruct axis, but the released Prophet code in
this workspace is LLaDA-oriented: eval_llada.py registers only llada_dist, the
generation scripts instantiate LLaDA, and no Dream-specific model path or
generate/eval wrapper is provided. This manifest converts that gap into exact
operational debt rather than leaving a vague unresolved blocker.
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
CAMPAIGN_DIR = RUNNER_DIR / "dream7b_table1_axis"
MANIFEST_PATH = CAMPAIGN_DIR / "dream7b_axis_campaign.json"
STATUS_MD = CAMPAIGN_DIR / "DREAM7B_AXIS_STATUS.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


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


def repository_evidence() -> dict[str, Any]:
    files = sorted(str(path.relative_to(REPO)) for path in REPO.glob("**/*") if path.is_file() and ".git/" not in str(path))
    dream_hits: list[str] = []
    for path in [REPO / "README.md", REPO / "eval_llada.py", REPO / "generate.py", REPO / "generate_earlyexit.py"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            if "Dream" in line or "dream" in line:
                dream_hits.append(f"{path.name}:{line_no}:{line.strip()[:240]}")
    return {
        "repo": str(REPO),
        "files": files,
        "dream_mentions": dream_hits,
        "llada_only_signals": [
            "eval_llada.py registers model name llada_dist",
            "generate.py and generate_earlyexit.py examples instantiate GSAI-ML/LLaDA-8B-Instruct",
            "no eval_dream.py, Dream-specific generate wrapper, or Dream model identifier is present in the release",
        ],
    }


def blocked_configs() -> list[dict[str, Any]]:
    benchmarks = [
        "MMLU",
        "ARC-C",
        "HellaSwag",
        "TruthfulQA",
        "WinoGrande",
        "PIQA",
        "GSM8K",
        "GPQA",
        "HumanEval",
        "MBPP",
        "Countdown",
        "Sudoku",
    ]
    return [
        {
            "id": "dream_model_identifier_and_loader",
            "paper_role": "Dream-7B-Instruct model axis",
            "status": "blocked_by_missing_exact_dream7b_model_identifier_and_loader",
            "reason": (
                "The paper reports Dream-7B-Instruct, but the released Prophet repo does not name a "
                "Dream model checkpoint or provide a Dream-specific AutoModel/eval wrapper."
            ),
        },
        {
            "id": "dream_generation_function_parity",
            "paper_role": "Dream-7B full-step and Prophet decoding mechanics",
            "status": "blocked_by_missing_dream_generate_and_earlyexit_parity_code",
            "reason": (
                "The generation code is documented and exemplified as LLaDA generation; using it for "
                "Dream without author parity checks could silently change mask IDs, scheduling, or logits semantics."
            ),
        },
        {
            "id": "dream_simple_evals_prompt_scorer_parity",
            "paper_role": "Dream-7B benchmark evaluator",
            "status": "blocked_by_missing_exact_simple_evals_prompt_and_answer_extractor_for_dream",
            "reason": (
                "The paper says LLaDA and Dream follow simple-evals prompts and generated-answer extraction, "
                "but the release only provides an lm-eval LLaDA integration."
            ),
        },
        {
            "id": "dream_table1_full_grid",
            "paper_role": "Dream-7B Table 1 benchmark rows",
            "status": "blocked_until_dream_loader_prompt_scorer_and_gpu_budget_are_resolved",
            "benchmarks": benchmarks,
            "reason": (
                "All Dream Table 1 rows require the exact Dream runner plus enough free GPU memory for "
                "full non-reduced benchmark execution."
            ),
        },
    ]


def build_manifest() -> dict[str, Any]:
    return {
        "artifact_kind": "prophet_dream7b_table1_axis_campaign",
        "created_at_utc": utc_now(),
        "paper_id": "ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding",
        "paper_title": "Diffusion Language Models Know the Answer Before Decoding",
        "convergence_role": (
            "professional artifact plan for Dream-7B Table 1 axis; no run can converge without "
            "exact Dream loader, generation parity, prompt/scorer parity, and GPU memory"
        ),
        "strict_policy": {
            "full_benchmark_required": True,
            "reduced_or_small_runs_allowed_to_converge": False,
            "paper_target_scores_visible_to_loop2": False,
            "exact_dream_model_parity_required": True,
        },
        "repository_evidence": repository_evidence(),
        "gpu_inventory": gpu_inventory(),
        "linked_existing_artifacts": [],
        "runnable_configs": [],
        "config_statuses": {},
        "blocked_configs": blocked_configs(),
    }


def render_status(manifest: dict[str, Any], launch_result: dict[str, Any] | None) -> None:
    lines = [
        "# Prophet Dream-7B Axis Campaign",
        "",
        f"- Updated: `{manifest['created_at_utc']}`",
        "- Policy: exact Dream-7B full-grid artifacts only; no reduced/proxy convergence.",
        f"- Runnable configs: `{len(manifest['runnable_configs'])}`",
        f"- Explicit blockers: `{len(manifest['blocked_configs'])}`",
    ]
    if launch_result:
        lines.append(f"- Launch: `{launch_result}`")
    lines += ["", "## Repository Evidence", ""]
    evidence = manifest["repository_evidence"]
    lines.append(f"- Repo: `{evidence['repo']}`")
    lines.append(f"- Dream mentions: `{len(evidence['dream_mentions'])}`")
    for signal in evidence["llada_only_signals"]:
        lines.append(f"- `{signal}`")
    lines += ["", "## GPU Inventory", ""]
    for gpu in manifest["gpu_inventory"]:
        lines.append(
            f"- GPU `{gpu['index']}` free=`{gpu['memory_free_mib']}` MiB "
            f"used=`{gpu['memory_used_mib']}` MiB util=`{gpu['utilization_gpu_pct']}`%"
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
            "reason": "no_runnable_configs_without_exact_dream7b_loader_and_generation_parity",
        }
    manifest = build_manifest()
    manifest["launch_result"] = launch_result
    write_json(MANIFEST_PATH, manifest)
    render_status(manifest, launch_result)
    print(json.dumps({"manifest": str(MANIFEST_PATH), "status": str(STATUS_MD), "launch_result": launch_result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
