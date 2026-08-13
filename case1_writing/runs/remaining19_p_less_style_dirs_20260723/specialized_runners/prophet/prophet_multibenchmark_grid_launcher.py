#!/usr/bin/env python3
"""Prepare or launch full Prophet Table 1 multi-benchmark jobs.

This campaign is professional debt tracking, not a reduced proxy. Runnable
entries use full benchmark tasks through the released lm-eval harness where a
task ID is available. The manifest also records exact-parity blockers for the
paper's simple-evals prompt/scorer path, planning tasks, and Dream-7B axis.
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
EVAL_LLADA = REPO / "eval_llada.py"
CAMPAIGN_DIR = RUNNER_DIR / "multibenchmark_table1_full"
MANIFEST_PATH = CAMPAIGN_DIR / "multibenchmark_grid_campaign.json"
STATUS_MD = CAMPAIGN_DIR / "MULTIBENCHMARK_GRID_STATUS.md"
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"


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


def paper_table1_tasks() -> list[dict[str, Any]]:
    general_config_candidates = [
        {
            "id": "main_text_L128_B16",
            "gen_length": 128,
            "steps": 128,
            "block_length": 16,
            "source": "main text says L=128 for general tasks; Appendix Table 6 extraction says L=64.",
        },
        {
            "id": "appendix_table6_L64_B16",
            "gen_length": 64,
            "steps": 64,
            "block_length": 16,
            "source": "Appendix Table 6 extraction; conflicts with main-text L=128 sentence.",
        },
    ]
    return [
        {
            "benchmark": "MMLU",
            "lm_eval_task": "mmlu",
            "domain": "general",
            "config_candidates": general_config_candidates,
        },
        {
            "benchmark": "ARC-C",
            "lm_eval_task": "arc_challenge",
            "domain": "general",
            "config_candidates": general_config_candidates,
        },
        {
            "benchmark": "HellaSwag",
            "lm_eval_task": "hellaswag",
            "domain": "general",
            "config_candidates": general_config_candidates,
        },
        {
            "benchmark": "TruthfulQA",
            "lm_eval_task": "truthfulqa",
            "domain": "general",
            "config_candidates": general_config_candidates,
        },
        {
            "benchmark": "WinoGrande",
            "lm_eval_task": "winogrande",
            "domain": "general",
            "config_candidates": general_config_candidates,
        },
        {
            "benchmark": "PIQA",
            "lm_eval_task": "piqa",
            "domain": "general",
            "config_candidates": general_config_candidates,
        },
        {
            "benchmark": "GPQA",
            "lm_eval_task": "gpqa",
            "domain": "math_science",
            "config_candidates": [
                {
                    "id": "table6_L256_B32",
                    "gen_length": 256,
                    "steps": 256,
                    "block_length": 32,
                    "source": "main text and Appendix Table 6 agree on L=256/T=256/B=32 for GPQA.",
                }
            ],
        },
        {
            "benchmark": "HumanEval",
            "lm_eval_task": "humaneval",
            "domain": "code",
            "config_candidates": [
                {
                    "id": "table6_L512_B32",
                    "gen_length": 512,
                    "steps": 512,
                    "block_length": 32,
                    "source": "main text and Appendix Table 6 agree on L=512/T=512/B=32 for code.",
                }
            ],
        },
        {
            "benchmark": "MBPP",
            "lm_eval_task": "mbpp",
            "domain": "code",
            "config_candidates": [
                {
                    "id": "table6_L512_B32",
                    "gen_length": 512,
                    "steps": 512,
                    "block_length": 32,
                    "source": "main text and Appendix Table 6 agree on L=512/T=512/B=32 for code.",
                }
            ],
        },
    ]


def constraints_for(config: dict[str, Any]) -> str:
    gen_length = int(config["gen_length"])
    answer_pos = max(0, gen_length - 56)
    return f"{answer_pos}:The|{answer_pos + 1}:answer|{answer_pos + 2}:is"


def runnable_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for task in paper_table1_tasks():
        for candidate in task["config_candidates"]:
            for variant, early_exit in [("baseline", "false"), ("prophet", "true")]:
                configs.append(
                    {
                        "id": (
                            f"llada8b_{task['benchmark'].lower().replace('-', '_')}_"
                            f"{candidate['id']}_{variant}"
                        ),
                        "paper_role": "Table 1 LLaDA-8B multi-benchmark axis",
                        "benchmark": task["benchmark"],
                        "lm_eval_task": task["lm_eval_task"],
                        "domain": task["domain"],
                        "variant": variant,
                        "model_id": MODEL_ID,
                        "enable_early_exit": early_exit,
                        "gen_length": candidate["gen_length"],
                        "steps": candidate["steps"],
                        "block_length": candidate["block_length"],
                        "constraints_text": constraints_for(candidate),
                        "answer_length": 5,
                        "remasking": "low_confidence",
                        "requires_full_benchmark": True,
                        "candidate_config_source": candidate["source"],
                        "professional_caveat": (
                            "Full lm-eval harness artifact, but not convergence unless "
                            "verifier accepts prompt/extraction parity with paper simple-evals."
                        ),
                    }
                )
    return configs


def linked_existing_artifacts() -> list[dict[str, Any]]:
    custom_dir = RUNNER_DIR / "custom_full_gsm8k_llada8b"
    return [
        {
            "id": "llada8b_gsm8k_custom_full_split",
            "paper_role": "Table 1 LLaDA-8B GSM8K row",
            "status_path": str(custom_dir / "status.json"),
            "summary_path": str(custom_dir / "summary.json"),
            "rows_path": str(custom_dir / "per_sample_results.jsonl"),
            "note": "Live custom full GSM8K runner covers this row and is not duplicated here.",
        }
    ]


def blocked_configs() -> list[dict[str, Any]]:
    return [
        {
            "id": "exact_simple_evals_prompt_and_answer_extractor",
            "paper_role": "Table 1 exact evaluator parity for general/math/code/planning tasks",
            "status": "blocked_until_exact_simple_evals_prompt_scorer_path_is_encoded_or_released",
            "reason": (
                "The paper says it follows simple-evals prompts and extracts generated final answers, "
                "but the released repo only contains lm-eval harness code. lm-eval full runs are useful "
                "operational parity candidates, not exact convergence evidence by themselves."
            ),
        },
        {
            "id": "main_text_vs_appendix_general_budget_conflict",
            "paper_role": "Table 1 general-task L/T/B configuration",
            "status": "blocked_until_verifier_resolves_L128_main_text_vs_L64_appendix_table6",
            "reason": (
                "Main text states L=128 for general tasks, while Appendix Table 6 extraction lists "
                "L=64/T=64/B=16. The campaign records both candidate budgets and requires verifier resolution."
            ),
        },
        {
            "id": "countdown_and_sudoku_simple_evals_planning_axis",
            "paper_role": "Table 1 planning tasks",
            "status": "blocked_by_missing_countdown_sudoku_task_ids_and_exact_8shot_simple_evals_runner",
            "reason": (
                "Countdown and Sudoku are not available in the local lm-eval task registry and the paper "
                "uses 8-shot simple-evals-style evaluation."
            ),
        },
        {
            "id": "dream7b_table1_axis",
            "paper_role": "Table 1 Dream-7B multi-benchmark axis",
            "status": "blocked_until_dream7b_exact_runner_model_loading_and_memory_budget_are_resolved",
            "reason": (
                "The active released path and custom runner are LLaDA-specific. Dream-7B requires exact model "
                "loading, prompt/config parity, and a free GPU memory window."
            ),
        },
    ]


def config_status(config: dict[str, Any]) -> dict[str, Any]:
    out_dir = CAMPAIGN_DIR / config["id"]
    status = read_json(out_dir / "status.json", {})
    pid_alive = process_alive(status.get("pid"))
    result_files = sorted(
        str(path)
        for path in out_dir.rglob("*.json")
        if path.name != "status.json" and not path.name.endswith(".tmp")
    )
    if pid_alive:
        status_label = "running"
    elif status.get("status") == "launched" and not result_files:
        status_label = "stopped_without_results"
    elif result_files:
        status_label = "completed_or_has_results_pending_verifier"
    else:
        status_label = "pending"
    return {
        "id": config["id"],
        "status": status_label,
        "pid": status.get("pid"),
        "pid_alive": pid_alive,
        "out_dir": str(out_dir),
        "result_files": result_files[:20],
        "updated_at_utc": status.get("updated_at_utc"),
    }


def build_model_args(config: dict[str, Any]) -> str:
    return ",".join(
        [
            f"model_path='{config['model_id']}'",
            f"enable_early_exit={config['enable_early_exit']}",
            f"constraints_text=\"{config['constraints_text']}\"",
            f"gen_length={config['gen_length']}",
            f"steps={config['steps']}",
            f"block_length={config['block_length']}",
            f"answer_length={config['answer_length']}",
            f"remasking={config['remasking']}",
        ]
    )


def build_manifest() -> dict[str, Any]:
    configs = runnable_configs()
    statuses = {config["id"]: config_status(config) for config in configs}
    return {
        "artifact_kind": "prophet_table1_multibenchmark_campaign",
        "created_at_utc": utc_now(),
        "paper_id": "ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding",
        "paper_title": "Diffusion Language Models Know the Answer Before Decoding",
        "convergence_role": (
            "professional artifact plan for Table 1 multi-benchmark debt; lm-eval jobs are full "
            "operational parity candidates and are not convergence evidence until verifier accepts "
            "prompt/scorer parity"
        ),
        "strict_policy": {
            "full_benchmark_required": True,
            "reduced_or_small_runs_allowed_to_converge": False,
            "paper_target_scores_visible_to_loop2": False,
            "exact_prompt_scorer_parity_required": True,
        },
        "runner": str(EVAL_LLADA),
        "campaign_dir": str(CAMPAIGN_DIR),
        "gpu_inventory": gpu_inventory(),
        "linked_existing_artifacts": linked_existing_artifacts(),
        "runnable_configs": configs,
        "blocked_configs": blocked_configs(),
        "config_statuses": statuses,
    }


def choose_next(manifest: dict[str, Any]) -> dict[str, Any] | None:
    statuses = manifest["config_statuses"]
    for config in manifest["runnable_configs"]:
        if statuses.get(config["id"], {}).get("status") == "pending":
            return config
    return None


def launch_config(config: dict[str, Any], gpu: str) -> dict[str, Any]:
    out_dir = CAMPAIGN_DIR / config["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stdout_stderr.log"
    cmd = [
        "accelerate",
        "launch",
        str(EVAL_LLADA),
        "--tasks",
        config["lm_eval_task"],
        "--model",
        "llada_dist",
        "--model_args",
        build_model_args(config),
        "--output_path",
        str(out_dir / "lm_eval_outputs"),
        "--log_samples",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("HF_HOME", "/tf/notebooks/.cache/huggingface")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("ab", buffering=0) as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    status = {
        "artifact_kind": "prophet_table1_multibenchmark_config_status",
        "status": "launched",
        "pid": proc.pid,
        "gpu": gpu,
        "config_id": config["id"],
        "cmd": cmd,
        "log_path": str(log_path),
        "out_dir": str(out_dir),
        "updated_at_utc": utc_now(),
    }
    write_json(out_dir / "status.json", status)
    return status | {"launched": True}


def render_status(manifest: dict[str, Any], launch_result: dict[str, Any] | None) -> None:
    lines = [
        "# Prophet Table 1 Multi-Benchmark Campaign",
        "",
        f"- Updated: `{manifest['created_at_utc']}`",
        "- Policy: full benchmark jobs only; no reduced/proxy convergence.",
        f"- Runnable lm-eval configs: `{len(manifest['runnable_configs'])}`",
        f"- Linked existing artifacts: `{len(manifest['linked_existing_artifacts'])}`",
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
        lines.append(f"- `{item['id']}` rows=`{item.get('rows_path')}` summary=`{item.get('summary_path')}`")
    lines += ["", "## Config Statuses", ""]
    for status in manifest["config_statuses"].values():
        lines.append(
            f"- `{status['id']}` status=`{status['status']}` pid=`{status.get('pid')}` out=`{status['out_dir']}`"
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
                "reason": "insufficient_free_gpu_memory_for_full_llada8b_lm_eval_job",
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
