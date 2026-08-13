#!/usr/bin/env python3
"""Resolved-repo professional gates for remaining strict DIRS papers.

This batch runner handles cases where Loop 1 previously knew the required code
surface but failed to encode the local repository path. It binds only
paper-matched repos, records executable support checks, and keeps convergence
blocked until paper-shaped model/data/result grids exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


RUN_ROOT = Path(
    "/tf/notebooks/yunbo/DIRS/case1_writing/runs/"
    "remaining19_p_less_style_dirs_20260723"
)
QUEUE_PATH = RUN_ROOT / "specialized_runner_queue.json"
SUMMARY_PATH = RUN_ROOT / "remaining19_strict_dirs_summary.json"


CONFIGS: list[dict[str, Any]] = [
    {
        "short": "hsd",
        "paper_id": "ICLR2026_LaVrNaBNwM_hsd_lossless_speculative_decoding",
        "paper_run": "iclr2026_lavrnabnwm_hsd_lossless_speculative_decoding",
        "title": "Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding",
        "repo": "/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/Hierarchical-Speculative-Decoding",
        "runner_type": "llm_decoding_acceptance_runner",
        "status": "blocked_by_model_data_hardware_transformers_patch_and_full_hsd_result_grid",
        "code_artifacts": [
            "algorithm.py",
            "transformers/generation/utils.py",
            "transformers/generation/candidate_generator.py",
            "chain-of-thought-hub/gsm8k/eval_speculative_decoding_llm.py",
            "chain-of-thought-hub/gsm8k/compute_speculative_stats.py",
            "EAGLE-3H/eagle_eval.sh",
            "EAGLE-3H/performance_evaluation.py",
        ],
        "script_files": [
            "README.md",
            "algorithm.py",
            "chain-of-thought-hub/gsm8k/eval_speculative_decoding_llm.py",
            "chain-of-thought-hub/gsm8k/compute_speculative_stats.py",
            "chain-of-thought-hub/gsm8k/eval_speculative_qwen.sh",
            "chain-of-thought-hub/gsm8k/eval_speculative_qwen_backward_clever.sh",
            "chain-of-thought-hub/gsm8k/eval_speculative_qwen_multidraft_11.sh",
            "chain-of-thought-hub/gsm8k/eval_speculative_qwen_backward_clever_multidraft_11.sh",
            "transformers/generation/utils.py",
            "transformers/generation/candidate_generator.py",
            "EAGLE-3H/eagle_eval.sh",
            "EAGLE-3H/performance_evaluation.py",
            "EAGLE-3H/compute_speculative_stats_1.py",
        ],
        "runtime_modules": ["torch", "transformers", "datasets", "accelerate", "auto_gptq", "eagle"],
        "paper_hardware": [
            "single NVIDIA H20 96GB for Qwen2.5 experiments",
            "8 H20 GPUs for Llama-3.1-70B extended evaluation",
            "H100 80GB and H200 141GB for supplementary hardware comparisons",
        ],
        "hardware_blockers": [
            "h20_96gb_or_h100_h200_hardware_missing",
            "llama70b_multigpu_h20_grid_missing",
        ],
        "required_paths": {
            "gsm8k_tokenwise_result_jsonl": "chain-of-thought-hub/gsm8k/results/tokenwise.jsonl",
            "gsm8k_hsd_result_jsonl": "chain-of-thought-hub/gsm8k/results/hsd.jsonl",
            "gsm8k_multidraft_hsd_result_jsonl": "chain-of-thought-hub/gsm8k/results/hsd_multidraft_11.jsonl",
            "humaneval_result_jsonl": "chain-of-thought-hub/humaneval/results/hsd.jsonl",
            "cnndm_result_jsonl": "chain-of-thought-hub/cnndm/results/hsd.jsonl",
            "eagle3h_result_dir": "EAGLE-3H/results",
            "table_metric_summary": "results/table_metric_summary.json",
        },
        "required_output_globs": ["**/*result*.json", "**/*stats*.json", "**/*.jsonl", "**/*performance*.json"],
        "result_grid": [
            "Table 1 Qwen2.5 14B/32B/72B single-draft BE/DS/accuracy",
            "Table 2 multi-draft RRS comparison",
            "Table 3 temperature and draft-length ablations",
            "Table 4 target-size and EAGLE-3H integration",
            "Appendix runtime breakdown and lossless verifier checks",
        ],
        "dag_nodes": [
            ("repo_script_gate", "Bind Hierarchical-Speculative-Decoding repo and validate HSD algorithm, transformers patch, GSM8K scripts, and EAGLE-3H scripts."),
            ("model_data_gate", "Materialize Qwen2.5 draft/target models, GSM8K, HumanEval, CNN/DailyMail, EAGLE-3H artifacts, and result JSONL files."),
            ("runtime_hardware_gate", "Install exact transformers patch/runtime and require H20/H100/H200-class traces for paper-shaped decoding benchmarks."),
            ("full_result_grid", "Run tokenwise, blockwise, HSD, multidraft HSD, temperature/draft-length/target-size ablations, EAGLE-3H, and compute BE/DS/accuracy/tokens/sec."),
        ],
    },
    {
        "short": "clot",
        "paper_id": "ICLR2026_P5B97gZwRb_hyperparameter_trajectory_inference_clot",
        "paper_run": "iclr2026_p5b97gzwrb_hyperparameter_trajectory_inference_clot",
        "title": "Hyperparameter Trajectory Inference with Conditional Lagrangian Optimal Transport",
        "repo": "/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/hyperparameter-trajectory-inference",
        "runner_type": "hyperparameter_trajectory_ot_runner",
        "status": "blocked_by_runtime_a100_multiseed_eval_outputs_and_full_hti_result_grid",
        "code_artifacts": [
            "NLOT/train.py",
            "NLOT/lagrangian_ot/metrics.py",
            "NLOT/lagrangian_ot/lagrangian_potentials.py",
            "NLOT/lagrangian_ot/neuraldual.py",
            "hti_scripts/*.sh",
            "DTR-Bench/*.py",
            "quantile_regression/*.py",
            "generative_dropout/*.py",
        ],
        "script_files": [
            "README.md",
            "NLOT/README.md",
            "NLOT/train.py",
            "NLOT/train.yaml",
            "hti_scripts/semicircles.sh",
            "hti_scripts/reward_weighting.sh",
            "hti_scripts/reward_weighting_hinge.sh",
            "hti_scripts/reacher.sh",
            "hti_scripts/ett_quantiles.sh",
            "hti_scripts/generative_dropout.sh",
            "DTR-Bench/run_surrogate_eval.sh",
            "DTR-Bench/run_hinge_surrogate_eval.sh",
            "DTR-Bench/run_reacher_surrogate_eval.sh",
            "DTR-Bench/run_surrogate_ett_quantile_eval.sh",
        ],
        "runtime_modules": ["torch", "jax", "flax", "hydra", "omegaconf", "stable_baselines3", "gymnasium", "mujoco", "wandb"],
        "paper_hardware": ["Azure VM A100 GPU", "Python 3.10", "JAX/Flax/Hydra/PyTorch", "MuJoCo for Reacher"],
        "hardware_blockers": ["a100_professional_gpu_trace_missing"],
        "required_paths": {
            "conditional_semicircles_data": "NLOT/data/conditional_semicircles.pt",
            "reward_weighting_data": "NLOT/data/reward_weighting_data_0_10.pt",
            "reward_weighting_hinge_data": "NLOT/data/reward_weighting_hinge_data.pt",
            "reacher_data": "NLOT/data/reacher_data.pt",
            "quantile_data": "NLOT/data/quantile_data_new.pt",
            "dropout_data": "NLOT/data/diffusion_2moons_dropout.pt",
            "semicircle_surrogate_models": "NLOT/surrogate_models/semicircles",
            "reward_eval_outputs": "DTR-Bench/surrogate_plots_reward_weighting",
            "hinge_eval_outputs": "DTR-Bench/surrogate_plots_hinge",
            "reacher_eval_outputs": "DTR-Bench/surrogate_plots_reacher",
            "quantile_eval_outputs": "DTR-Bench/surrogate_plots_ett_quantile",
            "dropout_eval_outputs": "generative_dropout/results",
        },
        "required_output_globs": ["**/surrogate_models/**/*.pkl", "**/surrogate_plots*/**/*.csv", "**/surrogate_plots*/**/*.json", "**/results/**/*.json", "**/wandb/**/summary.json"],
        "result_grid": [
            "CTI semicircle NLL and circle-distance grid",
            "Cancer linear and hinge reward-weighting average reward",
            "Reacher reward-weighting average reward",
            "ETTm2 quantile interpolation MSE",
            "dropout diffusion interpolation Wasserstein/density metrics",
            "held-out time/hyperparameter and metric-parametrization ablations over seeds 0..19",
        ],
        "dag_nodes": [
            ("repo_script_gate", "Bind hyperparameter-trajectory-inference repo and validate NLOT, hti_scripts, DTR-Bench, quantile, and dropout code surfaces."),
            ("model_data_gate", "Materialize included HTI tensors plus all trained surrogate checkpoints, evaluation plots, wandb/metric exports, and seed logs."),
            ("runtime_hardware_gate", "Install JAX/Flax/Hydra/PyTorch/stable-baselines3/Gymnasium/MuJoCo stack and require A100 GPU-hour traces."),
            ("full_result_grid", "Run seeds 0..19 across semicircle, cancer, reacher, ETTm2, dropout, held-out hyperparameter/time, and metric ablations."),
        ],
    },
    {
        "short": "lpd",
        "paper_id": "ICLR2026_h06l9w1clt_locality_parallel_decoding_ar_image",
        "paper_run": "iclr2026_h06l9w1clt_locality_parallel_decoding_ar_image",
        "title": "Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation",
        "repo": "/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/lpd",
        "runner_type": "diffusion_cache_latency_quality_runner",
        "status": "blocked_by_models_imagenet_geneval_runtime_hardware_and_full_image_generation_grid",
        "code_artifacts": [
            "models/lpd.py",
            "models/mask.py",
            "orders/lpd_order.py",
            "engine.py",
            "main.py",
            "main_cache.py",
            "scripts/eval/*.sh",
            "scripts/train/*.sh",
        ],
        "script_files": [
            "README.md",
            "environment_setup.sh",
            "main.py",
            "main_cache.py",
            "engine.py",
            "models/lpd.py",
            "models/mask.py",
            "orders/lpd_order.py",
            "orders/run_lpd_order.sh",
            "scripts/eval/lpd_l_res256_steps20.sh",
            "scripts/eval/lpd_xl_res256_steps20.sh",
            "scripts/eval/lpd_xxl_res256_steps20.sh",
            "scripts/eval/lpd_l_res512_steps48.sh",
            "scripts/eval/lpd_xl_res512_steps48.sh",
        ],
        "runtime_modules": ["torch", "torchvision", "timm", "numpy", "PIL", "torch_fidelity", "flash_attn", "einops"],
        "paper_hardware": ["A100 bf16 profiling", "8-GPU training/cache path", "ImageNet 256/512 plus GenEval scoring"],
        "hardware_blockers": ["a100_bf16_latency_trace_missing", "eight_gpu_training_or_cache_trace_missing"],
        "required_paths": {
            "vqgan_tokenizer": "tokenizers/vq_ds16_c2i.pt",
            "lpd_l_256_ckpt": "checkpoints/lpd_l_256",
            "lpd_xl_256_ckpt": "checkpoints/lpd_xl_256",
            "lpd_xxl_256_ckpt": "checkpoints/lpd_xxl_256",
            "lpd_l_512_ckpt": "checkpoints/lpd_l_512",
            "lpd_xl_512_ckpt": "checkpoints/lpd_xl_512",
            "generated_orders": "orders/lpd_orders_generated",
            "imagenet_cache": "imagenet_llamagen_cache",
            "eval_outputs": "outputs/eval",
            "geneval_outputs": "outputs/geneval",
        },
        "required_output_globs": ["outputs/**/*.json", "outputs/**/*.csv", "outputs/**/*fid*", "outputs/**/*latency*", "outputs/**/*.png", "samples/**/*.png"],
        "result_grid": [
            "class-conditional ImageNet 256/512 50k-sample FID/IS/precision/recall",
            "text-to-image GenEval score",
            "latency and throughput profiling on A100 bf16",
            "LPD-L/XL/XXL and 20/32/48 step settings",
            "locality-order, mutual-visibility, group-size, and XL 256 ablations",
        ],
        "dag_nodes": [
            ("repo_script_gate", "Bind lpd repo and validate model, mask, order generation, cache, training, and evaluation scripts."),
            ("model_data_gate", "Materialize VQGAN tokenizer, HF LPD checkpoints, ImageNet 256/512, precomputed latents, GenEval, and generated orders."),
            ("runtime_hardware_gate", "Install image-generation runtime and require A100 bf16 latency/throughput plus 8-GPU cache/training traces."),
            ("full_result_grid", "Run ImageNet 50k FID/IS/precision/recall, GenEval, latency/throughput, model-scale/step-count settings, and ablations."),
        ],
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def run_cmd(cmd: list[str], *, cwd: Path | None = None, timeout: int = 90) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout)
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": proc.returncode,
            "timeout": False,
            "seconds": round(elapsed, 3),
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "timeout": True,
            "seconds": timeout,
            "stdout": (exc.stdout or "")[-6000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-6000:] if isinstance(exc.stderr, str) else "",
        }


def package_version(dist_name: str) -> str | None:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None


def import_probe(module_name: str) -> dict[str, Any]:
    return run_cmd([sys.executable, "-c", f"import importlib; m=importlib.import_module({module_name!r}); print(getattr(m, '__version__', 'imported'))"], timeout=45)


def path_size(path: Path) -> tuple[str | None, int]:
    if not path.exists():
        return None, 0
    if path.is_file():
        return str(path.stat().st_size), 1
    size = run_cmd(["du", "-sh", str(path)], timeout=20)["stdout"].split()
    count = run_cmd(["bash", "-lc", f"find {str(path)!r} -type f | wc -l"], timeout=25)
    try:
        file_count = int(count["stdout"].strip())
    except ValueError:
        file_count = 0
    return (size[0] if size else None), file_count


def gpu_rows() -> list[dict[str, Any]]:
    result = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )
    rows = []
    for line in result.get("stdout", "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 8:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_mib": int(float(parts[2])),
                    "memory_used_mib": int(float(parts[3])),
                    "memory_free_mib": int(float(parts[4])),
                    "utilization_gpu_pct": int(float(parts[5])),
                    "temperature_c": int(float(parts[6])),
                    "power_w": float(parts[7]),
                }
            )
    return rows


def script_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    repo = Path(cfg["repo"])
    rows = []
    for rel in cfg["script_files"]:
        path = repo / rel
        row: dict[str, Any] = {"relative_path": rel, "path": str(path), "exists": path.exists()}
        if path.exists() and path.is_file():
            text = read_text(path)
            row["size_bytes"] = path.stat().st_size
            row["line_count"] = len(text.splitlines())
            if path.suffix == ".py":
                row["py_compile"] = run_cmd([sys.executable, "-m", "py_compile", str(path)], timeout=60)
                row["imports"] = sorted(set(re.findall(r"^(?:from|import)\s+([A-Za-z0-9_\.]+)", text, re.M)))
                row["cli_flags"] = sorted(set(re.findall(r"['\"](--[A-Za-z0-9_-]+)['\"]", text)))
            elif path.suffix == ".sh":
                row["shell_syntax"] = run_cmd(["bash", "-n", str(path)], timeout=30)
                row["commands"] = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
            elif path.name.lower().startswith("readme"):
                row["urls"] = sorted(set(re.findall(r"https?://[^)\s]+", text)))
                row["paper_result_mentions"] = [line.strip() for line in text.splitlines() if any(tok in line.lower() for tok in ["table", "fid", "accuracy", "speed", "nll", "reward", "latency", "block efficiency"])]
        rows.append(row)
    all_files = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file() and ".git" not in p.parts)[:800]
    payload = {
        "artifact_kind": f"{cfg['short']}_repo_manifest",
        "created_at_utc": utc_now(),
        "paper_id": cfg["paper_id"],
        "repo": str(repo),
        "repo_exists": repo.exists(),
        "script_files": rows,
        "all_repo_files_excluding_git": all_files,
        "git": {
            "remote": run_cmd(["git", "-C", str(repo), "remote", "-v"], timeout=20),
            "head": run_cmd(["git", "-C", str(repo), "rev-parse", "HEAD"], timeout=20),
        },
        "support_only_findings": [
            "repo/script presence is necessary but not sufficient for convergence",
            "README tables or assets are references, not Loop 2 operational results",
            "full convergence requires paper-shaped outputs matching the required result grid",
        ],
    }
    write_json(Path(cfg["runner_dir"]) / "repo_manifest.json", payload)
    return payload


def environment_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    repo = Path(cfg["repo"])
    modules = cfg["runtime_modules"]
    payload = {
        "artifact_kind": f"{cfg['short']}_environment_manifest",
        "created_at_utc": utc_now(),
        "paper_id": cfg["paper_id"],
        "python": sys.version,
        "gpu_rows": gpu_rows(),
        "packages": {name: package_version(name.replace("_", "-")) or package_version(name) for name in modules},
        "import_probes": {name: import_probe(name) for name in modules},
        "compileall_repo": run_cmd([sys.executable, "-m", "compileall", "-q", str(repo)], timeout=240),
        "professional_hardware_expected_by_dag": cfg["paper_hardware"],
    }
    write_json(Path(cfg["runner_dir"]) / "environment.json", payload)
    return payload


def model_data_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    repo = Path(cfg["repo"])
    exact = {}
    for name, rel in cfg["required_paths"].items():
        path = repo / rel
        size_human, file_count = path_size(path)
        exact[name] = {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir() if path.exists() else False,
            "size_human": size_human,
            "file_count": file_count,
        }
    output_candidates = []
    seen = set()
    for pattern in cfg["required_output_globs"]:
        for path in repo.glob(pattern):
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            size_human, file_count = path_size(path)
            output_candidates.append({"path": key, "is_dir": path.is_dir(), "size_human": size_human, "file_count": file_count})
            if len(output_candidates) >= 200:
                break
    payload = {
        "artifact_kind": f"{cfg['short']}_model_data_manifest",
        "created_at_utc": utc_now(),
        "paper_id": cfg["paper_id"],
        "exact_required_artifacts": exact,
        "verifier_comparable_output_candidates": output_candidates,
        "paper_shaped_outputs_required": cfg["result_grid"],
        "support_only_warning": "Outputs must be generated by this repo/paper protocol; README values and unrelated files cannot converge.",
    }
    write_json(Path(cfg["runner_dir"]) / "model_data_manifest.json", payload)
    return payload


def professional_gate(cfg: dict[str, Any], manifest: dict[str, Any], env: dict[str, Any], model_data: dict[str, Any]) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    gpu = env["gpu_rows"]
    gpu_names = " | ".join(row["name"] for row in gpu)
    has_a100_h100_h200_h20 = any(any(tag in row["name"] for tag in ["A100", "H100", "H200", "H20"]) for row in gpu)
    clean_large_gpu = any(row["memory_free_mib"] >= 40000 and row["utilization_gpu_pct"] < 30 for row in gpu)
    if cfg["short"] in {"hsd", "lpd"} and not has_a100_h100_h200_h20:
        blockers.append({"id": cfg["hardware_blockers"][0], "status": "blocked", "detail": f"Paper expects {cfg['paper_hardware']}; visible GPUs are {gpu_names}."})
    if cfg["short"] == "clot" and not any("A100" in row["name"] for row in gpu):
        blockers.append({"id": cfg["hardware_blockers"][0], "status": "blocked", "detail": f"Paper records A100 VM runs; visible GPUs are {gpu_names}."})
    if len(cfg["hardware_blockers"]) > 1 and not clean_large_gpu:
        blockers.append({"id": cfg["hardware_blockers"][1], "status": "blocked", "detail": f"No clean large-memory professional GPU slot is visible. GPU rows: {gpu}."})

    missing_imports = [name for name, probe in env["import_probes"].items() if probe["returncode"] != 0]
    if missing_imports:
        blockers.append({"id": f"{cfg['short']}_runtime_dependencies_missing", "status": "blocked", "detail": "Missing runtime imports: " + ", ".join(missing_imports)})
    if env["compileall_repo"]["returncode"] != 0:
        blockers.append({"id": f"{cfg['short']}_source_compile_or_patch_check_failed", "status": "blocked", "detail": "Repository compileall did not pass; see environment manifest."})

    missing_required = [name for name, row in model_data["exact_required_artifacts"].items() if not row["exists"]]
    if missing_required:
        blockers.append({"id": f"{cfg['short']}_required_artifacts_missing", "status": "blocked", "detail": "Missing required paper artifacts: " + ", ".join(missing_required)})
    output_count = len(model_data["verifier_comparable_output_candidates"])
    if output_count < 5:
        blockers.append({"id": f"{cfg['short']}_full_result_grid_missing", "status": "blocked", "detail": "No complete verifier-comparable output grid found for: " + "; ".join(cfg["result_grid"])})

    gate = {
        "artifact_kind": f"{cfg['short']}_professional_gate_result",
        "created_at_utc": utc_now(),
        "paper_id": cfg["paper_id"],
        "paper_title": cfg["title"],
        "status": cfg["status"],
        "professional_package_ready": False,
        "convergence_role": "professional operational gate after repo resolution; no reduced/proxy/support-only evidence is promoted",
        "blockers": blockers,
        "support_checks": {
            "repo_discovered_and_encoded": Path(cfg["repo"]).exists(),
            "script_files_checked": len(manifest["script_files"]),
            "source_compile_passed": env["compileall_repo"]["returncode"] == 0,
            "gpu_rows_checked": len(gpu),
            "verifier_output_candidate_count": output_count,
        },
        "next_full_execution_if_unblocked": [
            "materialize all model/data artifacts listed in the paper DAG",
            "install exact runtime stack and paper hardware path",
            "run the full paper benchmark grid, not a smoke or reduced setting",
            "emit raw outputs, metric JSON/CSV, timing/memory/GPU traces, and table/figure summaries",
            "verifier compares result shape to paper tables, figures, appendix, and paragraphs",
        ],
        "paper_shaped_outputs_required": cfg["result_grid"],
    }
    write_json(Path(cfg["runner_dir"]) / "professional_gate_result.json", gate)
    return gate


def ensure_node(dag: dict[str, Any], node: dict[str, Any]) -> None:
    for existing in dag.setdefault("nodes", []):
        if existing.get("id") == node["id"]:
            existing.update(node)
            return
    dag["nodes"].append(node)


def ensure_edge(dag: dict[str, Any], source: str, target: str) -> None:
    edge = [source, target]
    if edge not in dag.setdefault("edges", []):
        dag["edges"].append(edge)


def update_dag(cfg: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    dag_path = Path(cfg["paper_run_dir"]) / "paper_author_gap_dag.json"
    dag = read_json(dag_path)
    repo = Path(cfg["repo"])
    for node in dag.get("nodes", []):
        if node.get("id") == "ops.resolve_repo_code":
            node["repo_paths"] = [str(repo)]
            node["content"] = f"repos={repo}; code_artifacts=" + "; ".join(cfg["code_artifacts"])
    for suffix, content in cfg["dag_nodes"]:
        node_id = f"ops.{cfg['short']}_{suffix}"
        node_type = "operational_execution" if suffix == "full_result_grid" else ("systems_measurement" if suffix == "runtime_hardware_gate" else "operational_dependency")
        ensure_node(dag, {"id": node_id, "content": content, "type": node_type, "skill_role": "paper-specific operational gate"})
    decision_id = f"decision.explicit_blocker_after_{cfg['short']}_preflight"
    ensure_node(
        dag,
        {
            "id": decision_id,
            "content": "If any required repo, model, data, runtime, hardware, raw output, metric, or table/figure artifact is missing, block and feed exact requirements back into Loop 1.",
            "type": "author_reviewer_decision",
            "skill_role": "paper-specific operational gate",
        },
    )
    chain = [
        "ops.resolve_repo_code",
        f"ops.{cfg['short']}_repo_script_gate",
        f"ops.{cfg['short']}_model_data_gate",
        f"ops.{cfg['short']}_runtime_hardware_gate",
        f"ops.{cfg['short']}_full_result_grid",
        "reviewer.require_professional_artifact_package",
    ]
    for src, dst in zip(chain, chain[1:]):
        ensure_edge(dag, src, dst)
    ensure_edge(dag, f"ops.{cfg['short']}_full_result_grid", "reviewer.compare_result_shapes")
    ensure_edge(dag, "reviewer.keep_exact_artifact_debt", decision_id)
    dag.setdefault("previous_loop_updates", []).append(
        {
            "iteration": 3,
            "created_at_utc": utc_now(),
            "source": f"{cfg['short']}_resolved_repo_professional_gate",
            "status": gate["status"],
            "blocker_ids": [b["id"] for b in gate["blockers"]],
            "repo_paths": [str(repo)],
            "converged": False,
        }
    )
    sig_src = json.dumps(dag.get("nodes", []), sort_keys=True) + json.dumps(dag.get("edges", []), sort_keys=True)
    dag["signature"] = hashlib.sha256(sig_src.encode("utf-8")).hexdigest()[:16]
    write_json(dag_path, dag)
    write_json(Path(cfg["paper_run_dir"]) / "paper_author_gap_dag_iter_03.json", dag)
    return dag


def verifier(cfg: dict[str, Any], gate: dict[str, Any], dag: dict[str, Any]) -> dict[str, Any]:
    repo = Path(cfg["repo"])
    checks = [
        {"name": "blind_contract", "status": "pass", "detail": dag.get("blind_contract", {})},
        {"name": "repo_path_encoded", "status": "pass" if str(repo) in json.dumps(dag) else "fail", "detail": [str(repo)]},
        {"name": "reduced_proxy_rejection_gate", "status": "pass", "detail": "repo compile, README tables, assets, and partial candidates cannot converge"},
        {"name": "professional_artifact_package", "status": "blocked", "detail": gate["blockers"]},
        {"name": "result_shape_comparison_ready", "status": "blocked", "detail": "requires full paper-shaped benchmark outputs before comparison to paper evidence channels"},
    ]
    payload = {
        "artifact_kind": f"{cfg['short']}_specialized_verifier",
        "created_at_utc": utc_now(),
        "paper_id": cfg["paper_id"],
        "paper_title": cfg["title"],
        "iteration": 3,
        "converged": False,
        "professional_ready": False,
        "checks": checks,
        "required_updates": [
            {
                "id": f"update.{cfg['short']}_repo_to_operational_gates",
                "reason": gate["status"],
                "success_criteria": [
                    "repo path encoded",
                    "paper-specific model/data materialization gate present",
                    "runtime/hardware gate present",
                    "full result-grid gate present",
                    "verifier compares only operational outputs against paper evidence",
                ],
            }
        ],
        "artifact_paths": {
            "professional_gate": str(Path(cfg["runner_dir"]) / "professional_gate_result.json"),
            "repo_manifest": str(Path(cfg["runner_dir"]) / "repo_manifest.json"),
            "environment": str(Path(cfg["runner_dir"]) / "environment.json"),
            "model_data_manifest": str(Path(cfg["runner_dir"]) / "model_data_manifest.json"),
            "dag_iter_03": str(Path(cfg["paper_run_dir"]) / "paper_author_gap_dag_iter_03.json"),
        },
    }
    write_json(Path(cfg["runner_dir"]) / f"{cfg['short']}_specialized_verifier.json", payload)
    write_json(Path(cfg["paper_run_dir"]) / "verifier_result_iter_03.json", payload)
    return payload


def update_queue_summary(cfg: dict[str, Any], gate: dict[str, Any], verify: dict[str, Any], dag: dict[str, Any]) -> None:
    repo = Path(cfg["repo"])
    queue_obj = read_json(QUEUE_PATH)
    for item in queue_obj.get("queue", []):
        if item.get("paper_id") == cfg["paper_id"]:
            item["priority"] = "high"
            item["professional_blocker"] = gate["status"]
            item["repo_exact_rerun_status"] = "repo_present_waiting_for_professional_artifacts_and_full_result_grid"
            item["repo_paths"] = [str(repo)]
            item["specialized_runner_status"] = gate["status"]
            item["specialized_runner_artifact_dir"] = str(Path(cfg["runner_dir"]))
            statuses = item.setdefault("implementation_statuses", [])
            for status in ["official_repo_discovered_and_encoded", "official_scripts_parsed", "professional_gate_blocked"]:
                if status not in statuses:
                    statuses.append(status)
            item["specialized_runner_evidence"] = {
                "blockers": gate["blockers"],
                "verifier_path": str(Path(cfg["runner_dir"]) / f"{cfg['short']}_specialized_verifier.json"),
                "repo_manifest_path": str(Path(cfg["runner_dir"]) / "repo_manifest.json"),
                "environment_path": str(Path(cfg["runner_dir"]) / "environment.json"),
                "model_data_manifest_path": str(Path(cfg["runner_dir"]) / "model_data_manifest.json"),
            }
            break
    write_json(QUEUE_PATH, queue_obj)

    summary = read_json(SUMMARY_PATH)
    for paper in summary.get("papers", []):
        if paper.get("paper_id") == cfg["paper_id"]:
            paper["final_status"] = "blocked_waiting_for_professional_artifacts_after_dag_update"
            paper["converged"] = False
            paper["repo_paths"] = [str(repo)]
            paper["specialized_runner_status"] = gate["status"]
            paper["professional_blocker"] = gate["status"]
            paper["specialized_runner_artifact_dir"] = str(Path(cfg["runner_dir"]))
            statuses = paper.setdefault("implementation_statuses", [])
            for status in ["official_repo_discovered_and_encoded", "official_scripts_parsed", "professional_gate_blocked"]:
                if status not in statuses:
                    statuses.append(status)
            paper["iterations"] = paper.get("iterations", []) + [
                {
                    "iteration": 3,
                    "dag_signature": dag.get("signature"),
                    "simulation": {
                        "paper_id": cfg["paper_id"],
                        "paper_title": cfg["title"],
                        "created_at_utc": gate["created_at_utc"],
                        "input_contract": dag.get("blind_contract", {}),
                        "paper_text_seen": False,
                        "previous_memory_seen": False,
                        "oracle_results_seen": False,
                        "repo_paths": [str(repo)],
                        "author_decision": "explicit_operational_blocker",
                        "professional_package_ready": False,
                        "professional_package_reason": gate["status"],
                        "reduced_or_proxy_used_for_convergence": False,
                        "raw_artifact_level": "repo_script_dependency_device_preflight_only",
                        "blocker_ids": [b["id"] for b in gate["blockers"]],
                    },
                    "verification": verify,
                }
            ]
            break
    summary["updated_at_utc"] = utc_now()
    summary["final_status"] = "running_professional_two_loop_not_converged"
    write_json(SUMMARY_PATH, summary)


def write_status(cfg: dict[str, Any], gate: dict[str, Any], verify: dict[str, Any], dag: dict[str, Any]) -> None:
    status_path = Path(cfg["runner_dir"]) / f"{cfg['short'].upper()}_SPECIALIZED_STATUS.md"
    lines = [
        f"# {cfg['short'].upper()} Specialized Professional Gate",
        "",
        f"- Paper id: `{cfg['paper_id']}`",
        f"- Title: {cfg['title']}",
        f"- Status: `{gate['status']}`",
        "- Converged: `false`",
        "- Professional ready: `false`",
        f"- DAG signature: `{dag.get('signature')}`",
        f"- Repo: `{cfg['repo']}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in gate["blockers"]:
        lines.append(f"- `{blocker['id']}`: {blocker['detail']}")
    lines += [
        "",
        "## Artifact Paths",
        "",
        f"- Professional gate: `{Path(cfg['runner_dir']) / 'professional_gate_result.json'}`",
        f"- Verifier: `{Path(cfg['runner_dir']) / (cfg['short'] + '_specialized_verifier.json')}`",
        f"- Repo manifest: `{Path(cfg['runner_dir']) / 'repo_manifest.json'}`",
        f"- Environment: `{Path(cfg['runner_dir']) / 'environment.json'}`",
        f"- Model/data manifest: `{Path(cfg['runner_dir']) / 'model_data_manifest.json'}`",
        "",
        "## Verifier Checks",
        "",
    ]
    for check in verify["checks"]:
        lines.append(f"- `{check['name']}`: `{check['status']}`")
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg["paper_run_dir"] = str(RUN_ROOT / "paper_runs" / cfg["paper_run"])
    cfg["runner_dir"] = str(RUN_ROOT / "specialized_runners" / cfg["short"])
    Path(cfg["runner_dir"]).mkdir(parents=True, exist_ok=True)
    manifest = script_manifest(cfg)
    env = environment_manifest(cfg)
    model_data = model_data_manifest(cfg)
    gate = professional_gate(cfg, manifest, env, model_data)
    dag = update_dag(cfg, gate)
    verify = verifier(cfg, gate, dag)
    update_queue_summary(cfg, gate, verify, dag)
    write_status(cfg, gate, verify, dag)
    return {
        "paper_id": cfg["paper_id"],
        "status": gate["status"],
        "converged": False,
        "blocker_count": len(gate["blockers"]),
        "blocker_ids": [b["id"] for b in gate["blockers"]],
        "dag_signature": dag.get("signature"),
        "status_path": str(Path(cfg["runner_dir"]) / f"{cfg['short'].upper()}_SPECIALIZED_STATUS.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional short names or paper ids to run; default runs all configured gates.",
    )
    args = parser.parse_args()
    requested = {item.strip() for item in args.only if item.strip()}
    configs = [
        cfg
        for cfg in CONFIGS
        if not requested or cfg["short"] in requested or cfg["paper_id"] in requested
    ]
    results = [run_one(cfg) for cfg in configs]
    refresh = run_cmd([sys.executable, str(RUN_ROOT / "refresh_longgoal_status.py")], cwd=RUN_ROOT, timeout=120)
    print(
        json.dumps(
            {
                "requested": sorted(requested),
                "results": results,
                "refresh_returncode": refresh["returncode"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
