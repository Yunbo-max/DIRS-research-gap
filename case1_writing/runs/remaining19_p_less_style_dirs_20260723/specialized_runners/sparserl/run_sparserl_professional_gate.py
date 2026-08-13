#!/usr/bin/env python3
"""SparseRL professional gate runner for the remaining-19 DIRS loop.

This runner intentionally separates real operational evidence from convergence.
The DAG-only author simulation can use the DAG and encoded repo path to execute
repo code, but the verifier only accepts paper-shaped evidence when the required
model, datasets, baselines, metrics, and table/figure channels are present.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/SparseRL")
RUN_ROOT = Path(
    "/tf/notebooks/yunbo/DIRS/case1_writing/runs/"
    "remaining19_p_less_style_dirs_20260723/specialized_runners/sparserl"
)
PAPER_RUN = Path(
    "/tf/notebooks/yunbo/DIRS/case1_writing/runs/"
    "remaining19_p_less_style_dirs_20260723/paper_runs/"
    "iclr2026_vdleagpywt_sparserl_sparse_cuda_rl"
)
DAG_PATH = PAPER_RUN / "paper_author_gap_dag.json"
SPARSERL_HARD_BLOCKER = (
    "blocked_by_generated_policy_passatk_spmm_baselines_dataset_scale_and_v100_a100_grid"
)


CSR_KERNEL = r"""
__global__ void spmv_kernel(int m, const int* row_ptr, const int* col_idx, const float* val, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        float sum = 0.0f;
        int start = row_ptr[row];
        int end = row_ptr[row + 1];
        for (int idx = start; idx < end; ++idx) {
            sum += val[idx] * x[col_idx[idx]];
        }
        y[row] = sum;
    }
}
""".strip()

ELL_KERNEL = r"""
__global__ void spmv_kernel(int m, int ell_width, const int* col_idx, const float* val, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        float sum = 0.0f;
        for (int k = 0; k < ell_width; ++k) {
            int idx = row * ell_width + k;
            int col = col_idx[idx];
            if (col >= 0) {
                sum += val[idx] * x[col];
            }
        }
        y[row] = sum;
    }
}
""".strip()

SELL_KERNEL = r"""
__global__ void spmv_kernel(int m, int slice_height, const int* slice_ptr, const int* col_idx, const float* val, const float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        int slice = row / slice_height;
        int local = row % slice_height;
        int start = slice_ptr[slice];
        int end = slice_ptr[slice + 1];
        int width = (end - start) / slice_height;
        float sum = 0.0f;
        for (int k = 0; k < width; ++k) {
            int idx = start + local * width + k;
            int col = col_idx[idx];
            if (col >= 0) {
                sum += val[idx] * x[col];
            }
        }
        y[row] = sum;
    }
}
""".strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    start = time.time()
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": proc.returncode,
            "seconds": round(time.time() - start, 3),
            "stdout": stdout,
            "stderr": stderr,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        else:
            stdout, stderr = "", ""
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "seconds": round(time.time() - start, 3),
            "stdout": stdout or exc.stdout or "",
            "stderr": stderr or exc.stderr or "",
            "timeout": True,
        }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def load_dag() -> dict[str, Any]:
    return json.loads(DAG_PATH.read_text(encoding="utf-8"))


def collect_environment(gpu: str) -> dict[str, Any]:
    env = {
        "created_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "repo": str(REPO),
        "dag_path": str(DAG_PATH),
        "selected_cuda_visible_devices": gpu,
        "logical_gpu_mapping_note": (
            f"CUDA_VISIBLE_DEVICES={gpu} makes the selected physical GPU appear as "
            "logical cuda:0 inside PyTorch; OOM messages may therefore say GPU 0 "
            "while referring to the selected visible device."
        ),
        "which_nvcc": shutil.which("nvcc"),
        "which_python": shutil.which("python"),
    }
    env["nvidia_smi"] = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )
    env["nvcc_version"] = run_command(["nvcc", "--version"], timeout=30)
    env["python_deps"] = run_command(
        [
            "python",
            "-c",
            (
                "import torch, transformers, peft, scipy, numpy; "
                "print('torch', torch.__version__, 'cuda', torch.version.cuda); "
                "print('transformers', transformers.__version__); "
                "print('peft', peft.__version__); "
                "print('scipy', scipy.__version__); "
                "print('numpy', numpy.__version__)"
            ),
        ],
        timeout=60,
    )
    git_dir = REPO / ".git"
    env["git_head_file"] = (git_dir / "HEAD").read_text(encoding="utf-8").strip() if (git_dir / "HEAD").exists() else ""
    env["git_config_file_excerpt"] = compact_text((git_dir / "config").read_text(encoding="utf-8"), 2000) if (git_dir / "config").exists() else ""
    return env


def matrix_inventory() -> list[dict[str, Any]]:
    sys.path.insert(0, str(REPO))
    from scipy.io import mmread

    rows = []
    for path in sorted((REPO / "dataset").glob("*.mtx")):
        mtx = mmread(path).tocsr()
        rows.append(
            {
                "name": path.name,
                "shape": [int(mtx.shape[0]), int(mtx.shape[1])],
                "nnz": int(mtx.nnz),
                "artifact_path": str(path),
            }
        )
    return rows


def run_deterministic_cuda_eval(max_nnz: int, row_timeout: int) -> dict[str, Any]:
    worker = RUN_ROOT / "sparserl_cuda_eval_worker.py"
    formats = ["CSR", "ELL", "SELL"]
    results = []
    for matrix_path in sorted((REPO / "dataset").glob("*.mtx")):
        for fmt in formats:
            cmd = [
                "python",
                str(worker),
                "--matrix",
                str(matrix_path),
                "--format",
                fmt,
                "--max-nnz",
                str(max_nnz),
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
            proc = run_command(cmd, cwd=REPO, env=env, timeout=row_timeout)
            if proc["timeout"] or proc["returncode"] != 0:
                results.append(
                    {
                        "matrix": matrix_path.name,
                        "format": fmt,
                        "max_nnz": max_nnz,
                        "compile_success": False,
                        "run_success": False,
                        "correct": False,
                        "seconds": proc["seconds"],
                        "timeout": proc["timeout"],
                        "worker_returncode": proc["returncode"],
                        "worker_stdout": compact_text(proc["stdout"], 1200),
                        "worker_stderr": compact_text(proc["stderr"], 1200),
                    }
                )
                continue
            try:
                results.append(json.loads(proc["stdout"].strip().splitlines()[-1]))
            except (IndexError, json.JSONDecodeError) as exc:
                results.append(
                    {
                        "matrix": matrix_path.name,
                        "format": fmt,
                        "max_nnz": max_nnz,
                        "compile_success": False,
                        "run_success": False,
                        "correct": False,
                        "seconds": proc["seconds"],
                        "worker_returncode": proc["returncode"],
                        "parse_exception": repr(exc),
                        "worker_stdout": compact_text(proc["stdout"], 1200),
                        "worker_stderr": compact_text(proc["stderr"], 1200),
                    }
                )
    by_format = {}
    for fmt in formats:
        fmt_rows = [r for r in results if r["format"] == fmt]
        correct = [r for r in fmt_rows if r.get("correct")]
        by_format[fmt] = {
            "attempts": len(fmt_rows),
            "compile_success": sum(1 for r in fmt_rows if r.get("compile_success")),
            "correct": len(correct),
            "timeouts": sum(1 for r in fmt_rows if r.get("timeout")),
            "mean_base_ms": sum(r.get("base_ms", 0.0) for r in correct) / max(1, len(correct)),
            "mean_effective_candidate_ms": sum(
                r.get("effective_candidate_ms", 0.0) for r in correct
            )
            / max(1, len(correct)),
        }
    skipped_formats = {
        "COO": "Repo executor does not zero candidate output before COO atomicAdd; using it would create false correctness evidence.",
        "BSR": "Repo BSR signature lacks original row count, so odd-sized matrices risk out-of-bounds writes.",
        "SpMM": "The official minimal code path exposes SpMV executor only; SpMM paper table debt remains unresolved.",
    }
    return {
        "artifact_kind": "real_cuda_compile_run_timing",
        "convergence_role": "support_only_until_generated_policy_and_paper_grid_exist",
        "max_nnz_policy": "full_sample_matrix_nnz" if max_nnz >= 10**9 else "bounded",
        "row_timeout_seconds": row_timeout,
        "results": results,
        "summary_by_format": by_format,
        "skipped_formats": skipped_formats,
    }


def find_cache_snapshot(model_id: str) -> str | None:
    model_dir = "models--" + model_id.replace("/", "--")
    for root in [Path("/tf/notebooks/.cache/huggingface/hub"), Path("/root/.cache/huggingface/hub")]:
        snap_root = root / model_dir / "snapshots"
        if snap_root.exists():
            snaps = sorted(snap_root.glob("*"))
            for snap in reversed(snaps):
                has_weights = any(snap.glob("*.safetensors")) or any(snap.glob("*.bin"))
                has_index = any(snap.glob("*.safetensors.index.json")) or any(snap.glob("*.bin.index.json"))
                if has_weights or has_index:
                    return str(snap)
    return None


def run_model_attempt(
    gpu: str,
    exact_model_id: str,
    timeout: int,
    attempt_exact_model: bool,
) -> dict[str, Any]:
    cached_exact = find_cache_snapshot(exact_model_id)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["HF_HOME"] = "/tf/notebooks/.cache/huggingface"
    env["TRANSFORMERS_CACHE"] = "/tf/notebooks/.cache/huggingface/transformers"
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")

    if not attempt_exact_model:
        return {
            "attempted": False,
            "exact_model_id": exact_model_id,
            "cached_exact_snapshot": cached_exact,
            "reason": "exact_model_attempt_disabled_for_this_runner_invocation",
            "convergence_role": "blocker_until_exact_or_paper_declared_model_artifacts_run",
        }

    model_arg = cached_exact or exact_model_id
    cmd = [
        "python",
        "-m",
        "sparserl.train_minimal",
        "--steps",
        "1",
        "--max-nnz",
        "256",
        "--model-path",
        model_arg,
        "--tokenizer-path",
        model_arg,
    ]
    before = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        env=env,
        timeout=30,
    )
    result = run_command(cmd, cwd=REPO, env=env, timeout=timeout)
    after = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        env=env,
        timeout=30,
    )
    return {
        "attempted": True,
        "exact_model_id": exact_model_id,
        "cuda_visible_devices": gpu,
        "logical_gpu_mapping_note": (
            f"Inside this child process, physical GPU {gpu} is exposed as logical cuda:0."
        ),
        "model_arg": model_arg,
        "cached_exact_snapshot_before_run": cached_exact,
        "command_result": result,
        "gpu_before": before,
        "gpu_after": after,
        "convergence_role": (
            "support_only_repo_minimal_ppo"
            if result["returncode"] == 0
            else "blocker_evidence_model_or_runtime_failed"
        ),
        "non_convergence_reason": (
            "The repo PPO command uses max_nnz=256 and sample Matrix Market files; "
            "it does not cover the full paper grid, pass@k, SpMM, SuiteSparse/DL collections, "
            "or V100/A100 comparisons."
        ),
    }


def verify_against_paper_debt(
    dag: dict[str, Any],
    cuda_eval: dict[str, Any],
    model_attempt: dict[str, Any],
) -> dict[str, Any]:
    rows = cuda_eval["results"]
    correct_rows = [r for r in rows if r.get("correct")]
    has_real_cuda = bool(correct_rows)
    has_generated_policy = bool(
        model_attempt.get("attempted")
        and model_attempt.get("command_result", {}).get("returncode") == 0
        and "reward=" in model_attempt.get("command_result", {}).get("stdout", "")
    )
    unresolved = [
        {
            "id": "generated_kernel_policy_grid",
            "status": "unresolved" if not has_generated_policy else "partial_repo_minimal",
            "needed": "generated kernels from the pretrained/SFT/RL policy, not only hand-authored canonical kernels",
        },
        {
            "id": "pass_at_k_metrics",
            "status": "unresolved",
            "needed": "pass@1/pass@5/pass@1000 or equivalent sampling-grid measurements",
        },
        {
            "id": "paper_dataset_scale",
            "status": "unresolved",
            "needed": "SuiteSparse and Deep Learning Matrix Collection coverage, not only six released sample matrices",
        },
        {
            "id": "spmm_and_table_grid",
            "status": "unresolved",
            "needed": "SpMM executor and comparison rows in addition to SpMV",
        },
        {
            "id": "baseline_comparison",
            "status": "unresolved",
            "needed": "cuSPARSE/TVM-S/static LLM/codegen baselines with GFLOPS/TFLOPS/speedup",
        },
        {
            "id": "paper_hardware_match",
            "status": "unresolved",
            "needed": "paper-stated V100/A100 traces; current run records local RTX 4090 only",
        },
    ]
    semantic_nodes = [n for n in dag.get("nodes", []) if n.get("type") in {"gap_hypothesis", "mechanism_binding", "experiment_design"}]
    professional_package_ready = all(item["status"] != "unresolved" for item in unresolved)
    support_only_reasons = [
        "deterministic CUDA rows use canonical hand-authored CSR/ELL/SELL kernels, not generated SparseRL policy samples",
        "the local matrix inventory contains six released sample Matrix Market files, not the paper-scale SuiteSparse and Deep Learning Matrix Collection grids",
        "the failed Qwen3-8B repo-minimal attempt is an OOM/runtime blocker, not pass@k policy evidence",
        "no SpMM executor, cuSPARSE/TVM-S/static-codegen baseline table, or V100/A100 paper-hardware trace has been produced",
    ]
    return {
        "status": (
            "converged_professional_close_match"
            if professional_package_ready
            else SPARSERL_HARD_BLOCKER
        ),
        "created_at_utc": utc_now(),
        "blind_contract_checked": dag.get("blind_contract", {}),
        "semantic_dag_nodes_checked": semantic_nodes,
        "real_cuda_executor_evidence": {
            "has_real_cuda_compile_run": has_real_cuda,
            "correct_rows": len(correct_rows),
            "attempted_rows": len(rows),
        },
        "model_policy_evidence": {
            "has_generated_policy_run": has_generated_policy,
            "attempted": model_attempt.get("attempted", False),
            "returncode": model_attempt.get("command_result", {}).get("returncode"),
            "timeout": model_attempt.get("command_result", {}).get("timeout"),
        },
        "real_cuda_convergence_role": (
            "support_only" if has_real_cuda and not professional_package_ready else "professional_evidence"
        ),
        "support_only_reasons": support_only_reasons if not professional_package_ready else [],
        "unresolved_professional_debt": unresolved,
        "professional_package_ready": professional_package_ready,
        "convergence_decision": (
            "accepted" if professional_package_ready else "blocked_not_converged"
        ),
        "loop1_required_dag_update": [
            "Add an executor node that explicitly runs full-nnz SpMV compile/correctness/timing on released Matrix Market samples.",
            "Add an exact-model gate for Qwen3/DeepSeek-R1/static-codegen baselines before any pass@k claim.",
            "Add explicit SpMM debt because the official minimal repo executor is SpMV-only.",
            "Add paper-hardware gate distinguishing local RTX 4090 support from V100/A100 paper traces.",
            "Tell loop2 to keep running only after generated-policy, pass@k, baseline, and dataset-scale nodes are satisfied.",
            "Promote deterministic CUDA rows only as executor sanity evidence, never as final result-shape convergence.",
        ],
    }


def write_status(report: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# SparseRL Specialized Runner Status",
        "",
        f"- Updated: {utc_now()}",
        f"- Paper: `Mastering Sparse CUDA Generation through Pretrained Models and Deep Reinforcement Learning`",
        f"- Status: `{report['verifier']['status']}`",
        f"- Convergence decision: `{report['verifier']['convergence_decision']}`",
        f"- CUDA rows: `{report['verifier']['real_cuda_executor_evidence']['correct_rows']}` correct / `{report['verifier']['real_cuda_executor_evidence']['attempted_rows']}` attempted",
        f"- Physical GPU requested: `{report['environment']['selected_cuda_visible_devices']}`",
        f"- PyTorch mapping: {report['environment']['logical_gpu_mapping_note']}",
        f"- Model policy attempted: `{report['verifier']['model_policy_evidence']['attempted']}`",
        f"- Model policy returncode: `{report['verifier']['model_policy_evidence']['returncode']}`",
        f"- Real CUDA convergence role: `{report['verifier']['real_cuda_convergence_role']}`",
        "",
        "## Why Current GPU Evidence Cannot Converge",
    ]
    for reason in report["verifier"].get("support_only_reasons", []):
        lines.append(f"- {reason}")
    lines += [
        "",
        "## Unresolved Professional Debt",
    ]
    for item in report["verifier"]["unresolved_professional_debt"]:
        lines.append(f"- `{item['id']}`: {item['status']} - {item['needed']}")
    lines.extend(
        [
            "",
            "## Artifacts",
            f"- `environment.json`",
            f"- `matrix_inventory.json`",
            f"- `deterministic_cuda_kernel_eval.json`",
            f"- `model_policy_attempt.json`",
            f"- `sparserl_specialized_verifier.json`",
        ]
    )
    (out_dir / "SPARSERL_SPECIALIZED_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--max-nnz", type=int, default=10**9)
    parser.add_argument("--row-timeout", type=int, default=240)
    parser.add_argument("--reuse-cuda-eval", action="store_true")
    parser.add_argument("--reuse-model-attempt", action="store_true")
    parser.add_argument("--attempt-exact-model", action="store_true")
    parser.add_argument("--model-timeout", type=int, default=14400)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    dag = load_dag()
    environment = collect_environment(args.gpu)
    inventory = matrix_inventory()
    cuda_eval_path = RUN_ROOT / "deterministic_cuda_kernel_eval.json"
    if args.reuse_cuda_eval and cuda_eval_path.exists():
        cuda_eval = json.loads(cuda_eval_path.read_text(encoding="utf-8"))
    else:
        cuda_eval = run_deterministic_cuda_eval(args.max_nnz, args.row_timeout)
    model_attempt_path = RUN_ROOT / "model_policy_attempt.json"
    if args.reuse_model_attempt and model_attempt_path.exists():
        model_attempt = json.loads(model_attempt_path.read_text(encoding="utf-8"))
    else:
        model_attempt = run_model_attempt(
            args.gpu,
            "Qwen/Qwen3-8B",
            timeout=args.model_timeout,
            attempt_exact_model=args.attempt_exact_model,
        )
    verifier = verify_against_paper_debt(dag, cuda_eval, model_attempt)

    report = {
        "created_at_utc": utc_now(),
        "paper_id": "ICLR2026_VdLEaGPYWT_sparserl_sparse_cuda_rl",
        "paper_title": "Mastering Sparse CUDA Generation through Pretrained Models and Deep Reinforcement Learning",
        "runner_type": "sparse_cuda_kernel_quality_runner",
        "dag_path": str(DAG_PATH),
        "repo": str(REPO),
        "environment": environment,
        "matrix_inventory": inventory,
        "cuda_eval_summary": cuda_eval["summary_by_format"],
        "model_attempt_summary": {
            "attempted": model_attempt.get("attempted"),
            "model_arg": model_attempt.get("model_arg"),
            "returncode": model_attempt.get("command_result", {}).get("returncode"),
            "timeout": model_attempt.get("command_result", {}).get("timeout"),
            "seconds": model_attempt.get("command_result", {}).get("seconds"),
        },
        "verifier": verifier,
    }

    write_json(RUN_ROOT / "environment.json", environment)
    write_json(RUN_ROOT / "matrix_inventory.json", inventory)
    write_json(RUN_ROOT / "deterministic_cuda_kernel_eval.json", cuda_eval)
    write_json(RUN_ROOT / "model_policy_attempt.json", model_attempt)
    write_json(RUN_ROOT / "sparserl_specialized_verifier.json", report)
    write_status(report, RUN_ROOT)
    print(json.dumps(report["verifier"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
