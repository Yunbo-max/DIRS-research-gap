#!/usr/bin/env python3
"""One matrix/format CUDA eval worker for SparseRL.

The parent runner executes this file in a separate process group so nvcc and its
children can be killed cleanly if a full-nnz host program becomes pathological.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/SparseRL")


KERNELS = {
    "CSR": r"""
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
""".strip(),
    "ELL": r"""
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
""".strip(),
    "SELL": r"""
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
""".strip(),
}


def compact_text(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--format", required=True, choices=sorted(KERNELS))
    parser.add_argument("--max-nnz", type=int, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO))
    from sparserl.codegen import compile_and_run
    from sparserl.data import convert_format, load_mtx
    from sparserl.reward import compute_reward

    start = time.time()
    matrix_path = Path(args.matrix)
    fmt = args.format
    kernel = KERNELS[fmt]
    csr = load_mtx(str(matrix_path))
    try:
        fmt_payload, conversion_ms = convert_format(csr, fmt=fmt, max_nnz=args.max_nnz, seed=0)
        compile_res, run_res = compile_and_run(csr, fmt_payload, kernel, conversion_ms)
        reward = compute_reward(
            compile_res,
            run_res,
            kernel,
            0.5,
            -0.5,
            0.5,
            -0.5,
            0.2,
            0.2,
            48 * 1024,
        )
        payload = {
            "matrix": matrix_path.name,
            "format": fmt,
            "matrix_shape": [int(csr.shape[0]), int(csr.shape[1])],
            "matrix_nnz": int(csr.val.shape[0]),
            "max_nnz": int(args.max_nnz),
            "compile_success": bool(compile_res.success),
            "run_success": bool(run_res.success),
            "correct": bool(run_res.correct),
            "base_ms": float(run_res.base_ms),
            "candidate_ms": float(run_res.cand_ms),
            "conversion_ms": float(run_res.conversion_ms),
            "effective_candidate_ms": float(run_res.cand_ms + run_res.conversion_ms),
            "reward_total": float(reward.total),
            "reward_correctness": float(reward.correctness),
            "reward_efficiency": float(reward.efficiency),
            "reward_memory_penalty": float(reward.memory_penalty),
            "seconds": round(time.time() - start, 3),
            "compile_log": compact_text(compile_res.log),
            "run_log": compact_text(run_res.log),
            "kernel_code": kernel,
        }
    except Exception as exc:
        payload = {
            "matrix": matrix_path.name,
            "format": fmt,
            "matrix_shape": [int(csr.shape[0]), int(csr.shape[1])],
            "matrix_nnz": int(csr.val.shape[0]),
            "max_nnz": int(args.max_nnz),
            "compile_success": False,
            "run_success": False,
            "correct": False,
            "seconds": round(time.time() - start, 3),
            "exception": repr(exc),
            "kernel_code": kernel,
        }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
