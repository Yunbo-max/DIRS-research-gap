#!/usr/bin/env python3
"""Author-style GPU reproduction campaign for systems/token-efficiency gaps.

This is intentionally stricter than paper reading. It does three things:

1. Audits all 20 papers for exact-rerun readiness and local code availability.
2. Runs real GPU proxy experiments for the core systems motifs that recur in
   the paper set: KV/cache locality, token merging, speculative decoding,
   sampling truncation, quantization/compression, and sparse-kernel efficiency.
3. Converts measured failures/tradeoffs into research-gap evidence that a
   NeurIPS-style systems paper would need to handle.

The proxy experiments do not claim to reproduce a paper's leaderboard numbers.
They reproduce the measurement pressure: speed/memory/quality/correctness under
controlled baselines and stress axes on the local GPU.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable


RUN_DIR = Path(__file__).resolve().parent
DEFAULT_EVIDENCE = RUN_DIR / "paper_section_evidence_table.json"
OUTPUT_JSON = RUN_DIR / "author_style_gpu_reproduction_campaign.json"
OUTPUT_MD = RUN_DIR / "AUTHOR_STYLE_GPU_REPRODUCTION_CAMPAIGN.md"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 60) -> dict:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": proc.returncode,
            "elapsed_s": round(time.perf_counter() - start, 3),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "elapsed_s": round(time.perf_counter() - start, 3),
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }


def parse_local_repo_paths(paper: dict) -> list[Path]:
    paths: list[Path] = []
    footprint = paper.get("footprint") or {}
    repos = footprint.get("code_repositories") if isinstance(footprint, dict) else None
    if isinstance(repos, list):
        for item in repos:
            if isinstance(item, dict) and item.get("local_path"):
                p = Path(item["local_path"])
                if p.exists():
                    paths.append(p)

    chip_id = paper["chip_id"].lower()
    candidates = [
        Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/067_rdvq/RDVQ"),
        Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/052_seacache/SeaCache"),
        Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/053_sencache/SenCache"),
        Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/103_atoken/ml-atoken"),
        Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/103_atoken/ATOKEN-A-UNIFIED-TOKENIZER-FOR-VISION"),
        Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/FlashVID"),
        Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/Prophet"),
        Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/LoongRL"),
        Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/p-less"),
        Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/MrRoPE_OpenReviewSupp"),
        Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/SparseRL"),
    ]
    name_hints = {
        "rdvq": ["rdvq"],
        "seacache": ["seacache"],
        "sencache": ["sencache"],
        "atoken": ["atoken"],
        "flashvid": ["flashvid"],
        "prophet": ["prophet"],
        "loongrl": ["loongrl"],
        "p_less": ["p-less"],
        "mrrope": ["mrrope"],
        "sparserl": ["sparserl"],
    }
    for key, hints in name_hints.items():
        if key in chip_id or any(h in chip_id for h in hints):
            for c in candidates:
                if c.exists() and any(h.replace("-", "") in str(c).lower().replace("-", "") for h in hints):
                    paths.append(c)

    seen = set()
    unique = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            unique.append(p)
            seen.add(rp)
    return unique


def audit_repo(repo: Path, max_py: int = 500) -> dict:
    py_files = list(repo.rglob("*.py"))
    cu_files = list(repo.rglob("*.cu")) + list(repo.rglob("*.cuh")) + list(repo.rglob("*.cpp"))
    shell_files = list(repo.rglob("*.sh"))
    readmes = [p for p in repo.rglob("*") if p.is_file() and p.name.lower() in {"readme.md", "readme.rst", "readme.txt"}]
    selected_py = py_files[:max_py]
    syntax_start = time.perf_counter()
    syntax_errors = []
    for py_file in selected_py:
        try:
            ast.parse(py_file.read_text(errors="ignore"), filename=str(py_file))
        except SyntaxError as exc:
            syntax_errors.append({"file": str(py_file.relative_to(repo)), "line": exc.lineno, "msg": exc.msg})
        except ValueError as exc:
            syntax_errors.append({"file": str(py_file.relative_to(repo)), "line": None, "msg": str(exc)})
        except OSError as exc:
            syntax_errors.append({"file": str(py_file.relative_to(repo)), "line": None, "msg": str(exc)})
    syntax_elapsed = time.perf_counter() - syntax_start
    git = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo, timeout=10) if (repo / ".git").exists() else None
    return {
        "repo": str(repo),
        "exists": repo.exists(),
        "python_file_count": len(py_files),
        "cuda_cpp_file_count": len(cu_files),
        "shell_file_count": len(shell_files),
        "readme_count": len(readmes),
        "sample_entrypoints": [str(p.relative_to(repo)) for p in (selected_py[:20] + shell_files[:10])],
        "syntax_checked_python_files": len(selected_py),
        "syntax_check_ok": len(syntax_errors) == 0,
        "syntax_errors": syntax_errors[:20],
        "syntax_check_elapsed_s": round(syntax_elapsed, 3),
        "git_head": (git or {}).get("stdout_tail", "").strip() if git else None,
    }


def audit_papers(papers: list[dict]) -> list[dict]:
    rows = []
    for paper in papers:
        repos = parse_local_repo_paths(paper)
        repo_audits = [audit_repo(repo) for repo in repos]
        footprint = paper.get("footprint") or {}
        statuses = footprint.get("implementation_statuses", []) if isinstance(footprint, dict) else []
        status_text = json.dumps(statuses).lower()
        exact_rerun_status = "blocked"
        if any(a["syntax_check_ok"] for a in repo_audits) and "not_rerun" not in status_text and "missing" not in status_text:
            exact_rerun_status = "code_ready_needs_model_data"
        if "github_repository_not_found" in status_text or "source_code_missing" in status_text:
            exact_rerun_status = "paper_only_or_source_missing"
        rows.append(
            {
                "chip_id": paper["chip_id"],
                "title": paper["title"],
                "repo_count": len(repos),
                "repos": [str(p) for p in repos],
                "repo_audits": repo_audits,
                "implementation_statuses": statuses,
                "exact_rerun_status": exact_rerun_status,
            }
        )
    return rows


def pick_device() -> tuple[object, dict]:
    import torch

    if not torch.cuda.is_available():
        return torch.device("cpu"), {"cuda_available": False, "device": "cpu"}
    smi = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    gpus = []
    for line in smi.get("stdout_tail", "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": int(float(parts[2])),
                    "memory_used_mib": int(float(parts[3])),
                    "driver_version": parts[4],
                    "compute_capability": parts[5],
                }
            )
    if gpus:
        chosen = min(gpus, key=lambda g: g["memory_used_mib"])
        torch.cuda.set_device(chosen["index"])
        return torch.device(f"cuda:{chosen['index']}"), {"cuda_available": True, "gpus": gpus, "chosen": chosen}
    return torch.device("cuda:0"), {"cuda_available": True, "gpus": [], "chosen": {"index": 0, "name": torch.cuda.get_device_name(0)}}


@dataclass
class BenchTimer:
    device: object

    def time_ms(self, fn: Callable[[], object], warmup: int, iters: int) -> tuple[float, float]:
        import torch

        for _ in range(warmup):
            fn()
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        checksum = 0.0
        for _ in range(iters):
            out = fn()
            if hasattr(out, "float"):
                checksum += float(out.float().mean().detach().cpu())
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        return (time.perf_counter() - start) * 1000.0 / iters, checksum


def dense_decode(q, k, v):
    import torch

    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


def local_decode(q, k, v, window: int):
    return dense_decode(q, k[..., -window:, :], v[..., -window:, :])


def topk_decode(q, k, v, keep: int):
    import torch

    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    values, idx = torch.topk(scores, k=min(keep, scores.shape[-1]), dim=-1)
    v_expanded = v.unsqueeze(2).expand(-1, -1, idx.shape[2], -1, -1)
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, -1, v.shape[-1])
    v_selected = torch.gather(v_expanded, 3, gather_idx)
    probs = torch.softmax(values.float(), dim=-1).to(q.dtype)
    return torch.sum(probs.unsqueeze(-1) * v_selected, dim=3)


def kv_locality_experiment(device, seeds: list[int]) -> list[dict]:
    import torch

    rows = []
    timer = BenchTimer(device)
    for seed in seeds:
        torch.manual_seed(seed)
        for context in [512, 2048, 8192, 16384]:
            batch, heads, head_dim = 1, 16, 64
            q = torch.randn(batch, heads, 1, head_dim, device=device, dtype=torch.float16)
            k = torch.randn(batch, heads, context, head_dim, device=device, dtype=torch.float16)
            v = torch.randn(batch, heads, context, head_dim, device=device, dtype=torch.float16)
            base = dense_decode(q, k, v).detach()
            for method, fn in [
                ("dense_full_kv", lambda: dense_decode(q, k, v)),
                ("local_window_256", lambda: local_decode(q, k, v, 256)),
                ("local_window_1024", lambda: local_decode(q, k, v, 1024)),
                ("topk_128_after_full_scores", lambda: topk_decode(q, k, v, 128)),
            ]:
                ms, checksum = timer.time_ms(fn, warmup=10, iters=60)
                out = fn().detach()
                err = float((out.float() - base.float()).pow(2).mean().sqrt().cpu())
                rows.append(
                    {
                        "family": "kv_cache_locality",
                        "seed": seed,
                        "context_tokens": context,
                        "method": method,
                        "ms": round(ms, 4),
                        "items_per_second": round(1000.0 / ms, 2),
                        "quality_error_rmse_vs_dense": round(err, 6),
                        "checksum": round(checksum, 6),
                    }
                )
    return rows


def token_merge_experiment(device, seeds: list[int]) -> list[dict]:
    import torch

    rows = []
    timer = BenchTimer(device)
    for seed in seeds:
        torch.manual_seed(seed)
        for tokens in [1024, 4096, 8192]:
            dim = 64
            x = torch.randn(1, tokens, dim, device=device, dtype=torch.float16)
            q = x
            k = x
            v = x

            def attend(a, b, c):
                scale = dim ** -0.5
                scores = torch.matmul(a, b.transpose(-1, -2)) * scale
                probs = torch.softmax(scores.float(), dim=-1).to(a.dtype)
                return torch.matmul(probs, c)

            full = attend(q, k, v).detach()
            full_ms, full_checksum = timer.time_ms(lambda: attend(q, k, v), warmup=5, iters=20)
            rows.append(
                {
                    "family": "token_merging",
                    "seed": seed,
                    "tokens": tokens,
                    "method": "full_attention_baseline",
                    "kept_tokens": tokens,
                    "ms": round(full_ms, 4),
                    "items_per_second": round(1000.0 / full_ms, 2),
                    "quality_error_rmse_vs_full": 0.0,
                    "checksum": round(full_checksum, 6),
                }
            )
            for keep_ratio in [0.75, 0.5, 0.25]:
                kept = max(16, int(tokens * keep_ratio))
                # Stride control: cheap but oblivious.
                xs = x[:, :: max(1, tokens // kept), :][:, :kept, :].contiguous()
                # Norm control: keeps high-energy tokens but can erase low-energy semantic tokens.
                idx = torch.topk(x.float().norm(dim=-1), k=kept, dim=-1).indices.sort(dim=-1).values
                xn = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, dim)).contiguous()
                for method, xm in [("stride_merge_proxy", xs), ("norm_topk_merge_proxy", xn)]:
                    ms, checksum = timer.time_ms(lambda xm=xm: attend(q, xm, xm), warmup=5, iters=20)
                    approx = attend(q, xm, xm).detach()
                    err = float((approx.float() - full.float()).pow(2).mean().sqrt().cpu())
                    rows.append(
                        {
                            "family": "token_merging",
                            "seed": seed,
                            "tokens": tokens,
                            "method": method,
                            "kept_tokens": int(xm.shape[1]),
                            "keep_ratio": keep_ratio,
                            "ms": round(ms, 4),
                            "items_per_second": round(1000.0 / ms, 2),
                            "quality_error_rmse_vs_full": round(err, 6),
                            "checksum": round(checksum, 6),
                        }
                    )
    return rows


def speculative_decoding_experiment(device, seeds: list[int]) -> list[dict]:
    import torch

    rows = []
    timer = BenchTimer(device)
    vocab = 32768
    steps = 128
    block = 4
    for seed in seeds:
        torch.manual_seed(seed)
        target_logits = torch.randn(steps, vocab, device=device, dtype=torch.float16)
        target_probs = torch.softmax(target_logits.float(), dim=-1)
        target_tokens = torch.argmax(target_probs, dim=-1)
        target_ms, _ = timer.time_ms(lambda: torch.argmax(torch.softmax(target_logits.float(), dim=-1), dim=-1), 5, 40)
        for noise in [0.03, 0.08, 0.16, 0.32]:
            draft_logits = target_logits + noise * torch.randn_like(target_logits)
            draft_probs = torch.softmax(draft_logits.float(), dim=-1)
            draft_tokens = torch.argmax(draft_probs, dim=-1)
            accept = (draft_tokens == target_tokens).float()
            block_accept = accept.reshape(-1, block).prod(dim=-1)
            draft_ms, _ = timer.time_ms(lambda: torch.argmax(torch.softmax(draft_logits.float(), dim=-1), dim=-1), 5, 40)
            accepted_rate = float(accept.mean().cpu())
            block_rate = float(block_accept.mean().cpu())
            # A simple systems model: draft is cheap, target verifies one block.
            modeled_speedup = (block * target_ms) / max(draft_ms + target_ms, 1e-6)
            effective_speedup = modeled_speedup * max(block_rate, 1e-6)
            rows.append(
                {
                    "family": "speculative_decoding_proxy",
                    "seed": seed,
                    "vocab": vocab,
                    "steps": steps,
                    "block": block,
                    "draft_noise": noise,
                    "target_ms": round(target_ms, 4),
                    "draft_ms": round(draft_ms, 4),
                    "token_accept_rate": round(accepted_rate, 6),
                    "block_accept_rate": round(block_rate, 6),
                    "modeled_speedup_upper": round(modeled_speedup, 4),
                    "effective_speedup_after_acceptance": round(effective_speedup, 4),
                }
            )
    return rows


def sampling_experiment(device, seeds: list[int]) -> list[dict]:
    import torch

    rows = []
    timer = BenchTimer(device)
    vocab = 65536
    batch = 64
    for seed in seeds:
        torch.manual_seed(seed)
        logits = torch.randn(batch, vocab, device=device, dtype=torch.float16)
        full_probs = torch.softmax(logits.float(), dim=-1)
        full_entropy = -(full_probs * torch.log(full_probs + 1e-12)).sum(dim=-1).mean()
        full_ms, _ = timer.time_ms(lambda: torch.softmax(logits.float(), dim=-1), 5, 40)
        for method, param in [
            ("softmax_full", 1.0),
            ("top_p_0.95", 0.95),
            ("top_p_0.90", 0.90),
            ("top_k_128", 128),
            ("top_k_512", 512),
            ("p_less_entropy_proxy", 0.0),
        ]:
            if method == "softmax_full":
                ms, checksum = full_ms, 0.0
                kept = vocab
                mass = 1.0
                entropy_delta = 0.0
            elif method.startswith("top_k"):
                k = int(param)
                def topk():
                    vals, idx = torch.topk(logits, k=k, dim=-1)
                    return torch.softmax(vals.float(), dim=-1)
                ms, checksum = timer.time_ms(topk, 5, 40)
                vals, idx = torch.topk(full_probs, k=k, dim=-1)
                mass = float(vals.sum(dim=-1).mean().cpu())
                trunc_probs = vals / vals.sum(dim=-1, keepdim=True)
                ent = -(trunc_probs * torch.log(trunc_probs + 1e-12)).sum(dim=-1).mean()
                entropy_delta = float((full_entropy - ent).cpu())
                kept = k
            elif method.startswith("top_p"):
                threshold = float(param)
                def topp():
                    probs = torch.softmax(logits.float(), dim=-1)
                    sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    keep = cumsum <= threshold
                    return sorted_probs * keep
                ms, checksum = timer.time_ms(topp, 3, 15)
                sorted_probs, _ = torch.sort(full_probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                keep = cumsum <= threshold
                kept = float(keep.float().sum(dim=-1).mean().cpu())
                mass = float((sorted_probs * keep).sum(dim=-1).mean().cpu())
                kept_probs = sorted_probs * keep
                kept_probs = kept_probs / kept_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                ent = -(kept_probs * torch.log(kept_probs + 1e-12)).sum(dim=-1).mean()
                entropy_delta = float((full_entropy - ent).cpu())
            else:
                # Entropy-adaptive proxy: keep fewer tokens for low-entropy rows and more for high-entropy rows.
                row_entropy = -(full_probs * torch.log(full_probs + 1e-12)).sum(dim=-1)
                normalized = (row_entropy - row_entropy.min()) / (row_entropy.max() - row_entropy.min() + 1e-6)
                ks = (64 + normalized * 960).long()
                max_k = int(ks.max().item())
                def adaptive():
                    vals, _ = torch.topk(logits, k=max_k, dim=-1)
                    return torch.softmax(vals.float(), dim=-1)
                ms, checksum = timer.time_ms(adaptive, 5, 40)
                kept = float(ks.float().mean().cpu())
                vals, _ = torch.topk(full_probs, k=max_k, dim=-1)
                mass = float(vals.sum(dim=-1).mean().cpu())
                trunc_probs = vals / vals.sum(dim=-1, keepdim=True)
                ent = -(trunc_probs * torch.log(trunc_probs + 1e-12)).sum(dim=-1).mean()
                entropy_delta = float((full_entropy - ent).cpu())
            rows.append(
                {
                    "family": "sampling_truncation",
                    "seed": seed,
                    "method": method,
                    "batch": batch,
                    "vocab": vocab,
                    "ms": round(ms, 4),
                    "items_per_second": round(1000.0 / ms, 2),
                    "kept_tokens_mean": round(kept, 3),
                    "retained_probability_mass": round(mass, 6),
                    "entropy_delta_vs_full": round(entropy_delta, 6),
                    "checksum": round(checksum, 6),
                }
            )
    return rows


def quantization_experiment(device, seeds: list[int]) -> list[dict]:
    import torch

    rows = []
    timer = BenchTimer(device)
    for seed in seeds:
        torch.manual_seed(seed)
        for n in [1_000_000, 4_000_000, 16_000_000]:
            x = torch.randn(n, device=device, dtype=torch.float16)
            base_ms, base_checksum = timer.time_ms(lambda: x * 1.0001 + 0.01, 5, 50)
            rows.append(
                {
                    "family": "quantization_compression",
                    "seed": seed,
                    "elements": n,
                    "method": "fp16_baseline",
                    "bits": 16,
                    "ms": round(base_ms, 4),
                    "items_per_second": round(n / (base_ms / 1000.0), 2),
                    "rmse_vs_fp16": 0.0,
                    "checksum": round(base_checksum, 6),
                }
            )
            for bits in [8, 6, 4]:
                qmax = 2 ** (bits - 1) - 1
                scale = x.float().abs().max().clamp_min(1e-6) / qmax
                def quant_dequant():
                    q = torch.clamp(torch.round(x.float() / scale), -qmax, qmax)
                    return (q * scale).to(torch.float16)
                ms, checksum = timer.time_ms(quant_dequant, 5, 50)
                xd = quant_dequant()
                rmse = float((xd.float() - x.float()).pow(2).mean().sqrt().cpu())
                rows.append(
                    {
                        "family": "quantization_compression",
                        "seed": seed,
                        "elements": n,
                        "method": "symmetric_quant_dequant",
                        "bits": bits,
                        "ms": round(ms, 4),
                        "items_per_second": round(n / (ms / 1000.0), 2),
                        "rmse_vs_fp16": round(rmse, 6),
                        "checksum": round(checksum, 6),
                    }
                )
    return rows


def sparse_kernel_experiment(device, seeds: list[int]) -> list[dict]:
    import torch

    rows = []
    timer = BenchTimer(device)
    for seed in seeds:
        torch.manual_seed(seed)
        for n in [2048, 4096, 8192]:
            a = torch.randn(n, n, device=device, dtype=torch.float16)
            x = torch.randn(n, 1, device=device, dtype=torch.float16)
            dense_ms, dense_checksum = timer.time_ms(lambda: a @ x, 5, 20)
            dense_out = (a @ x).detach()
            rows.append(
                {
                    "family": "sparse_kernel_efficiency",
                    "seed": seed,
                    "n": n,
                    "density": 1.0,
                    "method": "dense_matvec",
                    "ms": round(dense_ms, 4),
                    "items_per_second": round(1000.0 / dense_ms, 2),
                    "rmse_vs_dense": 0.0,
                    "checksum": round(dense_checksum, 6),
                }
            )
            for density in [0.5, 0.25, 0.125]:
                mask = (torch.rand(n, n, device=device) < density).to(a.dtype)
                sparse_a = (a * mask).contiguous()
                ms, checksum = timer.time_ms(lambda: sparse_a @ x, 5, 20)
                out = sparse_a @ x
                err = float((out.float() - dense_out.float()).pow(2).mean().sqrt().cpu())
                rows.append(
                    {
                        "family": "sparse_kernel_efficiency",
                        "seed": seed,
                        "n": n,
                        "density": density,
                        "method": "masked_dense_matvec_control",
                        "ms": round(ms, 4),
                        "items_per_second": round(1000.0 / ms, 2),
                        "rmse_vs_dense": round(err, 6),
                        "checksum": round(checksum, 6),
                    }
                )
    return rows


def summarize_rows(rows: list[dict], group_keys: list[str], metric_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)
    out = []
    for key, vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["n"] = len(vals)
        for metric in metric_keys:
            xs = [v[metric] for v in vals if isinstance(v.get(metric), (int, float))]
            if xs:
                rec[f"{metric}_mean"] = round(mean(xs), 6)
                rec[f"{metric}_std"] = round(pstdev(xs), 6) if len(xs) > 1 else 0.0
        out.append(rec)
    return out


def infer_gaps(rows: list[dict], repo_audit: list[dict]) -> list[dict]:
    gaps = []
    blocked = [r for r in repo_audit if r["exact_rerun_status"] != "code_ready_needs_model_data"]
    if blocked:
        gaps.append(
            {
                "gap": "Exact paper-result reruns are structurally blocked for many papers without checkpoint/data/API provenance.",
                "evidence": f"{len(blocked)}/20 papers are not code-ready exact reruns under local constraints.",
                "paper_move": "A strict systems paper should publish a runnable benchmark harness with fallback tiny fixtures, not only full-scale scripts.",
            }
        )
    kv = [r for r in rows if r["family"] == "kv_cache_locality" and r["method"].startswith("local_window")]
    if kv:
        worst = max(kv, key=lambda r: r["quality_error_rmse_vs_dense"])
        gaps.append(
            {
                "gap": "KV/cache locality speedups need quality guards, because local windows can diverge from full-context attention.",
                "evidence": f"Max local-window RMSE vs dense was {worst['quality_error_rmse_vs_dense']} at context {worst['context_tokens']}.",
                "paper_move": "Report speed together with semantic/correctness deltas under long-context stress.",
            }
        )
    tm = [r for r in rows if r["family"] == "token_merging" and r["method"] != "full_attention_baseline"]
    if tm:
        worst = max(tm, key=lambda r: r["quality_error_rmse_vs_full"])
        gaps.append(
            {
                "gap": "Token merging methods need instance-adaptive retention, not only fixed token budgets.",
                "evidence": f"Max merge RMSE vs full attention was {worst['quality_error_rmse_vs_full']} with {worst.get('kept_tokens')} kept tokens.",
                "paper_move": "Tie merge ratio to failure cases and preserve a task-quality metric next to throughput.",
            }
        )
    spec = [r for r in rows if r["family"] == "speculative_decoding_proxy"]
    if spec:
        bad = min(spec, key=lambda r: r["effective_speedup_after_acceptance"])
        gaps.append(
            {
                "gap": "Speculative decoding speedup is acceptance-limited, so draft quality is a first-class systems variable.",
                "evidence": f"Lowest effective modeled speedup was {bad['effective_speedup_after_acceptance']} at draft noise {bad['draft_noise']}.",
                "paper_move": "Report acceptance distributions and not just raw draft/target latency.",
            }
        )
    samp = [r for r in rows if r["family"] == "sampling_truncation" and r["method"] != "softmax_full"]
    if samp:
        low_mass = min(samp, key=lambda r: r["retained_probability_mass"])
        gaps.append(
            {
                "gap": "Sampling/token-pruning claims need probability-mass and entropy audits.",
                "evidence": f"Lowest retained mass was {low_mass['retained_probability_mass']} for {low_mass['method']}.",
                "paper_move": "Frame hyperparameter-free decoding as a constrained mass/entropy problem, not only a recipe.",
            }
        )
    sparse = [r for r in rows if r["family"] == "sparse_kernel_efficiency" and r["method"] != "dense_matvec"]
    if sparse:
        no_speed = [r for r in sparse if r["items_per_second"] <= next((b["items_per_second"] for b in rows if b["family"] == "sparse_kernel_efficiency" and b["n"] == r["n"] and b["seed"] == r["seed"] and b["method"] == "dense_matvec"), 0)]
        if no_speed:
            gaps.append(
                {
                    "gap": "Unstructured sparsity is not automatically a kernel speedup.",
                    "evidence": f"{len(no_speed)} masked sparse controls failed to beat dense matvec despite lower density.",
                    "paper_move": "Require hardware-aware sparse layout/kernel measurements before claiming sparse efficiency.",
                }
            )
    return gaps


def run_campaign(args: argparse.Namespace) -> dict:
    import torch

    evidence = json.loads(args.evidence.read_text())
    papers = evidence["papers"]
    device, device_info = pick_device()
    random.seed(args.seed)
    seeds = [args.seed + i for i in range(args.repeats)]
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True

    repo_audit = audit_papers(papers)
    experiment_rows = []
    families = [
        ("kv_cache_locality", kv_locality_experiment),
        ("token_merging", token_merge_experiment),
        ("speculative_decoding_proxy", speculative_decoding_experiment),
        ("sampling_truncation", sampling_experiment),
        ("quantization_compression", quantization_experiment),
        ("sparse_kernel_efficiency", sparse_kernel_experiment),
    ]
    started = time.perf_counter()
    for family, fn in families:
        family_start = time.perf_counter()
        rows = fn(device, seeds)
        for row in rows:
            row["family_elapsed_s_at_finish"] = round(time.perf_counter() - family_start, 3)
        experiment_rows.extend(rows)

    summaries = {
        "kv_cache_locality": summarize_rows(
            [r for r in experiment_rows if r["family"] == "kv_cache_locality"],
            ["method", "context_tokens"],
            ["ms", "items_per_second", "quality_error_rmse_vs_dense"],
        ),
        "token_merging": summarize_rows(
            [r for r in experiment_rows if r["family"] == "token_merging"],
            ["method", "tokens", "kept_tokens"],
            ["ms", "items_per_second", "quality_error_rmse_vs_full"],
        ),
        "speculative_decoding_proxy": summarize_rows(
            [r for r in experiment_rows if r["family"] == "speculative_decoding_proxy"],
            ["draft_noise"],
            ["token_accept_rate", "block_accept_rate", "effective_speedup_after_acceptance"],
        ),
        "sampling_truncation": summarize_rows(
            [r for r in experiment_rows if r["family"] == "sampling_truncation"],
            ["method"],
            ["ms", "kept_tokens_mean", "retained_probability_mass", "entropy_delta_vs_full"],
        ),
        "quantization_compression": summarize_rows(
            [r for r in experiment_rows if r["family"] == "quantization_compression"],
            ["method", "bits", "elements"],
            ["ms", "items_per_second", "rmse_vs_fp16"],
        ),
        "sparse_kernel_efficiency": summarize_rows(
            [r for r in experiment_rows if r["family"] == "sparse_kernel_efficiency"],
            ["method", "n", "density"],
            ["ms", "items_per_second", "rmse_vs_dense"],
        ),
    }
    gaps = infer_gaps(experiment_rows, repo_audit)
    report = {
        "created_at_utc": now_utc(),
        "campaign_type": "author_style_gpu_reproduction_and_gap_search",
        "domain": evidence["domain"],
        "paper_count": len(papers),
        "seed": args.seed,
        "repeats": args.repeats,
        "device": str(device),
        "device_info": device_info,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "private_holdout_read": False,
        "paid_external_api_invoked": False,
        "repo_audit": repo_audit,
        "experiment_rows": experiment_rows,
        "summaries": summaries,
        "inferred_research_gaps": gaps,
        "strict_neurips_dag_delta": [
            "gap hypothesis must be paired with a runnable stress test",
            "experiment section must name backend, command, seed, data fixture, metric, and blocked artifacts",
            "results must separate exact rerun, proxy rerun, code audit, and paper-only evidence",
            "gap claims should emerge from measured tradeoff failures, not just related-work language",
            "appendix must include logs and machine-readable artifacts for every table claim",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True))
    write_markdown(report)
    return report


def write_markdown(report: dict) -> None:
    lines = [
        "# Author-Style GPU Reproduction Campaign",
        "",
        f"Date: `{report['created_at_utc']}`",
        f"Domain: `{report['domain']}`",
        f"Papers audited: `{report['paper_count']}`",
        f"Campaign: `{report['campaign_type']}`",
        f"Device: `{report['device']}`",
        f"Runtime: `{report['runtime_seconds']}s`",
        f"Private holdout read: `{str(report['private_holdout_read']).lower()}`",
        f"Paid/external API invoked: `{str(report['paid_external_api_invoked']).lower()}`",
        "",
        "## Device",
        "",
        "```json",
        json.dumps(report["device_info"], indent=2, sort_keys=True),
        "```",
        "",
        "## Repo And Exact-Rerun Audit",
        "",
    ]
    for row in report["repo_audit"]:
        ok = sum(1 for a in row["repo_audits"] if a["syntax_check_ok"])
        lines.append(
            f"- `{row['chip_id']}`: repos `{row['repo_count']}`, syntax-ready `{ok}`, status `{row['exact_rerun_status']}`"
        )
    lines += ["", "## Experiment Families", ""]
    for family, summary in report["summaries"].items():
        lines.append(f"### {family}")
        lines.append("")
        lines.append(f"Rows: `{sum(1 for r in report['experiment_rows'] if r['family'] == family)}`")
        lines.append("")
        for rec in summary[:12]:
            compact = ", ".join(f"{k}={v}" for k, v in rec.items() if k != "n")
            lines.append(f"- n `{rec['n']}`: {compact}")
        if len(summary) > 12:
            lines.append(f"- ... `{len(summary) - 12}` more grouped rows in JSON")
        lines.append("")
    lines += ["## Inferred Research Gaps", ""]
    for gap in report["inferred_research_gaps"]:
        lines.append(f"- Gap: {gap['gap']}")
        lines.append(f"  Evidence: {gap['evidence']}")
        lines.append(f"  Paper move: {gap['paper_move']}")
    lines += ["", "## Strict NeurIPS DAG Delta", ""]
    for item in report["strict_neurips_dag_delta"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- JSON: `{OUTPUT_JSON}`",
        f"- Markdown: `{OUTPUT_MD}`",
        "",
    ]
    OUTPUT_MD.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    report = run_campaign(args)
    print(
        json.dumps(
            {
                "json": str(OUTPUT_JSON),
                "markdown": str(OUTPUT_MD),
                "rows": len(report["experiment_rows"]),
                "gaps": len(report["inferred_research_gaps"]),
                "runtime_seconds": report["runtime_seconds"],
                "device": report["device"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
