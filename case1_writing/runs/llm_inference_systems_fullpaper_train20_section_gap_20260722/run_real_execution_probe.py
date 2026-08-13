#!/usr/bin/env python3
"""Collect local execution evidence for the systems/token-efficiency DAG.

The benchmark is deliberately lightweight: it avoids downloading models or
calling paid APIs, but it does exercise real CUDA kernels for decode-like
single-token attention over a KV cache.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
PROBE_PATH = RUN_DIR / "real_execution_probe.json"
BENCH_PATH = RUN_DIR / "gpu_microbenchmark.json"


def run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def parse_nvidia_smi() -> list[dict]:
    rc, out, err = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0:
        return [{"error": err or out or "nvidia-smi failed"}]
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, name, mem_total, mem_used, driver, compute_cap = parts[:6]
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "memory_total_mib": int(float(mem_total)),
                "memory_used_mib": int(float(mem_used)),
                "driver_version": driver,
                "compute_capability": compute_cap,
            }
        )
    return gpus


def torch_probe(gpus: list[dict]) -> tuple[dict, dict | None]:
    probe = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "openai_api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "paid_api_invoked": False,
        "api_policy": "Do not invoke paid/external APIs unless explicitly requested; record backend provenance instead.",
    }
    try:
        import torch

        probe.update(
            {
                "torch_available": True,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
            }
        )
        if not torch.cuda.is_available():
            return probe, None

        device_count = torch.cuda.device_count()
        smi_by_index = {gpu.get("index"): gpu for gpu in gpus if "index" in gpu}
        visible = []
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            visible.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": list(torch.cuda.get_device_capability(idx)),
                    "memory_total_mib": props.total_memory // (1024 * 1024),
                    "memory_used_mib_nvidia_smi": smi_by_index.get(idx, {}).get("memory_used_mib"),
                }
            )
        probe["torch_visible_gpus"] = visible

        candidates = [gpu for gpu in visible if isinstance(gpu.get("memory_used_mib_nvidia_smi"), int)]
        chosen = min(candidates or visible, key=lambda gpu: gpu.get("memory_used_mib_nvidia_smi") or 0)
        return probe, chosen
    except Exception as exc:  # pragma: no cover - diagnostic path
        probe.update({"torch_available": False, "torch_probe_error": f"{type(exc).__name__}: {exc}"})
        return probe, None


def run_microbenchmark(chosen_gpu: dict | None) -> dict:
    if chosen_gpu is None:
        return {
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "ran": False,
            "reason": "CUDA unavailable",
        }

    import torch

    device = torch.device(f"cuda:{chosen_gpu['index']}")
    dtype = torch.float16
    torch.cuda.set_device(device)
    torch.manual_seed(20260722)
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True

    # Decode-like single-token attention. Each iteration computes q @ K^T,
    # softmax, and weighted sum over V for one new token.
    batch = 1
    heads = 32
    head_dim = 128
    context_lengths = [512, 2048, 4096]
    warmup = 20
    timed_iters = 120
    rows = []

    for context in context_lengths:
        q = torch.randn(batch, heads, 1, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, heads, context, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, heads, context, head_dim, device=device, dtype=dtype)
        scale = head_dim ** -0.5

        def step() -> torch.Tensor:
            attn = torch.matmul(q, k.transpose(-1, -2)) * scale
            probs = torch.softmax(attn.float(), dim=-1).to(dtype)
            return torch.matmul(probs, v)

        for _ in range(warmup):
            step()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        checksum = 0.0
        for _ in range(timed_iters):
            out = step()
            checksum += float(out.float().mean().detach().cpu())
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        ms_per_token = elapsed * 1000.0 / timed_iters
        rows.append(
            {
                "context_tokens": context,
                "batch": batch,
                "heads": heads,
                "head_dim": head_dim,
                "dtype": str(dtype).replace("torch.", ""),
                "timed_iters": timed_iters,
                "ms_per_decode_token": round(ms_per_token, 4),
                "decode_tokens_per_second": round(1000.0 / ms_per_token, 2),
                "checksum": round(checksum, 6),
            }
        )

    memory = {
        "max_allocated_mib": round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2),
        "max_reserved_mib": round(torch.cuda.max_memory_reserved(device) / (1024 * 1024), 2),
    }
    return {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "ran": True,
        "benchmark_type": "synthetic_decode_kv_attention_microbenchmark",
        "chosen_gpu": chosen_gpu,
        "memory": memory,
        "results": rows,
    }


def main() -> None:
    gpus = parse_nvidia_smi()
    probe, chosen = torch_probe(gpus)
    probe.update(
        {
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "nvidia_smi_gpus": gpus,
            "selected_gpu_for_microbenchmark": chosen,
        }
    )
    PROBE_PATH.write_text(json.dumps(probe, indent=2, sort_keys=True))
    bench = run_microbenchmark(chosen)
    BENCH_PATH.write_text(json.dumps(bench, indent=2, sort_keys=True))
    print(json.dumps({"probe": str(PROBE_PATH), "benchmark": str(BENCH_PATH), "benchmark_ran": bench["ran"]}, indent=2))


if __name__ == "__main__":
    main()
