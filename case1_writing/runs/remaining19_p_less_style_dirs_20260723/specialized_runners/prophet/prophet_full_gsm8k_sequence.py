#!/usr/bin/env python3
"""Run full GSM8K baseline then Prophet lm-eval jobs sequentially on one GPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/Prophet")
RUN_ROOT = Path(
    "/tf/notebooks/yunbo/DIRS/case1_writing/runs/"
    "remaining19_p_less_style_dirs_20260723/specialized_runners/prophet"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_status(payload):
    (RUN_ROOT / "full_gsm8k_sequence_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_cmd(variant: str) -> list[str]:
    enable = "true" if variant == "prophet" else "false"
    out_dir = RUN_ROOT / f"lm_eval_gsm8k_full_{variant}"
    return [
        "accelerate",
        "launch",
        "--num_processes",
        "1",
        "eval_llada.py",
        "--tasks",
        "gsm8k_cot_zeroshot",
        "--model",
        "llada_dist",
        "--model_args",
        (
            "model_path='GSAI-ML/LLaDA-8B-Instruct',"
            f"enable_early_exit={enable},"
            "constraints_text=\"200:The|201:answer|202:is\","
            "gen_length=256,steps=256,block_length=32,answer_length=5"
        ),
        "--output_path",
        str(out_dir),
    ]


def run_variant(variant: str, gpu: str, status: dict) -> dict:
    out_dir = RUN_ROOT / f"lm_eval_gsm8k_full_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    cmd = build_cmd(variant)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["HF_HOME"] = "/tf/notebooks/.cache/huggingface"
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    start = time.time()
    row = {
        "variant": variant,
        "status": "running",
        "started_at_utc": utc_now(),
        "cmd": cmd,
        "cwd": str(REPO),
        "log_path": str(log_path),
        "output_dir": str(out_dir),
        "cuda_visible_devices": gpu,
        "logical_gpu_mapping_note": f"physical GPU {gpu} is logical cuda:0 inside this process",
    }
    status["runs"].append(row)
    status["current_variant"] = variant
    write_status(status)
    with log_path.open("ab", buffering=0) as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        row["pid"] = proc.pid
        write_status(status)
        returncode = proc.wait()
    row["finished_at_utc"] = utc_now()
    row["returncode"] = returncode
    row["seconds"] = round(time.time() - start, 3)
    row["status"] = "completed" if returncode == 0 else "failed"
    status["current_variant"] = None
    write_status(status)
    return row


def main() -> None:
    gpu = sys.argv[1] if len(sys.argv) > 1 else "3"
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    status = {
        "artifact_kind": "prophet_full_gsm8k_sequential_lm_eval",
        "convergence_role": "paper_shaped_long_running_evidence_until_verifier_accepts",
        "started_at_utc": utc_now(),
        "cuda_visible_devices": gpu,
        "runs": [],
        "current_variant": None,
    }
    write_status(status)
    for variant in ["baseline", "prophet"]:
        row = run_variant(variant, gpu, status)
        if row["returncode"] != 0:
            status["sequence_status"] = "failed_before_all_variants_finished"
            status["finished_at_utc"] = utc_now()
            write_status(status)
            return
    status["sequence_status"] = "completed_all_variants"
    status["finished_at_utc"] = utc_now()
    write_status(status)


if __name__ == "__main__":
    main()
