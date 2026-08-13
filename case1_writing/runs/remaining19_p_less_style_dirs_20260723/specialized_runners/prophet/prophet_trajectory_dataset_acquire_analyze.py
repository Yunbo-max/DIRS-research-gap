#!/usr/bin/env python3
"""Acquire and analyze the full Prophet trajectory artifact dataset.

This is a CPU/network operational node for the Prophet paper. It downloads the
released full trajectory dataset and computes the notebook's core
first-correct-answer-emergence statistics over all expected GSM8K and MMLU
trajectory folders. It is not a reduced sample path.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import HfApi, HfFolder, snapshot_download
from transformers import AutoTokenizer


RUN_ROOT = Path(
    "/tf/notebooks/yunbo/DIRS/case1_writing/runs/"
    "remaining19_p_less_style_dirs_20260723/specialized_runners/prophet"
)
OUT_DIR = RUN_ROOT / "trajectory_dataset_analysis"
SNAPSHOT_DIR = OUT_DIR / "DLM-Decoding-Analysis"
STATUS_PATH = OUT_DIR / "trajectory_dataset_status.json"
MANIFEST_PATH = OUT_DIR / "trajectory_dataset_manifest.json"
SUMMARY_PATH = OUT_DIR / "trajectory_analysis_summary.json"
ROWS_PATH = OUT_DIR / "trajectory_first_emergence_rows.jsonl"
LOG_PATH = OUT_DIR / "trajectory_acquire_analyze.log"

REPO_ID = "YefanZhou98/DLM-Decoding-Analysis"
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"

SETTINGS = [
    {
        "setting_id": "gsm8k_low_conf_none_block32",
        "domain": "gsm8k",
        "folder": "question_histories_low_conf_none_index_genlen_step256_blocklen32",
        "expected_count": 1319,
        "steps": 256,
        "needs_tokenizer_decode": False,
    },
    {
        "setting_id": "gsm8k_low_conf_constraint_block32",
        "domain": "gsm8k",
        "folder": "question_histories_low_conf_constraint_index_genlen_step256_blocklen32",
        "expected_count": 1319,
        "steps": 256,
        "needs_tokenizer_decode": False,
    },
    {
        "setting_id": "gsm8k_random_none_block256",
        "domain": "gsm8k",
        "folder": "question_histories_random_none_index_genlen_step256_blocklen256",
        "expected_count": 1319,
        "steps": 256,
        "needs_tokenizer_decode": False,
    },
    {
        "setting_id": "gsm8k_random_constraint_block256",
        "domain": "gsm8k",
        "folder": "question_histories_random_constraint_index_genlen_step256_blocklen256",
        "expected_count": 1319,
        "steps": 256,
        "needs_tokenizer_decode": False,
    },
    {
        "setting_id": "mmlu_low_confidence_none_block128",
        "domain": "mmlu",
        "folder": "question_histories_mmlu_low_confidence_none_index_genlen_step128_blocklen128",
        "expected_count": 3153,
        "steps": 128,
        "needs_tokenizer_decode": True,
    },
    {
        "setting_id": "mmlu_low_confidence_constraint_block128",
        "domain": "mmlu",
        "folder": "question_histories_mmlu_low_confidence_constraint_index_genlen_step128_blocklen128",
        "expected_count": 3153,
        "steps": 128,
        "needs_tokenizer_decode": True,
    },
    {
        "setting_id": "mmlu_random_none_block128",
        "domain": "mmlu",
        "folder": "question_histories_mmlu_random_none_index_genlen_step128_blocklen128",
        "expected_count": 3153,
        "steps": 128,
        "needs_tokenizer_decode": True,
    },
    {
        "setting_id": "mmlu_random_constraint_block128",
        "domain": "mmlu",
        "folder": "question_histories_mmlu_random_constraint_index_genlen_step128_blocklen128",
        "expected_count": 3153,
        "steps": 128,
        "needs_tokenizer_decode": True,
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def log(message: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def disk_snapshot() -> str:
    try:
        return subprocess.check_output(["df", "-h", str(OUT_DIR.parent)], text=True, timeout=10)
    except Exception as exc:
        return f"df unavailable: {type(exc).__name__}: {exc}"


def dataset_manifest(token: str | None = None) -> dict[str, Any]:
    api = HfApi(token=token)
    info = api.repo_info(REPO_ID, repo_type="dataset", files_metadata=True)
    files = []
    by_folder: dict[str, dict[str, Any]] = {}
    for sibling in info.siblings:
        size = getattr(sibling, "size", None) or 0
        name = sibling.rfilename
        folder = name.split("/")[0] if "/" in name else "."
        files.append({"path": name, "size_bytes": size})
        entry = by_folder.setdefault(folder, {"file_count": 0, "size_bytes": 0})
        entry["file_count"] += 1
        entry["size_bytes"] += size
    payload = {
        "artifact_kind": "prophet_dlm_decoding_analysis_hf_manifest",
        "created_at_utc": utc_now(),
        "repo_id": REPO_ID,
        "repo_type": "dataset",
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "by_folder": by_folder,
        "expected_settings": SETTINGS,
        "cached_hf_token_available": bool(HfFolder.get_token()),
        "token_explicitly_passed": bool(token),
        "files_head": files[:40],
        "files_tail": files[-40:],
    }
    write_json(MANIFEST_PATH, payload)
    return payload


def file_count_for_folder(folder: Path) -> int:
    return len(list(folder.glob("question_*_steps_*.pt"))) if folder.exists() else 0


def normalize_token_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().flatten().tolist()]
    if isinstance(value, np.ndarray):
        return [int(x) for x in value.flatten().tolist()]
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, tuple):
        return [int(x) for x in value]
    return []


def as_correct(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().item())
    return bool(value)


def normalize_number(num_str: Any) -> str:
    cleaned = str(num_str or "").replace("$", "").replace(",", "").strip()
    if not cleaned:
        return ""
    try:
        value = float(cleaned)
    except (TypeError, ValueError):
        return cleaned
    if value.is_integer():
        return str(int(value))
    return str(value).rstrip("0").rstrip(".")


def extract_answer_from_prediction(prediction: Any) -> str:
    text = str(prediction or "")
    match = re.search(r"Answer:([^\d-]*)?(-?\d+[\d,\.]*)", text, re.IGNORECASE)
    if match:
        return normalize_number(match.group(2))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return normalize_number(numbers[-1]) if numbers else ""


def author_notebook_correctness(data: dict[str, Any], setting: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    stored_correct = as_correct(data.get("correct", False))
    if setting["domain"] != "gsm8k":
        return stored_correct, {
            "correct_source": "stored_correct_flag",
            "stored_correct": stored_correct,
        }

    extracted = extract_answer_from_prediction(data.get("pred_text", ""))
    normalized_gt = normalize_number(data.get("gt_text", ""))
    recomputed_correct = extracted == normalized_gt
    return recomputed_correct, {
        "correct_source": "author_notebook_gsm8k_recomputed_final_answer",
        "stored_correct": stored_correct,
        "recomputed_correct": recomputed_correct,
        "extracted_answer": extracted,
        "normalized_gt_answer": normalized_gt,
    }


def load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def first_emergence_row(path: Path, setting: dict[str, Any], tokenizer) -> dict[str, Any]:
    data = load_pt(path)
    correct, correctness_meta = author_notebook_correctness(data, setting)
    row: dict[str, Any] = {
        "setting_id": setting["setting_id"],
        "domain": setting["domain"],
        "path": str(path),
        "question_index": int(path.stem.split("_")[1]),
        "correct": correct,
        "first_emergence_step": None,
        "first_emergence_pct": None,
        "durability_steps": None,
        "total_steps_observed": None,
        **correctness_meta,
    }
    if not correct:
        return row

    x0_history_blocks = data.get("x0_history", [])
    if isinstance(x0_history_blocks, torch.Tensor):
        all_steps = x0_history_blocks
    else:
        all_steps = torch.cat(list(x0_history_blocks), dim=0)
    if all_steps.ndim == 3 and all_steps.shape[1] == 1:
        all_steps = all_steps[:, 0, :]
    row["total_steps_observed"] = int(all_steps.shape[0])

    ans_pos = int(data.get("ans_posidx", -1))
    pred_token_id = normalize_token_list(data.get("pred_token_id", []))
    pred_ans = str(data.get("pred_ans", "")).strip()
    if ans_pos < 0 or not pred_token_id:
        row["failure_reason"] = "missing_answer_position_or_pred_token_id"
        return row

    appear_steps: list[int] = []
    pred_len = len(pred_token_id)
    for step_i in range(max(0, all_steps.shape[0] - 1)):
        cand = normalize_token_list(all_steps[step_i, ans_pos : ans_pos + pred_len])
        matched = cand == pred_token_id
        if not matched and tokenizer is not None and pred_ans:
            decoded = tokenizer.decode(cand)
            matched = decoded == pred_ans or decoded == " " + pred_ans
        if matched:
            appear_steps.append(step_i)
    appear_steps.append(int(all_steps.shape[0] - 1))

    first = int(appear_steps[0])
    row.update(
        {
            "first_emergence_step": first,
            "first_emergence_pct": first / float(setting["steps"]) * 100.0,
            "durability_steps": int(appear_steps[-1] - appear_steps[0]),
            "appear_step_count": len(appear_steps),
        }
    )
    return row


def summarize_setting(rows: list[dict[str, Any]], setting: dict[str, Any]) -> dict[str, Any]:
    correct_rows = [row for row in rows if row.get("correct")]
    stored_disagreements = [
        row
        for row in rows
        if row.get("correct_source") == "author_notebook_gsm8k_recomputed_final_answer"
        and row.get("stored_correct") != row.get("recomputed_correct")
    ]
    emergence = np.array(
        [row["first_emergence_pct"] for row in correct_rows if row["first_emergence_pct"] is not None],
        dtype=float,
    )
    durability = np.array(
        [row["durability_steps"] for row in correct_rows if row["durability_steps"] is not None],
        dtype=float,
    )
    hist_counts = None
    hist_bins = None
    if len(emergence):
        hist_counts, hist_bins = np.histogram(emergence, bins=20, range=(0.0, 100.0))
    return {
        "setting_id": setting["setting_id"],
        "domain": setting["domain"],
        "folder": setting["folder"],
        "expected_count": setting["expected_count"],
        "observed_count": len(rows),
        "correct_count": len(correct_rows),
        "emergence_count": int(len(emergence)),
        "correct_source": (
            "author_notebook_gsm8k_recomputed_final_answer"
            if setting["domain"] == "gsm8k"
            else "stored_correct_flag"
        ),
        "stored_correct_disagreement_count": len(stored_disagreements),
        "stored_correct_disagreement_examples": [
            {
                "question_index": row.get("question_index"),
                "stored_correct": row.get("stored_correct"),
                "recomputed_correct": row.get("recomputed_correct"),
                "extracted_answer": row.get("extracted_answer"),
                "normalized_gt_answer": row.get("normalized_gt_answer"),
            }
            for row in stored_disagreements[:8]
        ],
        "correct_rate_in_released_trajectory": len(correct_rows) / len(rows) if rows else None,
        "pct_correct_emerged_by_25pct_steps": float((emergence <= 25.0).mean() * 100.0)
        if len(emergence)
        else None,
        "pct_correct_emerged_by_50pct_steps": float((emergence <= 50.0).mean() * 100.0)
        if len(emergence)
        else None,
        "mean_first_emergence_pct": float(emergence.mean()) if len(emergence) else None,
        "median_first_emergence_pct": float(np.median(emergence)) if len(emergence) else None,
        "p90_first_emergence_pct": float(np.percentile(emergence, 90)) if len(emergence) else None,
        "mean_durability_steps": float(durability.mean()) if len(durability) else None,
        "hist_20bin_counts_0_100": hist_counts.tolist() if hist_counts is not None else None,
        "hist_20bin_edges_0_100": hist_bins.tolist() if hist_bins is not None else None,
    }


def analyze_snapshot(args) -> dict[str, Any]:
    tokenizer = None
    if any(setting["needs_tokenizer_decode"] for setting in SETTINGS):
        log(f"loading tokenizer {MODEL_ID} for MMLU answer emergence checks")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    ROWS_PATH.write_text("", encoding="utf-8")
    for setting in SETTINGS:
        folder = SNAPSHOT_DIR / setting["folder"]
        files = sorted(folder.glob("question_*_steps_*.pt"))
        log(f"analyzing {setting['setting_id']} files={len(files)} expected={setting['expected_count']}")
        setting_rows = []
        if len(files) != setting["expected_count"]:
            log(f"count mismatch for {setting['setting_id']}: observed {len(files)}")
        with ROWS_PATH.open("a", encoding="utf-8") as handle:
            for path in files:
                row = first_emergence_row(path, setting, tokenizer if setting["needs_tokenizer_decode"] else None)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                setting_rows.append(row)
                all_rows.append(row)
        summaries.append(summarize_setting(setting_rows, setting))
        write_json(
            STATUS_PATH,
            {
                "artifact_kind": "prophet_trajectory_dataset_status",
                "status": "analyzing",
                "updated_at_utc": utc_now(),
                "current_setting": setting["setting_id"],
                "settings_completed": len(summaries),
                "total_settings": len(SETTINGS),
                "rows_written": len(all_rows),
                "snapshot_dir": str(SNAPSHOT_DIR),
                "summary_path": str(SUMMARY_PATH),
            },
        )
        gc.collect()

    payload = {
        "artifact_kind": "prophet_full_trajectory_emergence_analysis",
        "created_at_utc": utc_now(),
        "repo_id": REPO_ID,
        "snapshot_dir": str(SNAPSHOT_DIR),
        "rows_path": str(ROWS_PATH),
        "manifest_path": str(MANIFEST_PATH),
        "settings": summaries,
        "professional_gate": {
            "full_dataset_expected_files": sum(setting["expected_count"] for setting in SETTINGS),
            "full_dataset_observed_files": sum(item["observed_count"] for item in summaries),
            "all_expected_counts_present": all(
                item["observed_count"] == item["expected_count"] for item in summaries
            ),
            "not_reduced": True,
            "paper_figure_logic": (
                "first correct answer emergence percentage over correct trajectories at 25% and 50% decoding thresholds; "
                "GSM8K correctness is recomputed from generated pred_text vs gt_text following analysis/visualize.ipynb"
            ),
        },
    }
    write_json(SUMMARY_PATH, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--token", default="auto", choices=["auto", "true", "false"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    cached_token = HfFolder.get_token()
    token = cached_token if args.token == "true" else None if args.token == "false" else cached_token
    manifest = dataset_manifest(token=token)
    write_json(
        STATUS_PATH,
        {
            "artifact_kind": "prophet_trajectory_dataset_status",
            "status": "starting",
            "started_at_utc": started,
            "updated_at_utc": started,
            "repo_id": REPO_ID,
            "snapshot_dir": str(SNAPSHOT_DIR),
            "manifest_path": str(MANIFEST_PATH),
            "summary_path": str(SUMMARY_PATH),
            "rows_path": str(ROWS_PATH),
            "expected_file_count": manifest["file_count"],
            "expected_size_bytes": manifest["total_size_bytes"],
            "cached_hf_token_available": bool(cached_token),
            "passing_hf_token": bool(token),
            "disk_snapshot": disk_snapshot(),
            "pid": os.getpid(),
        },
    )
    if not args.skip_download:
        log(f"downloading full trajectory dataset {REPO_ID} to {SNAPSHOT_DIR}")
        write_json(
            STATUS_PATH,
            {
                "artifact_kind": "prophet_trajectory_dataset_status",
                "status": "downloading",
                "started_at_utc": started,
                "updated_at_utc": utc_now(),
                "repo_id": REPO_ID,
                "snapshot_dir": str(SNAPSHOT_DIR),
                "expected_file_count": manifest["file_count"],
                "expected_size_bytes": manifest["total_size_bytes"],
                "cached_hf_token_available": bool(cached_token),
                "passing_hf_token": bool(token),
                "max_workers": args.max_workers,
                "disk_snapshot": disk_snapshot(),
                "pid": os.getpid(),
            },
        )
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(SNAPSHOT_DIR),
            allow_patterns=["README.md", "question_histories_*/*.pt"],
            max_workers=args.max_workers,
            resume_download=True,
            token=token,
        )
    counts = {
        setting["setting_id"]: file_count_for_folder(SNAPSHOT_DIR / setting["folder"])
        for setting in SETTINGS
    }
    write_json(
        STATUS_PATH,
        {
            "artifact_kind": "prophet_trajectory_dataset_status",
            "status": "downloaded" if not args.download_only else "download_only_completed",
            "started_at_utc": started,
            "updated_at_utc": utc_now(),
            "repo_id": REPO_ID,
            "snapshot_dir": str(SNAPSHOT_DIR),
            "folder_file_counts": counts,
            "all_expected_counts_present": all(
                counts[setting["setting_id"]] == setting["expected_count"] for setting in SETTINGS
            ),
            "disk_snapshot": disk_snapshot(),
            "pid": os.getpid(),
        },
    )
    if args.download_only:
        log("download-only mode complete")
        return
    log("starting full trajectory emergence analysis")
    summary = analyze_snapshot(args)
    write_json(
        STATUS_PATH,
        {
            "artifact_kind": "prophet_trajectory_dataset_status",
            "status": "completed",
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "repo_id": REPO_ID,
            "snapshot_dir": str(SNAPSHOT_DIR),
            "summary_path": str(SUMMARY_PATH),
            "rows_path": str(ROWS_PATH),
            "professional_gate": summary["professional_gate"],
            "disk_snapshot": disk_snapshot(),
            "pid": os.getpid(),
        },
    )
    log("trajectory analysis complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(
            STATUS_PATH,
            {
                "artifact_kind": "prophet_trajectory_dataset_status",
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "pid": os.getpid(),
                "snapshot_dir": str(SNAPSHOT_DIR),
                "summary_path": str(SUMMARY_PATH),
                "rows_path": str(ROWS_PATH),
                "disk_snapshot": disk_snapshot(),
            },
        )
        log(f"failed: {type(exc).__name__}: {exc}")
        raise
