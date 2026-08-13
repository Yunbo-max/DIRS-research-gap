#!/usr/bin/env python3
"""Full GSM8K Prophet reproduction runner.

This runner exists because the official lm-eval integration can fail before it
writes any benchmark artifacts in the current environment. It keeps the same
paper-shaped evidence contract: full GSM8K test split, released LLaDA-8B
checkpoint, baseline full-step decoding, Prophet early-exit decoding, official
zero-shot GSM8K prompt shape, paper constraints, per-sample raw outputs, timing,
steps, and aggregate accuracy/speed summaries.
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

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


REPO = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/Prophet")
RUN_ROOT = Path(
    "/tf/notebooks/yunbo/DIRS/case1_writing/runs/"
    "remaining19_p_less_style_dirs_20260723/specialized_runners/prophet"
)
OUT_DIR = RUN_ROOT / "custom_full_gsm8k_llada8b"
ROWS_PATH = OUT_DIR / "per_sample_results.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"
STATUS_PATH = OUT_DIR / "status.json"
LOG_PATH = OUT_DIR / "runner.log"

sys.path.insert(0, str(REPO))

from generate import generate as baseline_generate  # noqa: E402
from generate_earlyexit import generate as prophet_generate  # noqa: E402


PROMPT_TEMPLATES = {
    "official_zero_shot": "Q: {question}\nA: Let's think step by step.",
    "trajectory_gsm8k_cot": (
        "Solve the following math problem step by step. The last line of your "
        "response should be of the form Answer: $ANSWER.\n\n{question}"
    ),
}
STOP_SEQUENCES = ["Q:", "</s>", "<|im_end|>"]


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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_answer(text: str) -> str:
    text = text.strip().replace("$", "").replace(",", "")
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".")
    number = re.search(r"-?\d+(?:\.\d+)?", text)
    return number.group(0) if number else text.lower()


def extract_gold(answer: str) -> str:
    match = re.search(r"####\s*(-?[$0-9.,]+)", answer)
    if match:
        return normalize_answer(match.group(1))
    numbers = re.findall(r"-?[$0-9.,]+", answer)
    return normalize_answer(numbers[-1]) if numbers else ""


def extract_strict_answer(text: str) -> str:
    answer_colon = re.search(r"Answer:\s*\$?(-?[0-9\.,]+)", text, flags=re.IGNORECASE)
    if answer_colon:
        return normalize_answer(answer_colon.group(1))
    boxed = re.search(r"The answer is\s+\$?\\boxed\{([^}]+)\}\$?\.?", text)
    if boxed:
        return normalize_answer(boxed.group(1))
    match = re.search(r"The answer is (\-?[0-9\.\,]+)\.", text)
    return normalize_answer(match.group(1)) if match else ""


def extract_flexible_answer(text: str) -> str:
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    for candidate in reversed(boxed_matches):
        normalized = normalize_answer(candidate)
        if re.search(r"\d", normalized):
            return normalized
    matches = re.findall(r"-?\$?[0-9][0-9,]*(?:\.[0-9]+)?", text)
    for candidate in reversed(matches):
        normalized = normalize_answer(candidate)
        if re.search(r"\d", normalized):
            return normalized
    return ""


def parse_constraints(text: str, tokenizer) -> dict[int, int]:
    constraints: dict[int, int] = {}
    if not text:
        return constraints
    for part in text.split("|"):
        if ":" not in part:
            continue
        pos_str, word = part.split(":", 1)
        try:
            pos = int(pos_str.strip())
        except ValueError:
            continue
        ids = tokenizer.encode(" " + word.strip(), add_special_tokens=False)
        for offset, token_id in enumerate(ids):
            constraints[pos + offset] = int(token_id)
    return constraints


def decode_generation(tokenizer, output, prompt_len: int) -> str:
    generated_ids = output[0, prompt_len:].detach().cpu().tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    for stop in STOP_SEQUENCES:
        if stop in generated_text:
            generated_text = generated_text.split(stop)[0]
    cleaned_ids = tokenizer(generated_text)["input_ids"]
    return tokenizer.decode(cleaned_ids, skip_special_tokens=True)


def nvidia_smi_short() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        return subprocess.check_output(cmd, text=True, timeout=10).strip()
    except Exception as exc:
        return f"nvidia-smi unavailable: {type(exc).__name__}: {exc}"


def existing_completed_indices(path: Path, variants: list[str]) -> set[int]:
    completed_variants = existing_completed_variants(path, variants)
    return {idx for idx, got in completed_variants.items() if all(v in got for v in variants)}


def existing_completed_variants(path: Path, variants: list[str]) -> dict[int, set[str]]:
    if not path.exists():
        return {}
    seen: dict[int, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = int(row.get("sample_index", -1))
            variant = row.get("variant")
            if idx >= 0 and variant in variants:
                seen.setdefault(idx, set()).add(variant)
    return seen


def summarize_rows(path: Path, total_samples: int, variants: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    for row in rows:
        variant = row.get("variant")
        if variant in by_variant:
            by_variant[variant].append(row)

    summary: dict[str, Any] = {
        "artifact_kind": "prophet_custom_full_gsm8k_summary",
        "created_at_utc": utc_now(),
        "status": "running_or_partial" if any(len(v) < total_samples for v in by_variant.values()) else "completed",
        "dataset": "openai/gsm8k main test",
        "total_samples": total_samples,
        "variants_requested": variants,
        "rows_path": str(path),
        "aggregates": {},
    }
    for variant, variant_rows in by_variant.items():
        completed = len(variant_rows)
        strict_correct = sum(1 for row in variant_rows if row.get("strict_exact_match"))
        flexible_correct = sum(1 for row in variant_rows if row.get("flexible_exact_match"))
        seconds = [float(row.get("seconds", 0.0)) for row in variant_rows]
        actual_steps = [
            int(row.get("exit_info", {}).get("actual_steps", row.get("steps", 0)))
            for row in variant_rows
        ]
        early_exits = sum(
            1 for row in variant_rows if row.get("exit_info", {}).get("early_exit_triggered")
        )
        summary["aggregates"][variant] = {
            "completed_samples": completed,
            "strict_exact_match": strict_correct / completed if completed else None,
            "flexible_exact_match": flexible_correct / completed if completed else None,
            "strict_correct_count": strict_correct,
            "flexible_correct_count": flexible_correct,
            "mean_seconds": sum(seconds) / completed if completed else None,
            "mean_actual_steps": sum(actual_steps) / completed if completed else None,
            "early_exit_count": early_exits,
            "early_exit_rate": early_exits / completed if completed else None,
        }
    if all(v in summary["aggregates"] for v in ("baseline", "prophet")):
        base = summary["aggregates"]["baseline"]
        prop = summary["aggregates"]["prophet"]
        if base["mean_seconds"] and prop["mean_seconds"]:
            summary["paired_shape"] = {
                "speedup_mean_seconds": base["mean_seconds"] / prop["mean_seconds"],
                "flexible_accuracy_delta": (
                    prop["flexible_exact_match"] - base["flexible_exact_match"]
                    if prop["flexible_exact_match"] is not None
                    and base["flexible_exact_match"] is not None
                    else None
                ),
                "mean_step_reduction": (
                    base["mean_actual_steps"] - prop["mean_actual_steps"]
                    if prop["mean_actual_steps"] is not None
                    and base["mean_actual_steps"] is not None
                    else None
                ),
            }
    return summary


def run_variant(model, tokenizer, prompt_ids, variant: str, args) -> tuple[str, dict[str, Any]]:
    constraints = parse_constraints(args.constraints_text, tokenizer)
    torch.cuda.synchronize()
    start = time.time()
    with torch.inference_mode():
        if variant == "baseline":
            output = baseline_generate(
                model,
                prompt_ids,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=0.0,
                cfg_scale=0.0,
                remasking=args.remasking,
                mask_id=args.mask_id,
                constraints=constraints,
            )
            exit_info = {
                "early_exit_triggered": False,
                "exit_decision_step": None,
                "total_steps": args.steps,
                "actual_steps": args.steps,
            }
        else:
            answer_start = max(constraints.keys()) + 2 if constraints else args.answer_start_offset
            output, gap_data = prophet_generate(
                model,
                prompt_ids,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=0.0,
                cfg_scale=0.0,
                remasking=args.remasking,
                mask_id=args.mask_id,
                constraints=constraints,
                analyze_gap=True,
                tokenizer=tokenizer,
                answer_start_pos=prompt_ids.shape[1] + answer_start,
                early_exit_thresholds={
                    "early": args.early_threshold,
                    "mid": args.mid_threshold,
                    "late": args.late_threshold,
                },
                measure_time=False,
            )
            exit_info = gap_data["exit_info"]
    torch.cuda.synchronize()
    seconds = time.time() - start
    generated_text = decode_generation(tokenizer, output, int(prompt_ids.shape[1]))
    return generated_text, {
        "seconds": round(seconds, 4),
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "remasking": args.remasking,
        "constraints_text": args.constraints_text,
        "prompt_profile": args.prompt_profile,
        "exit_info": exit_info,
    }


def prompt_template_from_args(args) -> str:
    if args.prompt_template:
        return args.prompt_template
    try:
        return PROMPT_TEMPLATES[args.prompt_profile]
    except KeyError as exc:
        raise ValueError(
            f"unsupported prompt profile {args.prompt_profile!r}; "
            f"available={sorted(PROMPT_TEMPLATES)}"
        ) from exc


def main() -> None:
    global OUT_DIR, ROWS_PATH, SUMMARY_PATH, STATUS_PATH, LOG_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "3"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--run-label", default="full_gsm8k_llada8b")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--constraints-text", default="200:The|201:answer|202:is")
    parser.add_argument("--answer-start-offset", type=int, default=200)
    parser.add_argument("--prompt-profile", default="official_zero_shot")
    parser.add_argument("--prompt-template", default=None)
    parser.add_argument("--early-threshold", type=float, default=7.5)
    parser.add_argument("--mid-threshold", type=float, default=5.0)
    parser.add_argument("--late-threshold", type=float, default=2.5)
    parser.add_argument("--mask-id", type=int, default=126336)
    parser.add_argument("--variants", default="baseline,prophet")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--status-every", type=int, default=1)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    invalid = [v for v in variants if v not in {"baseline", "prophet"}]
    if invalid:
        raise ValueError(f"unsupported variants: {invalid}")

    OUT_DIR = Path(args.out_dir)
    ROWS_PATH = OUT_DIR / "per_sample_results.jsonl"
    SUMMARY_PATH = OUT_DIR / "summary.json"
    STATUS_PATH = OUT_DIR / "status.json"
    LOG_PATH = OUT_DIR / "runner.log"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HOME", "/tf/notebooks/.cache/huggingface")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_seed(args.seed)
    prompt_template = prompt_template_from_args(args)

    status: dict[str, Any] = {
        "artifact_kind": "prophet_custom_full_gsm8k_status",
        "status": "starting",
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "logical_gpu_mapping_note": (
            f"physical GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '')} is logical cuda:0 "
            f"when CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
        ),
        "model_id": args.model_id,
        "dataset": "openai/gsm8k main test",
        "run_label": args.run_label,
        "variants": variants,
        "prompt_profile": args.prompt_profile,
        "prompt_template_source": "cli_prompt_template" if args.prompt_template else "named_prompt_profile",
        "convergence_role": (
            "full non-reduced paper-shaped operational evidence for Prophet "
            "full-step vs early-exit comparison"
        ),
        "lm_eval_failure_predecessor": str(RUN_ROOT / "full_gsm8k_sequence_status.json"),
        "rows_path": str(ROWS_PATH),
        "summary_path": str(SUMMARY_PATH),
        "gpu_snapshot": nvidia_smi_short(),
    }
    write_json(STATUS_PATH, status)
    log("loading dataset openai/gsm8k main test")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    total_samples = len(dataset)
    limit_stop = total_samples if args.max_samples is None else min(total_samples, args.start_index + args.max_samples)
    status.update(
        {
            "status": "loading_model",
            "updated_at_utc": utc_now(),
            "total_samples": total_samples,
            "effective_start_index": args.start_index,
            "effective_stop_index_exclusive": limit_stop,
            "full_split_requested": args.max_samples is None and args.start_index == 0,
            "gpu_snapshot": nvidia_smi_short(),
        }
    )
    write_json(STATUS_PATH, status)
    log(f"loading model {args.model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    completed_variants = existing_completed_variants(ROWS_PATH, variants)
    completed = {idx for idx, got in completed_variants.items() if all(v in got for v in variants)}
    log(
        f"starting full GSM8K loop: total={total_samples} "
        f"range=[{args.start_index},{limit_stop}) already_completed={len(completed)}"
    )
    status.update(
        {
            "status": "running",
            "updated_at_utc": utc_now(),
            "device": device,
            "already_completed_samples": len(completed),
            "gpu_snapshot": nvidia_smi_short(),
        }
    )
    write_json(STATUS_PATH, status)

    with ROWS_PATH.open("a", encoding="utf-8") as rows_handle:
        for idx in range(args.start_index, limit_stop):
            if idx in completed:
                continue
            already_done_variants = completed_variants.setdefault(idx, set())
            item = dataset[idx]
            prompt_text = prompt_template.format(question=item["question"])
            prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
            gold = extract_gold(item["answer"])
            sample_rows = []
            status.update(
                {
                    "status": "running",
                    "updated_at_utc": utc_now(),
                    "current_sample_index": idx,
                    "completed_sample_indices": len(completed),
                    "gpu_snapshot": nvidia_smi_short()
                    if idx % max(1, args.status_every) == 0
                    else status.get("gpu_snapshot"),
                }
            )
            write_json(STATUS_PATH, status)
            for variant in variants:
                if variant in already_done_variants:
                    log(f"sample {idx:04d} {variant}: already present in rows; skipping duplicate on resume")
                    continue
                generated_text, metrics = run_variant(model, tokenizer, prompt_ids, variant, args)
                strict = extract_strict_answer(generated_text)
                flexible = extract_flexible_answer(generated_text)
                row = {
                    "artifact_kind": "prophet_custom_full_gsm8k_per_sample",
                    "created_at_utc": utc_now(),
                    "sample_index": idx,
                    "variant": variant,
                    "question": item["question"],
                    "gold_answer": gold,
                    "strict_predicted_answer": strict,
                    "flexible_predicted_answer": flexible,
                    "strict_exact_match": strict == gold and gold != "",
                    "flexible_exact_match": flexible == gold and gold != "",
                    "generated_text": generated_text,
                    "prompt_profile": args.prompt_profile,
                    **metrics,
                }
                rows_handle.write(json.dumps(row, sort_keys=True) + "\n")
                rows_handle.flush()
                os.fsync(rows_handle.fileno())
                sample_rows.append(row)
                log(
                    "sample "
                    f"{idx:04d} {variant}: strict={row['strict_predicted_answer']!r} "
                    f"flex={row['flexible_predicted_answer']!r} gold={gold!r} "
                    f"ok={row['flexible_exact_match']} "
                    f"steps={row['exit_info'].get('actual_steps')} "
                    f"sec={row['seconds']:.2f}"
                )
                already_done_variants.add(variant)
            if all(v in already_done_variants for v in variants):
                completed.add(idx)
            if idx % max(1, args.status_every) == 0:
                summary = summarize_rows(ROWS_PATH, total_samples, variants)
                write_json(SUMMARY_PATH, summary)
                status.update(
                    {
                        "updated_at_utc": utc_now(),
                        "completed_sample_indices": len(completed),
                        "last_rows": sample_rows,
                        "running_summary": summary,
                    }
                )
                write_json(STATUS_PATH, status)
            torch.cuda.empty_cache()
            gc.collect()

    summary = summarize_rows(ROWS_PATH, total_samples, variants)
    summary["status"] = "completed" if len(completed) >= (limit_stop - args.start_index) else summary["status"]
    summary["finished_at_utc"] = utc_now()
    write_json(SUMMARY_PATH, summary)
    status.update(
        {
            "status": "completed",
            "finished_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "completed_sample_indices": len(completed),
            "summary": summary,
            "gpu_snapshot": nvidia_smi_short(),
        }
    )
    write_json(STATUS_PATH, status)
    log("completed full GSM8K custom run")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_kind": "prophet_custom_full_gsm8k_status",
            "status": "failed",
            "failed_at_utc": utc_now(),
            "pid": os.getpid(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "rows_path": str(ROWS_PATH),
            "summary_path": str(SUMMARY_PATH),
            "gpu_snapshot": nvidia_smi_short(),
        }
        write_json(STATUS_PATH, failure)
        log(f"failed: {type(exc).__name__}: {exc}")
        raise
