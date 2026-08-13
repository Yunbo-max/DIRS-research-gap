#!/usr/bin/env python3
"""Professional-scale p-less author simulation runner.

This runner is intentionally not a smoke test. It follows the paper-shaped DAG:

- official p-less sampler code
- target paper model family where available
- multiple benchmark families
- all paper temperatures
- all paper sampler baselines
- raw generations, scoring rows, token timing, CPU/RAM traces, and status files

It may run for many hours and checkpoints after each generation. Reduced runs
are never marked as convergence evidence by this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import psutil
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


RUN_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_DIR = RUN_DIR / "operational_dag_reproduction_loop" / "iter_02" / "blind_workspace" / "external" / "p-less-sampling"
DEFAULT_OUT_DIR = RUN_DIR / "professional_scale_author_simulation"
DEFAULT_HF_HOME = RUN_DIR / "operational_shared_hf_cache"
STATUS_MD = RUN_DIR / "LONGGOAL_STATUS.md"


TEMPERATURES = [0.5, 0.7, 1.0, 1.5, 2.0]
SAMPLERS = ["top_p", "min_p", "epsilon", "eta", "mirostat", "p_less", "p_lessnorm"]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def run_cmd(cmd: list[str], timeout: int = 20) -> dict:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        return {"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
    except Exception as exc:
        return {"returncode": None, "error": repr(exc)}


def gpu_snapshot() -> dict:
    result = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    rows = []
    for line in result.get("stdout", "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 6:
            rows.append(
                {
                    "index": int(float(parts[0])),
                    "name": parts[1],
                    "memory_total_mib": int(float(parts[2])),
                    "memory_used_mib": int(float(parts[3])),
                    "memory_free_mib": int(float(parts[4])),
                    "utilization_gpu_percent": int(float(parts[5])),
                }
            )
    return {"created_at_utc": now_utc(), "gpus": rows, "raw": result}


def load_official_sampler(repo_dir: Path):
    sampler_file = repo_dir / "p_less_samplers.py"
    spec = importlib.util.spec_from_file_location("official_p_less_samplers", sampler_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {sampler_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick_next_token(probs: torch.Tensor, sampler_name: str, official) -> torch.Tensor:
    probs = probs.clone()
    if sampler_name == "p_less":
        return official.p_less_decode(probs.unsqueeze(0))[0]
    if sampler_name == "p_lessnorm":
        return official.p_less_norm_decode(probs.unsqueeze(0))[0]
    if sampler_name == "top_p":
        values, indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(values, dim=-1)
        keep = cumsum <= 0.90
        keep[0] = True
        crossing = torch.argmax((cumsum >= 0.90).to(torch.int64))
        keep[crossing] = True
        masked = torch.zeros_like(probs)
        masked[indices[keep]] = probs[indices[keep]]
        masked = masked / masked.sum()
        return torch.multinomial(masked, 1)
    if sampler_name == "min_p":
        mask = probs >= probs.max() * 0.05
        masked = probs * mask
        masked = masked / masked.sum()
        return torch.multinomial(masked, 1)
    if sampler_name == "epsilon":
        mask = probs >= 0.0005
        if not mask.any():
            mask[torch.argmax(probs)] = True
        masked = probs * mask
        masked = masked / masked.sum()
        return torch.multinomial(masked, 1)
    if sampler_name == "eta":
        entropy = -(probs * torch.log(torch.clamp(probs, min=1e-30))).sum()
        threshold = min(0.0005, (0.0005 ** 0.5) * float(torch.exp(-entropy)))
        mask = probs >= threshold
        if not mask.any():
            mask[torch.argmax(probs)] = True
        masked = probs * mask
        masked = masked / masked.sum()
        return torch.multinomial(masked, 1)
    if sampler_name == "mirostat":
        surprisal = -torch.log(torch.clamp(probs, min=1e-30))
        mask = (surprisal >= 3.5) & (surprisal <= 6.5)
        if not mask.any():
            mask[torch.argmax(probs)] = True
        masked = probs * mask
        masked = masked / masked.sum()
        return torch.multinomial(masked, 1)
    raise ValueError(f"unknown sampler {sampler_name}")


def choice_text(choices: object) -> str:
    if isinstance(choices, dict):
        labels = choices.get("label") or choices.get("labels") or []
        texts = choices.get("text") or choices.get("texts") or []
        return "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    if isinstance(choices, list):
        rows = []
        for idx, choice in enumerate(choices):
            label = chr(ord("A") + idx)
            text = str(choice)
            if isinstance(choice, dict):
                label = str(choice.get("label", label))
                text = str(choice.get("text", choice.get("answer", choice)))
            rows.append(f"{label}. {text}")
        return "\n".join(rows)
    return str(choices)


def gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        answer = answer.split("####")[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
    return nums[-1] if nums else answer.strip()


def extract_number(text: str) -> str | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def extract_choice(text: str) -> str | None:
    match = re.search(r"(?:answer is|answer:|option)\s*([A-J])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-J])\b", text)
    return match.group(1).upper() if match else None


def load_examples(dataset_name: str, split: str, limit: int, offset: int = 0) -> list[dict]:
    ds = load_dataset(dataset_name, split=split, streaming=True)
    out = []
    for row in itertools.islice(ds, offset, offset + limit):
        out.append(dict(row))
    return out


def load_named_examples(name: str, limit: int) -> tuple[list[dict], list[dict], dict]:
    meta = {"name": name, "status": "pass", "limit": limit}
    try:
        if name == "gsm8k":
            train = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
            test = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
            shots = list(itertools.islice(train, 8))
            rows = [dict(row) for row in itertools.islice(test, limit)]
        elif name == "csqa":
            train = load_dataset("tau/commonsense_qa", split="train", streaming=True)
            val = load_dataset("tau/commonsense_qa", split="validation", streaming=True)
            shots = list(itertools.islice(train, 8))
            rows = [dict(row) for row in itertools.islice(val, limit)]
        elif name == "qasc":
            train = load_dataset("allenai/qasc", split="train", streaming=True)
            val = load_dataset("allenai/qasc", split="validation", streaming=True)
            shots = list(itertools.islice(train, 8))
            rows = [dict(row) for row in itertools.islice(val, limit)]
        elif name == "writingprompts":
            train = load_dataset("euclaise/writingprompts", split="train", streaming=True)
            shots = []
            rows = [dict(row) for row in itertools.islice(train, limit)]
        else:
            raise ValueError(name)
        return [dict(row) for row in shots], rows, meta
    except Exception as exc:
        meta["status"] = "blocked"
        meta["error"] = str(exc).splitlines()[0][:500]
        return [], [], meta


def build_prompt(dataset: str, shots: list[dict], row: dict) -> tuple[str, str | None, str]:
    if dataset == "gsm8k":
        shot_text = []
        for item in shots:
            answer = str(item["answer"])
            rationale = answer.split("####")[0].strip()
            final = gsm8k_gold(answer)
            shot_text.append(f"Question: {item['question']}\nLet's think step by step.\n{rationale}\nThe answer is {final}.")
        prompt = "\n\n".join(shot_text)
        prompt += f"\n\nQuestion: {row['question']}\nLet's think step by step."
        return prompt, gsm8k_gold(str(row["answer"])), "numeric_exact"
    if dataset in {"csqa", "qasc"}:
        shot_text = []
        for item in shots:
            choices = choice_text(item.get("choices"))
            question = item.get("question") or item.get("formatted_question") or ""
            answer = item.get("answerKey") or item.get("answerkey")
            shot_text.append(f"Question: {question}\nChoices:\n{choices}\nAnswer: {answer}")
        choices = choice_text(row.get("choices"))
        question = row.get("question") or row.get("formatted_question") or ""
        prompt = "\n\n".join(shot_text)
        prompt += f"\n\nQuestion: {question}\nChoices:\n{choices}\nAnswer:"
        return prompt, str(row.get("answerKey") or row.get("answerkey")), "choice_exact"
    if dataset == "writingprompts":
        prompt = f"Write a coherent story from this prompt:\n{row.get('prompt', '')}\nStory:"
        return prompt, None, "writing_proxy"
    raise ValueError(dataset)


def generate_one(
    model,
    tokenizer,
    prompt: str,
    sampler_name: str,
    temperature: float,
    max_new_tokens: int,
    device: str,
    official,
    seed: int,
) -> tuple[str, list[dict], list[dict]]:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    process = psutil.Process(os.getpid())
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    generated_tokens: list[int] = []
    timing_rows: list[dict] = []
    cpu_ram_rows: list[dict] = []
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        forward_start = time.perf_counter()
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        prefill_seconds = time.perf_counter() - forward_start
        logits = output.logits[:, -1, :]
        past = output.past_key_values

        for step in range(max_new_tokens):
            rss_before = process.memory_info().rss
            token_start = time.perf_counter()
            probs = torch.softmax((logits[0] / float(temperature)).float(), dim=-1)
            sample_start = time.perf_counter()
            next_token = pick_next_token(probs, sampler_name, official).reshape(1, 1).to(device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            sample_seconds = time.perf_counter() - sample_start

            forward_start = time.perf_counter()
            output = model(input_ids=next_token, past_key_values=past, use_cache=True)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - forward_start
            past = output.past_key_values
            logits = output.logits[:, -1, :]
            generated_tokens.append(int(next_token.item()))
            rss_after = process.memory_info().rss
            timing_rows.append(
                {
                    "step": step,
                    "sampler": sampler_name,
                    "temperature": temperature,
                    "prefill_seconds": prefill_seconds if step == 0 else 0.0,
                    "sample_seconds": sample_seconds,
                    "forward_seconds": forward_seconds,
                    "token_seconds": time.perf_counter() - token_start,
                }
            )
            cpu_ram_rows.append(
                {
                    "step": step,
                    "sampler": sampler_name,
                    "temperature": temperature,
                    "rss_gb": rss_after / (1024**3),
                    "rss_delta_gb": (rss_after - rss_before) / (1024**3),
                }
            )
            if eos_id is not None and int(next_token.item()) == int(eos_id):
                break

    return tokenizer.decode(generated_tokens, skip_special_tokens=True), timing_rows, cpu_ram_rows


def score_generation(dataset: str, scoring_kind: str, gold: str | None, text: str) -> dict:
    if scoring_kind == "numeric_exact":
        predicted = extract_number(text)
        return {"predicted": predicted, "gold": gold, "correct": bool(predicted is not None and gold is not None and predicted == gold)}
    if scoring_kind == "choice_exact":
        predicted = extract_choice(text)
        return {"predicted": predicted, "gold": gold, "correct": bool(predicted is not None and gold is not None and predicted == gold)}
    tokens = text.split()
    unique_bigrams = len(set(zip(tokens, tokens[1:]))) if len(tokens) > 1 else 0
    repetition_diversity = unique_bigrams / max(1, len(tokens) - 1)
    return {
        "gold": None,
        "token_count": len(tokens),
        "repetition_diversity": repetition_diversity,
        "usable_story_proxy": bool(len(tokens) >= 20 and repetition_diversity >= 0.55),
    }


def read_completed_keys(path: Path) -> set[str]:
    keys = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add(row["run_key"])
    return keys


def summarize_scores(score_path: Path) -> dict:
    rows = []
    if score_path.exists():
        with score_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    by_cell: dict[str, dict] = {}
    for row in rows:
        cell = f"{row['dataset']}|{row['temperature']}|{row['sampler']}"
        entry = by_cell.setdefault(cell, {"dataset": row["dataset"], "temperature": row["temperature"], "sampler": row["sampler"], "n": 0, "correct": 0, "usable": 0})
        entry["n"] += 1
        entry["correct"] += int(bool(row.get("correct")))
        entry["usable"] += int(bool(row.get("usable_story_proxy")))
    for entry in by_cell.values():
        if entry["dataset"] == "writingprompts":
            entry["metric"] = "usable_story_proxy_rate"
            entry["value"] = entry["usable"] / max(1, entry["n"])
        else:
            entry["metric"] = "accuracy"
            entry["value"] = entry["correct"] / max(1, entry["n"])
    return {"cell_count": len(by_cell), "cells": sorted(by_cell.values(), key=lambda r: (r["dataset"], r["temperature"], r["sampler"]))}


def scale_gate(args, dataset_records: dict, planned_count: int) -> dict:
    loaded_dataset_count = sum(1 for row in dataset_records.values() if row.get("loaded_rows", 0) > 0)
    loaded_prompt_count = sum(int(row.get("loaded_rows", 0)) for row in dataset_records.values())
    has_all_samplers = len(SAMPLERS) >= 7
    has_all_temperatures = len(TEMPERATURES) >= 5
    professional = (
        loaded_dataset_count >= 3
        and loaded_prompt_count >= 75
        and args.max_prompts_per_dataset >= 25
        and args.max_new_tokens >= 32
        and has_all_samplers
        and has_all_temperatures
        and planned_count >= 2500
    )
    reasons = []
    if loaded_dataset_count < 3:
        reasons.append("fewer_than_three_loaded_benchmark_families")
    if loaded_prompt_count < 75:
        reasons.append("fewer_than_75_total_prompts")
    if args.max_prompts_per_dataset < 25:
        reasons.append("max_prompts_per_dataset_below_25")
    if args.max_new_tokens < 32:
        reasons.append("max_new_tokens_below_32")
    if not has_all_samplers:
        reasons.append("missing_sampler_family")
    if not has_all_temperatures:
        reasons.append("missing_temperature_grid")
    if planned_count < 2500:
        reasons.append("planned_generation_count_below_2500")
    return {
        "scale": "professional_paper_shaped_long_run" if professional else "preflight_or_debug_only",
        "professional_scale_for_gap_convergence": professional,
        "reduced_or_small_run": not professional,
        "loaded_dataset_count": loaded_dataset_count,
        "loaded_prompt_count": loaded_prompt_count,
        "planned_generations": planned_count,
        "gate_reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO_DIR))
    parser.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "csqa", "qasc", "writingprompts"])
    parser.add_argument("--max-prompts-per-dataset", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--wall-clock-hours", type=float, default=24.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_generations.jsonl"
    score_path = out_dir / "scores.jsonl"
    timing_path = out_dir / "sampling_time_by_token.jsonl"
    cpu_ram_path = out_dir / "cpu_ram_profile.jsonl"
    status_path = out_dir / "professional_scale_status.json"
    manifest_path = out_dir / "professional_scale_manifest.json"

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    official = load_official_sampler(Path(args.repo_dir))
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model.to(args.device)
    model.eval()

    dataset_records = {}
    examples_by_dataset: dict[str, list[dict]] = {}
    for dataset_name in args.datasets:
        shots, rows, meta = load_named_examples(dataset_name, args.max_prompts_per_dataset)
        dataset_records[dataset_name] = meta | {"loaded_rows": len(rows), "shot_count": len(shots)}
        examples_by_dataset[dataset_name] = []
        for idx, row in enumerate(rows):
            prompt, gold, scoring_kind = build_prompt(dataset_name, shots, row)
            examples_by_dataset[dataset_name].append({"dataset": dataset_name, "prompt_idx": idx, "prompt": prompt, "gold": gold, "scoring_kind": scoring_kind})

    planned = []
    max_prompt_count = max((len(rows) for rows in examples_by_dataset.values()), default=0)
    for prompt_idx in range(max_prompt_count):
        for dataset_name in args.datasets:
            dataset_examples = examples_by_dataset.get(dataset_name, [])
            if prompt_idx >= len(dataset_examples):
                continue
            item = dataset_examples[prompt_idx]
            for temperature in TEMPERATURES:
                for sampler in SAMPLERS:
                    planned.append(
                        {
                            "dataset": item["dataset"],
                            "prompt_idx": item["prompt_idx"],
                            "sampler": sampler,
                            "temperature": temperature,
                            "prompt": item["prompt"],
                            "gold": item["gold"],
                            "scoring_kind": item["scoring_kind"],
                        }
                    )
    for idx, row in enumerate(planned):
        row["run_key"] = f"{row['dataset']}|{row['prompt_idx']}|{row['temperature']}|{row['sampler']}"
        row["planned_index"] = idx

    scale = scale_gate(args, dataset_records, len(planned))
    completed = read_completed_keys(raw_path)
    start = time.perf_counter()
    deadline = start + args.wall_clock_hours * 3600.0
    manifest = {
        "created_at_utc": now_utc(),
        "scale": scale["scale"],
        "professional_scale_for_gap_convergence": scale["professional_scale_for_gap_convergence"],
        "reduced_or_small_run": scale["reduced_or_small_run"],
        "scale_gate": scale,
        "model_id": args.model_id,
        "datasets": args.datasets,
        "dataset_records": dataset_records,
        "temperatures": TEMPERATURES,
        "samplers": SAMPLERS,
        "max_prompts_per_dataset": args.max_prompts_per_dataset,
        "max_new_tokens": args.max_new_tokens,
        "schedule": "interleaved_by_prompt_index_then_dataset_then_temperature_then_sampler",
        "planned_generations": len(planned),
        "device": args.device,
        "hf_home": os.environ.get("HF_HOME"),
        "repo_dir": str(Path(args.repo_dir)),
        "gpu_snapshot_start": gpu_snapshot(),
        "outputs": {
            "raw_generations": str(raw_path),
            "scores": str(score_path),
            "sampling_time_by_token": str(timing_path),
            "cpu_ram_profile": str(cpu_ram_path),
            "status": str(status_path),
        },
    }
    write_json(manifest_path, manifest)

    stop_reason = "completed"
    try:
        for row in planned:
            if row["run_key"] in completed:
                continue
            if time.perf_counter() >= deadline:
                stop_reason = "wall_clock_limit_reached"
                break
            run_seed = args.seed + int(row["planned_index"])
            try:
                text, timing, cpu_ram = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=row["prompt"],
                    sampler_name=row["sampler"],
                    temperature=float(row["temperature"]),
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    official=official,
                    seed=run_seed,
                )
                score = score_generation(row["dataset"], row["scoring_kind"], row["gold"], text)
                generation_row = {
                    "created_at_utc": now_utc(),
                    "run_key": row["run_key"],
                    "planned_index": row["planned_index"],
                    "dataset": row["dataset"],
                    "prompt_idx": row["prompt_idx"],
                    "sampler": row["sampler"],
                    "temperature": row["temperature"],
                    "model_id": args.model_id,
                    "seed": run_seed,
                    "prompt": row["prompt"],
                    "generated_text": text,
                }
                append_jsonl(raw_path, generation_row)
                append_jsonl(score_path, generation_row | score)
                for timing_row in timing:
                    append_jsonl(timing_path, {"run_key": row["run_key"], "dataset": row["dataset"], "prompt_idx": row["prompt_idx"], **timing_row})
                for ram_row in cpu_ram:
                    append_jsonl(cpu_ram_path, {"run_key": row["run_key"], "dataset": row["dataset"], "prompt_idx": row["prompt_idx"], **ram_row})
                completed.add(row["run_key"])
            except Exception as exc:
                append_jsonl(out_dir / "errors.jsonl", {"created_at_utc": now_utc(), "run_key": row["run_key"], "error": repr(exc)})
            completed_count = len(completed)
            if completed_count % 5 == 0:
                status = {
                    "created_at_utc": now_utc(),
                    "status": "running",
                    "scale": scale["scale"],
                    "professional_scale_for_gap_convergence": scale["professional_scale_for_gap_convergence"],
                    "scale_gate": scale,
                    "completed_generations": completed_count,
                    "planned_generations": len(planned),
                    "coverage": completed_count / max(1, len(planned)),
                    "elapsed_seconds": round(time.perf_counter() - start, 3),
                    "gpu_snapshot": gpu_snapshot(),
                    "score_summary": summarize_scores(score_path),
                }
                write_json(status_path, status)
    finally:
        completed_count = len(read_completed_keys(raw_path))
        status = {
            "created_at_utc": now_utc(),
            "status": "completed" if completed_count >= len(planned) else "incomplete",
            "stop_reason": stop_reason,
            "scale": scale["scale"],
            "professional_scale_for_gap_convergence": scale["professional_scale_for_gap_convergence"],
            "scale_gate": scale,
            "completed_generations": completed_count,
            "planned_generations": len(planned),
            "coverage": completed_count / max(1, len(planned)),
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "gpu_snapshot_end": gpu_snapshot(),
            "score_summary": summarize_scores(score_path),
        }
        write_json(status_path, status)


if __name__ == "__main__":
    main()
