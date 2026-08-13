#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_official_sampler(repo_dir):
    import importlib.util
    sampler_file = Path(repo_dir) / "p_less_samplers.py"
    spec = importlib.util.spec_from_file_location("official_p_less_samplers", sampler_file)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def pick_next_token(probs, sampler_name, official):
    probs = probs.clone()
    if sampler_name == "p_less":
        return official.p_less_decode(probs.unsqueeze(0))[0]
    if sampler_name == "p_lessnorm":
        return official.p_less_norm_decode(probs.unsqueeze(0))[0]
    if sampler_name == "top_p":
        values, indices = torch.sort(probs, descending=True)
        keep = torch.cumsum(values, dim=-1) <= 0.90
        keep[0] = True
        crossing = torch.argmax((torch.cumsum(values, dim=-1) >= 0.90).to(torch.int64))
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
        # Lightweight practical fallback for the harness: sample around target surprisal.
        surprisal = -torch.log(torch.clamp(probs, min=1e-30))
        mask = (surprisal >= 3.5) & (surprisal <= 6.5)
        if not mask.any():
            mask[torch.argmax(probs)] = True
        masked = probs * mask
        masked = masked / masked.sum()
        return torch.multinomial(masked, 1)
    raise ValueError(f"unknown sampler {sampler_name}")


def generate_one(model, tokenizer, prompt, sampler_name, max_new_tokens, temperature, official, device):
    process = psutil.Process(os.getpid())
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids
    timing = []
    cpu_ram = []
    for step in range(max_new_tokens):
        start_mem = process.memory_info().rss
        start = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=generated)
            logits = out.logits[0, -1] / float(temperature)
            probs = torch.softmax(logits.float(), dim=-1)
            sample_start = time.perf_counter()
            next_token = pick_next_token(probs, sampler_name, official).reshape(1, 1).to(device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            sample_elapsed = time.perf_counter() - sample_start
        generated = torch.cat([generated, next_token], dim=-1)
        if int(next_token.item()) == int(tokenizer.eos_token_id or -1):
            break
        elapsed = time.perf_counter() - start
        timing.append({"step": step, "sampler": sampler_name, "sample_seconds": sample_elapsed, "token_seconds": elapsed})
        cpu_ram.append({"step": step, "sampler": sampler_name, "rss_gb": process.memory_info().rss / (1024**3), "rss_delta_gb": (process.memory_info().rss - start_mem) / (1024**3)})
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text, timing, cpu_ram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--sampler", action="append", default=[])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    official = load_official_sampler(args.repo_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model.to(args.device)
    model.eval()

    prompts = args.prompt or ["Solve 1 + 2 - 3 * 4 / 5. End with The answer is [ANSWER]."]
    samplers = args.sampler or ["p_less", "p_lessnorm"]
    generations = []
    timing_rows = []
    cpu_ram_rows = []
    for prompt_idx, prompt in enumerate(prompts):
        for sampler in samplers:
            text, timing, cpu_ram = generate_one(model, tokenizer, prompt, sampler, args.max_new_tokens, args.temperature, official, args.device)
            generations.append({"prompt_idx": prompt_idx, "prompt": prompt, "sampler": sampler, "temperature": args.temperature, "text": text})
            timing_rows.extend({"prompt_idx": prompt_idx, **row} for row in timing)
            cpu_ram_rows.extend({"prompt_idx": prompt_idx, **row} for row in cpu_ram)
    with (out_dir / "raw_generations.jsonl").open("w") as f:
        for row in generations:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (out_dir / "sampling_time_by_token.jsonl").open("w") as f:
        for row in timing_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (out_dir / "cpu_ram_profile.jsonl").open("w") as f:
        for row in cpu_ram_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "model_id": args.model_id,
        "samplers": samplers,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "prompt_count": len(prompts),
        "output_files": ["raw_generations.jsonl", "sampling_time_by_token.jsonl", "cpu_ram_profile.jsonl"],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
