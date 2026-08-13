#!/usr/bin/env python3
"""Exact-model Prophet generation probe.

This is not convergence evidence. It proves that the DAG-selected model, repo
code, GPU, and full-step Prophet parameters can execute before the long
paper-shaped benchmark harness is launched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


REPO = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/Prophet")
sys.path.insert(0, str(REPO))

from generate import generate as baseline_generate  # noqa: E402
from generate_earlyexit import generate as prophet_generate  # noqa: E402


QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()


def parse_constraints(text: str, tokenizer) -> dict[int, int]:
    constraints: dict[int, int] = {}
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


def extract_answer(text: str) -> str:
    match = re.search(r"(?i)Answer\s*:\s*([^\n]+)", text)
    if match:
        return match.group(1).replace("$", "").strip()
    nums = re.findall(r"-?\\d+[\\d,]*", text)
    return nums[-1].replace(",", "") if nums else ""


def run_one(model, tokenizer, mode: str, prompt_ids, args) -> dict:
    constraints = parse_constraints(args.constraints_text, tokenizer) if args.constraints_text else None
    start = time.time()
    if mode == "baseline":
        output = baseline_generate(
            model,
            prompt_ids,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=0.0,
            cfg_scale=0.0,
            remasking=args.remasking,
            constraints=constraints,
        )
        gap_data = {
            "exit_info": {
                "early_exit_triggered": False,
                "exit_decision_step": None,
                "total_steps": args.steps,
                "actual_steps": args.steps,
            }
        }
    else:
        answer_start_pos = prompt_ids.shape[1] + args.answer_start_offset
        output, gap_data = prophet_generate(
            model,
            prompt_ids,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=0.0,
            cfg_scale=0.0,
            remasking=args.remasking,
            constraints=constraints,
            analyze_gap=True,
            tokenizer=tokenizer,
            answer_start_pos=answer_start_pos,
            early_exit_thresholds={
                "early": args.early_threshold,
                "mid": args.mid_threshold,
                "late": args.late_threshold,
            },
            measure_time=False,
        )
    torch.cuda.synchronize()
    generated_ids = output[0, prompt_ids.shape[1] :].detach().cpu().tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "mode": mode,
        "seconds": round(time.time() - start, 3),
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "remasking": args.remasking,
        "constraints_text": args.constraints_text,
        "exit_info": gap_data["exit_info"],
        "generated_text": generated_text,
        "predicted_answer": extract_answer(generated_text),
        "generated_token_count": len(generated_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--constraints-text", default="200:The|201:answer|202:is")
    parser.add_argument("--answer-start-offset", type=int, default=200)
    parser.add_argument("--early-threshold", type=float, default=7.5)
    parser.add_argument("--mid-threshold", type=float, default=5.0)
    parser.add_argument("--late-threshold", type=float, default=2.5)
    parser.add_argument(
        "--question",
        default="Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    prompt_text = QUERY_TEMPLATE.format(Question=args.question)
    messages = [{"role": "user", "content": prompt_text}]
    chat_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(chat_prompt, return_tensors="pt")["input_ids"].to(device)

    before = torch.cuda.mem_get_info() if torch.cuda.is_available() else None
    rows = []
    for mode in ["baseline", "prophet"]:
        rows.append(run_one(model, tokenizer, mode, prompt_ids, args))
    after = torch.cuda.mem_get_info() if torch.cuda.is_available() else None
    payload = {
        "artifact_kind": "exact_llada_full_parameter_single_question_probe",
        "convergence_role": "support_only_not_full_benchmark_grid",
        "model_id": args.model_id,
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "logical_gpu_mapping_note": "When CUDA_VISIBLE_DEVICES=3, PyTorch logical cuda:0 is physical GPU 3.",
        "prompt_token_count": int(prompt_ids.shape[1]),
        "question": args.question,
        "expected_answer": "72",
        "gpu_mem_get_info_before": list(before) if before else None,
        "gpu_mem_get_info_after": list(after) if after else None,
        "results": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
