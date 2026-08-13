#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_cmd(cmd, cwd=None, timeout=120, env=None):
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": proc.returncode,
            "elapsed_s": round(time.perf_counter() - start, 3),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "elapsed_s": round(time.perf_counter() - start, 3),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }


def choose_device():
    try:
        import torch
    except Exception as exc:
        return "cpu", {"cuda_available": False, "reason": f"torch import failed: {exc}"}
    if not torch.cuda.is_available():
        return "cpu", {"cuda_available": False, "reason": "torch.cuda.is_available false"}
    smi = run_cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], timeout=10)
    candidates = []
    for line in smi.get("stdout_tail", "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            candidates.append({
                "index": int(float(parts[0])),
                "name": parts[1],
                "memory_total_mib": int(float(parts[2])),
                "memory_used_mib": int(float(parts[3])),
                "memory_free_mib": int(float(parts[4])),
                "utilization_gpu_percent": int(float(parts[5])),
            })
    if candidates:
        chosen = max(candidates, key=lambda g: (g["memory_free_mib"], -g["utilization_gpu_percent"]))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen["index"])
        return "cuda:0", {"cuda_available": True, "visible_physical_gpu": chosen["index"], "gpus": candidates, "chosen": chosen}
    return "cuda:0", {"cuda_available": True, "reason": "nvidia-smi parse unavailable"}


def clone_repo(dag, workspace):
    source = dag["source_code"]
    repo_dir = workspace / "external" / "p-less-sampling"
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    attempts = []
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    for url in [source["primary_repo_url"], source["fallback_repo_url"]]:
        result = run_cmd(["git", "clone", "--depth", "1", url, str(repo_dir)], timeout=180)
        attempts.append(result)
        if result["returncode"] == 0:
            break
    head = None
    if repo_dir.exists() and (repo_dir / ".git").exists():
        head_result = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir, timeout=30)
        head = head_result.get("stdout_tail", "").strip() if head_result["returncode"] == 0 else None
    files = sorted(str(p.relative_to(repo_dir)) for p in repo_dir.glob("*") if p.is_file()) if repo_dir.exists() else []
    return {
        "status": "pass" if repo_dir.exists() and (repo_dir / source["official_sampler_file"]).exists() else "blocked",
        "repo_dir": str(repo_dir),
        "attempts": attempts,
        "head": head,
        "files": files,
    }


def check_dependencies():
    deps = ["torch", "transformers", "datasets", "accelerate", "huggingface_hub", "psutil"]
    rows = []
    for dep in deps:
        try:
            mod = __import__(dep)
            rows.append({"package": dep, "status": "pass", "version": getattr(mod, "__version__", "unknown")})
        except Exception as exc:
            rows.append({"package": dep, "status": "blocked", "error": str(exc)})
    return {"status": "pass" if all(r["status"] == "pass" for r in rows) else "blocked", "packages": rows}


def validate_sampler(repo_dir, dag):
    source = dag["source_code"]
    sampler_file = Path(repo_dir) / source["official_sampler_file"]
    if not sampler_file.exists():
        return {"status": "blocked", "reason": f"missing {source['official_sampler_file']}"}
    try:
        spec = importlib.util.spec_from_file_location("official_p_less_samplers", sampler_file)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot create import spec")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        missing = [name for name in source["official_sampler_functions"] if not hasattr(mod, name)]
        if missing:
            return {"status": "blocked", "reason": f"missing functions: {missing}"}
        import torch
        probs = torch.tensor([[0.70, 0.20, 0.07, 0.03]], dtype=torch.float32)
        _ = mod.p_less_decode(probs.clone())
        _ = mod.p_less_norm_decode(probs.clone())
        return {"status": "pass", "functions": source["official_sampler_functions"], "unit_check": "sampled from valid toy probability tensor"}
    except Exception as exc:
        return {"status": "blocked", "reason": repr(exc)}


def probe_models(dag, attempt_downloads=False, max_downloads=0, cache_dir=None):
    from huggingface_hub import model_info, snapshot_download

    rows = []
    downloads = 0
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    for model in dag["models"]["target_exact"]:
        model_row = {"paper_name": model["paper_name"], "candidates": [], "access_note": model["access_note"]}
        for hf_id in model["hf_candidates"]:
            item = {"hf_id": hf_id, "info_status": "unknown", "gated": None, "download_status": "not_attempted"}
            try:
                info = model_info(hf_id)
                item["info_status"] = "pass"
                item["gated"] = getattr(info, "gated", None)
                item["private"] = getattr(info, "private", None)
                item["sha"] = getattr(info, "sha", None)
                if item["gated"] in (True, "manual") and not token_present:
                    item["download_status"] = "blocked_missing_hf_token_or_manual_access"
                elif attempt_downloads and downloads < max_downloads and item["gated"] not in (True, "manual"):
                    # Download metadata/tokenizer/config only. Weight download belongs to target generation stage.
                    path = snapshot_download(
                        hf_id,
                        cache_dir=cache_dir,
                        allow_patterns=["*.json", "tokenizer.*", "*.model", "*.txt", "*.py"],
                        ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.gguf", "*.onnx"],
                    )
                    item["download_status"] = "metadata_downloaded"
                    item["snapshot_path"] = path
                    downloads += 1
            except Exception as exc:
                item["info_status"] = "blocked"
                item["error"] = str(exc).splitlines()[0][:500]
            model_row["candidates"].append(item)
        rows.append(model_row)
    exact_ready = all(any(c.get("download_status") in {"metadata_downloaded"} or (c.get("info_status") == "pass" and c.get("gated") not in (True, "manual")) for c in row["candidates"]) for row in rows)
    gated_blockers = [row["paper_name"] for row in rows for c in row["candidates"] if c.get("download_status") == "blocked_missing_hf_token_or_manual_access"]
    return {"status": "partial" if exact_ready else "blocked", "token_present": token_present, "target_models": rows, "gated_blockers": gated_blockers}


def probe_datasets(dag, load_smoke=False):
    import itertools
    from datasets import get_dataset_config_names, load_dataset

    rows = []
    for group_name, group in dag["datasets"].items():
        for data in group:
            data_row = {"paper_name": data["paper_name"], "group": group_name, "candidates": []}
            for cand in data["hf_candidates"]:
                item = {"id": cand["id"], "config": cand.get("config"), "split": cand.get("split"), "metadata_status": "unknown", "smoke_load_status": "not_attempted"}
                try:
                    # Metadata check; some datasets have no configs.
                    try:
                        item["configs"] = list(get_dataset_config_names(cand["id"])[:20])
                    except Exception as cfg_exc:
                        item["config_probe_warning"] = str(cfg_exc).splitlines()[0][:300]
                    item["metadata_status"] = "pass"
                    if load_smoke:
                        kwargs = {}
                        if cand.get("config"):
                            kwargs["name"] = cand["config"]
                        split = cand.get("split") or "train"
                        ds = load_dataset(cand["id"], **kwargs, split=split, streaming=True)
                        sample = next(iter(ds))
                        item["smoke_load_status"] = "pass"
                        item["sample_keys"] = list(sample.keys())[:20]
                except Exception as exc:
                    item["metadata_status"] = "blocked"
                    item["smoke_load_status"] = "blocked" if load_smoke else "not_attempted"
                    item["error"] = str(exc).splitlines()[0][:500]
                data_row["candidates"].append(item)
            rows.append(data_row)
    all_metadata = all(any(c["metadata_status"] == "pass" for c in row["candidates"]) for row in rows)
    all_smoke = all(any(c["smoke_load_status"] == "pass" for c in row["candidates"]) for row in rows) if load_smoke else False
    return {"status": "pass" if all_smoke else ("partial" if all_metadata else "blocked"), "datasets": rows, "load_smoke": load_smoke}


HARNESS_SOURCE = r"""#!/usr/bin/env python3
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
"""


def write_harness(workspace):
    harness = workspace / "operational_p_less_generation_harness.py"
    harness.write_text(HARNESS_SOURCE)
    return {"status": "pass", "path": str(harness), "script_name": harness.name}


def run_smoke_generation(workspace, repo_dir, device, smoke_model, max_new_tokens):
    out_dir = workspace / "artifacts" / "smoke_real_model_generation"
    cmd = [
        sys.executable,
        "operational_p_less_generation_harness.py",
        "--repo-dir",
        str(repo_dir),
        "--model-id",
        smoke_model,
        "--output-dir",
        str(out_dir),
        "--sampler",
        "p_less",
        "--sampler",
        "p_lessnorm",
        "--temperature",
        "1.0",
        "--max-new-tokens",
        str(max_new_tokens),
        "--device",
        device,
        "--prompt",
        "In one short sentence, language model decoding is",
    ]
    result = run_cmd(cmd, cwd=workspace, timeout=900)
    status = "pass" if result["returncode"] == 0 and (out_dir / "raw_generations.jsonl").exists() else "blocked"
    return {"status": status, "cmd_result": result, "output_dir": str(out_dir)}


def first_gsm8k_prompt():
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
        sample = next(iter(ds))
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        return {
            "status": "pass",
            "prompt": f"Question: {question}\nLet's think step by step.",
            "sample_keys": sorted(sample.keys()),
            "answer_present_for_later_scoring": bool(answer),
        }
    except Exception as exc:
        return {"status": "blocked", "error": str(exc).splitlines()[0][:500]}


def run_feasible_target_generation(workspace, repo_dir, device, model_id, max_new_tokens):
    prompt_info = first_gsm8k_prompt()
    if prompt_info.get("status") != "pass":
        return {"status": "blocked", "reason": "could_not_load_gsm8k_prompt", "prompt_info": prompt_info}
    out_dir = workspace / "artifacts" / "reduced_mistral_gsm8k_generation"
    cmd = [
        sys.executable,
        "operational_p_less_generation_harness.py",
        "--repo-dir",
        str(repo_dir),
        "--model-id",
        model_id,
        "--output-dir",
        str(out_dir),
        "--sampler",
        "p_less",
        "--sampler",
        "p_lessnorm",
        "--sampler",
        "top_p",
        "--sampler",
        "min_p",
        "--temperature",
        "1.0",
        "--max-new-tokens",
        str(max_new_tokens),
        "--device",
        device,
        "--prompt",
        prompt_info["prompt"],
    ]
    result = run_cmd(cmd, cwd=workspace, timeout=7200)
    status = "pass" if result["returncode"] == 0 and (out_dir / "raw_generations.jsonl").exists() else "blocked"
    return {
        "status": status,
        "model_id": model_id,
        "dataset": "openai/gsm8k/main/test",
        "prompt_info": {k: v for k, v in prompt_info.items() if k != "prompt"},
        "cmd_result": result,
        "output_dir": str(out_dir),
        "exact_paper_claim": False,
        "reason_not_exact": "single-model single-temperature reduced generation, not full paper grid",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dag", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--attempt-model-downloads", action="store_true")
    ap.add_argument("--max-model-downloads", type=int, default=0)
    ap.add_argument("--load-data-smoke", action="store_true")
    ap.add_argument("--run-smoke", action="store_true")
    ap.add_argument("--smoke-model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--smoke-max-new-tokens", type=int, default=12)
    ap.add_argument("--run-feasible-target", action="store_true")
    ap.add_argument("--target-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--target-max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    workspace = Path.cwd()
    dag = json.loads(args.dag.read_text())
    artifacts = workspace / "artifacts"
    artifacts.mkdir(exist_ok=True)
    cache_root = workspace / "hf_cache"
    os.environ.setdefault("HF_HOME", str(cache_root))
    device, device_info = choose_device()

    trace = []
    trace.append({"node": "root.operational_author_loop", "decision": "execute DAG nodes in topological paper workflow order", "created_at_utc": now_utc()})
    repo = clone_repo(dag, workspace)
    trace.append({"node": "source.clone_official_repo", "decision": "download official code before experiments", "result_status": repo["status"]})
    deps = check_dependencies()
    trace.append({"node": "env.verify_dependencies", "decision": "check package availability before model/data downloads", "result_status": deps["status"]})
    sampler = validate_sampler(repo.get("repo_dir", ""), dag)
    trace.append({"node": "source.validate_sampler_code", "decision": "use official sampler functions rather than reimplemented proxy for p-less/p-lessnorm", "result_status": sampler["status"]})
    model_probe = probe_models(dag, args.attempt_model_downloads, args.max_model_downloads, str(cache_root))
    trace.append({"node": "model.resolve_target_checkpoints", "decision": "probe/download exact paper model candidates named by DAG", "result_status": model_probe["status"]})
    data_probe = probe_datasets(dag, args.load_data_smoke)
    trace.append({"node": "data.resolve_benchmark_splits", "decision": "probe/load exact benchmark splits named by DAG", "result_status": data_probe["status"]})
    harness = write_harness(workspace)
    trace.append({"node": "harness.write_generation_runner", "decision": "write the concrete python file that later paper experiments must run", "result_status": harness["status"]})

    smoke = {"status": "not_requested"}
    if args.run_smoke:
        smoke = run_smoke_generation(workspace, Path(repo.get("repo_dir", "")), device, args.smoke_model, args.smoke_max_new_tokens)
    trace.append({"node": "harness.smoke_real_model_generation", "decision": "real model smoke validates harness only and cannot satisfy paper-result gate", "result_status": smoke["status"]})

    reduced_target = {"status": "not_requested"}
    if args.run_feasible_target:
        reduced_target = run_feasible_target_generation(
            workspace,
            Path(repo.get("repo_dir", "")),
            device,
            args.target_model,
            args.target_max_new_tokens,
        )
    trace.append({"node": "harness.reduced_mistral_gsm8k_generation", "decision": "real target-model/data generation is useful but remains non-exact until full grid runs", "result_status": reduced_target["status"]})

    paper_experiments = {}
    for exp_id, exp in dag["experiments"].items():
        paper_experiments[exp_id] = {
            "status": "blocked",
            "reason": "not_run_exact_grid_yet",
            "kind": exp["kind"],
            "minimum_for_exact_claim": exp.get("minimum_for_exact_claim"),
        }
        trace.append({"node": f"exp.{exp_id}", "decision": "paper experiment remains blocked unless exact raw artifacts exist", "result_status": "blocked"})

    result = {
        "created_at_utc": now_utc(),
        "dag_id": dag["dag_id"],
        "dag_signature": dag.get("signature"),
        "only_input_file_read": str(args.dag.name),
        "workspace": str(workspace),
        "device": device,
        "device_info": device_info,
        "repo": repo,
        "dependencies": deps,
        "sampler_validation": sampler,
        "model_probe": model_probe,
        "dataset_probe": data_probe,
        "generated_harness": harness,
        "smoke_real_model_generation": smoke,
        "reduced_target_model_generation": reduced_target,
        "paper_experiments": paper_experiments,
        "decision_trace": trace,
        "claim_gate": {
            "status": "blocked",
            "reason": "operational stages did not produce exact paper table/figure/appendix artifact package",
            "proxy_sampler_measurement_used": False,
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
