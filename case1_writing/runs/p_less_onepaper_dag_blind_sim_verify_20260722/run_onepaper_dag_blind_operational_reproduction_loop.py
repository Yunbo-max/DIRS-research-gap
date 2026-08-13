#!/usr/bin/env python3
"""Operational DAG-only author/reviewer reproduction loop for p-less.

This is the correction to the sampler-proxy loop.  Loop 2 is treated as an
author-style executable workflow:

- The blind executor receives only an operational DAG JSON.
- The DAG names code repositories, model ids, dataset ids, scripts, commands,
  artifact paths, metrics, and success gates.
- The executor clones/checks code, probes/downloads model/data assets when
  enabled, writes a runnable generation harness, and runs real model-generation
  smoke tests when enabled.
- The verifier, outside the blind workspace, compares produced artifacts with
  hidden paper evidence channels and converts mismatches into DAG updates.

Proxy sampler measurements are explicitly disallowed for convergence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ORACLE_PATH = RUN_DIR / "paper_oracle_results.json"
BASE_DAG_PATH = RUN_DIR / "paper_author_operational_dag.json"
EXECUTOR_PATH = RUN_DIR / "blind_operational_reproduction_executor.py"
LOOP_DIR = RUN_DIR / "operational_dag_reproduction_loop"
OUTPUT_JSON = RUN_DIR / "onepaper_dag_blind_operational_reproduction_summary.json"
OUTPUT_MD = RUN_DIR / "ONEPAPER_DAG_BLIND_OPERATIONAL_REPRODUCTION_REPORT.md"
STATUS_MD = RUN_DIR / "LONGGOAL_STATUS.md"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:16]


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 60, env: dict[str, str] | None = None) -> dict:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
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


def build_operational_dag() -> dict:
    """Build a DAG detailed enough for a blind executable reproduction agent.

    The DAG intentionally contains protocol and execution details, but not the
    verifier's hidden numeric oracle values.
    """

    dag = {
        "dag_id": "ICLR2026_ItFuNJQGH4_p_less_sampling_operational_author_dag_v1",
        "created_at_utc": now_utc(),
        "title": "p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding",
        "blind_simulation_contract": {
            "only_input_file": "paper_author_operational_dag.json",
            "forbidden_inputs": [
                "paper_oracle_results.json",
                "paper_section_evidence_table.json",
                "author_style_gpu_reproduction_campaign.json",
                "onepaper_dag_blind_gpu_update_loop_summary.json",
                "paper text files",
                "prior reports or traces",
            ],
            "proxy_sampler_measurements_allowed_for_convergence": False,
            "real_execution_required": True,
        },
        "operational_policy": {
            "claim_gate": "paper_table_paragraph_figure_appendix_claims_require_real_reproduction_artifacts_or_blocked_status",
            "do_not_accept": [
                "synthetic random-logit sampler timing as paper timing",
                "shape-only results as exact paper table reproduction",
                "model-info probe as model-generation experiment",
                "tiny-model smoke test as paper result",
            ],
            "allowed_intermediate": [
                "repo clone",
                "dependency check",
                "model access/download check",
                "dataset access/download check",
                "tiny real-model smoke test only for harness validation",
                "reduced feasible target-model generation test marked non-exact",
            ],
        },
        "resources": {
            "preferred_gpu": "select_lowest_used_cuda_device",
            "local_gpu_class": "single RTX 4090 is enough for Mistral-7B microbenchmark and smoke reruns; exact Llama3-70B requires multi-GPU or a smaller verified reproduction profile",
            "cache_root_env": "HF_HOME",
        },
        "source_code": {
            "primary_repo_url": "https://github.com/ryttry/p-less-sampling.git",
            "fallback_repo_url": "https://github.com/ryttry/p-less.git",
            "expected_head_if_available": "a681f23682a329099306eea9cf7b1dd0447e2eec",
            "required_files": ["README.md", "p_less_samplers.py", "p_less_examples.ipynb"],
            "official_sampler_file": "p_less_samplers.py",
            "official_sampler_functions": ["p_less_decode", "p_less_norm_decode"],
        },
        "models": {
            "target_exact": [
                {
                    "paper_name": "Llama-2-7B-Chat",
                    "hf_candidates": ["meta-llama/Llama-2-7b-chat-hf"],
                    "access_note": "manual gated; exact rerun blocks without accepted HF access token",
                },
                {
                    "paper_name": "Mistral-7B-Instruct",
                    "hf_candidates": ["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.3"],
                    "access_note": "ungated; feasible on one 4090 for reduced sample rerun",
                },
                {
                    "paper_name": "Llama3-70B-Instruct",
                    "hf_candidates": ["meta-llama/Meta-Llama-3-70B-Instruct"],
                    "access_note": "manual gated and likely multi-GPU/quantization required",
                },
            ],
            "smoke_only": [
                {
                    "hf_id": "sshleifer/tiny-gpt2",
                    "purpose": "real Transformers generation path validation only; never accepted as paper result",
                }
            ],
        },
        "datasets": {
            "reasoning": [
                {"paper_name": "CSQA", "hf_candidates": [{"id": "tau/commonsense_qa", "config": None, "split": "validation"}]},
                {"paper_name": "GPQA", "hf_candidates": [{"id": "Idavidrein/gpqa", "config": "gpqa_diamond", "split": "train"}]},
                {"paper_name": "GSM8K", "hf_candidates": [{"id": "openai/gsm8k", "config": "main", "split": "test"}]},
                {"paper_name": "QASC", "hf_candidates": [{"id": "allenai/qasc", "config": None, "split": "validation"}]},
            ],
            "writing": [
                {"paper_name": "Writing Prompts", "hf_candidates": [{"id": "euclaise/writingprompts", "config": None, "split": "train"}]}
            ],
        },
        "samplers": {
            "paper_methods": ["top_p", "min_p", "epsilon", "eta", "mirostat", "p_less", "p_lessnorm"],
            "implemented_from_official_repo": ["p_less", "p_lessnorm"],
            "implemented_in_generated_harness": ["top_p", "min_p", "epsilon", "eta", "mirostat"],
        },
        "generated_scripts": {
            "generation_harness": {
                "path": "operational_p_less_generation_harness.py",
                "purpose": "run real Transformers generation with official p-less sampler functions and baseline samplers",
                "required_outputs": [
                    "raw_generations.jsonl",
                    "sampling_time_by_token.jsonl",
                    "cpu_ram_profile.jsonl",
                    "run_manifest.json",
                ],
            }
        },
        "experiments": {
            "table1_reasoning_auc": {
                "kind": "real_model_generation_evaluation",
                "models": ["Llama-2-7B-Chat", "Mistral-7B-Instruct", "Llama3-70B-Instruct"],
                "datasets": ["CSQA", "GPQA", "GSM8K", "QASC"],
                "temperatures": [0.5, 0.7, 1.0, 1.5, 2.0],
                "samplers": ["top_p", "min_p", "epsilon", "eta", "mirostat", "p_less", "p_lessnorm"],
                "metric": "accuracy-vs-temperature AUC",
                "minimum_for_exact_claim": "all target models, all four datasets, all temperatures, all samplers, required seed policy",
            },
            "figure2_temperature_curves": {
                "kind": "figure_from_table1_runs",
                "metric": "accuracy by temperature curves",
                "minimum_for_exact_claim": "same raw runs as Table 1 plus plotting artifact",
            },
            "table2_writing_prompts": {
                "kind": "real_model_generation_preference_or_length_controlled_scoring",
                "dataset": "Writing Prompts",
                "temperatures": [0.5, 0.7, 1.0, 1.5, 2.0],
                "samplers": ["top_p", "min_p", "epsilon", "eta", "mirostat", "p_less", "p_lessnorm"],
                "metric": "length-controlled win rate against default sampling reference",
                "minimum_for_exact_claim": "100 prompts, reference generations, scoring method, all samplers",
            },
            "table3_sampling_time": {
                "kind": "full_generation_timing",
                "model": "Mistral-7B-Instruct",
                "datasets": ["GSM8K", "GPQA"],
                "metric": "average sampling seconds per generated token during model generation",
                "minimum_for_exact_claim": "instrument per-token time inside the model generation loop; do not use sampler-only tensor timing",
            },
            "figures16_17_table15_cpu_ram": {
                "kind": "generation_cpu_ram_profile",
                "methods": ["top_p", "min_p", "p_less"],
                "metric": "CPU processing time and RAM usage during generation",
                "minimum_for_exact_claim": "CPU/RAM traces binned into figure-style summaries and Table 15-style values",
            },
        },
        "nodes": [
            {"id": "root.operational_author_loop", "type": "author_loop", "action": "execute the paper's reproduction workflow from DAG instructions"},
            {"id": "source.clone_official_repo", "type": "download_code", "action": "clone primary repo then fallback if needed"},
            {"id": "source.validate_sampler_code", "type": "code_validation", "action": "import official p_less_samplers.py and verify expected functions"},
            {"id": "env.verify_dependencies", "type": "environment_check", "action": "verify torch, transformers, datasets, accelerate, huggingface_hub, psutil"},
            {"id": "model.resolve_target_checkpoints", "type": "model_download_plan", "action": "probe or download exact target model checkpoints named by DAG"},
            {"id": "data.resolve_benchmark_splits", "type": "data_download_plan", "action": "probe or load benchmark splits named by DAG"},
            {"id": "harness.write_generation_runner", "type": "code_generation", "action": "write operational_p_less_generation_harness.py from DAG schema"},
            {"id": "harness.smoke_real_model_generation", "type": "smoke_real_execution", "action": "optional tiny real-model generation; cannot satisfy paper result gate"},
            {"id": "harness.reduced_mistral_gsm8k_generation", "type": "reduced_target_execution", "action": "optional real Mistral-7B + GSM8K generation to produce non-exact but paper-model/paper-data artifacts"},
            {"id": "exp.table1_reasoning_auc", "type": "paper_experiment", "action": "run exact target model/dataset/sampler/temperature grid and compute AUC"},
            {"id": "exp.figure2_temperature_curves", "type": "paper_figure_reproduction", "action": "plot accuracy-temperature curves from Table 1 raw runs"},
            {"id": "exp.table2_writing_prompts", "type": "paper_experiment", "action": "run writing prompt generations and length-controlled scoring"},
            {"id": "exp.table3_full_generation_timing", "type": "paper_experiment", "action": "measure sampling time per token inside full Mistral generation"},
            {"id": "exp.figures16_17_table15_cpu_ram", "type": "paper_appendix_experiment", "action": "measure CPU time and RAM during generation"},
            {"id": "eval.score_reasoning", "type": "evaluation", "action": "score exact-answer reasoning tasks and aggregate AUC"},
            {"id": "eval.score_writing", "type": "evaluation", "action": "score writing generations with length-controlled preference protocol"},
            {"id": "verifier.package_artifacts", "type": "artifact_package", "action": "emit manifest, raw generations, timing traces, CPU/RAM traces, and table summaries"},
            {"id": "decision.claim_gate", "type": "author_reviewer_decision", "action": "claim only exact reproduced channels; otherwise block and request DAG update"},
        ],
        "edges": [
            ["root.operational_author_loop", "source.clone_official_repo"],
            ["source.clone_official_repo", "source.validate_sampler_code"],
            ["source.validate_sampler_code", "env.verify_dependencies"],
            ["env.verify_dependencies", "model.resolve_target_checkpoints"],
            ["env.verify_dependencies", "data.resolve_benchmark_splits"],
            ["model.resolve_target_checkpoints", "harness.write_generation_runner"],
            ["data.resolve_benchmark_splits", "harness.write_generation_runner"],
            ["harness.write_generation_runner", "harness.smoke_real_model_generation"],
            ["harness.write_generation_runner", "harness.reduced_mistral_gsm8k_generation"],
            ["harness.write_generation_runner", "exp.table1_reasoning_auc"],
            ["exp.table1_reasoning_auc", "exp.figure2_temperature_curves"],
            ["harness.write_generation_runner", "exp.table2_writing_prompts"],
            ["harness.write_generation_runner", "exp.table3_full_generation_timing"],
            ["harness.write_generation_runner", "exp.figures16_17_table15_cpu_ram"],
            ["exp.table1_reasoning_auc", "eval.score_reasoning"],
            ["exp.table2_writing_prompts", "eval.score_writing"],
            ["eval.score_reasoning", "verifier.package_artifacts"],
            ["eval.score_writing", "verifier.package_artifacts"],
            ["exp.table3_full_generation_timing", "verifier.package_artifacts"],
            ["exp.figures16_17_table15_cpu_ram", "verifier.package_artifacts"],
            ["verifier.package_artifacts", "decision.claim_gate"],
        ],
        "convergence_gate": {
            "requires": [
                "official repo cloned at recorded commit",
                "official p-less sampler imported",
                "target model checkpoints available or blocked with exact access reason",
                "benchmark data splits available or blocked with exact access reason",
                "raw generations produced for all required paper experiment cells",
                "reasoning AUC, writing win-rate, timing, CPU/RAM artifacts emitted",
                "verifier comparison over tables, paragraph claims/values, figures, and appendix artifacts",
            ],
            "proxy_only_result": "automatic_blocked",
        },
    }
    dag["signature"] = stable_hash({"nodes": dag["nodes"], "edges": dag["edges"], "experiments": dag["experiments"]})
    return dag


BLIND_EXECUTOR_SOURCE = r'''#!/usr/bin/env python3
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
'''


def write_executor() -> None:
    EXECUTOR_PATH.write_text(BLIND_EXECUTOR_SOURCE)
    EXECUTOR_PATH.chmod(0o755)


def append_node_once(dag: dict, node: dict) -> None:
    if not any(existing.get("id") == node["id"] for existing in dag["nodes"]):
        dag["nodes"].append(node)


def append_edge_once(dag: dict, edge: list[str]) -> None:
    if edge not in dag["edges"]:
        dag["edges"].append(edge)


def run_blind_executor(dag: dict, iteration_dir: Path, args: argparse.Namespace) -> tuple[dict, float]:
    blind = iteration_dir / "blind_workspace"
    if blind.exists():
        shutil.rmtree(blind)
    blind.mkdir(parents=True)
    dag_path = blind / "paper_author_operational_dag.json"
    dag_path.write_text(json.dumps(dag, indent=2, sort_keys=True))
    shutil.copy2(EXECUTOR_PATH, blind / "blind_operational_reproduction_executor.py")
    cmd = [
        sys.executable,
        "blind_operational_reproduction_executor.py",
        "--dag",
        "paper_author_operational_dag.json",
        "--output",
        "blind_operational_reproduction_result.json",
        "--max-model-downloads",
        str(args.max_model_downloads),
        "--smoke-max-new-tokens",
        str(args.smoke_max_new_tokens),
    ]
    if args.attempt_model_downloads:
        cmd.append("--attempt-model-downloads")
    if args.load_data_smoke:
        cmd.append("--load-data-smoke")
    if args.run_smoke:
        cmd.append("--run-smoke")
    if args.smoke_model:
        cmd.extend(["--smoke-model", args.smoke_model])
    if args.run_feasible_target:
        cmd.append("--run-feasible-target")
    if args.target_model:
        cmd.extend(["--target-model", args.target_model])
    cmd.extend(["--target-max-new-tokens", str(args.target_max_new_tokens)])
    start = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("HF_HOME", str((RUN_DIR / "operational_shared_hf_cache").resolve()))
    proc = subprocess.run(cmd, cwd=blind, env=env, text=True, capture_output=True, check=False, timeout=args.executor_timeout_seconds)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        result = {
            "created_at_utc": now_utc(),
            "dag_id": dag["dag_id"],
            "subprocess_failed": True,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
        (blind / "blind_operational_reproduction_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        return result, elapsed
    result = json.loads((blind / "blind_operational_reproduction_result.json").read_text())
    result["subprocess"] = {"returncode": proc.returncode, "elapsed_s": round(elapsed, 3), "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
    (blind / "blind_operational_reproduction_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result, elapsed


def verify_operational_result(result: dict, oracle: dict) -> dict:
    checks: list[dict] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    add(
        "blind_contract_only_dag_input",
        "pass" if result.get("only_input_file_read") == "paper_author_operational_dag.json" else "fail",
        f"only_input_file_read={result.get('only_input_file_read')}",
    )
    add(
        "proxy_sampler_disallowed",
        "pass" if result.get("claim_gate", {}).get("proxy_sampler_measurement_used") is False else "fail",
        "proxy sampler measurements are not accepted for convergence",
    )
    repo = result.get("repo", {})
    add(
        "official_repo_downloaded",
        "pass" if repo.get("status") == "pass" else "blocked",
        f"status={repo.get('status')} head={repo.get('head')} files={repo.get('files')}",
    )
    sampler = result.get("sampler_validation", {})
    add(
        "official_sampler_imported",
        "pass" if sampler.get("status") == "pass" else "blocked",
        sampler.get("reason") or json.dumps(sampler)[:500],
    )
    deps = result.get("dependencies", {})
    add(
        "environment_dependencies",
        "pass" if deps.get("status") == "pass" else "blocked",
        json.dumps(deps.get("packages", []))[:1000],
    )
    model_probe = result.get("model_probe", {})
    gated = model_probe.get("gated_blockers", [])
    add(
        "target_model_access_download_plan",
        "blocked" if gated else ("partial" if model_probe.get("status") == "partial" else "blocked"),
        f"status={model_probe.get('status')} gated_blockers={gated}; exact paper models require checkpoint availability",
    )
    data_probe = result.get("dataset_probe", {})
    add(
        "benchmark_dataset_access_plan",
        "pass" if data_probe.get("status") == "pass" else ("partial" if data_probe.get("status") == "partial" else "blocked"),
        f"status={data_probe.get('status')} load_smoke={data_probe.get('load_smoke')}",
    )
    harness = result.get("generated_harness", {})
    add(
        "generation_harness_script_written",
        "pass" if harness.get("status") == "pass" else "blocked",
        f"path={harness.get('path')} script={harness.get('script_name')}",
    )
    smoke = result.get("smoke_real_model_generation", {})
    add(
        "real_model_generation_smoke",
        "pass" if smoke.get("status") == "pass" else ("blocked" if smoke.get("status") != "not_requested" else "partial"),
        f"status={smoke.get('status')} output_dir={smoke.get('output_dir')}",
    )
    reduced_target = result.get("reduced_target_model_generation", {})
    add(
        "reduced_target_model_generation_artifact",
        "pass" if reduced_target.get("status") == "pass" else ("blocked" if reduced_target.get("status") != "not_requested" else "partial"),
        f"status={reduced_target.get('status')} model={reduced_target.get('model_id')} dataset={reduced_target.get('dataset')} output_dir={reduced_target.get('output_dir')} exact={reduced_target.get('exact_paper_claim')}",
    )
    paper_experiments = result.get("paper_experiments", {})
    for exp_name in [
        "table1_reasoning_auc",
        "figure2_temperature_curves",
        "table2_writing_prompts",
        "table3_sampling_time",
        "figures16_17_table15_cpu_ram",
    ]:
        exp = paper_experiments.get(exp_name, {})
        add(
            f"paper_{exp_name}_real_artifacts",
            "pass" if exp.get("status") == "pass" else "blocked",
            f"status={exp.get('status')} reason={exp.get('reason')} minimum={exp.get('minimum_for_exact_claim')}",
        )
    add(
        "paper_evidence_channel_comparison_gate",
        "blocked",
        "verifier cannot compare exact table values, paragraph claims, figures, and appendix artifacts until raw generation/timing/CPU-RAM artifacts exist",
    )

    total = len(checks)
    passed = sum(1 for c in checks if c["status"] == "pass")
    partial = sum(1 for c in checks if c["status"] == "partial")
    score = round((passed + 0.5 * partial) / max(total, 1), 6)
    converged = all(c["status"] == "pass" for c in checks)
    required_updates = build_updates_from_checks(checks, oracle)
    return {
        "created_at_utc": now_utc(),
        "score": score,
        "converged": converged,
        "checks": checks,
        "evidence_channels": {
            "tables": ["Table 1", "Table 2", "Table 3", "Table 15"],
            "paragraphs": ["gap/motivation", "method mechanism", "high-temperature result discussion", "efficiency explanation"],
            "figures": ["Figure 2", "Figure 15", "Figures 16 and 17"],
            "appendix_artifacts": ["Appendix C.11 CPU/RAM profiling", "official sampler code snippet"],
        },
        "oracle_numeric_anchor_keys_compared_by_verifier_only": sorted((oracle.get("reported_numeric_anchors") or {}).keys()),
        "required_updates": required_updates,
    }


def build_updates_from_checks(checks: list[dict], oracle: dict) -> list[dict]:
    status = {c["name"]: c["status"] for c in checks}
    detail = {c["name"]: c["detail"] for c in checks}
    updates = []
    if status.get("target_model_access_download_plan") != "pass":
        updates.append(
            {
                "id": "update.operational_exact_model_download_and_access",
                "reason": "The blind agent did not have all exact paper target checkpoints available for generation.",
                "success_criteria": [
                    "record HF ids, access/gating status, snapshot commit, cache path, and disk footprint for Llama-2-7B-Chat, Mistral-7B-Instruct, and Llama3-70B-Instruct",
                    "obtain manual-gated model access where required or declare a paper-faithful substituted-model profile before claiming any result",
                    "load at least the feasible 7B target model on GPU and emit raw generation/timing artifacts",
                ],
                "verifier_detail": detail.get("target_model_access_download_plan"),
            }
        )
    if status.get("benchmark_dataset_access_plan") != "pass":
        updates.append(
            {
                "id": "update.operational_dataset_download_and_prompt_builder",
                "reason": "The blind agent did not load every benchmark split/prompt source required by the paper.",
                "success_criteria": [
                    "download/load CSQA, GPQA, GSM8K, QASC, and Writing Prompts splits",
                    "write prompt builders including 8-shot chain-of-thought demonstrations where required",
                    "store sampled prompt ids, seeds, and serialized prompts for verifier comparison",
                ],
                "verifier_detail": detail.get("benchmark_dataset_access_plan"),
            }
        )
    if status.get("reduced_target_model_generation_artifact") != "pass":
        updates.append(
            {
                "id": "update.run_feasible_mistral_gsm8k_generation_node",
                "reason": "The blind agent did not produce even a reduced real target-model/data generation artifact.",
                "success_criteria": [
                    "load an ungated feasible paper target model such as Mistral-7B-Instruct on GPU",
                    "load at least one real benchmark prompt such as GSM8K from the DAG-named dataset",
                    "run the generated harness with official p-less/p-lessnorm and baseline samplers",
                    "store raw generations, per-token sampling timing, CPU/RAM trace, and manifest while marking the result non-exact",
                ],
                "verifier_detail": detail.get("reduced_target_model_generation_artifact"),
            }
        )
    for exp_id, update_id in [
        ("paper_table1_reasoning_auc_real_artifacts", "update.run_table1_reasoning_auc_exact_grid"),
        ("paper_figure2_temperature_curves_real_artifacts", "update.render_figure2_from_raw_table1_runs"),
        ("paper_table2_writing_prompts_real_artifacts", "update.run_table2_writing_prompt_scoring"),
        ("paper_table3_sampling_time_real_artifacts", "update.run_table3_full_generation_timing"),
        ("paper_figures16_17_table15_cpu_ram_real_artifacts", "update.run_figures16_17_table15_cpu_ram_profile"),
    ]:
        if status.get(exp_id) != "pass":
            updates.append(
                {
                    "id": update_id,
                    "reason": f"Missing exact reproduction artifact for {exp_id}.",
                    "success_criteria": [
                        "run the DAG-named model/data/sampler command instead of any synthetic proxy",
                        "store raw generations, per-token timing, CPU/RAM traces, scoring outputs, and aggregation code",
                        "allow verifier to compare against hidden paper tables, paragraph values, figures, and appendix artifacts",
                    ],
                    "verifier_detail": detail.get(exp_id),
                }
            )
    if status.get("paper_evidence_channel_comparison_gate") != "pass":
        updates.append(
            {
                "id": "update.require_verifier_ready_artifact_package",
                "reason": "The verifier cannot compare all evidence channels without a complete result package.",
                "success_criteria": [
                    "emit table1_reasoning_auc.json, figure2_temperature_curves.json/png, table2_writing_prompts.json, table3_sampling_time.json, figures16_17_cpu_ram.json/png, table15_cpu_ram.json",
                    "include raw inputs/outputs and hardware logs",
                    "do not mark Loop 2 converged until every evidence-channel check is pass",
                ],
                "verifier_detail": detail.get("paper_evidence_channel_comparison_gate"),
            }
        )
    return updates


def update_dag(dag: dict, verification: dict, iteration: int) -> tuple[dict, list[dict]]:
    next_dag = copy.deepcopy(dag)
    updates = verification["required_updates"]
    if not updates:
        return next_dag, []
    next_dag.setdefault("verifier_feedback_history", []).append(
        {"iteration": iteration, "created_at_utc": now_utc(), "updates": updates}
    )
    for update in updates:
        node_id = update["id"].replace("update.", "required.")
        append_node_once(
            next_dag,
            {
                "id": node_id,
                "type": "required_operational_update",
                "action": update["reason"],
                "success_criteria": update["success_criteria"],
            },
        )
        append_edge_once(next_dag, ["decision.claim_gate", node_id])
    next_dag["dag_id"] = f"{dag['dag_id']}_iter_{iteration}_updated"
    next_dag["signature"] = stable_hash(
        {
            "nodes": next_dag["nodes"],
            "edges": next_dag["edges"],
            "feedback": next_dag.get("verifier_feedback_history", []),
            "experiments": next_dag["experiments"],
        }
    )
    return next_dag, updates


def blocker_signature(updates: list[dict]) -> str:
    return stable_hash([(u["id"], u["reason"]) for u in updates])


def write_report(summary: dict) -> None:
    lines = [
        "# One-Paper DAG-Only Operational Reproduction Loop",
        "",
        f"Date: `{summary['created_at_utc']}`",
        f"Target: `{summary['target']}`",
        f"Status: `{summary['status']}`",
        f"Iterations: `{len(summary['iterations'])}`",
        f"Total executor runtime seconds: `{summary['total_executor_runtime_seconds']}`",
        "",
        "## Correction",
        "",
        "This run rejects GPU sampler proxies as Loop 2 convergence evidence. The blind agent receives only the operational DAG and must execute code/model/data/evaluation nodes.",
        "",
        "## Iterations",
        "",
    ]
    for item in summary["iterations"]:
        statuses = ", ".join(f"{c['name']}={c['status']}" for c in item["verification"]["checks"])
        lines.append(f"- Iteration `{item['iteration']}` score `{item['verification']['score']}` updates `{len(item['dag_updates'])}`: {statuses}")
    lines += ["", "## Final Required DAG Updates", ""]
    for update in summary["final_required_updates"]:
        lines.append(f"- `{update['id']}`: {update['reason']}")
        lines.append(f"  Success criteria: {'; '.join(update['success_criteria'])}")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Operational DAG: `{BASE_DAG_PATH}`",
        f"- Blind executor: `{EXECUTOR_PATH}`",
        f"- Summary JSON: `{OUTPUT_JSON}`",
        f"- Loop directory: `{LOOP_DIR}`",
    ]
    OUTPUT_MD.write_text("\n".join(lines))


def update_status(summary: dict) -> None:
    marker = "## 2026-07-22 Correction: Operational DAG-Only Reproduction Loop"
    block = [
        marker,
        "",
        "The sampler-proxy Loop 2 was rejected as insufficient. The corrected run makes the DAG operational: the blind agent sees only `paper_author_operational_dag.json`, then follows DAG nodes to clone/download code, check dependencies, resolve models and datasets, write a concrete generation harness, attempt real model generation, and package evaluation artifacts.",
        "",
        "Operational result:",
        "",
        f"- Final status: `{summary['status']}`",
        f"- Iterations: `{len(summary['iterations'])}`",
        f"- Total executor runtime: `{summary['total_executor_runtime_seconds']}` seconds",
        "- Proxy sampler convergence: `disallowed`",
        "",
        "The verifier compares against hidden evidence channels:",
        "",
        "- Tables: Table 1, Table 2, Table 3, Table 15.",
        "- Paragraphs: gap/motivation, method mechanism, high-temperature discussion, efficiency explanation.",
        "- Figures: Figure 2, Figure 15, Figures 16 and 17.",
        "- Appendix artifacts: Appendix C.11 CPU/RAM profiling and official sampler code.",
        "",
        "Required DAG updates:",
        "",
    ]
    for update in summary["final_required_updates"]:
        block.append(f"- `{update['id']}`")
    block += [
        "",
        "The loop therefore does not call the one-paper simulation converged. It blocks until exact model/data/generation/scoring/timing/CPU-RAM artifacts exist and pass verifier comparison.",
        "",
    ]
    text = STATUS_MD.read_text() if STATUS_MD.exists() else ""
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n\n"
    text += "\n".join(block)
    STATUS_MD.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--stable-blocker-window", type=int, default=2)
    parser.add_argument("--attempt-model-downloads", action="store_true")
    parser.add_argument("--max-model-downloads", type=int, default=1)
    parser.add_argument("--load-data-smoke", action="store_true")
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--smoke-model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--smoke-max-new-tokens", type=int, default=12)
    parser.add_argument("--run-feasible-target", action="store_true")
    parser.add_argument("--target-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--target-max-new-tokens", type=int, default=16)
    parser.add_argument("--executor-timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    dag = build_operational_dag()
    BASE_DAG_PATH.write_text(json.dumps(dag, indent=2, sort_keys=True))
    write_executor()
    oracle = json.loads(ORACLE_PATH.read_text())
    LOOP_DIR.mkdir(parents=True, exist_ok=True)

    iterations = []
    stable_count = 0
    last_sig = None
    total_runtime = 0.0
    final_updates: list[dict] = []
    status = "max_iterations_reached"

    for iteration in range(1, args.max_iterations + 1):
        iteration_dir = LOOP_DIR / f"iter_{iteration:02d}"
        result, runtime = run_blind_executor(dag, iteration_dir, args)
        total_runtime += runtime
        verification = verify_operational_result(result, oracle)
        next_dag, updates = update_dag(dag, verification, iteration)
        final_updates = updates
        sig = blocker_signature(updates)
        stable_count = stable_count + 1 if sig == last_sig else 1
        last_sig = sig
        (iteration_dir / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True))
        (iteration_dir / "dag_update_request.json").write_text(json.dumps(updates, indent=2, sort_keys=True))
        (iteration_dir / "paper_author_operational_dag.updated.json").write_text(json.dumps(next_dag, indent=2, sort_keys=True))
        iterations.append(
            {
                "iteration": iteration,
                "executor_runtime_seconds": round(runtime, 3),
                "verification": verification,
                "dag_updates": updates,
                "blocking_status_signature": sig,
                "stable_blocker_count": stable_count,
                "paths": {
                    "iteration_dir": str(iteration_dir),
                    "blind_workspace": str(iteration_dir / "blind_workspace"),
                    "dag_update_request": str(iteration_dir / "dag_update_request.json"),
                },
            }
        )
        dag = next_dag
        if verification["converged"]:
            status = "converged"
            break
        if updates and stable_count >= args.stable_blocker_window:
            status = "blocked_waiting_for_operational_artifacts_after_dag_update"
            break

    summary = {
        "created_at_utc": now_utc(),
        "target": "ICLR2026_ItFuNJQGH4_p_less_sampling",
        "status": status,
        "blind_executor_only_input": "paper_author_operational_dag.json",
        "total_executor_runtime_seconds": round(total_runtime, 3),
        "iterations": iterations,
        "final_required_updates": final_updates,
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_report(summary)
    update_status(summary)
    print(f"status={status}")
    print(f"iterations={len(iterations)}")
    print(f"total_executor_runtime_seconds={summary['total_executor_runtime_seconds']}")
    print(f"summary={OUTPUT_JSON}")
    print(f"report={OUTPUT_MD}")


if __name__ == "__main__":
    main()
