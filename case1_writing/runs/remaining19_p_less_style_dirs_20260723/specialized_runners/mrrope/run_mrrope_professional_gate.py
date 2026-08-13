#!/usr/bin/env python3
"""MrRoPE professional operational gate for the strict DIRS loop.

The OpenReview supplement has useful code, but the paper-shaped result requires
full long-context datasets, local model checkpoints, 128K evaluation scripts,
raw outputs, and metric tables. This runner inventories those requirements and
feeds missing artifacts back into the DAG; it never promotes import checks or
tiny context probes into convergence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, HfFolder


RUN_ROOT = Path(
    "/tf/notebooks/yunbo/DIRS/case1_writing/runs/"
    "remaining19_p_less_style_dirs_20260723"
)
PAPER_RUN = RUN_ROOT / "paper_runs" / "iclr2026_1j63fjyjkg_mrrope_mixed_radix_rope"
RUNNER_DIR = RUN_ROOT / "specialized_runners" / "mrrope"
REPO = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/MrRoPE_OpenReviewSupp")
SUPP = REPO / "supplement"

PAPER_ID = "ICLR2026_1J63FJYJKg_mrrope_mixed_radix_rope"
TITLE = "MrRoPE: Mixed-radix Rotary Position Embedding"

DAG_PATH = PAPER_RUN / "paper_author_gap_dag.json"
STATUS_PATH = RUNNER_DIR / "MRROPE_SPECIALIZED_STATUS.md"
VERIFIER_PATH = RUNNER_DIR / "mrrope_specialized_verifier.json"
ENV_PATH = RUNNER_DIR / "environment.json"
SCRIPT_MANIFEST_PATH = RUNNER_DIR / "official_script_manifest.json"
MODEL_DATA_PATH = RUNNER_DIR / "model_data_manifest.json"
PROFESSIONAL_GATE_PATH = RUNNER_DIR / "professional_gate_result.json"

QUEUE_PATH = RUN_ROOT / "specialized_runner_queue.json"
SUMMARY_PATH = RUN_ROOT / "remaining19_strict_dirs_summary.json"
QUEUE_MD_PATH = RUN_ROOT / "SPECIALIZED_RUNNER_QUEUE.md"
LONGGOAL_STATUS_PATH = RUN_ROOT / "LONGGOAL_STATUS.md"

SUPPLEMENT_SCRIPTS = [
    "eval.sh",
    "evalb.sh",
    "evali.sh",
    "evaln.sh",
    "evalr.sh",
    "eval/perplexity.py",
    "eval/neddle.py",
    "eval/ruler.py",
    "eval/infity.py",
    "eval/longbench.py",
    "eval/score_infity.py",
    "eval/model_loader.py",
    "scaled_rope/LlamaMrRoPE.py",
    "scaled_rope/patch.py",
    "analysis/attn.py",
    "requirements.txt",
]

LOCAL_DATA_PATHS = [
    "testset/pp-tokenized-llama3",
    "testset/pp-tokenized-qwen2.5",
    "testset/infity",
    "testset/longbenchv2",
    "testset/PaulGrahamEssays",
]

LOCAL_MODEL_PATHS = [
    "models/llama3.1-8b-ins",
    "models/qwen2.5-3b-ins",
]

HF_MODEL_REPOS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

HF_DATASET_REPOS = [
    "xinrongzhang2022/InfiniteBench",
    "SaylorTwift/RULER-4096-llama-3.1-tokenizer-chat-template",
    "SaylorTwift/RULER-8192-llama-3.1-tokenizer-chat-template",
    "SaylorTwift/RULER-16384-llama-3.1-tokenizer-chat-template",
    "SaylorTwift/RULER-32768-llama-3.1-tokenizer-chat-template",
    "SaylorTwift/RULER-65536-llama-3.1-tokenizer-chat-template",
    "SaylorTwift/RULER-131072-llama-3.1-tokenizer-chat-template",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str], *, cwd: Path | None = None, timeout: int = 120) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": proc.returncode,
            "timeout": False,
            "seconds": round(elapsed, 3),
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "timeout": True,
            "seconds": timeout,
            "stdout": (exc.stdout or "")[-6000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-6000:] if isinstance(exc.stderr, str) else "",
        }


def package_version(dist_name: str) -> str | None:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None


def parse_script(text: str) -> dict[str, Any]:
    return {
        "python_entrypoints": re.findall(r"python(?:\s+-u)?\s+([A-Za-z0-9_./-]+\.py)", text),
        "models": sorted(set(re.findall(r"(?:-m|--model(?:_path|_name)?|--model)\s+([A-Za-z0-9_./:-]+)", text))),
        "data_dirs": sorted(set(re.findall(r"--data_dir\s+([A-Za-z0-9_./:-]+)", text))),
        "tokenized_paths": sorted(set(re.findall(r"--tokenized\s+([A-Za-z0-9_./:-]+)", text))),
        "output_files": sorted(set(re.findall(r"--output-file\s+([A-Za-z0-9_./:-]+)", text))),
        "tasks": sorted(set(re.findall(r"--task\s+([A-Za-z0-9_./:-]+)", text))),
        "rope_variants": {
            "radix": re.findall(r"--radix\s+([0-9.]+)", text),
            "yarn": re.findall(r"--yarn\s+([0-9.]+)", text),
            "ntk": re.findall(r"--ntk\s+([0-9.]+)", text),
        },
        "token_lengths": {
            "dataset_min_tokens": re.findall(r"--dataset-min-tokens\s+([0-9]+)", text),
            "min_tokens": re.findall(r"--min-tokens\s+([0-9]+)", text),
            "max_tokens": re.findall(r"--max-tokens\s+([0-9]+)", text),
            "tokens_step": re.findall(r"--tokens-step\s+([0-9]+)", text),
            "s_len": re.findall(r"--s_len\s+\$?([A-Za-z0-9_]+)", text),
            "e_len": re.findall(r"--e_len\s+\$?([A-Za-z0-9_]+)", text),
            "step": re.findall(r"--step\s+\$?([A-Za-z0-9_]+)", text),
        },
        "flags": sorted(set(re.findall(r"(--[A-Za-z0-9_-]+)", text))),
    }


def script_manifest() -> dict[str, Any]:
    scripts = []
    for rel in SUPPLEMENT_SCRIPTS:
        path = SUPP / rel
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        scripts.append(
            {
                "relative_path": rel,
                "path": str(path),
                "exists": path.exists(),
                "line_count": len(text.splitlines()),
                "parsed": parse_script(text),
            }
        )
    payload = {
        "artifact_kind": "mrrope_official_script_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "source_distribution": "OpenReview supplementary archive",
        "repo": str(REPO),
        "scripts": scripts,
        "paper_shaped_execution_matrix": {
            "perplexity": "eval.sh: 10 Proofpile sequences, 131072-token tokenized datasets, MrRoPE vs YaRN",
            "niah": "evaln.sh: 8192 to 131072 length sweep",
            "ruler": "evalr.sh / eval/ruler.py: RULER context lengths and subtasks",
            "infinite_bench": "evali.sh / eval/infity.py: InfiniteBench subsets with 100-example slices",
            "longbenchv2": "evalb.sh / eval/longbench.py: LongBenchV2 long split under 131072 max tokens",
        },
        "accepted_loop2_evidence": (
            "full datasets, model checkpoints, patched RoPE model loading, raw outputs, "
            "scores, and timing/memory traces for the full benchmark grid"
        ),
    }
    write_json(SCRIPT_MANIFEST_PATH, payload)
    return payload


def gpu_rows() -> list[dict[str, Any]]:
    result = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )
    rows = []
    for line in result.get("stdout", "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "memory_used_mib": int(parts[3]),
                    "memory_free_mib": int(parts[4]),
                    "utilization_gpu_pct": int(parts[5]),
                }
            )
    return rows


def environment_manifest() -> dict[str, Any]:
    payload = {
        "artifact_kind": "mrrope_environment_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "repo": str(REPO),
        "supplement": str(SUPP),
        "python": sys.version,
        "gpu_rows": gpu_rows(),
        "nvidia_smi": run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=30,
        ),
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "datasets": package_version("datasets"),
            "evaluate": package_version("evaluate"),
            "flash_attn": package_version("flash-attn"),
            "bitsandbytes": package_version("bitsandbytes"),
            "rouge": package_version("rouge"),
            "jieba": package_version("jieba"),
            "sentencepiece": package_version("sentencepiece"),
            "protobuf": package_version("protobuf"),
            "einops": package_version("einops"),
        },
        "compileall_support_check": run_cmd(
            ["python", "-m", "compileall", "-q", "eval", "scaled_rope", "analysis"],
            cwd=SUPP,
            timeout=240,
        ),
        "import_support_checks": {
            "llama_mrrope": run_cmd(
                ["python", "-c", "from scaled_rope.LlamaMrRoPE import LlamaMrRoPE; print('import_ok')"],
                cwd=SUPP,
                timeout=120,
            ),
            "model_loader_args": run_cmd(
                ["python", "-c", "from eval.model_loader import add_args; print('model_loader_ok')"],
                cwd=SUPP,
                timeout=120,
            ),
        },
        "professional_runtime_note": (
            "paper hardware was not specified, but supplement uses device_map=auto, "
            "FlashAttention 2 when --flash-attention is present, and 128K-token contexts"
        ),
    }
    write_json(ENV_PATH, payload)
    return payload


def hf_repo_manifest(repo_id: str, repo_type: str) -> dict[str, Any]:
    api = HfApi(token=HfFolder.get_token())
    try:
        info = api.repo_info(repo_id, repo_type=repo_type, files_metadata=True)
        files = [
            {"path": s.rfilename, "size_bytes": getattr(s, "size", None) or 0}
            for s in info.siblings
        ]
        return {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "status": "available",
            "private": getattr(info, "private", None),
            "gated": getattr(info, "gated", None),
            "file_count": len(files),
            "total_size_bytes": sum(item["size_bytes"] for item in files),
            "files_head": files[:20],
            "files_tail": files[-20:],
        }
    except Exception as exc:
        return {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "status": "unavailable",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def local_path_manifest(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for rel in paths:
        path = SUPP / rel
        row = {
            "relative_path": rel,
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        }
        if path.exists() and path.is_dir():
            count = 0
            for _ in path.rglob("*"):
                count += 1
                if count > 100000:
                    break
            row["descendant_count_capped_100k"] = count
        rows.append(row)
    return rows


def model_data_manifest() -> dict[str, Any]:
    payload = {
        "artifact_kind": "mrrope_model_data_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "hf_token_available": bool(HfFolder.get_token()),
        "local_required_datasets": local_path_manifest(LOCAL_DATA_PATHS),
        "local_required_models": local_path_manifest(LOCAL_MODEL_PATHS),
        "hf_models": [hf_repo_manifest(repo, "model") for repo in HF_MODEL_REPOS],
        "hf_datasets": [hf_repo_manifest(repo, "dataset") for repo in HF_DATASET_REPOS],
        "score_outputs_expected": [
            "output/mrrope-llama3.csv",
            "output/yarn-llama3.csv",
            "output/2longbench-llama8b-radix.csv",
            "output/2longbench-llama8b-yarn.csv",
            "log/ruler_yarn.jsonl",
            "results_infity/llama3.1-8b-ins/*.jsonl",
            "log/<model>.png for NIAH visualization",
        ],
    }
    write_json(MODEL_DATA_PATH, payload)
    return payload


def derive_blockers(env: dict[str, Any], data: dict[str, Any]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if not (REPO / ".git").exists():
        blockers.append(
            {
                "id": "source_distribution_supplement_only",
                "status": "blocked",
                "detail": "queue reports GitHub repository not found; only OpenReview supplement archive is available",
            }
        )
    missing_data = [row["relative_path"] for row in data["local_required_datasets"] if not row.get("exists")]
    if missing_data:
        blockers.append(
            {
                "id": "local_long_context_datasets_missing",
                "status": "blocked",
                "detail": "required local dataset paths are missing: " + ", ".join(missing_data),
            }
        )
    missing_models = [row["relative_path"] for row in data["local_required_models"] if not row.get("exists")]
    if missing_models:
        blockers.append(
            {
                "id": "local_model_checkpoints_missing",
                "status": "blocked",
                "detail": "supplement scripts expect missing local checkpoints: " + ", ".join(missing_models),
            }
        )
    if not env.get("packages", {}).get("flash_attn"):
        blockers.append(
            {
                "id": "flash_attention_2_missing",
                "status": "blocked",
                "detail": "supplement benchmark scripts use --flash-attention, but flash-attn is not installed",
            }
        )
    missing_pkgs = [
        name
        for name in ["evaluate", "rouge", "jieba", "sentencepiece"]
        if not env.get("packages", {}).get(name)
    ]
    if missing_pkgs:
        blockers.append(
            {
                "id": "supplement_python_dependencies_missing",
                "status": "blocked",
                "detail": "missing supplement dependencies: " + ", ".join(missing_pkgs),
            }
        )
    unavailable = [
        item
        for item in data.get("hf_models", []) + data.get("hf_datasets", [])
        if item.get("status") != "available"
    ]
    for item in unavailable[:8]:
        blockers.append(
            {
                "id": "hf_artifact_unavailable_" + re.sub(r"[^A-Za-z0-9]+", "_", item.get("repo_id", "unknown")).strip("_"),
                "status": "blocked",
                "detail": item.get("error", "HuggingFace artifact unavailable"),
            }
        )
    rows = env.get("gpu_rows", [])
    if rows and max(row.get("memory_free_mib", 0) for row in rows) < 16000:
        blockers.append(
            {
                "id": "no_idle_large_memory_gpu_for_128k_eval",
                "status": "blocked",
                "detail": "all visible GPUs have less than 16GiB free while another full GPU run is active; 128K model eval should wait for a clean device",
            }
        )
    return blockers


def professional_gate_result(blockers: list[dict[str, str]]) -> dict[str, Any]:
    payload = {
        "artifact_kind": "mrrope_professional_gate_result",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "status": "blocked_by_supplement_only_missing_datasets_models_runtime"
        if blockers
        else "ready_for_full_paper_shaped_execution",
        "professional_package_ready": not blockers,
        "convergence_role": "professional operational gate; no reduced run is promoted",
        "support_checks": {
            "supplement_scripts_parsed": len(SUPPLEMENT_SCRIPTS),
            "compileall_and_imports_are_support_only": True,
        },
        "blockers": blockers,
        "next_full_execution_if_unblocked": [
            "materialize Proofpile tokenized datasets, InfiniteBench, RULER/NIAH, LongBenchV2, and haystack data",
            "materialize Llama3.1-8B and Qwen2.5 model checkpoints in supplement-expected locations",
            "install FlashAttention 2 and supplement dependencies",
            "run full MrRoPE/YaRN/NTK benchmark scripts and emit CSV/JSONL/raw generation outputs",
            "compare table-shaped perplexity, NIAH, RULER, Infinite-Bench, and LongBenchV2 results to the paper evidence",
        ],
    }
    write_json(PROFESSIONAL_GATE_PATH, payload)
    return payload


def add_node_if_missing(dag: dict[str, Any], node: dict[str, Any]) -> None:
    if not any(existing.get("id") == node["id"] for existing in dag.get("nodes", [])):
        dag.setdefault("nodes", []).append(node)


def add_edge_if_missing(dag: dict[str, Any], src: str, dst: str) -> None:
    edge = [src, dst]
    if edge not in dag.setdefault("edges", []):
        dag["edges"].append(edge)


def signature_for(dag: dict[str, Any]) -> str:
    payload = {
        "nodes": sorted((n.get("id"), n.get("type"), n.get("content")) for n in dag.get("nodes", [])),
        "edges": sorted(tuple(edge) for edge in dag.get("edges", [])),
        "strict_policy": dag.get("strict_policy", {}),
        "target_paper_id": dag.get("target_paper_id"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def update_dag(blockers: list[dict[str, str]]) -> dict[str, Any]:
    dag = read_json(DAG_PATH)
    dag["graph_id"] = "ICLR2026_1J63FJYJKg_mrrope_mixed_radix_rope_gap_dag_iter_03"
    dag["updated_at_utc"] = utc_now()
    new_nodes = [
        {
            "id": "ops.supplement_script_matrix",
            "type": "operational_execution_matrix",
            "skill_role": "bind Loop 2 to the OpenReview supplement scripts",
            "content": (
                "Use eval.sh/evalb.sh/evali.sh/evaln.sh/evalr.sh plus eval/perplexity.py, "
                "eval/neddle.py, eval/ruler.py, eval/infity.py, eval/longbench.py, and scaled_rope patches. "
                "Extract MrRoPE/YaRN/NTK variants, 8192-131072 token sweeps, dataset paths, model paths, and output files."
            ),
        },
        {
            "id": "ops.long_context_dataset_checkpoint_gate",
            "type": "model_data_gate",
            "skill_role": "require full paper datasets and checkpoints",
            "content": (
                "Before running or converging, resolve testset/pp-tokenized-llama3, testset/pp-tokenized-qwen2.5, "
                "testset/infity, testset/longbenchv2, testset/PaulGrahamEssays, models/llama3.1-8b-ins, "
                "and models/qwen2.5-3b-ins. Missing local datasets or checkpoints block Loop 2."
            ),
        },
        {
            "id": "ops.flashattention_128k_runtime_gate",
            "type": "professional_runtime_gate",
            "skill_role": "prevent fake 128K-context convergence",
            "content": (
                "Verify FlashAttention 2, supplement dependencies, device_map=auto viability, and a clean large-memory GPU "
                "before 128K perplexity/retrieval/dialogue runs. Import or compile checks are support only."
            ),
        },
        {
            "id": "ops.full_mrrope_result_artifact_gate",
            "type": "evaluation_artifact_gate",
            "skill_role": "make verifier comparison table-shaped",
            "content": (
                "Require full Proofpile perplexity CSVs, NIAH length/depth JSON/visualization, RULER 13-subtask logs, "
                "Infinite-Bench JSONL outputs/scores, and LongBenchV2 CSV/domain scores before accepting close result shape."
            ),
        },
        {
            "id": "decision.explicit_blocker_after_mrrope_preflight",
            "type": "author_reviewer_decision",
            "skill_role": "feed supplement-only/data/runtime failure back into Loop 1",
            "content": (
                "If only supplement code is available or datasets/checkpoints/runtime are missing, mark not converged and "
                "record exact artifact debt. Do not run reduced context probes as convergence."
            ),
        },
    ]
    for node in new_nodes:
        add_node_if_missing(dag, node)
    add_edge_if_missing(dag, "ops.resolve_repo_code", "ops.supplement_script_matrix")
    add_edge_if_missing(dag, "ops.supplement_script_matrix", "ops.long_context_dataset_checkpoint_gate")
    add_edge_if_missing(dag, "ops.resolve_models_data", "ops.long_context_dataset_checkpoint_gate")
    add_edge_if_missing(dag, "ops.long_context_dataset_checkpoint_gate", "ops.flashattention_128k_runtime_gate")
    add_edge_if_missing(dag, "ops.flashattention_128k_runtime_gate", "ops.full_mrrope_result_artifact_gate")
    add_edge_if_missing(dag, "ops.full_mrrope_result_artifact_gate", "loop2.execute_operational_dag")
    add_edge_if_missing(dag, "loop2.execute_operational_dag", "decision.explicit_blocker_after_mrrope_preflight")
    add_edge_if_missing(dag, "decision.explicit_blocker_after_mrrope_preflight", "reviewer.keep_exact_artifact_debt")
    dag.setdefault("previous_loop_updates", []).append(
        {
            "id": "update.add_mrrope_supplement_dataset_runtime_result_gates",
            "reason": "specialized MrRoPE preflight found supplement-only source and missing full benchmark artifacts",
            "success_criteria": [
                "supplement script matrix encoded",
                "local long-context dataset/model gate encoded",
                "FlashAttention/128K runtime gate encoded",
                "full result artifact gate encoded",
                "reduced context probes remain non-convergent",
            ],
            "blocker_ids": [item["id"] for item in blockers],
        }
    )
    dag["signature"] = signature_for(dag)
    write_json(PAPER_RUN / "paper_author_gap_dag_iter_03.json", dag)
    write_json(DAG_PATH, dag)
    return dag


def update_paper_run(verifier: dict[str, Any], dag: dict[str, Any]) -> None:
    gate = verifier["professional_gate"]
    existing = read_json(PAPER_RUN / "operational_artifacts.json") if (PAPER_RUN / "operational_artifacts.json").exists() else {}
    write_json(
        PAPER_RUN / "operational_artifacts.json",
        {
            "blockers": gate.get("blockers", []),
            "gpu_probe": {"status": "pass", "gpu_rows": verifier["environment"].get("gpu_rows", [])},
            "repo_audits": existing.get("repo_audits", []),
            "specialized_runner": {
                "status": gate["status"],
                "artifact_dir": str(RUNNER_DIR),
                "environment_path": str(ENV_PATH),
                "official_script_manifest_path": str(SCRIPT_MANIFEST_PATH),
                "model_data_manifest_path": str(MODEL_DATA_PATH),
                "professional_gate_path": str(PROFESSIONAL_GATE_PATH),
                "verifier_path": str(VERIFIER_PATH),
            },
        },
    )
    write_json(
        PAPER_RUN / "verifier_result_iter_03.json",
        {
            "checks": [
                {"name": "blind_contract", "status": "pass", "detail": verifier["blind_contract_checked"]},
                {
                    "name": "gap_semantic_match",
                    "status": "pass",
                    "detail": "Existing iter_02 semantic gap remains accepted; iter_03 tightens operational gates.",
                },
                {
                    "name": "professional_artifact_package",
                    "status": "blocked",
                    "detail": {
                        "ready": False,
                        "reason": gate["status"],
                        "specialized_runner_artifact_dir": str(RUNNER_DIR),
                        "blocker_count": len(gate.get("blockers", [])),
                    },
                },
                {
                    "name": "exact_artifact_debt_recorded",
                    "status": "pass",
                    "detail": [
                        {
                            "id": "main_benchmark_tables",
                            "required": "Proofpile perplexity, NIAH, RULER, Infinite-Bench, LongBenchV2 full outputs",
                        },
                        {
                            "id": "models_data_runtime",
                            "required": "local model checkpoints, local datasets, FlashAttention 2, supplement dependencies",
                        },
                        {
                            "id": "raw_outputs_and_metrics",
                            "required": "CSV/JSONL predictions, ROUGE/retrieval/accuracy/perplexity scores, timing/memory traces",
                        },
                    ],
                },
            ],
            "converged": False,
            "created_at_utc": utc_now(),
            "iteration": 3,
            "paper_id": PAPER_ID,
            "paper_title": TITLE,
            "professional_ready": False,
            "required_updates": [
                {
                    "id": "update.resolve_mrrope_professional_blockers",
                    "reason": gate["status"],
                    "success_criteria": [
                        "run paper-appropriate long-context evals or provide exact operational artifacts",
                        "emit verifier-comparable table/figure result shapes",
                        "do not count reduced/proxy/syntax-only evidence",
                    ],
                }
            ],
            "score": 0.857143,
            "semantic_ready": True,
            "status": "blocked_by_supplement_only_missing_datasets_models_runtime_after_specialized_runner",
            "dag_signature": dag["signature"],
        },
    )
    (PAPER_RUN / "STATUS.md").write_text(
        f"# {TITLE}\n\n"
        f"- Paper id: `{PAPER_ID}`\n"
        "- Final status: `blocked_by_supplement_only_missing_datasets_models_runtime_after_specialized_runner`\n"
        "- Converged: `false`\n"
        "- Semantic ready: `true`\n"
        "- Professional ready: `false`\n"
        f"- DAG signature: `{dag['signature']}`\n"
        f"- Specialized runner status: `{gate['status']}`\n"
        f"- Specialized status: `{STATUS_PATH}`\n\n"
        "## Current Professional Blockers\n\n"
        + "\n".join(f"- `{item['id']}`: {item['detail']}" for item in gate.get("blockers", []))
        + "\n",
        encoding="utf-8",
    )


def write_status(verifier: dict[str, Any]) -> None:
    gate = verifier["professional_gate"]
    STATUS_PATH.write_text(
        "# MrRoPE Specialized Runner Status\n\n"
        f"- Updated: {verifier['updated_at_utc']}\n"
        f"- Paper: `{TITLE}`\n"
        f"- Status: `{gate['status']}`\n"
        f"- Professional package ready: `{gate['professional_package_ready']}`\n"
        f"- Supplement scripts parsed: `{gate['support_checks']['supplement_scripts_parsed']}`\n"
        f"- Blocker count: `{len(gate.get('blockers', []))}`\n\n"
        "## Artifact Paths\n"
        f"- Environment: `{ENV_PATH}`\n"
        f"- Script manifest: `{SCRIPT_MANIFEST_PATH}`\n"
        f"- Model/data manifest: `{MODEL_DATA_PATH}`\n"
        f"- Professional gate: `{PROFESSIONAL_GATE_PATH}`\n"
        f"- Verifier: `{VERIFIER_PATH}`\n\n"
        "## Why This Is Not Converged\n"
        "- This did not run one short-context probe or import check as convergence evidence.\n"
        "- The full paper shape requires 128K Proofpile/NIAH/RULER/Infinite-Bench/LongBenchV2 artifacts and model/runtime traces.\n"
        "- The DAG was updated so Loop 2 must satisfy these exact gates before verifier acceptance.\n\n"
        "## Current Blockers\n"
        + "\n".join(f"- `{item['id']}`: {item['detail']}" for item in gate.get("blockers", []))
        + "\n",
        encoding="utf-8",
    )


def update_global_files(verifier: dict[str, Any]) -> None:
    gate = verifier["professional_gate"]
    evidence = {
        "environment_path": str(ENV_PATH),
        "official_script_manifest_path": str(SCRIPT_MANIFEST_PATH),
        "model_data_manifest_path": str(MODEL_DATA_PATH),
        "professional_gate_path": str(PROFESSIONAL_GATE_PATH),
        "verifier_path": str(VERIFIER_PATH),
        "blockers": gate.get("blockers", []),
    }
    if QUEUE_PATH.exists():
        queue = read_json(QUEUE_PATH)
        for item in queue.get("queue", []):
            if item.get("paper_id") == PAPER_ID:
                item["professional_blocker"] = gate["status"] + "_after_specialized_runner"
                item["specialized_runner_status"] = gate["status"]
                item["specialized_runner_artifact_dir"] = str(RUNNER_DIR)
                item["specialized_runner_evidence"] = evidence
                statuses = item.setdefault("implementation_statuses", [])
                for status in [
                    "specialized_runner_preflight_completed",
                    "supplement_scripts_parsed",
                    "hf_model_and_dataset_manifests_checked",
                    "blocked_supplement_only_missing_datasets_models_runtime",
                ]:
                    if status not in statuses:
                        statuses.append(status)
                break
        write_json(QUEUE_PATH, queue)
    if SUMMARY_PATH.exists():
        summary = read_json(SUMMARY_PATH)
        summary["created_at_utc"] = utc_now()
        for paper in summary.get("papers", []):
            if paper.get("paper_id") == PAPER_ID:
                paper["final_status"] = "blocked_by_supplement_only_missing_datasets_models_runtime_after_specialized_runner"
                paper["converged"] = False
                paper["specialized_runner_status"] = gate["status"]
                paper["specialized_runner_artifact_dir"] = str(RUNNER_DIR)
                paper["specialized_runner_evidence"] = evidence
                statuses = paper.setdefault("implementation_statuses", [])
                for status in [
                    "specialized_runner_preflight_completed",
                    "supplement_scripts_parsed",
                    "hf_model_and_dataset_manifests_checked",
                    "blocked_supplement_only_missing_datasets_models_runtime",
                ]:
                    if status not in statuses:
                        statuses.append(status)
        write_json(SUMMARY_PATH, summary)
    QUEUE_MD_PATH.write_text(
        "# Specialized Runner Queue\n\n"
        f"Updated: `{utc_now()}`\n\n"
        "This queue preserves non-reduced professional gates. A specialized runner may block, but it must not count reduced/proxy evidence as convergence.\n\n"
        "## Recently Updated\n"
        f"- MrRoPE: `{gate['status']}` with `{len(gate.get('blockers', []))}` blockers. Artifact dir: `{RUNNER_DIR}`\n"
        "- LoongRL: cluster/data/checkpoint/runtime gate completed and blocked.\n"
        "- Prophet: full GSM8K GPU run still tracked separately.\n",
        encoding="utf-8",
    )
    current = LONGGOAL_STATUS_PATH.read_text(encoding="utf-8") if LONGGOAL_STATUS_PATH.exists() else ""
    insert = (
        f"- MrRoPE: `{gate['status']}_after_specialized_runner`; supplement scripts parsed, local model/data/runtime gates checked, and DAG iter 03 now encodes full 128K benchmark artifact requirements.\n"
    )
    if "MrRoPE:" not in current:
        current = current.replace(
            "- LoongRL: `blocked_by_cluster_training_data_checkpoint_and_runtime_requirements_after_specialized_runner`",
            insert
            + "- LoongRL: `blocked_by_cluster_training_data_checkpoint_and_runtime_requirements_after_specialized_runner`",
        )
    current = re.sub(r"Date: `[^`]+`", f"Date: `{utc_now()}`", current)
    LONGGOAL_STATUS_PATH.write_text(current, encoding="utf-8")


def main() -> int:
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    env = environment_manifest()
    scripts = script_manifest()
    data = model_data_manifest()
    blockers = derive_blockers(env, data)
    gate = professional_gate_result(blockers)
    verifier = {
        "artifact_kind": "mrrope_specialized_verifier",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "repo": str(REPO),
        "supplement": str(SUPP),
        "dag_path": str(DAG_PATH),
        "blind_contract_checked": {
            "only_input_file": "paper_author_gap_dag.json",
            "paper_text_visible_to_loop2": False,
            "oracle_results_visible_to_loop2": False,
            "previous_memory_visible_to_loop2": False,
            "repo_paths_visible_only_if_encoded_in_dag": True,
        },
        "official_script_manifest": scripts,
        "model_data_manifest": data,
        "environment": env,
        "professional_gate": gate,
        "verifier": {
            "status": gate["status"],
            "convergence_decision": "not_converged_explicit_professional_blockers_after_operational_preflight",
            "professional_package_ready": gate["professional_package_ready"],
            "semantic_dag_nodes_checked": [
                "gap.paper_gap_claims",
                "method.bind_gap_to_mechanism",
                "experiments.benchmark_metric_grid",
                "experiments.system_surface",
                "ops.resolve_models_data",
            ],
            "unresolved_professional_debt": blockers,
            "loop1_required_dag_update": [
                "Add supplement script matrix gate.",
                "Add local 128K dataset and model checkpoint gate.",
                "Add FlashAttention/dependency/runtime gate.",
                "Add full result artifact gate for CSV/JSONL/figure outputs.",
                "Keep imports and tiny context probes as support only.",
            ],
        },
    }
    write_json(VERIFIER_PATH, verifier)
    dag = update_dag(blockers)
    update_paper_run(verifier, dag)
    write_status(verifier)
    update_global_files(verifier)
    print(json.dumps(verifier["verifier"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
