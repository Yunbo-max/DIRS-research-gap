#!/usr/bin/env python3
"""PRISM professional operational gate for the strict DIRS loop.

PRISM has a real local repository and an author pipeline, but convergence
requires fMRI datasets, generated structured-text artifacts, checkpoints, API
stages, reconstruction/QA metrics, and paper-shaped result grids. This gate
turns the previous vague "no repo encoded" blocker into explicit operational
nodes without promoting script parsing or unrelated local files to evidence.
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


RUN_ROOT = Path(
    "/tf/notebooks/yunbo/DIRS/case1_writing/runs/"
    "remaining19_p_less_style_dirs_20260723"
)
PAPER_RUN = RUN_ROOT / "paper_runs" / "iclr2026_88zlp7xyxw_prism_fmri_structured_text"
RUNNER_DIR = RUN_ROOT / "specialized_runners" / "prism"
REPO = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/PRISM")

PAPER_ID = "ICLR2026_88ZLp7xYxw_prism_fmri_structured_text"
TITLE = "Seeing Through the Brain: New Insights from Decoding Visual Stimuli with fMRI"

DAG_PATH = PAPER_RUN / "paper_author_gap_dag.json"
QUEUE_PATH = RUN_ROOT / "specialized_runner_queue.json"
SUMMARY_PATH = RUN_ROOT / "remaining19_strict_dirs_summary.json"
STATUS_PATH = RUNNER_DIR / "PRISM_SPECIALIZED_STATUS.md"
VERIFIER_PATH = RUNNER_DIR / "prism_specialized_verifier.json"
PROFESSIONAL_GATE_PATH = RUNNER_DIR / "professional_gate_result.json"
REPO_MANIFEST_PATH = RUNNER_DIR / "repo_manifest.json"
ENV_PATH = RUNNER_DIR / "environment.json"
MODEL_DATA_PATH = RUNNER_DIR / "model_data_manifest.json"

SCRIPT_FILES = [
    "README.md",
    "src/env.yml",
    "src/quick_start.sh",
    "src/create_data.py",
    "src/new_nsd_data.py",
    "src/search_prompt/keyword_generator.py",
    "src/search_prompt/keyword_search.py",
    "src/get_gpt_data.py",
    "src/gpt_rewrite.py",
    "src/my_model.py",
    "src/train_model.py",
    "src/gen_pipe.py",
    "src/run_gen.py",
    "src/text_prompt_utils.py",
]

EXPECTED_OUTPUT_SURFACES = [
    "NSD, BOLD5000, and Generic Object Decoding fMRI reconstruction benchmarks",
    "COCO-derived QA pairs and Qwen2.5 QA scoring on reconstructed images",
    "five-run subject-averaged metrics and standardized fMRI train/test splits",
    "PixCorr, SSIM, LPIPS, CLIP, Inception V3, QA accuracy, CKA, CCA, and generalization-gap outputs",
    "latent-space ablation comparing CLIP text latent, LDM latent, and PRISM structured text",
    "module ablation for attribute/relationship search and object-centric cross-attention generation",
    "raw reconstructed images, prompts/structured descriptions, checkpoints, metric JSON/CSV, and GPU traces",
]

PAPER_ARTIFACT_PATTERNS = {
    "nsd_artifacts": ["*nsd*", "*NSD*"],
    "bold5000_artifacts": ["*bold5000*", "*BOLD5000*"],
    "god_artifacts": ["*god*", "*GOD*", "*generic*object*decoding*"],
    "coco_artifacts": ["*coco*", "*COCO*"],
    "structured_text_npz": ["*structured*.npz", "*response*.npz", "*gpt*.npz", "*.npz"],
    "keyword_artifacts": ["best_keyword.json", "*keyword*.json"],
    "checkpoint_artifacts": ["*.pt", "*.pth", "*.ckpt", "*.safetensors"],
    "metric_outputs": ["*pixcorr*.json", "*ssim*.json", "*lpips*.json", "*clip*.json", "*inception*.json", "*qa*.json", "*cka*.json", "*cca*.json", "*metric*.json", "*result*.json", "*.csv"],
    "generated_images": ["*.png", "*.jpg", "*.jpeg"],
}

SEARCH_ROOTS = [
    REPO,
    Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h"),
    RUNNER_DIR,
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


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def run_cmd(cmd: list[str], *, cwd: Path | None = None, timeout: int = 90) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout)
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


def import_probe(module_name: str) -> dict[str, Any]:
    return run_cmd(
        [sys.executable, "-c", f"import importlib; m=importlib.import_module({module_name!r}); print(getattr(m, '__version__', 'imported'))"],
        timeout=45,
    )


def path_size(path: Path) -> tuple[str | None, int]:
    if not path.exists():
        return None, 0
    if path.is_file():
        return str(path.stat().st_size), 1
    size = run_cmd(["du", "-sh", str(path)], timeout=20)["stdout"].split()
    count = run_cmd(["bash", "-lc", f"find {str(path)!r} -type f | wc -l"], timeout=25)
    try:
        file_count = int(count["stdout"].strip())
    except ValueError:
        file_count = 0
    return (size[0] if size else None), file_count


def gpu_rows() -> list[dict[str, Any]]:
    result = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )
    rows = []
    for line in result.get("stdout", "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 8:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_mib": int(float(parts[2])),
                    "memory_used_mib": int(float(parts[3])),
                    "memory_free_mib": int(float(parts[4])),
                    "utilization_gpu_pct": int(float(parts[5])),
                    "temperature_c": int(float(parts[6])),
                    "power_w": float(parts[7]),
                }
            )
    return rows


def repo_manifest() -> dict[str, Any]:
    files = []
    for rel in SCRIPT_FILES:
        path = REPO / rel
        row: dict[str, Any] = {"relative_path": rel, "path": str(path), "exists": path.exists()}
        if path.exists() and path.is_file():
            text = read_text(path)
            row["size_bytes"] = path.stat().st_size
            row["line_count"] = len(text.splitlines())
            if path.suffix == ".py":
                row["py_compile"] = run_cmd([sys.executable, "-m", "py_compile", str(path)], timeout=45)
                row["imports"] = sorted(set(re.findall(r"^(?:from|import)\s+([A-Za-z0-9_\.]+)", text, re.M)))
                row["cli_flags"] = sorted(set(re.findall(r"['\"](--[A-Za-z0-9_-]+)['\"]", text)))
            elif path.suffix == ".sh":
                row["shell_syntax"] = run_cmd(["bash", "-n", str(path)], timeout=30)
                row["commands"] = [line.strip() for line in text.splitlines() if line.strip().startswith("python ")]
                row["contains_api_placeholder"] = "{api key}" in text
            elif path.name == "README.md":
                row["urls"] = sorted(set(re.findall(r"https?://[^)\s]+", text)))
                row["mentions_nsd_access_form"] = "NSD Data Access form" in text
                row["mentions_l40_cuda_12_2"] = "NVIDIA L40" in text and "CUDA Version 12.2" in text
                row["pipeline_steps"] = [line.strip("` ") for line in text.splitlines() if line.strip().startswith("python ")]
            elif path.name == "env.yml":
                row["head"] = "\n".join(text.splitlines()[:80])
                row["declared_packages"] = [
                    line.strip().lstrip("- ").strip()
                    for line in text.splitlines()
                    if line.strip().startswith("- ") and not line.strip().endswith(":")
                ]
        files.append(row)

    all_repo_files = sorted(str(p.relative_to(REPO)) for p in REPO.rglob("*") if p.is_file() and ".git" not in p.parts)
    payload = {
        "artifact_kind": "prism_repo_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "repo": str(REPO),
        "repo_exists": REPO.exists(),
        "script_files": files,
        "all_repo_files_excluding_git": all_repo_files[:400],
        "git": {
            "remote": run_cmd(["git", "-C", str(REPO), "remote", "-v"], timeout=20),
            "head": run_cmd(["git", "-C", str(REPO), "rev-parse", "HEAD"], timeout=20),
        },
        "support_only_findings": [
            "README, source compile, and quick_start.sh syntax prove executable intent only",
            "OpenAI API stages and access-gated fMRI data prevent a faithful end-to-end run until artifacts are materialized",
            "unrelated local COCO/checkpoints do not count unless tied to the PRISM result grid",
        ],
    }
    write_json(REPO_MANIFEST_PATH, payload)
    return payload


def environment_manifest() -> dict[str, Any]:
    packages = {
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "transformers": package_version("transformers"),
        "diffusers": package_version("diffusers"),
        "xformers": package_version("xformers"),
        "openai": package_version("openai"),
        "langchain": package_version("langchain"),
        "langchain-openai": package_version("langchain-openai"),
        "lpips": package_version("lpips"),
        "retry": package_version("retry"),
        "scikit-learn": package_version("scikit-learn"),
        "pillow": package_version("pillow"),
    }
    payload = {
        "artifact_kind": "prism_environment_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "python": sys.version,
        "gpu_rows": gpu_rows(),
        "openai_api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "openai_api_key_value_recorded": False,
        "packages": packages,
        "import_probes": {
            "torch": import_probe("torch"),
            "torchvision": import_probe("torchvision"),
            "transformers": import_probe("transformers"),
            "diffusers": import_probe("diffusers"),
            "xformers": import_probe("xformers"),
            "openai": import_probe("openai"),
            "langchain": import_probe("langchain"),
            "langchain_openai": import_probe("langchain_openai"),
            "lpips": import_probe("lpips"),
            "retry": import_probe("retry"),
            "sklearn": import_probe("sklearn"),
            "PIL": import_probe("PIL"),
        },
        "compileall_src": run_cmd([sys.executable, "-m", "compileall", "-q", str(REPO / "src")], timeout=180),
        "quick_start_syntax": run_cmd(["bash", "-n", str(REPO / "src/quick_start.sh")], timeout=30),
        "professional_hardware_expected_by_dag": [
            "two NVIDIA L40 48GB GPUs",
            "CUDA 12.2",
            "OpenAI API access for keyword generation and structured text generation",
        ],
    }
    write_json(ENV_PATH, payload)
    return payload


def collect_candidates(kind: str, patterns: list[str], limit: int = 80) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if ".git" in path.parts or "__pycache__" in path.parts:
                    continue
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                size_human, file_count = path_size(path)
                rows.append(
                    {
                        "path": str(path),
                        "relative_to_search_root": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                        "search_root": str(root),
                        "is_dir": path.is_dir(),
                        "size_human": size_human,
                        "file_count": file_count,
                        "support_only_until_grid_bound": True,
                    }
                )
                if len(rows) >= limit:
                    return rows
    return rows


def model_data_manifest() -> dict[str, Any]:
    candidates = {kind: collect_candidates(kind, patterns) for kind, patterns in PAPER_ARTIFACT_PATTERNS.items()}
    # PRISM-specific artifacts expected by the released scripts.
    exact_paths = {
        "best_keyword_json": REPO / "src/search_prompt/best_keyword.json",
        "train_structured_npz": REPO / "src/data/train_response.npz",
        "val_structured_npz": REPO / "src/data/val_response.npz",
        "test_structured_npz": REPO / "src/data/test_response.npz",
        "trained_checkpoint_cur_best": REPO / "src/cur_best_large_lr.pt",
        "generated_output_dir": REPO / "src/results",
        "metric_output_dir": REPO / "src/metrics",
    }
    exact = {}
    for name, path in exact_paths.items():
        size_human, file_count = path_size(path)
        exact[name] = {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir() if path.exists() else False,
            "size_human": size_human,
            "file_count": file_count,
        }
    payload = {
        "artifact_kind": "prism_model_data_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "candidate_artifacts": candidates,
        "exact_prism_pipeline_artifacts": exact,
        "paper_shaped_outputs_required": EXPECTED_OUTPUT_SURFACES,
        "support_only_warning": "candidate artifacts are not verifier-comparable unless bound to PRISM paper splits, metrics, tables, figures, and ablations",
    }
    write_json(MODEL_DATA_PATH, payload)
    return payload


def professional_gate(manifest: dict[str, Any], env: dict[str, Any], model_data: dict[str, Any]) -> dict[str, Any]:
    blockers = []
    gpu = env["gpu_rows"]
    gpu_names = " | ".join(row["name"] for row in gpu)
    l40_48_count = sum("L40" in row["name"] and row["memory_total_mib"] >= 45000 for row in gpu)
    clean_l40_like = any(row["memory_total_mib"] >= 45000 and row["memory_free_mib"] >= 30000 and row["utilization_gpu_pct"] < 30 for row in gpu)
    if l40_48_count < 2:
        blockers.append({"id": "two_l40_48gb_hardware_missing", "status": "blocked", "detail": f"README/DAG expect two NVIDIA L40 48GB GPUs; visible devices are {gpu_names}."})
    if not clean_l40_like:
        blockers.append({"id": "clean_large_fmri_generation_gpu_missing", "status": "blocked", "detail": f"No clean >=45GB GPU is available for the paper-shaped PRISM reconstruction run. GPU rows: {gpu}."})

    missing_runtime = [
        name for name, result in env["import_probes"].items()
        if result["returncode"] != 0 and name in {"diffusers", "xformers", "langchain_openai", "lpips", "retry"}
    ]
    if missing_runtime:
        blockers.append({"id": "prism_runtime_dependencies_missing", "status": "blocked", "detail": "Missing imports for the released PRISM pipeline: " + ", ".join(missing_runtime)})
    if not env["openai_api_key_present"]:
        blockers.append({"id": "openai_api_pipeline_key_missing", "status": "blocked", "detail": "quick_start.sh and README require OpenAI API key stages for keyword and structured-text generation."})

    exact = model_data["exact_prism_pipeline_artifacts"]
    missing_exact = [name for name, row in exact.items() if not row["exists"]]
    if missing_exact:
        blockers.append({"id": "prism_pipeline_artifacts_missing", "status": "blocked", "detail": "Missing PRISM pipeline artifacts: " + ", ".join(missing_exact)})

    candidate = model_data["candidate_artifacts"]
    prism_bound = {
        kind: [
            row for row in rows
            if row["path"].startswith(str(REPO))
            and not row["path"].endswith((".py", ".pyc", ".md", ".yml", ".yaml"))
        ]
        for kind, rows in candidate.items()
    }
    prism_metric_grid = [
        row for row in prism_bound["metric_outputs"]
        if any(metric in row["path"].lower() for metric in ["pixcorr", "ssim", "lpips", "clip", "inception", "qa", "cka", "cca", "metric", "result"])
    ]
    prism_god_artifacts = [
        row for row in prism_bound["god_artifacts"]
        if "godel" not in row["path"].lower() and ("god" in row["path"].lower() or "generic" in row["path"].lower())
    ]
    if not prism_bound["nsd_artifacts"]:
        blockers.append({"id": "nsd_access_gated_fmri_data_missing", "status": "blocked", "detail": "No materialized NSD fMRI data for PRISM was found; README requires NSD terms/form access."})
    if not prism_bound["bold5000_artifacts"]:
        blockers.append({"id": "bold5000_benchmark_artifacts_missing", "status": "blocked", "detail": "No BOLD5000 benchmark artifacts found for PRISM verifier comparison."})
    if not prism_god_artifacts:
        blockers.append({"id": "generic_object_decoding_artifacts_missing", "status": "blocked", "detail": "No Generic Object Decoding benchmark artifacts found for PRISM verifier comparison."})
    if not prism_bound["coco_artifacts"]:
        blockers.append({"id": "coco_qa_artifacts_missing", "status": "blocked", "detail": "No PRISM-bound COCO image/caption/QA artifacts found for verifier comparison."})

    if len(prism_metric_grid) < 8:
        blockers.append({"id": "full_fmri_reconstruction_qa_metric_grid_missing", "status": "blocked", "detail": "No complete PixCorr/SSIM/LPIPS/CLIP/Inception/QA/CKA/CCA/generalization-gap result grid was found."})

    gate = {
        "artifact_kind": "prism_professional_gate_result",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "status": "ready_for_full_prism_fmri_reconstruction_grid_not_converged" if not blockers else "blocked_by_fmri_data_api_runtime_hardware_pipeline_artifacts_and_metric_grid",
        "professional_package_ready": not blockers,
        "convergence_role": "professional operational gate; no script syntax, API placeholder, or unrelated local artifact evidence is promoted",
        "blockers": blockers,
        "support_checks": {
            "repo_discovered_and_encoded": REPO.exists(),
            "source_compile_passed": env["compileall_src"]["returncode"] == 0,
            "quick_start_syntax_passed": env["quick_start_syntax"]["returncode"] == 0,
            "openai_api_key_present": env["openai_api_key_present"],
            "gpu_rows_checked": len(gpu),
            "candidate_artifact_families_checked": sorted(candidate.keys()),
            "prism_bound_candidate_counts": {kind: len(rows) for kind, rows in prism_bound.items()},
            "prism_metric_grid_candidate_count": len(prism_metric_grid),
        },
        "next_full_execution_if_unblocked": [
            "materialize NSD under accepted terms plus BOLD5000, GOD, COCO image/caption/QA assets",
            "install exact PRISM environment including diffusers, xformers, langchain-openai, lpips, and retry",
            "run keyword_generator.py and get_gpt_data.py/gpt_rewrite.py with an API key to produce best_keyword.json and structured response npz files",
            "train train_model.py on PRISM splits and save fMRI-to-T5 checkpoint artifacts",
            "run run_gen.py/gen_pipe.py to generate reconstructed images for NSD/BOLD5000/GOD",
            "score PixCorr, SSIM, LPIPS, CLIP, Inception V3, Qwen2.5 QA, CKA, CCA, and generalization gap",
            "run latent-space and module ablations, emit raw outputs, metrics, GPU traces, and compare to paper tables/figures/paragraphs",
        ],
        "paper_shaped_outputs_required": EXPECTED_OUTPUT_SURFACES,
    }
    write_json(PROFESSIONAL_GATE_PATH, gate)
    return gate


def ensure_node(dag: dict[str, Any], node: dict[str, Any]) -> None:
    for existing in dag.setdefault("nodes", []):
        if existing.get("id") == node["id"]:
            existing.update(node)
            return
    dag["nodes"].append(node)


def ensure_edge(dag: dict[str, Any], source: str, target: str) -> None:
    edge = [source, target]
    if edge not in dag.setdefault("edges", []):
        dag["edges"].append(edge)


def update_dag(gate: dict[str, Any]) -> dict[str, Any]:
    dag = read_json(DAG_PATH)
    for node in dag.get("nodes", []):
        if node.get("id") == "ops.resolve_repo_code":
            node["repo_paths"] = [str(REPO)]
            node["content"] = (
                "repos=/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/PRISM; "
                "code_artifacts=src/quick_start.sh; src/search_prompt/keyword_generator.py; "
                "src/get_gpt_data.py; src/gpt_rewrite.py; src/train_model.py; src/run_gen.py; "
                "src/my_model.py; src/gen_pipe.py; src/env.yml; README.md"
            )
    nodes = [
        ("ops.prism_repo_script_gate", "Resolve PRISM repo, parse quick_start.sh/env.yml, compile source, and record the keyword, GPT-data, training, and generation entrypoints.", "operational_dependency"),
        ("ops.prism_fmri_dataset_access_gate", "Materialize NSD under access terms plus BOLD5000, Generic Object Decoding, and COCO image/caption/QA artifacts on PRISM splits.", "operational_dependency"),
        ("ops.prism_openai_structured_text_gate", "Run API-backed keyword and structured-text stages to produce best_keyword.json and train/val/test structured response npz artifacts.", "operational_dependency"),
        ("ops.prism_runtime_hardware_gate", "Install diffusers/xformers/langchain-openai/lpips/retry stack and require two L40-48GB/CUDA-12.2-class execution traces for paper-shaped reconstruction.", "systems_measurement"),
        ("ops.prism_full_reconstruction_qa_grid", "Run fMRI-to-structured-text training, image generation, latent/module ablations, PixCorr/SSIM/LPIPS/CLIP/Inception/QA/CKA/CCA/generalization-gap scoring, and GPU traces.", "operational_execution"),
        ("decision.explicit_blocker_after_prism_preflight", "If fMRI data, API artifacts, runtime, hardware, checkpoints, metric outputs, or ablation grid are missing, block and feed exact requirements back into Loop 1.", "author_reviewer_decision"),
    ]
    for node_id, content, typ in nodes:
        ensure_node(dag, {"id": node_id, "content": content, "type": typ, "skill_role": "paper-specific operational gate"})
    for source, target in [
        ("ops.resolve_repo_code", "ops.prism_repo_script_gate"),
        ("ops.prism_repo_script_gate", "ops.prism_fmri_dataset_access_gate"),
        ("ops.prism_fmri_dataset_access_gate", "ops.prism_openai_structured_text_gate"),
        ("ops.prism_openai_structured_text_gate", "ops.prism_runtime_hardware_gate"),
        ("ops.prism_runtime_hardware_gate", "ops.prism_full_reconstruction_qa_grid"),
        ("ops.prism_full_reconstruction_qa_grid", "reviewer.require_professional_artifact_package"),
        ("ops.prism_full_reconstruction_qa_grid", "reviewer.compare_result_shapes"),
        ("reviewer.keep_exact_artifact_debt", "decision.explicit_blocker_after_prism_preflight"),
    ]:
        ensure_edge(dag, source, target)
    dag.setdefault("previous_loop_updates", []).append(
        {
            "iteration": 3,
            "created_at_utc": utc_now(),
            "source": "prism_specialized_professional_gate",
            "status": gate["status"],
            "blocker_ids": [b["id"] for b in gate["blockers"]],
            "repo_paths": [str(REPO)],
            "converged": False,
        }
    )
    sig_src = json.dumps(dag.get("nodes", []), sort_keys=True) + json.dumps(dag.get("edges", []), sort_keys=True)
    dag["signature"] = hashlib.sha256(sig_src.encode("utf-8")).hexdigest()[:16]
    write_json(DAG_PATH, dag)
    write_json(PAPER_RUN / "paper_author_gap_dag_iter_03.json", dag)
    return dag


def verifier(gate: dict[str, Any], dag: dict[str, Any]) -> dict[str, Any]:
    checks = [
        {"name": "blind_contract", "status": "pass", "detail": dag.get("blind_contract", {})},
        {"name": "repo_path_encoded", "status": "pass" if str(REPO) in json.dumps(dag) else "fail", "detail": [str(REPO)]},
        {"name": "reduced_proxy_rejection_gate", "status": "pass", "detail": "script syntax, source compile, API placeholder, and unrelated local artifacts cannot converge"},
        {"name": "professional_artifact_package", "status": "pass" if gate["professional_package_ready"] else "blocked", "detail": gate["blockers"]},
        {"name": "result_shape_comparison_ready", "status": "blocked", "detail": "requires PRISM fMRI reconstruction/QA metric grid before comparison to paper evidence channels"},
    ]
    payload = {
        "artifact_kind": "prism_specialized_verifier",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "iteration": 3,
        "converged": False,
        "professional_ready": gate["professional_package_ready"],
        "checks": checks,
        "required_updates": [
            {
                "id": "update.prism_repo_to_full_fmri_operational_gates",
                "reason": gate["status"],
                "success_criteria": [
                    "repo path encoded",
                    "fMRI dataset/materialization gate present",
                    "OpenAI structured-text artifact gate present",
                    "runtime/hardware gate present",
                    "full reconstruction/QA/ablation metric grid gate present",
                ],
            }
        ],
        "artifact_paths": {
            "professional_gate": str(PROFESSIONAL_GATE_PATH),
            "repo_manifest": str(REPO_MANIFEST_PATH),
            "environment": str(ENV_PATH),
            "model_data_manifest": str(MODEL_DATA_PATH),
            "dag_iter_03": str(PAPER_RUN / "paper_author_gap_dag_iter_03.json"),
        },
    }
    write_json(VERIFIER_PATH, payload)
    write_json(PAPER_RUN / "verifier_result_iter_03.json", payload)
    return payload


def update_queue_summary(gate: dict[str, Any], verify: dict[str, Any], dag: dict[str, Any]) -> None:
    queue_obj = read_json(QUEUE_PATH)
    for item in queue_obj.get("queue", []):
        if item.get("paper_id") == PAPER_ID:
            item["priority"] = "high"
            item["professional_blocker"] = gate["status"]
            item["repo_exact_rerun_status"] = "repo_present_needs_fmri_data_api_runtime_hardware_and_full_metric_grid"
            item["repo_paths"] = [str(REPO)]
            item["specialized_runner_status"] = gate["status"]
            item["specialized_runner_artifact_dir"] = str(RUNNER_DIR)
            statuses = item.setdefault("implementation_statuses", [])
            for status in ["official_repo_discovered_and_encoded", "official_scripts_parsed", "source_compile_checked", "professional_gate_blocked"]:
                if status not in statuses:
                    statuses.append(status)
            item["specialized_runner_evidence"] = {
                "blockers": gate["blockers"],
                "verifier_path": str(VERIFIER_PATH),
                "repo_manifest_path": str(REPO_MANIFEST_PATH),
                "environment_path": str(ENV_PATH),
                "model_data_manifest_path": str(MODEL_DATA_PATH),
            }
            break
    write_json(QUEUE_PATH, queue_obj)

    summary = read_json(SUMMARY_PATH)
    for paper in summary.get("papers", []):
        if paper.get("paper_id") == PAPER_ID:
            paper["final_status"] = "blocked_waiting_for_professional_artifacts_after_dag_update"
            paper["converged"] = False
            paper["repo_paths"] = [str(REPO)]
            paper["specialized_runner_status"] = gate["status"]
            paper["professional_blocker"] = gate["status"]
            paper["specialized_runner_artifact_dir"] = str(RUNNER_DIR)
            statuses = paper.setdefault("implementation_statuses", [])
            for status in ["official_repo_discovered_and_encoded", "official_scripts_parsed", "source_compile_checked", "professional_gate_blocked"]:
                if status not in statuses:
                    statuses.append(status)
            paper["iterations"] = paper.get("iterations", []) + [
                {
                    "iteration": 3,
                    "dag_signature": dag.get("signature"),
                    "simulation": {
                        "paper_id": PAPER_ID,
                        "paper_title": TITLE,
                        "created_at_utc": gate["created_at_utc"],
                        "input_contract": dag.get("blind_contract", {}),
                        "paper_text_seen": False,
                        "previous_memory_seen": False,
                        "oracle_results_seen": False,
                        "repo_paths": [str(REPO)],
                        "author_decision": "explicit_operational_blocker",
                        "professional_package_ready": gate["professional_package_ready"],
                        "professional_package_reason": gate["status"],
                        "reduced_or_proxy_used_for_convergence": False,
                        "raw_artifact_level": "repo_script_dependency_device_preflight_only",
                        "blocker_ids": [b["id"] for b in gate["blockers"]],
                    },
                    "verification": verify,
                }
            ]
            break
    summary["updated_at_utc"] = utc_now()
    summary["final_status"] = "running_professional_two_loop_not_converged"
    write_json(SUMMARY_PATH, summary)


def write_status(gate: dict[str, Any], verify: dict[str, Any], dag: dict[str, Any]) -> None:
    lines = [
        "# PRISM Specialized Professional Gate",
        "",
        f"- Paper id: `{PAPER_ID}`",
        f"- Title: {TITLE}",
        f"- Status: `{gate['status']}`",
        "- Converged: `false`",
        f"- Professional ready: `{str(gate['professional_package_ready']).lower()}`",
        f"- DAG signature: `{dag.get('signature')}`",
        f"- Repo: `{REPO}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in gate["blockers"]:
        lines.append(f"- `{blocker['id']}`: {blocker['detail']}")
    lines += [
        "",
        "## Artifact Paths",
        "",
        f"- Professional gate: `{PROFESSIONAL_GATE_PATH}`",
        f"- Verifier: `{VERIFIER_PATH}`",
        f"- Repo manifest: `{REPO_MANIFEST_PATH}`",
        f"- Environment: `{ENV_PATH}`",
        f"- Model/data manifest: `{MODEL_DATA_PATH}`",
        "",
        "## Verifier Checks",
        "",
    ]
    for check in verify["checks"]:
        lines.append(f"- `{check['name']}`: `{check['status']}`")
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    manifest = repo_manifest()
    env = environment_manifest()
    model_data = model_data_manifest()
    gate = professional_gate(manifest, env, model_data)
    dag = update_dag(gate)
    verify = verifier(gate, dag)
    update_queue_summary(gate, verify, dag)
    write_status(gate, verify, dag)
    refresh = run_cmd([sys.executable, str(RUN_ROOT / "refresh_longgoal_status.py")], cwd=RUN_ROOT, timeout=120)
    print(
        json.dumps(
            {
                "paper_id": PAPER_ID,
                "status": gate["status"],
                "converged": False,
                "professional_ready": gate["professional_package_ready"],
                "blocker_count": len(gate["blockers"]),
                "blocker_ids": [b["id"] for b in gate["blockers"]],
                "dag_signature": dag.get("signature"),
                "status_path": str(STATUS_PATH),
                "verifier_path": str(VERIFIER_PATH),
                "refresh_returncode": refresh["returncode"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
