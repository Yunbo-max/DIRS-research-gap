#!/usr/bin/env python3
"""InfoTok professional operational gate for the strict DIRS loop.

InfoTok has released code, scripts, configs, default videos, and public
Hugging Face model/data repositories. This gate records those concrete author
simulation steps while refusing to converge from script parsing, README tables,
or default-video demos. Full convergence requires paper-shaped TokenBench/DAVIS
reconstruction, metric, latency, decoder-pass, ablation, and hardware artifacts.
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
PAPER_RUN = RUN_ROOT / "paper_runs" / "iclr2026_jeywpfgzvn_infotok_adaptive_video_tokenizer"
RUNNER_DIR = RUN_ROOT / "specialized_runners" / "infotok"
REPO = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/InfoTok")

PAPER_ID = "ICLR2026_JEYWpFGzvn_infotok_adaptive_video_tokenizer"
TITLE = "InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression"

DAG_PATH = PAPER_RUN / "paper_author_gap_dag.json"
QUEUE_PATH = RUN_ROOT / "specialized_runner_queue.json"
SUMMARY_PATH = RUN_ROOT / "remaining19_strict_dirs_summary.json"
LONG_STATUS_PATH = RUN_ROOT / "LONGGOAL_STATUS.md"
SPECIALIZED_QUEUE_MD = RUN_ROOT / "SPECIALIZED_RUNNER_QUEUE.md"

STATUS_PATH = RUNNER_DIR / "INFOTOK_SPECIALIZED_STATUS.md"
VERIFIER_PATH = RUNNER_DIR / "infotok_specialized_verifier.json"
PROFESSIONAL_GATE_PATH = RUNNER_DIR / "professional_gate_result.json"
SCRIPT_MANIFEST_PATH = RUNNER_DIR / "official_script_manifest.json"
ENV_PATH = RUNNER_DIR / "environment.json"
MODEL_DATA_PATH = RUNNER_DIR / "model_data_manifest.json"

SCRIPT_FILES = [
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "infotok.yaml",
    "exp_scripts/infotok_inference.sh",
    "exp_scripts/infotok_posttrain.sh",
    "cosmos_predict1/tokenizer/inference/video_cli.py",
    "cosmos_predict1/tokenizer/inference/video_cli_compare.py",
    "cosmos_predict1/tokenizer/inference/video_lib.py",
    "cosmos_predict1/tokenizer/networks/ours_discrete_video.py",
    "cosmos_predict1/tokenizer/training/train.py",
    "cosmos_predict1/tokenizer/training/metrics.py",
    "cosmos_predict1/tokenizer/training/configs/config.py",
]

HF_ARTIFACTS = [
    {"id": "infotok_flex_checkpoint", "repo_id": "qyoo/infotok-flex", "repo_type": "model", "local_hint": "infotok-flex/infotok_mse.pt"},
    {"id": "tokenbench_240p", "repo_id": "qyoo/tokenbench_240p", "repo_type": "dataset", "local_hint": "tokenbench_240p"},
    {"id": "davis_240p", "repo_id": "qyoo/davis_240p", "repo_type": "dataset", "local_hint": "davis_240p"},
]

EXPECTED_OUTPUT_SURFACES = [
    "TokenBench and DAVIS reconstruction at matched BPP16/token-rate",
    "PSNR, SSIM, LPIPS, FVD, BPP16, latency seconds per video, decoder-pass counts",
    "global_elbo versus elbo routing comparison",
    "fixed Cosmos-DV and ElasticTok/OmniTokenizer/Open-MAGVIT2 baseline comparison",
    "router optimality ablation",
    "compressor ablation",
    "post-training/full-training trace for 32-H100 scale claim",
    "raw reconstructed videos, masks/token bars, metric JSON, timing/GPU traces, and table summaries",
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


def run_cmd(cmd: list[str], *, cwd: Path | None = None, timeout: int = 120) -> dict[str, Any]:
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


def path_size(path: Path) -> tuple[str | None, int]:
    if not path.exists():
        return None, 0
    if path.is_file():
        return str(path.stat().st_size), 1
    size = run_cmd(["du", "-sh", str(path)], timeout=20)["stdout"].split()
    count = run_cmd(["bash", "-lc", f"find {str(path)!r} -type f | wc -l"], timeout=30)
    try:
        file_count = int(count["stdout"].strip())
    except ValueError:
        file_count = 0
    return (size[0] if size else None), file_count


def import_probe(module_name: str) -> dict[str, Any]:
    return run_cmd([sys.executable, "-c", f"import importlib; m=importlib.import_module({module_name!r}); print(getattr(m, '__version__', 'imported'))"], timeout=45)


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


def parse_readme_table(text: str) -> list[dict[str, str]]:
    rows = []
    for line in text.splitlines():
        if line.startswith("| TokenBench") or line.startswith("| DAVIS"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 5:
                rows.append({"dataset": cells[0], "token_rate": cells[1], "temporal_window": cells[2], "psnr": cells[3], "ssim": cells[4]})
    return rows


def script_manifest() -> dict[str, Any]:
    files = []
    for rel in SCRIPT_FILES:
        path = REPO / rel
        row: dict[str, Any] = {"relative_path": rel, "path": str(path), "exists": path.exists()}
        if path.exists() and path.is_file():
            row["size_bytes"] = path.stat().st_size
            text = read_text(path)
            row["line_count"] = len(text.splitlines())
            if path.suffix == ".sh":
                row["shell_syntax"] = run_cmd(["bash", "-n", str(path)], timeout=30)
                row["commands"] = [line.strip() for line in text.splitlines() if "python" in line or "torch.distributed.run" in line]
            elif path.suffix == ".py":
                row["py_compile"] = run_cmd([sys.executable, "-m", "py_compile", str(path)], timeout=45)
                row["imports"] = sorted(set(re.findall(r"^(?:from|import)\s+([A-Za-z0-9_\.]+)", text, re.M)))
                row["cli_flags"] = sorted(set(re.findall(r"['\"](--[A-Za-z0-9_-]+)['\"]", text)))
            elif path.name == "README.md":
                row["urls"] = sorted(set(re.findall(r"https?://[^)\s]+", text)))
                row["expected_result_table"] = parse_readme_table(text)
                row["mentions_checkpoint"] = "infotok_mse.pt" in text
                row["mentions_tokenbench"] = "tokenbench_240p" in text
                row["mentions_davis"] = "davis_240p" in text
            elif path.name in {"requirements.txt", "infotok.yaml", "pyproject.toml"}:
                row["head"] = "\n".join(text.splitlines()[:80])
        files.append(row)
    payload = {
        "artifact_kind": "infotok_official_script_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "repo": str(REPO),
        "files": files,
        "default_videos": sorted(str(p) for p in (REPO / "default_videos").glob("*.mp4")) if (REPO / "default_videos").exists() else [],
        "support_only_findings": [
            "default_videos and README expected-result rows are smoke/support only",
            "infotok_posttrain.sh contains single-GPU debug mode; debug mode cannot converge the paper",
            "full paper result requires TokenBench/DAVIS full evaluation plus baselines/ablations/latency/decoder-pass traces",
        ],
    }
    write_json(SCRIPT_MANIFEST_PATH, payload)
    return payload


def hf_info(repo_id: str, repo_type: str) -> dict[str, Any]:
    code = (
        "from huggingface_hub import HfApi; import json; "
        f"info=HfApi().repo_info({repo_id!r}, repo_type={repo_type!r}, files_metadata=True); "
        "files=getattr(info,'siblings',[]) or []; "
        "print(json.dumps({'repo_id':info.id, 'files_first50':[getattr(f,'rfilename','') for f in files[:50]], "
        "'total_size_bytes':sum((getattr(f,'size',0) or 0) for f in files), "
        "'file_count':len(files)}))"
    )
    result = run_cmd([sys.executable, "-c", code], timeout=60)
    parsed = None
    if result["returncode"] == 0:
        try:
            parsed = json.loads(result["stdout"].strip())
        except json.JSONDecodeError:
            parsed = None
    return {"probe": result, "parsed": parsed}


def model_data_manifest() -> dict[str, Any]:
    artifacts = []
    for item in HF_ARTIFACTS:
        local_path = REPO / item["local_hint"]
        size_human, file_count = path_size(local_path)
        artifacts.append(
            {
                **item,
                "local_path": str(local_path),
                "materialized_locally": local_path.exists() and (local_path.is_file() or file_count > 10),
                "local_size_human": size_human,
                "local_file_count": file_count,
                "hf_info": hf_info(item["repo_id"], item["repo_type"]),
            }
        )
    metric_output_candidates = []
    support_media_candidates = []
    for root in [REPO / "outputs", RUNNER_DIR]:
        if not root.exists():
            continue
        for pattern in ["**/*psnr*.json", "**/*ssim*.json", "**/*lpips*.json", "**/*fvd*.json", "**/*metric*.json", "**/*result*.json", "**/*.csv"]:
            for candidate in list(root.glob(pattern))[:80]:
                size_human, file_count = path_size(candidate)
                metric_output_candidates.append({"path": str(candidate), "is_dir": candidate.is_dir(), "size_human": size_human, "file_count": file_count})
    for root in [REPO / "default_videos", REPO / "assets", REPO / "outputs"]:
        if not root.exists():
            continue
        for pattern in ["**/*.mp4", "**/*.gif", "**/*.png"]:
            for candidate in list(root.glob(pattern))[:80]:
                size_human, file_count = path_size(candidate)
                support_media_candidates.append({"path": str(candidate), "is_dir": candidate.is_dir(), "size_human": size_human, "file_count": file_count})
    payload = {
        "artifact_kind": "infotok_model_data_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "hf_artifacts": artifacts,
        "verifier_comparable_metric_outputs": metric_output_candidates[:120],
        "support_only_media_candidates": support_media_candidates[:120],
        "paper_shaped_outputs_required": EXPECTED_OUTPUT_SURFACES,
    }
    write_json(MODEL_DATA_PATH, payload)
    return payload


def environment_manifest() -> dict[str, Any]:
    packages = {
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "transformers": package_version("transformers"),
        "decord": package_version("decord"),
        "diffusers": package_version("diffusers"),
        "hydra-core": package_version("hydra-core"),
        "imageio": package_version("imageio"),
        "mediapy": package_version("mediapy"),
        "megatron-core": package_version("megatron-core"),
        "omegaconf": package_version("omegaconf"),
        "pynvml": package_version("pynvml"),
        "scikit-image": package_version("scikit-image"),
        "termcolor": package_version("termcolor"),
        "apex": package_version("apex"),
        "transformer-engine": package_version("transformer-engine"),
    }
    payload = {
        "artifact_kind": "infotok_environment_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "python": sys.version,
        "gpu_rows": gpu_rows(),
        "packages": packages,
        "import_probes": {
            "torch": import_probe("torch"),
            "torchvision": import_probe("torchvision"),
            "cosmos_predict1.tokenizer.inference.video_cli": import_probe("cosmos_predict1.tokenizer.inference.video_cli"),
            "decord": import_probe("decord"),
            "diffusers": import_probe("diffusers"),
            "hydra": import_probe("hydra"),
            "imageio": import_probe("imageio"),
            "mediapy": import_probe("mediapy"),
            "megatron": import_probe("megatron"),
            "omegaconf": import_probe("omegaconf"),
            "pynvml": import_probe("pynvml"),
            "skimage": import_probe("skimage"),
            "termcolor": import_probe("termcolor"),
            "apex": import_probe("apex"),
            "transformer_engine": import_probe("transformer_engine"),
            "token_bench": import_probe("token_bench"),
        },
        "compileall_tokenizer": run_cmd(
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                str(REPO / "cosmos_predict1/tokenizer"),
                str(REPO / "cosmos_predict1/utils"),
                str(REPO / "cosmos_predict1/checkpointer"),
            ],
            timeout=180,
        ),
        "professional_hardware_expected_by_dag": [
            "H100-80GB or A100-80GB recommended for inference",
            "32 H100 GPUs reported for paper-scale training",
            "8-GPU post-training path in script for non-debug training",
        ],
    }
    write_json(ENV_PATH, payload)
    return payload


def professional_gate(manifest: dict[str, Any], env: dict[str, Any], model_data: dict[str, Any]) -> dict[str, Any]:
    blockers = []
    gpu = env["gpu_rows"]
    gpu_names = " | ".join(row["name"] for row in gpu)
    h100_count = sum("H100" in row["name"] for row in gpu)
    has_80gb = any(row["memory_total_mib"] >= 75000 and ("H100" in row["name"] or "A100" in row["name"]) for row in gpu)
    clean_large_gpu = any(row["memory_free_mib"] >= 30000 and row["utilization_gpu_pct"] < 30 for row in gpu)
    if not has_80gb:
        blockers.append({"id": "h100_or_a100_80gb_runtime_missing", "status": "blocked", "detail": f"README recommends H100-80GB/A100-80GB; visible GPUs are {gpu_names}."})
    if h100_count < 32:
        blockers.append({"id": "paper_scale_32_h100_training_missing", "status": "blocked", "detail": f"DAG records 32 H100 GPUs for paper-scale training; visible H100 count is {h100_count}."})
    if not clean_large_gpu:
        blockers.append({"id": "clean_large_gpu_slot_missing", "status": "blocked", "detail": f"No clean >=30GB free GPU is visible for official long-window inference. GPU rows: {gpu}."})

    missing_runtime = [
        name for name, result in env["import_probes"].items()
        if result["returncode"] != 0 and name in {"diffusers", "hydra", "imageio", "mediapy", "megatron", "omegaconf", "pynvml", "skimage", "termcolor", "apex", "transformer_engine", "token_bench"}
    ]
    if missing_runtime:
        blockers.append({"id": "infotok_runtime_dependencies_missing", "status": "blocked", "detail": "Missing imports for full inference/training/evaluation path: " + ", ".join(missing_runtime)})

    local_missing = [a["id"] for a in model_data["hf_artifacts"] if not a["materialized_locally"]]
    if local_missing:
        blockers.append({"id": "infotok_hf_artifacts_not_materialized", "status": "blocked", "detail": "Public HF artifacts are reachable but not locally materialized: " + ", ".join(local_missing)})

    output_count = len(model_data["verifier_comparable_metric_outputs"])
    if output_count < 10:
        blockers.append({"id": "full_reconstruction_metric_grid_missing", "status": "blocked", "detail": "No complete TokenBench/DAVIS metric JSON, FVD/LPIPS, latency, decoder-pass, ablation, and baseline table outputs were found."})

    gate = {
        "artifact_kind": "infotok_professional_gate_result",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "status": "ready_for_full_infotok_grid_execution_not_converged" if not blockers else "blocked_by_hf_artifacts_runtime_hardware_training_scale_and_result_grid_requirements",
        "professional_package_ready": not blockers,
        "convergence_role": "professional operational gate; no default-video or README-table evidence is promoted",
        "blockers": blockers,
        "support_checks": {
            "repo_discovered_and_encoded": REPO.exists(),
            "script_files_checked": len(manifest["files"]),
            "inference_script_syntax_ok": next((f.get("shell_syntax", {}).get("returncode") == 0 for f in manifest["files"] if f["relative_path"] == "exp_scripts/infotok_inference.sh"), False),
            "posttrain_script_syntax_ok": next((f.get("shell_syntax", {}).get("returncode") == 0 for f in manifest["files"] if f["relative_path"] == "exp_scripts/infotok_posttrain.sh"), False),
            "compileall_tokenizer_passed": env["compileall_tokenizer"]["returncode"] == 0,
            "hf_artifacts_checked": len(model_data["hf_artifacts"]),
        },
        "next_full_execution_if_unblocked": [
            "materialize qyoo/infotok-flex, qyoo/tokenbench_240p, and qyoo/davis_240p",
            "install exact InfoTok/Cosmos runtime plus TokenBench metric repo",
            "run exp_scripts/infotok_inference.sh only as smoke, not convergence",
            "run full TokenBench/DAVIS reconstruction with avg_rate 0.75 and 0.5 plus global_elbo/elbo strategy variants",
            "compute PSNR, SSIM, LPIPS, FVD, BPP16, latency seconds/video, and decoder-pass counts",
            "run router/compressor ablations and baseline comparisons where released code/data permit",
            "emit raw videos, mask visualizations, metric JSON, timing/GPU traces, and compare to paper tables/figures/paragraphs",
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
                "repos=/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/InfoTok; "
                "code_artifacts=infotok_mse.pt checkpoint; exp_scripts/infotok_inference.sh; "
                "exp_scripts/infotok_posttrain.sh; cosmos_predict1/tokenizer/inference/video_cli.py; "
                "cosmos_predict1/tokenizer/training/train.py; infotok.yaml; requirements.txt"
            )
    nodes = [
        ("ops.infotok_repo_script_gate", "Resolve InfoTok repo and parse inference/post-training scripts, tokenizer config, runtime requirements, and default-video support artifacts.", "operational_dependency"),
        ("ops.infotok_hf_materialization_gate", "Materialize qyoo/infotok-flex checkpoint plus qyoo/tokenbench_240p and qyoo/davis_240p; HF reachability alone is not convergence.", "operational_dependency"),
        ("ops.infotok_runtime_metric_gate", "Install exact Cosmos/InfoTok runtime and TokenBench metric stack for PSNR, SSIM, LPIPS, FVD, BPP16, latency, and decoder-pass scoring.", "operational_dependency"),
        ("ops.infotok_hardware_scale_gate", "Require H100-80GB/A100-80GB inference readiness and record 32-H100 paper-scale training as non-reproducible unless actual traces exist.", "systems_measurement"),
        ("ops.infotok_full_reconstruction_grid", "Run TokenBench/DAVIS avg_rate 0.75/0.5, elbo/global_elbo, baselines, router/compressor ablations, raw videos, masks, metric JSON, timing and GPU traces.", "operational_execution"),
        ("decision.explicit_blocker_after_infotok_preflight", "If checkpoint, dataset, runtime, hardware, baseline/ablation, or result-grid artifacts are missing, block and feed exact requirements back into Loop 1.", "author_reviewer_decision"),
    ]
    for node_id, content, typ in nodes:
        ensure_node(dag, {"id": node_id, "content": content, "type": typ, "skill_role": "paper-specific operational gate"})
    for source, target in [
        ("ops.resolve_repo_code", "ops.infotok_repo_script_gate"),
        ("ops.infotok_repo_script_gate", "ops.infotok_hf_materialization_gate"),
        ("ops.infotok_hf_materialization_gate", "ops.infotok_runtime_metric_gate"),
        ("ops.infotok_runtime_metric_gate", "ops.infotok_hardware_scale_gate"),
        ("ops.infotok_hardware_scale_gate", "ops.infotok_full_reconstruction_grid"),
        ("ops.infotok_full_reconstruction_grid", "reviewer.require_professional_artifact_package"),
        ("ops.infotok_full_reconstruction_grid", "reviewer.compare_result_shapes"),
        ("reviewer.keep_exact_artifact_debt", "decision.explicit_blocker_after_infotok_preflight"),
    ]:
        ensure_edge(dag, source, target)
    dag.setdefault("previous_loop_updates", []).append(
        {
            "iteration": 3,
            "created_at_utc": utc_now(),
            "source": "infotok_specialized_professional_gate",
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
        {"name": "reduced_proxy_rejection_gate", "status": "pass", "detail": "default videos, README expected rows, script syntax, and HF reachability are support only"},
        {"name": "professional_artifact_package", "status": "pass" if gate["professional_package_ready"] else "blocked", "detail": gate["blockers"]},
        {"name": "result_shape_comparison_ready", "status": "blocked", "detail": "requires full TokenBench/DAVIS/baseline/ablation metric outputs before comparing to paper evidence channels"},
    ]
    payload = {
        "artifact_kind": "infotok_specialized_verifier",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "iteration": 3,
        "converged": False,
        "professional_ready": gate["professional_package_ready"],
        "checks": checks,
        "required_updates": [
            {
                "id": "update.infotok_released_code_to_operational_gates",
                "reason": gate["status"],
                "success_criteria": [
                    "repo path encoded",
                    "HF checkpoint/dataset materialization gate present",
                    "runtime/metric stack gate present",
                    "H100/A100/32-H100 scale gate present",
                    "full reconstruction/baseline/ablation grid gate present",
                ],
            }
        ],
        "artifact_paths": {
            "professional_gate": str(PROFESSIONAL_GATE_PATH),
            "script_manifest": str(SCRIPT_MANIFEST_PATH),
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
            item["repo_exact_rerun_status"] = "released_code_needs_hf_artifacts_runtime_hardware_and_full_grid"
            item["repo_paths"] = [str(REPO)]
            item["specialized_runner_status"] = gate["status"]
            item["specialized_runner_artifact_dir"] = str(RUNNER_DIR)
            statuses = item.setdefault("implementation_statuses", [])
            for status in ["official_repo_discovered_and_encoded", "official_scripts_parsed", "hf_manifests_reachable", "professional_gate_blocked"]:
                if status not in statuses:
                    statuses.append(status)
            item["specialized_runner_evidence"] = {
                "blockers": gate["blockers"],
                "verifier_path": str(VERIFIER_PATH),
                "script_manifest_path": str(SCRIPT_MANIFEST_PATH),
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
            for status in ["official_repo_discovered_and_encoded", "official_scripts_parsed", "hf_manifests_reachable", "professional_gate_blocked"]:
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
                        "raw_artifact_level": "repo_script_hf_manifest_dependency_device_preflight_only",
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
        "# InfoTok Specialized Professional Gate",
        "",
        f"- Paper id: `{PAPER_ID}`",
        f"- Title: {TITLE}",
        f"- Status: `{gate['status']}`",
        f"- Converged: `false`",
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
        f"- Script manifest: `{SCRIPT_MANIFEST_PATH}`",
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


def refresh_global_status() -> None:
    summary = read_json(SUMMARY_PATH)
    papers = summary.get("papers", [])
    accepted = sum(1 for p in papers if p.get("converged"))
    blocked = sum(1 for p in papers if not p.get("converged"))
    visible = [p for p in papers if p.get("specialized_runner_status") or p.get("repo_paths") or p.get("final_status")]
    lines = [
        "# Remaining 19 Strict DIRS Long Goal Status",
        "",
        f"- Updated: `{utc_now()}`",
        f"- Final status: `{summary.get('final_status')}`",
        f"- Accepted/converged papers: `{accepted}`",
        f"- Not yet converged papers: `{blocked}`",
        "- Policy: no reduced/small/proxy/syntax-only evidence can converge a paper.",
        "",
        "## Active / Specialized Runs",
        "",
    ]
    for paper in visible[:24]:
        status = paper.get("specialized_runner_status") or paper.get("professional_blocker") or paper.get("final_status") or "unknown"
        lines.append(f"- `{paper.get('paper_id')}`: `{status}` repo_paths={paper.get('repo_paths', [])}")
    prophet = RUN_ROOT / "specialized_runners/prophet/custom_full_gsm8k_llada8b/status.json"
    if prophet.exists():
        ps = read_json(prophet)
        lines += [
            "",
            "## Prophet Live GPU Run",
            "",
            f"- Status: `{ps.get('status')}`",
            f"- Samples: `{ps.get('completed_sample_indices')}/{ps.get('total_samples')}`",
            f"- GPU: `{ps.get('cuda_visible_devices')}`",
            f"- Updated: `{ps.get('updated_at_utc')}`",
        ]
    LONG_STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    queue = read_json(QUEUE_PATH)
    qlines = ["# Specialized Runner Queue", ""]
    for item in queue.get("queue", []):
        qlines.append(
            f"- `{item.get('paper_id')}` | priority=`{item.get('priority')}` | "
            f"status=`{item.get('specialized_runner_status') or item.get('professional_blocker')}` | "
            f"runner=`{item.get('runner_type')}` | repos={item.get('repo_paths', [])}"
        )
    SPECIALIZED_QUEUE_MD.write_text("\n".join(qlines) + "\n", encoding="utf-8")


def main() -> None:
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    manifest = script_manifest()
    env = environment_manifest()
    model_data = model_data_manifest()
    gate = professional_gate(manifest, env, model_data)
    dag = update_dag(gate)
    verify = verifier(gate, dag)
    update_queue_summary(gate, verify, dag)
    write_status(gate, verify, dag)
    refresh_global_status()
    print(
        json.dumps(
            {
                "paper_id": PAPER_ID,
                "status": gate["status"],
                "blocker_count": len(gate["blockers"]),
                "dag_signature": dag.get("signature"),
                "status_path": str(STATUS_PATH),
                "verifier_path": str(VERIFIER_PATH),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
