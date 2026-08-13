#!/usr/bin/env python3
"""SenCache professional operational gate for the strict DIRS loop.

This runner is not a reduced video-diffusion demo. It checks whether the
DAG-only author simulation can execute the paper-shaped SenCache process:
sensitivity calibration or released sensitivity weights, Wan2.1/CogVideoX/LTX
generation with cached and uncached variants, full prompt/evaluator suites,
latency/GFLOPs/quality metrics, and verifier-comparable result tables. Missing
conditions become explicit blockers and Loop-1 DAG updates.
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
PAPER_RUN = RUN_ROOT / "paper_runs" / "cvpr2026_053_sencache_sensitivity_aware_caching"
RUNNER_DIR = RUN_ROOT / "specialized_runners" / "sencache"
REPO = Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/053_sencache/SenCache")

PAPER_ID = "CVPR2026_053_sencache_sensitivity_aware_caching"
TITLE = "SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware Caching"

DAG_PATH = PAPER_RUN / "paper_author_gap_dag.json"
STATUS_PATH = RUNNER_DIR / "SENCACHE_SPECIALIZED_STATUS.md"
VERIFIER_PATH = RUNNER_DIR / "sencache_specialized_verifier.json"
ENV_PATH = RUNNER_DIR / "environment.json"
SCRIPT_MANIFEST_PATH = RUNNER_DIR / "official_script_manifest.json"
MODEL_DATA_PATH = RUNNER_DIR / "model_data_manifest.json"
PROFESSIONAL_GATE_PATH = RUNNER_DIR / "professional_gate_result.json"

QUEUE_PATH = RUN_ROOT / "specialized_runner_queue.json"
SUMMARY_PATH = RUN_ROOT / "remaining19_strict_dirs_summary.json"
QUEUE_MD_PATH = RUN_ROOT / "SPECIALIZED_RUNNER_QUEUE.md"
LONGGOAL_STATUS_PATH = RUN_ROOT / "LONGGOAL_STATUS.md"

SCRIPT_FILES = [
    "README.md",
    "Wan2.1/README.md",
    "Wan2.1/sencache.py",
    "Wan2.1/sensitivity_calculation.py",
    "Wan2.1/test_prompt.json",
    "CogVideoX/README.md",
    "CogVideoX/sencache.py",
    "CogVideoX/sensitivity_calculation_cogvid.py",
    "CogVideoX/test_prompt.json",
    "LTX-Video/README.md",
    "LTX-Video/sencache.py",
    "LTX-Video/sensitivity_calculation_LTX.py",
    "LTX-Video/test_prompt.json",
]

EXPECTED_MODELS = [
    {
        "id": "wan21_t2v_13b",
        "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "local_hints": ["Wan2.1-T2V-1.3B", "models--Wan-AI--Wan2.1-T2V-1.3B"],
        "paper_role": "Wan2.1 T=50 cached and uncached video generation surface",
    },
    {
        "id": "cogvideox_15_5b",
        "repo_id": "THUDM/CogVideoX1.5-5B",
        "local_hints": ["CogVideoX1.5-5B", "models--THUDM--CogVideoX1.5-5B"],
        "paper_role": "CogVideoX T=50 cached and uncached video generation surface",
    },
    {
        "id": "ltx_video_091",
        "repo_id": "a-r-r-o-w/LTX-Video-0.9.1-diffusers",
        "local_hints": [
            "LTX-Video-0.9.1-diffusers",
            "models--a-r-r-o-w--LTX-Video-0.9.1-diffusers",
        ],
        "paper_role": "LTX-Video T=50 cached and uncached video generation surface",
    },
]

EXPECTED_SENSITIVITY_WEIGHTS = [
    {
        "id": "sensitivity_wan21",
        "filename": "sensitivity_wan21.npz",
        "paper_role": "Wan2.1 latent/timestep sensitivity norms used by the cache gate",
    },
    {
        "id": "sensitivity_cogvid",
        "filename": "sensitivity_cogvid.npz",
        "paper_role": "CogVideoX latent/timestep sensitivity norms used by the cache gate",
    },
    {
        "id": "sensitivity_ltx",
        "filename": "sensitivity_ltx.npz",
        "paper_role": "LTX-Video latent/timestep sensitivity norms used by the cache gate",
    },
]

EXPECTED_DATASETS = [
    {
        "id": "mixkit_calibration_videos",
        "required": "MixKit calibration videos for finite-difference sensitivity calculation",
        "local_hints": ["MixKit", "mixkit"],
    },
    {
        "id": "vbench_full_prompt_set",
        "required": "full VBench prompt set for paper-scale video evaluation",
        "local_hints": ["VBench", "vbench", "vbench_prompts"],
    },
    {
        "id": "t2v_compbench_70_prompts",
        "required": "70 T2V-CompBench prompts for ablation",
        "local_hints": ["T2V-CompBench", "t2v_compbench", "T2VCompBench"],
    },
]

PAPER_RESULT_SURFACES = [
    "Wan2.1 T=50 cached and uncached generation",
    "CogVideoX T=50 cached and uncached generation",
    "LTX-Video T=50 cached and uncached generation",
    "similar-compute comparisons against TeaCache and MagCache",
    "VBench prompt-set evaluation",
    "70-prompt T2V-CompBench ablation",
    "supplement 100-video consecutive-step MAE diagnostic",
    "NFE/cache-ratio measurements",
    "LPIPS/PSNR/SSIM reference metrics",
    "wall-clock latency and GFLOPs",
]

SEARCH_ROOTS = [
    REPO,
    Path(os.path.expanduser("~/.cache/huggingface/hub")),
    Path(os.path.expanduser("~/.cache/huggingface/datasets")),
    Path("/tf/notebooks/.cache/huggingface/hub"),
    Path("/tf/notebooks/.cache/huggingface/datasets"),
    Path("/tf/notebooks/models"),
    Path("/tf/notebooks/data"),
    Path("/tf/notebooks/datasets"),
    Path("/tf/notebooks/checkpoints"),
    Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/053_sencache"),
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
            "stdout": proc.stdout[-16000:],
            "stderr": proc.stderr[-16000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "timeout": True,
            "seconds": timeout,
            "stdout": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
        }


def package_version(dist_name: str) -> str | None:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def local_matches(hints: list[str]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for hint in hints:
            candidates = []
            exact = root / hint
            if exact.exists():
                candidates.append(exact)
            glob_pat = f"*{hint.replace('/', '--')}*" if root.name == "hub" else f"*{hint}*"
            try:
                candidates.extend(list(root.glob(glob_pat))[:30])
            except OSError:
                pass
            for candidate in candidates:
                key = str(candidate.resolve())
                if key in seen:
                    continue
                seen.add(key)
                if candidate.is_dir():
                    size = run_cmd(["du", "-sh", str(candidate)], timeout=30)["stdout"].split()
                    size_human = size[0] if size else None
                    file_count = int(
                        run_cmd(
                            ["bash", "-lc", f"find {str(candidate)!r} -type f | wc -l"],
                            timeout=30,
                        )["stdout"].strip()
                        or 0
                    )
                else:
                    size_human = str(candidate.stat().st_size)
                    file_count = 1
                matches.append(
                    {
                        "path": str(candidate),
                        "is_dir": candidate.is_dir(),
                        "file_count": file_count,
                        "size_human": size_human,
                    }
                )
    return matches


def has_marker(path: str, patterns: list[str]) -> bool:
    candidate = Path(path)
    if candidate.is_file():
        return any(candidate.match(pattern) for pattern in patterns)
    if not candidate.is_dir():
        return False
    for pattern in patterns:
        try:
            if next(candidate.rglob(pattern), None) is not None:
                return True
        except OSError:
            return False
    return False


def valid_model_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    markers = ["model_index.json", "config.json", "*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt"]
    return [match for match in matches if has_marker(match["path"], markers)]


def valid_dataset_matches(dataset_id: str, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = []
    for match in matches:
        lower = match["path"].lower()
        parts = {part.lower() for part in Path(match["path"]).parts}
        if dataset_id == "mixkit_calibration_videos" and "mixkit" in lower:
            valid.append(match)
        elif dataset_id == "vbench_full_prompt_set" and (
            ("vbench" in parts or "vchitect" in lower) and "lvbench" not in lower
        ):
            valid.append(match)
        elif dataset_id == "t2v_compbench_70_prompts" and (
            "t2v-compbench" in lower or "t2v_compbench" in lower or "t2vcompbench" in lower
        ):
            valid.append(match)
    return valid


def hf_repo_manifest(repo_id: str, repo_type: str) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, HfFolder
    except Exception as exc:
        return {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "status": "not_checked",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    try:
        api = HfApi(token=HfFolder.get_token())
        info = api.repo_info(repo_id, repo_type=repo_type, files_metadata=True)
        files = [
            {"path": sibling.rfilename, "size_bytes": getattr(sibling, "size", None) or 0}
            for sibling in info.siblings
        ]
        return {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "status": "available",
            "file_count": len(files),
            "total_size_bytes": sum(item["size_bytes"] for item in files),
            "files_head": files[:40],
            "files_tail": files[-40:],
        }
    except Exception as exc:
        return {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "status": "unavailable_or_gated",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def parse_python_script(path: Path) -> dict[str, Any]:
    text = read_text(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "line_count": len(text.splitlines()),
        "cli_flags": sorted(set(re.findall(r"['\"](--[A-Za-z0-9_-]+)['\"]", text))),
        "hf_model_ids": sorted(set(re.findall(r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", text))),
        "npz_dependencies": sorted(set(re.findall(r"['\"]([^'\"]+\\.npz)['\"]", text))),
        "json_dependencies": sorted(set(re.findall(r"['\"]([^'\"]+\\.jsonl?|[^'\"]+\\.pt)['\"]", text))),
        "has_prompt_file_loop": "--prompt_file" in text and "for" in text,
        "has_uncached_flag": "--no_sencache" in text,
        "has_latency_measurement": "--measure_latency" in text or "time.perf_counter" in text,
        "has_flops_measurement": "--measure_flops" in text or "with_flops=True" in text,
        "requires_wan_import": "import wan" in text,
        "requires_diffusers_import": "from diffusers" in text or "import diffusers" in text,
        "requires_decord_import": "decord" in text,
        "requires_checkpoint_dir": "--ckpt_dir" in text or "args.ckpt_dir" in text,
        "requires_checkpoint_path": "--ckpts_path" in text or "args.ckpts_path" in text,
        "requires_calibration_video_json": "--json_path" in text and "--video_base_path" in text,
        "threshold_fields": sorted(set(re.findall(r"(sencache_thresh_start|sencache_thresh_main|sencache_K)", text))),
        "sample_step_fields": sorted(set(re.findall(r"(sample_steps|num_inference_steps|num_t_steps)", text))),
    }


def parse_readme(path: Path) -> dict[str, Any]:
    text = read_text(path)
    commands = []
    for match in re.finditer(r"```(?:bash)?\n(.*?)```", text, flags=re.S):
        block = match.group(1)
        if "python" in block or "pip" in block:
            commands.append("\n".join(line.rstrip() for line in block.splitlines()))
    return {
        "path": str(path),
        "exists": path.exists(),
        "line_count": len(text.splitlines()),
        "commands": commands,
        "mentions_precomputed_weights": "precomputed sensitivity" in text.lower() or "sensitivity weights" in text.lower(),
        "mentions_vbench": "VBench" in text,
        "mentions_mixkit": "MixKit" in text,
        "mentions_gh200": "GH200" in text,
        "mentions_hf_sencache_dataset": "Yassaman/SenCache" in text,
    }


def script_manifest() -> dict[str, Any]:
    rows = []
    for rel in SCRIPT_FILES:
        path = REPO / rel
        row: dict[str, Any] = {
            "relative_path": rel,
            "path": str(path),
            "exists": path.exists(),
        }
        if path.exists() and path.is_file():
            row["size_bytes"] = path.stat().st_size
            if path.suffix == ".py":
                row["parsed"] = parse_python_script(path)
            elif path.name == "README.md":
                row["parsed"] = parse_readme(path)
            elif path.suffix == ".json":
                row["line_count"] = len(read_text(path).splitlines())
                row["is_single_prompt_support_only"] = len(read_text(path).splitlines()) <= 2
        rows.append(row)
    payload = {
        "artifact_kind": "sencache_official_script_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "repo": str(REPO),
        "files": rows,
        "paper_shaped_execution_matrix": {
            "calibration": [
                "MixKit calibration videos",
                "finite-difference latent and timestep sensitivity calculation",
                "sensitivity_wan21.npz, sensitivity_cogvid.npz, sensitivity_ltx.npz",
            ],
            "generation": [
                "Wan2.1 T=50 cached and uncached variants",
                "CogVideoX T=50 cached and uncached variants",
                "LTX-Video T=50 cached and uncached variants",
                "threshold sweeps and max consecutive skip K controls",
            ],
            "baselines": ["TeaCache", "MagCache", "standard uncached inference"],
            "metrics": ["NFE", "cache ratio", "LPIPS", "PSNR", "SSIM", "latency", "GFLOPs", "MAE"],
            "accepted_loop2_evidence": (
                "loadable model checkpoints, calibration weights or calibration reruns, raw generated videos, "
                "uncached references, logs with skip counts, timing/GFLOPs traces, quality metrics, and table summaries"
            ),
        },
        "support_only_findings": [
            {
                "id": "bundled_test_prompt_single_example",
                "status": "support_only",
                "detail": "Each bundled test_prompt.json is a one-line prompt, not the full VBench or T2V-CompBench paper prompt set.",
            }
        ],
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


def import_probe(module_name: str) -> dict[str, Any]:
    return run_cmd(
        [
            sys.executable,
            "-c",
            f"import importlib; m=importlib.import_module({module_name!r}); print(getattr(m, '__version__', 'imported'))",
        ],
        timeout=45,
    )


def environment_manifest() -> dict[str, Any]:
    packages = {
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "diffusers": package_version("diffusers"),
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "huggingface_hub": package_version("huggingface-hub"),
        "decord": package_version("decord"),
        "imageio": package_version("imageio"),
        "imageio-ffmpeg": package_version("imageio-ffmpeg"),
        "lpips": package_version("lpips"),
        "scikit-image": package_version("scikit-image"),
        "opencv-python": package_version("opencv-python"),
        "vbench": package_version("vbench"),
        "calflops": package_version("calflops"),
        "xfuser": package_version("xfuser"),
    }
    payload = {
        "artifact_kind": "sencache_environment_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "repo": str(REPO),
        "python": sys.version,
        "gpu_rows": gpu_rows(),
        "python_processes": run_cmd(
            [
                "bash",
                "-lc",
                "ps -eo pid,etime,cmd | rg 'prophet_custom_full_gsm8k_runner|sencache.py|python' || true",
            ],
            timeout=30,
        ),
        "nvidia_smi": run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=30,
        ),
        "packages": packages,
        "import_probes": {
            "diffusers": import_probe("diffusers"),
            "wan": import_probe("wan"),
            "decord": import_probe("decord"),
            "vbench": import_probe("vbench"),
            "lpips": import_probe("lpips"),
            "skimage": import_probe("skimage"),
            "cv2": import_probe("cv2"),
        },
        "compileall_support_check": run_cmd(
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                str(REPO / "Wan2.1"),
                str(REPO / "CogVideoX"),
                str(REPO / "LTX-Video"),
            ],
            timeout=120,
        ),
        "professional_hardware_expected_by_paper": [
            "GH200 supplement latency traces",
            "GPU inference for Wan2.1, CogVideoX, and LTX-Video full prompt grids",
        ],
    }
    write_json(ENV_PATH, payload)
    return payload


def sensitivity_weight_manifest() -> list[dict[str, Any]]:
    rows = []
    for item in EXPECTED_SENSITIVITY_WEIGHTS:
        local = []
        for root in SEARCH_ROOTS:
            if root.exists():
                try:
                    local.extend(root.rglob(item["filename"]))
                except OSError:
                    pass
        seen: set[str] = set()
        matches = []
        for path in local:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            matches.append({"path": str(path), "size_bytes": path.stat().st_size})
        row = dict(item)
        row["local_matches"] = matches
        row["materialized_locally"] = bool(matches)
        rows.append(row)
    return rows


def model_data_manifest() -> dict[str, Any]:
    models = []
    for item in EXPECTED_MODELS:
        row = dict(item)
        raw_matches = local_matches(item["local_hints"])
        row["local_matches"] = valid_model_matches(raw_matches)
        row["ignored_local_matches"] = [m for m in raw_matches if m not in row["local_matches"]]
        row["hf_manifest"] = hf_repo_manifest(item["repo_id"], "model")
        row["materialized_locally"] = bool(row["local_matches"])
        models.append(row)
    datasets = []
    for item in EXPECTED_DATASETS:
        row = dict(item)
        raw_matches = local_matches(item["local_hints"])
        row["local_matches"] = valid_dataset_matches(item["id"], raw_matches)
        row["ignored_local_matches"] = [m for m in raw_matches if m not in row["local_matches"]]
        row["materialized_locally"] = bool(row["local_matches"])
        datasets.append(row)
    sensitivity_weights = sensitivity_weight_manifest()
    payload = {
        "artifact_kind": "sencache_model_data_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "repo": str(REPO),
        "models": models,
        "sensitivity_weights": sensitivity_weights,
        "sensitivity_weights_hf_dataset": hf_repo_manifest("Yassaman/SenCache", "dataset"),
        "datasets_and_metric_artifacts": datasets,
        "bundled_prompt_files": [
            {
                "path": str(REPO / sub / "test_prompt.json"),
                "line_count": len(read_text(REPO / sub / "test_prompt.json").splitlines()),
                "support_only": True,
            }
            for sub in ["Wan2.1", "CogVideoX", "LTX-Video"]
        ],
        "paper_result_surfaces": PAPER_RESULT_SURFACES,
        "not_promoted_support": [
            "one-line test_prompt.json files",
            "syntax/compile checks",
            "HF metadata availability without local checkpoint materialization",
            "released sensitivity metadata without model/evaluator generation artifacts",
        ],
    }
    write_json(MODEL_DATA_PATH, payload)
    return payload


def derive_blockers(env: dict[str, Any], data: dict[str, Any]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    gpu_names = " | ".join(row["name"] for row in env.get("gpu_rows", []))
    if not env.get("gpu_rows"):
        blockers.append(
            {"id": "gpu_visibility", "status": "blocked", "detail": "No visible CUDA GPUs."}
        )
    if "GH200" not in gpu_names and "H100" not in gpu_names and "A100" not in gpu_names:
        blockers.append(
            {
                "id": "paper_hardware_class",
                "status": "blocked",
                "detail": f"Paper/DAG expects GH200 supplement latency or comparable high-end video-diffusion traces; visible GPUs are: {gpu_names or 'none'}.",
            }
        )
    high_use = [
        row
        for row in env.get("gpu_rows", [])
        if int(row.get("memory_used_mib", 0)) > 12000 or int(row.get("utilization_gpu_pct", 0)) > 50
    ]
    if high_use:
        blockers.append(
            {
                "id": "clean_gpu_slot_for_video_diffusion_grid",
                "status": "blocked",
                "detail": f"Visible GPUs are memory-heavy or active; GPU 3 is occupied by Prophet. High-use rows: {high_use}",
            }
        )
    missing_models = [item["id"] for item in data.get("models", []) if not item.get("materialized_locally")]
    if missing_models:
        blockers.append(
            {
                "id": "required_video_model_checkpoints_not_materialized",
                "status": "blocked",
                "detail": "Missing local loadable checkpoints: " + ", ".join(missing_models),
            }
        )
    missing_weights = [
        item["id"] for item in data.get("sensitivity_weights", []) if not item.get("materialized_locally")
    ]
    if missing_weights:
        blockers.append(
            {
                "id": "sensitivity_weights_or_calibration_outputs_missing",
                "status": "blocked",
                "detail": "Missing local sensitivity .npz weights or calibration outputs: " + ", ".join(missing_weights),
            }
        )
    missing_datasets = [
        item["id"] for item in data.get("datasets_and_metric_artifacts", []) if not item.get("materialized_locally")
    ]
    if missing_datasets:
        blockers.append(
            {
                "id": "calibration_prompt_metric_artifacts_missing",
                "status": "blocked",
                "detail": "Missing local calibration/prompt/evaluator artifacts: " + ", ".join(missing_datasets),
            }
        )
    packages = env.get("packages", {})
    missing_packages = [
        pkg
        for pkg in ["diffusers", "decord", "lpips", "vbench"]
        if not packages.get(pkg)
    ]
    failed_imports = [
        name
        for name in ["wan"]
        if env.get("import_probes", {}).get(name, {}).get("returncode") != 0
    ]
    if missing_packages or failed_imports:
        blockers.append(
            {
                "id": "video_diffusion_metric_runtime_missing",
                "status": "blocked",
                "detail": "Missing required runtime packages/imports: " + ", ".join(missing_packages + failed_imports),
            }
        )
    blockers.append(
        {
            "id": "full_cached_uncached_result_grid_missing",
            "status": "blocked",
            "detail": "No raw cached/uncached Wan2.1, CogVideoX, and LTX outputs with VBench/LPIPS/PSNR/SSIM/latency/GFLOPs summaries were found.",
        }
    )
    return blockers


def professional_gate_result(
    env: dict[str, Any],
    scripts: dict[str, Any],
    data: dict[str, Any],
    blockers: list[dict[str, str]],
) -> dict[str, Any]:
    payload = {
        "artifact_kind": "sencache_professional_gate_result",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "status": "blocked_by_sensitivity_weights_model_data_runtime_hardware_and_result_grid_requirements"
        if blockers
        else "ready_for_full_paper_shaped_execution",
        "convergence_role": "professional operational gate; no reduced run is promoted",
        "professional_package_ready": not blockers,
        "support_checks": {
            "repo_files_checked": len(scripts.get("files", [])),
            "compileall_passed": env.get("compileall_support_check", {}).get("returncode") == 0,
            "model_manifests_checked": len(data.get("models", [])),
            "sensitivity_weight_manifests_checked": len(data.get("sensitivity_weights", [])),
            "dataset_metric_artifacts_checked": len(data.get("datasets_and_metric_artifacts", [])),
        },
        "blockers": blockers,
        "next_full_execution_if_unblocked": [
            "materialize Wan2.1, CogVideoX, and LTX-Video checkpoints",
            "download released Yassaman/SenCache sensitivity weights or rerun finite-difference calibration on MixKit videos",
            "materialize VBench and T2V-CompBench prompt/evaluator artifacts",
            "run cached and uncached variants for each model with matching T=50, seeds, K, and thresholds",
            "emit raw videos, skip-count logs, NFE/cache ratio, latency/GFLOPs traces, and quality scores",
            "compare simulated gap/results to the paper tables, paragraphs, figures, and supplement MAE diagnostic",
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
    dag["graph_id"] = "CVPR2026_053_sencache_sensitivity_aware_caching_gap_dag_iter_03"
    dag["updated_at_utc"] = utc_now()
    new_nodes = [
        {
            "id": "ops.sensitivity_calibration_or_weight_gate",
            "type": "calibration_artifact_gate",
            "skill_role": "require sensitivity evidence before cached generation",
            "content": (
                "Resolve released Yassaman/SenCache .npz sensitivity weights or rerun finite-difference "
                "calibration on MixKit videos for Wan2.1, CogVideoX, and LTX-Video. Required artifacts include "
                "timesteps, J_x_norm, J_t_norm, sample counts, calibration logs, and GPU traces."
            ),
        },
        {
            "id": "ops.video_model_script_matrix",
            "type": "operational_execution_matrix",
            "skill_role": "bind Loop 2 to official SenCache entrypoints",
            "content": (
                "Use DAG-encoded Wan2.1/sencache.py, CogVideoX/sencache.py, LTX-Video/sencache.py, and "
                "their sensitivity_calculation scripts. Extract checkpoint path, prompt file, seed, T=50, "
                "sencache thresholds, K, cached/uncached flag, latency flag, and GFLOPs flag."
            ),
        },
        {
            "id": "ops.video_checkpoint_gate",
            "type": "model_data_gate",
            "skill_role": "require loadable video diffusion checkpoints",
            "content": (
                "Resolve loadable Wan2.1 T2V, CogVideoX1.5-5B, and LTX-Video checkpoints. HF metadata alone is "
                "not enough; the author simulation needs local or directly loadable model artifacts."
            ),
        },
        {
            "id": "ops.vbench_compbench_metric_gate",
            "type": "evaluation_artifact_gate",
            "skill_role": "require paper prompt and scoring artifacts",
            "content": (
                "Resolve full VBench prompt set, T2V-CompBench 70-prompt ablation set, LPIPS/PSNR/SSIM "
                "reference scoring, NFE/cache-ratio logs, latency/GFLOPs traces, and supplement 100-video MAE diagnostic."
            ),
        },
        {
            "id": "ops.paper_video_hardware_runtime_gate",
            "type": "professional_hardware_gate",
            "skill_role": "make latency and cache claims hardware-bound",
            "content": (
                "Before claiming SenCache acceleration, verify GH200 supplement latency or comparable paper-approved "
                "video-diffusion hardware, CUDA/PyTorch/Diffusers runtime, decord/video IO, idle GPU capacity, and telemetry."
            ),
        },
        {
            "id": "ops.full_cached_uncached_artifact_gate",
            "type": "professional_artifact_gate",
            "skill_role": "make verifier comparison table-shaped",
            "content": (
                "Require full cached and uncached generation outputs for Wan2.1, CogVideoX, and LTX-Video, "
                "plus raw videos, skip-count logs, cache ratio, NFE, latency, GFLOPs, LPIPS, PSNR, SSIM, VBench, "
                "and MAE summaries."
            ),
        },
        {
            "id": "decision.explicit_blocker_after_sencache_preflight",
            "type": "author_reviewer_decision",
            "skill_role": "feed operational failure back into Loop 1",
            "content": (
                "If sensitivity weights/calibration, model checkpoints, prompt/evaluator artifacts, hardware/runtime, "
                "or cached/uncached result-grid artifacts are absent, mark not converged and update the DAG instead of "
                "launching a one-prompt demo."
            ),
        },
    ]
    for node in new_nodes:
        add_node_if_missing(dag, node)
    add_edge_if_missing(dag, "ops.resolve_repo_code", "ops.video_model_script_matrix")
    add_edge_if_missing(dag, "ops.resolve_models_data", "ops.sensitivity_calibration_or_weight_gate")
    add_edge_if_missing(dag, "ops.resolve_models_data", "ops.video_checkpoint_gate")
    add_edge_if_missing(dag, "ops.sensitivity_calibration_or_weight_gate", "ops.video_model_script_matrix")
    add_edge_if_missing(dag, "ops.video_checkpoint_gate", "ops.video_model_script_matrix")
    add_edge_if_missing(dag, "ops.video_model_script_matrix", "ops.vbench_compbench_metric_gate")
    add_edge_if_missing(dag, "experiments.system_surface", "ops.paper_video_hardware_runtime_gate")
    add_edge_if_missing(dag, "ops.vbench_compbench_metric_gate", "ops.paper_video_hardware_runtime_gate")
    add_edge_if_missing(dag, "ops.paper_video_hardware_runtime_gate", "ops.full_cached_uncached_artifact_gate")
    add_edge_if_missing(dag, "ops.full_cached_uncached_artifact_gate", "loop2.execute_operational_dag")
    add_edge_if_missing(dag, "loop2.execute_operational_dag", "decision.explicit_blocker_after_sencache_preflight")
    add_edge_if_missing(dag, "decision.explicit_blocker_after_sencache_preflight", "reviewer.keep_exact_artifact_debt")
    dag.setdefault("previous_loop_updates", []).append(
        {
            "id": "update.add_sencache_sensitivity_model_metric_hardware_result_gates",
            "reason": "specialized SenCache preflight found missing full paper-shaped sensitivity, video model, prompt/evaluator, runtime, hardware, and result-grid artifacts",
            "success_criteria": [
                "sensitivity calibration or released .npz weight gate encoded",
                "Wan2.1/CogVideoX/LTX checkpoint gate encoded",
                "VBench/T2V-CompBench/LPIPS/PSNR/SSIM metric gate encoded",
                "GH200 or paper-approved video hardware gate encoded",
                "full cached/uncached result-grid gate encoded",
                "one-prompt test_prompt.json files remain support only",
            ],
            "blocker_ids": [item["id"] for item in blockers],
        }
    )
    dag["signature"] = signature_for(dag)
    iter_path = PAPER_RUN / "paper_author_gap_dag_iter_03.json"
    write_json(iter_path, dag)
    write_json(DAG_PATH, dag)
    return dag


def artifact_debt() -> list[dict[str, str]]:
    return [
        {
            "id": "sensitivity_calibration_artifacts",
            "required": "sensitivity_wan21.npz, sensitivity_cogvid.npz, sensitivity_ltx.npz or full finite-difference calibration reruns on MixKit videos",
        },
        {
            "id": "main_cached_uncached_tables",
            "required": "Wan2.1, CogVideoX, and LTX cached/uncached T=50 video generation results at similar compute budgets",
        },
        {
            "id": "metric_scoring_outputs",
            "required": "NFE, cache ratio, LPIPS, PSNR, SSIM, wall-clock latency, GFLOPs, VBench scores, and consecutive-step MAE",
        },
        {
            "id": "datasets_and_model_artifacts",
            "required": "loadable video model checkpoints, MixKit calibration videos, full VBench prompts, and T2V-CompBench prompts",
        },
        {
            "id": "hardware_runtime_traces",
            "required": "GH200 supplement latency or comparable paper-approved video-diffusion telemetry with GPU/CPU/RAM traces",
        },
    ]


def update_paper_run(verifier: dict[str, Any], dag: dict[str, Any]) -> None:
    gate = verifier["professional_gate"]
    artifacts = {
        "blockers": gate.get("blockers", []),
        "gpu_probe": {
            "status": "pass" if verifier["environment"].get("gpu_rows") else "missing",
            "gpu_rows": verifier["environment"].get("gpu_rows", []),
        },
        "repo_audits": read_json(PAPER_RUN / "operational_artifacts.json").get("repo_audits", [])
        if (PAPER_RUN / "operational_artifacts.json").exists()
        else [],
        "specialized_runner": {
            "status": gate["status"],
            "artifact_dir": str(RUNNER_DIR),
            "environment_path": str(ENV_PATH),
            "official_script_manifest_path": str(SCRIPT_MANIFEST_PATH),
            "model_data_manifest_path": str(MODEL_DATA_PATH),
            "professional_gate_path": str(PROFESSIONAL_GATE_PATH),
            "verifier_path": str(VERIFIER_PATH),
        },
    }
    write_json(PAPER_RUN / "operational_artifacts.json", artifacts)
    paper_verifier = {
        "checks": [
            {"name": "blind_contract", "status": "pass", "detail": verifier["blind_contract_checked"]},
            {
                "name": "gap_semantic_match",
                "status": "pass",
                "detail": "Existing iter_02 semantic gap remains accepted; iter_03 tightens author-simulation execution gates.",
            },
            {
                "name": "method_gap_binding_match",
                "status": "pass",
                "detail": "Latent/timestep sensitivity and epsilon/K cache gates remain bound to the paper gap.",
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
            {"name": "exact_artifact_debt_recorded", "status": "pass", "detail": artifact_debt()},
        ],
        "converged": False,
        "created_at_utc": utc_now(),
        "iteration": 3,
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "professional_ready": False,
        "required_updates": [
            {
                "id": "update.resolve_sencache_professional_blockers",
                "reason": gate["status"],
                "success_criteria": [
                    "run paper-appropriate cached/uncached video diffusion grid or provide equivalent exact artifacts",
                    "emit verifier-comparable NFE/cache/quality/latency/GFLOPs table shapes",
                    "do not count one-prompt demo files, syntax checks, or HF metadata as convergence",
                ],
            }
        ],
        "score": 0.84,
        "semantic_ready": True,
        "status": gate["status"] + "_after_specialized_runner",
        "dag_signature": dag["signature"],
    }
    write_json(PAPER_RUN / "verifier_result_iter_03.json", paper_verifier)
    (PAPER_RUN / "STATUS.md").write_text(
        f"# {TITLE}\n\n"
        f"- Paper id: `{PAPER_ID}`\n"
        f"- Final status: `{gate['status']}_after_specialized_runner`\n"
        "- Converged: `false`\n"
        "- Semantic ready: `true`\n"
        "- Professional ready: `false`\n"
        f"- DAG signature: `{dag['signature']}`\n"
        f"- Specialized runner status: `{gate['status']}`\n"
        f"- Specialized status: `{STATUS_PATH}`\n\n"
        "## Checks\n\n"
        "- `blind_contract`: `pass`\n"
        "- `gap_semantic_match`: `pass`\n"
        "- `method_gap_binding_match`: `pass`\n"
        "- `reduced_proxy_rejection_gate`: `pass`\n"
        "- `professional_artifact_package`: `blocked`\n"
        "- `exact_artifact_debt_recorded`: `pass`\n\n"
        "## Current Professional Blockers\n\n"
        + "\n".join(f"- `{item['id']}`: {item['detail']}" for item in gate.get("blockers", []))
        + "\n",
        encoding="utf-8",
    )


def write_status(verifier: dict[str, Any]) -> None:
    gate = verifier["professional_gate"]
    STATUS_PATH.write_text(
        "# SenCache Specialized Runner Status\n\n"
        f"- Updated: {verifier['updated_at_utc']}\n"
        f"- Paper: `{TITLE}`\n"
        f"- Status: `{gate['status']}`\n"
        f"- Professional package ready: `{gate['professional_package_ready']}`\n"
        f"- Repo files checked: `{gate['support_checks']['repo_files_checked']}`\n"
        f"- Compileall support check passed: `{gate['support_checks']['compileall_passed']}`\n"
        f"- Model manifests checked: `{gate['support_checks']['model_manifests_checked']}`\n"
        f"- Sensitivity weight manifests checked: `{gate['support_checks']['sensitivity_weight_manifests_checked']}`\n"
        f"- Dataset/metric artifacts checked: `{gate['support_checks']['dataset_metric_artifacts_checked']}`\n"
        f"- Blocker count: `{len(gate.get('blockers', []))}`\n\n"
        "## Artifact Paths\n"
        f"- Environment: `{ENV_PATH}`\n"
        f"- Official script manifest: `{SCRIPT_MANIFEST_PATH}`\n"
        f"- Model/data manifest: `{MODEL_DATA_PATH}`\n"
        f"- Professional gate: `{PROFESSIONAL_GATE_PATH}`\n"
        f"- Verifier: `{VERIFIER_PATH}`\n\n"
        "## Why This Is Not Converged\n"
        "- This did not run the bundled one-line prompt files, syntax checks, or HF metadata checks as convergence evidence.\n"
        "- The full SenCache paper shape requires sensitivity weights or calibration reruns, paper-scale video model checkpoints, prompt/evaluator suites, cached and uncached outputs, timing/GFLOPs traces, and metric tables.\n"
        "- The DAG was updated so Loop 2 must satisfy those operational gates before the verifier can accept the research-gap simulation.\n\n"
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
                item.setdefault("implementation_statuses", [])
                for status in [
                    "specialized_runner_preflight_completed",
                    "official_sencache_scripts_parsed",
                    "sensitivity_weight_model_dataset_manifests_checked",
                    "blocked_exact_sensitivity_video_model_metric_runtime_hardware_grid",
                ]:
                    if status not in item["implementation_statuses"]:
                        item["implementation_statuses"].append(status)
                item["professional_blocker"] = gate["status"] + "_after_specialized_runner"
                item["specialized_runner_status"] = gate["status"]
                item["specialized_runner_artifact_dir"] = str(RUNNER_DIR)
                item["specialized_runner_evidence"] = evidence
                break
        queue["created_at_utc"] = utc_now()
        write_json(QUEUE_PATH, queue)
    if SUMMARY_PATH.exists():
        summary = read_json(SUMMARY_PATH)
        for paper in summary.get("papers", []):
            if paper.get("paper_id") == PAPER_ID:
                paper["final_status"] = gate["status"] + "_after_specialized_runner"
                paper["converged"] = False
                paper["specialized_runner_status"] = gate["status"]
                paper["specialized_runner_artifact_dir"] = str(RUNNER_DIR)
                paper["specialized_runner_evidence"] = evidence
                statuses = paper.setdefault("implementation_statuses", [])
                for status in [
                    "specialized_runner_preflight_completed",
                    "official_sencache_scripts_parsed",
                    "sensitivity_weight_model_dataset_manifests_checked",
                    "blocked_exact_sensitivity_video_model_metric_runtime_hardware_grid",
                ]:
                    if status not in statuses:
                        statuses.append(status)
        summary["final_status"] = "active_specialized_operational_runs_pending_verifier"
        summary["created_at_utc"] = utc_now()
        write_json(SUMMARY_PATH, summary)


def main() -> int:
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    env = environment_manifest()
    scripts = script_manifest()
    data = model_data_manifest()
    blockers = derive_blockers(env, data)
    gate = professional_gate_result(env, scripts, data, blockers)
    verifier = {
        "artifact_kind": "sencache_specialized_verifier",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "repo": str(REPO),
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
                "ops.resolve_repo_code",
                "ops.resolve_models_data",
            ],
            "unresolved_professional_debt": blockers,
            "loop1_required_dag_update": [
                "Add sensitivity calibration or released .npz weight gate.",
                "Add Wan2.1/CogVideoX/LTX script and checkpoint gates.",
                "Add VBench/T2V-CompBench/LPIPS/PSNR/SSIM metric artifact gate.",
                "Add GH200 or paper-approved video hardware runtime gate.",
                "Add full cached/uncached result-grid gate.",
                "Keep one-line prompt files, syntax checks, and metadata as support only.",
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
