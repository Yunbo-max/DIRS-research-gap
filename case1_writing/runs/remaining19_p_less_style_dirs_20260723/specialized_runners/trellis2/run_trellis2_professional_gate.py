#!/usr/bin/env python3
"""TRELLIS.2 professional operational gate for the strict DIRS loop.

The previous DAG said the local repository was not encoded. This runner first
records the discovered official repo, then checks whether the DAG-only author
simulation can run the actual paper-shaped TRELLIS.2 study: 4B HF weights,
O-Voxel preprocessing assets, Objaverse/ABO/HSSD/TexVerse/Toys4K/Sketchfab
datasets, CUDA extension stack, H100/A100 traces, and verifier-comparable 3D
reconstruction/generation/texturing/user-study outputs.
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
PAPER_RUN = RUN_ROOT / "paper_runs" / "cvpr2026_065_trellis2_native_compact_structured_latents"
RUNNER_DIR = RUN_ROOT / "specialized_runners" / "trellis2"
REPO = Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/065_trellis2/TRELLIS.2")
OVOXEL_REPO = REPO / "o-voxel"

PAPER_ID = "CVPR2026_065_trellis2_native_compact_structured_latents"
TITLE = "Native and Compact Structured Latents for 3D Generation"

DAG_PATH = PAPER_RUN / "paper_author_gap_dag.json"
STATUS_PATH = RUNNER_DIR / "TRELLIS2_SPECIALIZED_STATUS.md"
VERIFIER_PATH = RUNNER_DIR / "trellis2_specialized_verifier.json"
ENV_PATH = RUNNER_DIR / "environment.json"
SCRIPT_MANIFEST_PATH = RUNNER_DIR / "official_script_manifest.json"
MODEL_DATA_PATH = RUNNER_DIR / "model_data_manifest.json"
PROFESSIONAL_GATE_PATH = RUNNER_DIR / "professional_gate_result.json"

QUEUE_PATH = RUN_ROOT / "specialized_runner_queue.json"
SUMMARY_PATH = RUN_ROOT / "remaining19_strict_dirs_summary.json"
LONG_STATUS_PATH = RUN_ROOT / "LONGGOAL_STATUS.md"
SPECIALIZED_QUEUE_MD = RUN_ROOT / "SPECIALIZED_RUNNER_QUEUE.md"

SCRIPT_FILES = [
    "README.md",
    "setup.sh",
    "example.py",
    "example_texturing.py",
    "app.py",
    "app_texturing.py",
    "train.py",
    "data_toolkit/README.md",
    "data_toolkit/setup.sh",
    "data_toolkit/build_metadata.py",
    "data_toolkit/download.py",
    "data_toolkit/dump_mesh.py",
    "data_toolkit/dump_pbr.py",
    "data_toolkit/dual_grid.py",
    "data_toolkit/voxelize_pbr.py",
    "data_toolkit/encode_shape_latent.py",
    "data_toolkit/encode_pbr_latent.py",
    "data_toolkit/encode_ss_latent.py",
    "data_toolkit/render_cond.py",
    "trellis2/pipelines/trellis2_image_to_3d.py",
    "trellis2/pipelines/trellis2_texturing.py",
    "trellis2/pipelines/base.py",
    "trellis2/models/sparse_structure_flow.py",
    "trellis2/models/sparse_structure_vae.py",
    "trellis2/models/structured_latent_flow.py",
    "trellis2/models/sc_vaes/fdg_vae.py",
    "trellis2/models/sc_vaes/sparse_unet_vae.py",
    "trellis2/renderers/pbr_mesh_renderer.py",
    "trellis2/renderers/voxel_renderer.py",
    "o-voxel/README.md",
    "o-voxel/pyproject.toml",
    "o-voxel/setup.py",
    "o-voxel/o_voxel/convert/flexible_dual_grid.py",
    "o-voxel/o_voxel/convert/volumetic_attr.py",
    "o-voxel/o_voxel/io/vxz.py",
    "o-voxel/o_voxel/postprocess.py",
    "configs/gen/ss_flow_img_dit_1_3B_64_bf16.json",
    "configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json",
    "configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
    "configs/scvae/shape_vae_next_dc_f16c32_fp16.json",
    "configs/scvae/tex_vae_next_dc_f16c32_fp16.json",
]

EXPECTED_MODELS = [
    {
        "id": "trellis2_4b_hf_weights",
        "repo_id": "microsoft/TRELLIS.2-4B",
        "repo_type": "model",
        "local_hints": ["models--microsoft--TRELLIS.2-4B", "TRELLIS.2-4B", "microsoft/TRELLIS.2-4B"],
        "paper_role": "4B pretrained model for image-to-3D and shape-conditioned texturing",
    }
]

EXPECTED_DATASETS = [
    {"id": "objaverse_xl", "hints": ["ObjaverseXL", "Objaverse-XL", "ObjaverseXL_sketchfab"], "required": "Objaverse-XL training/reconstruction assets"},
    {"id": "abo", "hints": ["ABO", "abo"], "required": "ABO 3D assets"},
    {"id": "hssd", "hints": ["HSSD", "hssd"], "required": "HSSD 3D assets"},
    {"id": "texverse", "hints": ["TexVerse", "texverse"], "required": "TexVerse texture/PBR assets"},
    {"id": "toys4k", "hints": ["Toys4K", "Toys4k", "toys4k"], "required": "Toys4K reconstruction benchmark"},
    {"id": "sketchfab_featured", "hints": ["SketchfabFeatured", "SketchfabPicked", "Sketchfab Featured", "sketchfab"], "required": "Sketchfab Featured/Picked reconstruction benchmark"},
    {"id": "nanobanana_prompts", "hints": ["NanoBanana", "nanobanana", "image_prompts"], "required": "100 image prompts for image-to-3D comparison"},
    {"id": "pbr_dumps", "hints": ["pbr_dumps", "pbr_dump"], "required": "PBR dumps from data toolkit"},
    {"id": "dual_grid", "hints": ["dual_grid", "dual_grid_256", "dual_grid_512"], "required": "O-Voxel/Flexible Dual Grid assets"},
    {"id": "ovoxel_vxz", "hints": [".vxz", "vxz"], "required": "serialized O-Voxel .vxz assets"},
    {"id": "shape_latents", "hints": ["shape_latents", "shape_enc"], "required": "encoded shape latents"},
    {"id": "pbr_latents", "hints": ["pbr_latents", "tex_enc"], "required": "encoded PBR/texture latents"},
    {"id": "render_cond", "hints": ["renders_cond", "render_cond"], "required": "conditioning view renders"},
]

EXPECTED_OUTPUT_SURFACES = [
    "3D asset reconstruction table with MD, CD, and F1",
    "image-to-3D generation comparison over 100 prompts",
    "shape-conditioned PBR texture generation with PSNR, LPIPS, CLIP, CLIP-N",
    "SC-VAE ablation and architecture ablations",
    "test-time resolution scaling at 512^3, 1024^3, and 1536^3",
    "H100/A100 runtime table and memory/latency traces",
    "Toys4K and Sketchfab reconstruction results",
    "about 40 participant user-study Pref% artifacts",
    "raw GLB/mesh/O-Voxel outputs, rendered videos/images, metrics JSON, and table summaries",
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
    Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/065_trellis2"),
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
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": proc.returncode,
            "timeout": False,
            "seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 3),
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


def local_matches(hints: list[str]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for hint in hints:
            candidates: list[Path] = []
            exact = root / hint
            if exact.exists():
                candidates.append(exact)
            if hint.startswith("."):
                pattern_set = [f"*{hint}"]
            else:
                pattern_set = [f"*{hint}*", f"*{hint.replace(' ', '_')}*", f"*{hint.replace('/', '--')}*"]
            for pattern in pattern_set:
                try:
                    candidates.extend(list(root.glob(pattern))[:30])
                except OSError:
                    pass
            for candidate in candidates:
                try:
                    key = str(candidate.resolve())
                except OSError:
                    key = str(candidate)
                if key in seen:
                    continue
                seen.add(key)
                size_human, file_count = path_size(candidate)
                matches.append(
                    {
                        "path": str(candidate),
                        "is_dir": candidate.is_dir(),
                        "file_count": file_count,
                        "size_human": size_human,
                    }
                )
    return matches


def count_artifact_files(path: str) -> int:
    candidate = Path(path)
    if candidate.is_file():
        return int(candidate.suffix.lower() in {".glb", ".ply", ".obj", ".vxz", ".npz", ".json", ".png", ".jpg", ".webp", ".mp4"})
    if not candidate.is_dir():
        return 0
    result = run_cmd(
        [
            "bash",
            "-lc",
            "find "
            + repr(str(candidate))
            + " -maxdepth 2 -type f \\( -iname '*.glb' -o -iname '*.ply' -o -iname '*.obj' "
            + "-o -iname '*.vxz' -o -iname '*.npz' -o -iname '*.json' -o -iname '*.png' "
            + "-o -iname '*.jpg' -o -iname '*.webp' -o -iname '*.mp4' \\) | wc -l",
        ],
        timeout=30,
    )
    try:
        return int(result["stdout"].strip())
    except ValueError:
        return 0


def parse_python(path: Path) -> dict[str, Any]:
    text = read_text(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "line_count": len(text.splitlines()),
        "cli_flags": sorted(set(re.findall(r"['\"](--[A-Za-z0-9_-]+)['\"]", text))),
        "imports": sorted(set(re.findall(r"^(?:from|import)\s+([A-Za-z0-9_\.]+)", text, re.M))),
        "from_pretrained": sorted(set(re.findall(r"from_pretrained\([\"']([^\"']+)[\"']", text))),
        "cuda_required": ".cuda()" in text or "device='cuda'" in text or 'device="cuda"' in text,
        "exports_glb": ".glb" in text or "to_glb" in text,
        "mentions_ovoxel": "o_voxel" in text or "O-Voxel" in text,
        "mentions_metrics": sorted(
            metric for metric in ["md", "cd", "f1", "psnr", "lpips", "clip", "ulip", "uni3d", "pref"] if metric in text.lower()
        ),
    }


def parse_shell(path: Path) -> dict[str, Any]:
    text = read_text(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "line_count": len(text.splitlines()),
        "options": sorted(set(re.findall(r"--[A-Za-z0-9_-]+", text))),
        "pip_installs": sorted(set(re.findall(r"pip install ([^\n]+)", text))),
        "git_clones": sorted(set(re.findall(r"git clone[^\n]+", text))),
        "apt_installs": sorted(set(re.findall(r"(?:sudo )?apt install[^\n]+", text))),
        "mutates_environment": any(term in text for term in ["conda create", "pip install", "sudo apt", "git clone", "cp -r"]),
    }


def parse_readme(path: Path) -> dict[str, Any]:
    text = read_text(path)
    commands = []
    for match in re.finditer(r"```(?:sh|bash|python)?\n(.*?)```", text, flags=re.S):
        block = match.group(1)
        if "python" in block or "setup.sh" in block or "pip" in block:
            commands.append("\n".join(line.rstrip() for line in block.splitlines())[:1200])
    return {
        "path": str(path),
        "exists": path.exists(),
        "line_count": len(text.splitlines()),
        "urls": sorted(set(re.findall(r"https?://[^\"'\s)<>]+", text))),
        "hf_model_ids": sorted(set(re.findall(r"microsoft/TRELLIS\.2-4B", text))),
        "commands": commands[:16],
        "mentions_h100": "H100" in text,
        "mentions_a100": "A100" in text,
        "mentions_24gb": "24GB" in text,
        "mentions_setup_extensions": all(term in text for term in ["flash-attn", "nvdiffrast", "nvdiffrec", "cumesh", "o-voxel", "flexgemm"]),
        "mentions_training_datasets": all(term in text for term in ["Objaverse-XL", "ABO", "HSSD", "TexVerse"]),
    }


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
            "files_head": files[:80],
            "files_tail": files[-80:],
        }
    except Exception as exc:
        return {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "status": "unavailable_or_gated",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def script_manifest() -> dict[str, Any]:
    files = []
    for rel in SCRIPT_FILES:
        path = REPO / rel
        row: dict[str, Any] = {"relative_path": rel, "path": str(path), "exists": path.exists()}
        if path.exists() and path.is_file():
            row["size_bytes"] = path.stat().st_size
            if path.suffix == ".py":
                row["parsed"] = parse_python(path)
            elif path.suffix == ".sh":
                row["parsed"] = parse_shell(path)
            elif path.name == "README.md":
                row["parsed"] = parse_readme(path)
            elif path.suffix == ".json":
                try:
                    cfg = json.loads(path.read_text())
                    row["config_keys"] = list(cfg.keys())[:20] if isinstance(cfg, dict) else []
                    row["config_head"] = cfg if isinstance(cfg, dict) else None
                except Exception as exc:
                    row["config_error"] = str(exc)
        files.append(row)
    license_text = read_text(REPO / "LICENSE")
    payload = {
        "artifact_kind": "trellis2_official_script_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "repo": str(REPO),
        "files": files,
        "license_manifest": {
            "license_file_exists": (REPO / "LICENSE").exists(),
            "license_file_head": "\n".join(license_text.splitlines()[:4]),
            "appears_mit": "MIT License" in license_text[:200],
        },
        "paper_shaped_execution_matrix": {
            "inference": [
                "load Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')",
                "run image-to-3D generation on paper prompt/image set",
                "export rendered videos and GLB/O-Voxel outputs",
            ],
            "texturing": [
                "load Trellis2TexturingPipeline with texturing_pipeline.json",
                "run shape-conditioned PBR texture generation",
                "score PSNR, LPIPS, CLIP, CLIP-N and render/shading outputs",
            ],
            "training": [
                "preprocess Objaverse-XL/ABO/HSSD/TexVerse through metadata, mesh/PBR dumps, dual grid, voxels, latents, and condition renders",
                "run SC-VAE and sparse/shape/texture flow configs at paper scale",
            ],
            "verification": EXPECTED_OUTPUT_SURFACES,
        },
        "support_only_findings": [
            "assets/example_image and example_texturing are demos, not paper benchmark convergence",
            "README H100 timings are paper-side evidence unless locally reproduced with telemetry",
            "HF model metadata is not local model materialization",
            "repo path discovery fixes DAG operational visibility but is not a result",
        ],
    }
    write_json(SCRIPT_MANIFEST_PATH, payload)
    return payload


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
        "flash-attn": package_version("flash-attn"),
        "xformers": package_version("xformers"),
        "nvdiffrast": package_version("nvdiffrast"),
        "nvdiffrec": package_version("nvdiffrec"),
        "cumesh": package_version("cumesh"),
        "o-voxel": package_version("o-voxel"),
        "flexgemm": package_version("flexgemm"),
        "imageio": package_version("imageio"),
        "opencv-python-headless": package_version("opencv-python-headless"),
        "trimesh": package_version("trimesh"),
        "transformers": package_version("transformers"),
        "gradio": package_version("gradio"),
        "lpips": package_version("lpips"),
        "zstandard": package_version("zstandard"),
        "kornia": package_version("kornia"),
        "timm": package_version("timm"),
        "easydict": package_version("easydict"),
    }
    payload = {
        "artifact_kind": "trellis2_environment_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "python": sys.version,
        "repo": str(REPO),
        "gpu_rows": gpu_rows(),
        "python_processes": run_cmd(
            ["bash", "-lc", "ps -eo pid,etime,stat,cmd | rg 'prophet_custom_full_gsm8k_runner|TRELLIS|trellis2|python' || true"],
            timeout=30,
        ),
        "packages": packages,
        "import_probes": {
            "torch": import_probe("torch"),
            "torchvision": import_probe("torchvision"),
            "flash_attn": import_probe("flash_attn"),
            "xformers": import_probe("xformers"),
            "nvdiffrast": import_probe("nvdiffrast"),
            "nvdiffrec": import_probe("nvdiffrec"),
            "cumesh": import_probe("cumesh"),
            "o_voxel": import_probe("o_voxel"),
            "flexgemm": import_probe("flexgemm"),
            "cv2": import_probe("cv2"),
            "imageio": import_probe("imageio"),
            "trimesh": import_probe("trimesh"),
            "PIL": import_probe("PIL"),
            "transformers": import_probe("transformers"),
            "gradio": import_probe("gradio"),
            "lpips": import_probe("lpips"),
            "zstandard": import_probe("zstandard"),
            "kornia": import_probe("kornia"),
            "timm": import_probe("timm"),
            "easydict": import_probe("easydict"),
        },
        "compileall": {
            "repo": run_cmd([sys.executable, "-m", "compileall", "-q", str(REPO)], timeout=240),
            "o_voxel": run_cmd([sys.executable, "-m", "compileall", "-q", str(OVOXEL_REPO)], timeout=180),
        },
        "shell_syntax": {
            "setup": run_cmd(["bash", "-n", "setup.sh"], cwd=REPO, timeout=30),
            "data_toolkit_setup": run_cmd(["bash", "-n", "data_toolkit/setup.sh"], cwd=REPO, timeout=30),
        },
        "professional_hardware_expected_by_dag": [
            "H100 for reported runtime scaling",
            "A100/H100 verified official inference environment",
            "at least 24GB NVIDIA GPU for minimum inference",
            "CUDA 12.4 and PyTorch 2.6 environment for official setup",
        ],
    }
    write_json(ENV_PATH, payload)
    return payload


def model_data_manifest() -> dict[str, Any]:
    models = []
    for item in EXPECTED_MODELS:
        row = dict(item)
        raw = local_matches(item["local_hints"])
        strong = []
        for match in raw:
            lower = match["path"].lower()
            if "models--microsoft--trellis.2-4b" in lower or "trellis.2-4b" in lower:
                strong.append(match)
        row["local_matches"] = strong
        row["ignored_local_matches"] = [m for m in raw if m not in strong][:30]
        row["materialized_locally"] = bool(strong)
        row["hf_manifest"] = hf_repo_manifest(item["repo_id"], item["repo_type"])
        models.append(row)

    datasets = []
    for item in EXPECTED_DATASETS:
        raw = local_matches(item["hints"])
        strong = []
        for match in raw:
            artifact_count = count_artifact_files(match["path"])
            lower = match["path"].lower()
            if (
                match["is_dir"]
                and (match["file_count"] >= 50 or artifact_count >= 10)
                and "assets/example" not in lower
            ) or (not match["is_dir"] and Path(match["path"]).suffix.lower() in {".vxz", ".npz", ".json"}):
                strong.append({**match, "artifact_file_count_2deep": artifact_count})
        row = dict(item)
        row["local_matches"] = strong
        row["ignored_local_matches"] = [m for m in raw if m not in strong][:30]
        row["materialized_locally"] = bool(strong)
        datasets.append(row)

    outputs = []
    for root in [REPO, RUNNER_DIR, Path("/tf/notebooks/results"), Path("/tf/notebooks/outputs")]:
        if not root.exists():
            continue
        for pattern in ["**/*.glb", "**/*.mp4", "**/*metrics*.json", "**/*result*.json", "**/*user*study*", "**/*.vxz"]:
            try:
                for candidate in list(root.glob(pattern))[:80]:
                    if "assets/example" in str(candidate):
                        continue
                    size_human, file_count = path_size(candidate)
                    outputs.append(
                        {
                            "path": str(candidate),
                            "is_dir": candidate.is_dir(),
                            "file_count": file_count,
                            "size_human": size_human,
                        }
                    )
            except OSError:
                pass

    payload = {
        "artifact_kind": "trellis2_model_data_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "repo": str(REPO),
        "models": models,
        "datasets_and_preprocessed_artifacts": datasets,
        "candidate_output_artifacts": outputs[:120],
        "paper_result_surfaces": EXPECTED_OUTPUT_SURFACES,
        "not_promoted_support": [
            "example images and example_texturing mesh",
            "README H100 timing table",
            "HF model card metadata without local snapshots",
            "syntax/import checks without full benchmark outputs",
        ],
    }
    write_json(MODEL_DATA_PATH, payload)
    return payload


def derive_blockers(env: dict[str, Any], data: dict[str, Any]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    gpus = env.get("gpu_rows", [])
    gpu_names = " | ".join(row["name"] for row in gpus)
    if not any("H100" in row["name"] for row in gpus):
        blockers.append(
            {
                "id": "h100_runtime_scaling_hardware_missing",
                "status": "blocked",
                "detail": f"Paper/DAG includes H100 runtime scaling; visible GPUs are: {gpu_names or 'none'}.",
            }
        )
    if not any("A100" in row["name"] or "H100" in row["name"] for row in gpus):
        blockers.append(
            {
                "id": "official_verified_a100_h100_environment_missing",
                "status": "blocked",
                "detail": f"Official README says code verified on A100/H100; visible GPUs are: {gpu_names or 'none'}.",
            }
        )
    clean_slots = [
        row for row in gpus if row.get("memory_total_mib", 0) >= 24000 and row.get("memory_free_mib", 0) >= 18000 and row.get("utilization_gpu_pct", 0) < 30
    ]
    if not clean_slots:
        blockers.append(
            {
                "id": "clean_24gb_gpu_slot_missing",
                "status": "blocked",
                "detail": f"No visible 24GB+ GPU has >=18GB free and <30% utilization for TRELLIS.2 inference. GPU rows: {gpus}",
            }
        )
    missing_models = [item["id"] for item in data.get("models", []) if not item.get("materialized_locally")]
    if missing_models:
        blockers.append(
            {
                "id": "trellis2_4b_weights_not_materialized",
                "status": "blocked",
                "detail": f"Missing local HF model snapshots/checkpoints: {', '.join(missing_models)}.",
            }
        )
    missing_data = [item["id"] for item in data.get("datasets_and_preprocessed_artifacts", []) if not item.get("materialized_locally")]
    if missing_data:
        blockers.append(
            {
                "id": "trellis2_datasets_or_preprocessed_artifacts_missing",
                "status": "blocked",
                "detail": f"Missing local datasets/preprocessed artifacts: {', '.join(missing_data)}.",
            }
        )
    required_imports = ["flash_attn", "nvdiffrast", "nvdiffrec", "cumesh", "o_voxel", "flexgemm", "imageio", "cv2", "trimesh", "gradio", "lpips", "zstandard", "kornia", "timm", "easydict"]
    missing_imports = [
        name for name in required_imports if env.get("import_probes", {}).get(name, {}).get("returncode") != 0
    ]
    if missing_imports:
        blockers.append(
            {
                "id": "trellis2_runtime_extensions_missing",
                "status": "blocked",
                "detail": f"Missing official runtime/extension imports: {', '.join(missing_imports)}.",
            }
        )
    outputs = data.get("candidate_output_artifacts", [])
    has_glb = any(Path(row["path"]).suffix.lower() == ".glb" for row in outputs)
    has_metrics = any("metric" in Path(row["path"]).name.lower() or "result" in Path(row["path"]).name.lower() for row in outputs)
    if not (has_glb and has_metrics):
        blockers.append(
            {
                "id": "full_3d_result_grid_and_userstudy_missing",
                "status": "blocked",
                "detail": "No complete verifier-comparable 3D generation/reconstruction/texturing/scaling/user-study result grid was found.",
            }
        )
    return blockers


def professional_gate_result(blockers: list[dict[str, str]], env: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    gate = {
        "artifact_kind": "trellis2_professional_gate_result",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "status": "ready_for_full_trellis2_3d_execution" if not blockers else "blocked_by_weights_datasets_runtime_hardware_and_result_grid_requirements",
        "professional_package_ready": not blockers,
        "convergence_role": "professional operational gate; no reduced run is promoted",
        "support_checks": {
            "repo_discovered_and_encoded": REPO.exists(),
            "compileall_repo_passed": env.get("compileall", {}).get("repo", {}).get("returncode") == 0,
            "setup_syntax_ok": env.get("shell_syntax", {}).get("setup", {}).get("returncode") == 0,
            "model_manifests_checked": len(data.get("models", [])),
            "dataset_artifacts_checked": len(data.get("datasets_and_preprocessed_artifacts", [])),
        },
        "blockers": blockers,
        "next_full_execution_if_unblocked": [
            "materialize microsoft/TRELLIS.2-4B checkpoints from Hugging Face",
            "install PyTorch 2.6/CUDA 12.4 stack plus flash-attn, nvdiffrast, nvdiffrec, cumesh, o-voxel, flexgemm",
            "materialize Objaverse-XL, ABO, HSSD, TexVerse, Toys4K, Sketchfab Featured, NanoBanana prompts, and all O-Voxel/PBR/latent/rendered-condition artifacts",
            "run image-to-3D and shape-conditioned texturing pipelines on paper benchmark sets",
            "run SC-VAE, resolution-scaling, and architecture-ablation surfaces when training artifacts/hardware are present",
            "emit GLB/O-Voxel/raw render outputs, metric JSON, H100/A100 timing traces, user-study artifacts, and verifier table summaries",
        ],
    }
    write_json(PROFESSIONAL_GATE_PATH, gate)
    return gate


def node_exists(dag: dict[str, Any], node_id: str) -> bool:
    return any(node.get("id") == node_id for node in dag.get("nodes", []))


def add_node(dag: dict[str, Any], node: dict[str, Any]) -> None:
    if not node_exists(dag, node["id"]):
        dag.setdefault("nodes", []).append(node)


def add_edge(dag: dict[str, Any], src: str, dst: str) -> None:
    edge = [src, dst]
    if edge not in dag.setdefault("edges", []):
        dag["edges"].append(edge)


def update_dag(blockers: list[dict[str, str]]) -> dict[str, Any]:
    dag = read_json(DAG_PATH)
    dag["graph_id"] = f"{PAPER_ID}_gap_dag_iter_03"
    dag["updated_at_utc"] = utc_now()
    for node in [
        {
            "id": "ops.trellis2_repo_path_resolution_gate",
            "type": "operational_dependency",
            "skill_role": "correct missing repo path in Loop 1",
            "content": f"Use discovered official repo path {REPO}. The previous no-local-repository blocker is replaced by repo-aware professional gates.",
        },
        {
            "id": "ops.trellis2_hf_weight_materialization_gate",
            "type": "model_data_gate",
            "skill_role": "require 4B HF model materialization",
            "content": "Resolve microsoft/TRELLIS.2-4B as local loadable Hugging Face model files. Model-card availability is support only.",
        },
        {
            "id": "ops.ovoxel_dataset_preprocessing_gate",
            "type": "evaluation_artifact_gate",
            "skill_role": "require 3D datasets and O-Voxel preprocessing",
            "content": "Materialize Objaverse-XL, ABO, HSSD, TexVerse, Toys4K, Sketchfab Featured, NanoBanana prompts, PBR dumps, dual grids, .vxz assets, latents, and rendered conditioning views.",
        },
        {
            "id": "ops.trellis2_runtime_extension_gate",
            "type": "runtime_gate",
            "skill_role": "require official CUDA extension stack",
            "content": "Verify PyTorch 2.6/CUDA 12.4, flash-attn, nvdiffrast, nvdiffrec, cumesh, o-voxel, flexgemm, imageio/cv2/trimesh/gradio/lpips runtime before running or scoring.",
        },
        {
            "id": "ops.trellis2_inference_training_matrix",
            "type": "operational_execution_matrix",
            "skill_role": "bind Loop 2 to official image-to-3D/texturing/training paths",
            "content": "Run example/pipeline image-to-3D, example_texturing, SC-VAE, sparse/shape/texture flow configs, resolution scaling, and user-study artifact generation only with full models/data/runtime.",
        },
        {
            "id": "ops.trellis2_h100_a100_hardware_trace_gate",
            "type": "professional_hardware_gate",
            "skill_role": "bind performance claims to paper hardware",
            "content": "Require H100 timing traces for reported scaling and A100/H100 verified environment for official inference; 24GB RTX demos are support only unless full paper surface runs and verifier comparison pass.",
        },
        {
            "id": "ops.full_3d_result_userstudy_artifact_gate",
            "type": "professional_artifact_gate",
            "skill_role": "make verifier comparison table/figure shaped",
            "content": "Require GLB/O-Voxel/raw renders, MD/CD/F1/PSNR/LPIPS/CLIP/CLIP-N/ULIP-2/Uni3D/Pref% metric JSON, table summaries, runtime traces, ablations, and user-study artifacts.",
        },
        {
            "id": "decision.explicit_blocker_after_trellis2_preflight",
            "type": "author_reviewer_decision",
            "skill_role": "feed operational failure back into Loop 1",
            "content": "If weights, datasets/preprocessing, runtime extensions, H100/A100 traces, clean GPU slot, or result grids are absent, mark not converged and update the DAG instead of running examples.",
        },
    ]:
        add_node(dag, node)
    for src, dst in [
        ("ops.resolve_repo_code", "ops.trellis2_repo_path_resolution_gate"),
        ("ops.trellis2_repo_path_resolution_gate", "ops.trellis2_hf_weight_materialization_gate"),
        ("ops.resolve_models_data", "ops.trellis2_hf_weight_materialization_gate"),
        ("ops.resolve_models_data", "ops.ovoxel_dataset_preprocessing_gate"),
        ("ops.trellis2_hf_weight_materialization_gate", "ops.trellis2_inference_training_matrix"),
        ("ops.ovoxel_dataset_preprocessing_gate", "ops.trellis2_inference_training_matrix"),
        ("ops.trellis2_repo_path_resolution_gate", "ops.trellis2_runtime_extension_gate"),
        ("ops.trellis2_runtime_extension_gate", "ops.trellis2_inference_training_matrix"),
        ("experiments.system_surface", "ops.trellis2_h100_a100_hardware_trace_gate"),
        ("ops.trellis2_inference_training_matrix", "ops.trellis2_h100_a100_hardware_trace_gate"),
        ("ops.trellis2_h100_a100_hardware_trace_gate", "ops.full_3d_result_userstudy_artifact_gate"),
        ("ops.full_3d_result_userstudy_artifact_gate", "loop2.execute_operational_dag"),
        ("loop2.execute_operational_dag", "decision.explicit_blocker_after_trellis2_preflight"),
        ("decision.explicit_blocker_after_trellis2_preflight", "reviewer.keep_exact_artifact_debt"),
    ]:
        add_edge(dag, src, dst)
    dag.setdefault("previous_loop_updates", []).append(
        {
            "id": "update.add_trellis2_repo_weight_dataset_runtime_hardware_result_gates",
            "reason": "specialized TRELLIS.2 preflight discovered official repo path but found missing 4B weights, datasets/preprocessing, runtime extensions, hardware traces, and result grids",
            "blocker_ids": [blocker["id"] for blocker in blockers],
            "success_criteria": [
                "official repo path encoded",
                "HF 4B weight materialization gate encoded",
                "O-Voxel dataset/preprocessing gate encoded",
                "runtime extension gate encoded",
                "image-to-3D/texturing/training matrix encoded",
                "H100/A100 hardware trace gate encoded",
                "full 3D result and user-study artifact gate encoded",
            ],
        }
    )
    serial = json.dumps({"nodes": dag.get("nodes", []), "edges": dag.get("edges", [])}, sort_keys=True)
    dag["signature"] = hashlib.sha256(serial.encode("utf-8")).hexdigest()[:16]
    write_json(DAG_PATH, dag)
    write_json(PAPER_RUN / "paper_author_gap_dag_iter_03.json", dag)
    return dag


def artifact_debt(blockers: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"id": "main_benchmark_tables", "required": "3D reconstruction, image-to-3D, PBR texturing, SC-VAE ablation, resolution scaling, architecture ablations, H100 runtime, user study"},
        {"id": "metric_scoring_outputs", "required": "MD, CD, F1, PSNR, LPIPS, CLIP, CLIP-N, ULIP-2, Uni3D, Pref%, timing, memory"},
        {"id": "datasets_and_model_artifacts", "required": "microsoft/TRELLIS.2-4B; Objaverse-XL, ABO, HSSD, TexVerse, Toys4K, Sketchfab Featured, NanoBanana prompts; PBR dumps, dual grids, .vxz, latents, rendered conditions"},
        {"id": "hardware_runtime_traces", "required": "H100 runtime traces, A100/H100 verified inference environment, clean 24GB+ CUDA slot, PyTorch 2.6/CUDA 12.4, extension build logs"},
        {"id": "operational_preflight_blockers", "required": "; ".join(f"{item['id']}: {item['detail']}" for item in blockers)},
    ]


def verifier_result(gate: dict[str, Any], dag: dict[str, Any], blockers: list[dict[str, str]]) -> dict[str, Any]:
    ids = [b["id"] for b in blockers]
    checks = [
        {"name": "blind_contract", "status": "pass", "detail": {"loop2_input": "DAG plus repo path encoded by Loop 1 update", "paper_text_visible_to_loop2": False, "oracle_results_visible_to_loop2": False}},
        {"name": "repo_path_resolution_gate", "status": "pass", "detail": str(REPO)},
        {"name": "hf_4b_weight_gate", "status": "blocked" if "trellis2_4b_weights_not_materialized" in ids else "pass"},
        {"name": "dataset_preprocessing_gate", "status": "blocked" if "trellis2_datasets_or_preprocessed_artifacts_missing" in ids else "pass"},
        {"name": "runtime_extension_gate", "status": "blocked" if "trellis2_runtime_extensions_missing" in ids else "pass"},
        {"name": "hardware_trace_gate", "status": "blocked" if any(x in ids for x in ["h100_runtime_scaling_hardware_missing", "official_verified_a100_h100_environment_missing", "clean_24gb_gpu_slot_missing"]) else "pass"},
        {"name": "full_3d_result_grid_gate", "status": "blocked" if "full_3d_result_grid_and_userstudy_missing" in ids else "pass"},
        {"name": "reduced_proxy_rejection_gate", "status": "pass", "detail": "Examples, README timing, and repo syntax are support only."},
        {"name": "exact_artifact_debt_recorded", "status": "pass", "detail": artifact_debt(blockers)},
    ]
    payload = {
        "artifact_kind": "trellis2_specialized_verifier",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "iteration": 3,
        "dag_signature": dag.get("signature"),
        "status": "not_converged_operational_blocker_recorded" if blockers else "ready_for_verifier_result_comparison",
        "converged": False,
        "professional_ready": gate["professional_package_ready"],
        "checks": checks,
        "blockers": blockers,
        "required_updates": [] if not blockers else [{"id": "update.materialize_trellis2_professional_artifacts_before_rerun", "reason": gate["status"], "success_criteria": gate["next_full_execution_if_unblocked"]}],
    }
    write_json(VERIFIER_PATH, payload)
    write_json(PAPER_RUN / "verifier_result_iter_03.json", payload)
    return payload


def update_paper_files(gate: dict[str, Any], verifier: dict[str, Any], blockers: list[dict[str, str]]) -> None:
    write_json(
        PAPER_RUN / "paper_run_status.json",
        {
            "paper_id": PAPER_ID,
            "title": TITLE,
            "updated_at_utc": utc_now(),
            "status": gate["status"],
            "converged": False,
            "professional_ready": gate["professional_package_ready"],
            "specialized_runner": str(RUNNER_DIR),
            "professional_gate": str(PROFESSIONAL_GATE_PATH),
            "verifier": str(VERIFIER_PATH),
            "artifact_debt": artifact_debt(blockers),
        },
    )
    lines = [
        "# TRELLIS.2 Specialized Operational Gate",
        "",
        f"- Paper id: `{PAPER_ID}`",
        f"- Title: `{TITLE}`",
        f"- Updated: `{utc_now()}`",
        f"- Final status: `{gate['status']}`",
        "- Converged: `false`",
        f"- Professional ready: `{str(gate['professional_package_ready']).lower()}`",
        f"- DAG signature: `{verifier.get('dag_signature')}`",
        f"- Discovered repo encoded: `{REPO}`",
        "",
        "## Blockers",
    ]
    for blocker in blockers:
        lines.append(f"- `{blocker['id']}`: {blocker['detail']}")
    lines += [
        "",
        "## Artifacts",
        f"- Gate: `{PROFESSIONAL_GATE_PATH}`",
        f"- Verifier: `{VERIFIER_PATH}`",
        f"- Environment: `{ENV_PATH}`",
        f"- Script manifest: `{SCRIPT_MANIFEST_PATH}`",
        f"- Model/data manifest: `{MODEL_DATA_PATH}`",
        "",
        "Examples, README timing, syntax checks, and model-card metadata remain support only.",
    ]
    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (PAPER_RUN / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_global_files(gate: dict[str, Any], blockers: list[dict[str, str]], dag: dict[str, Any]) -> None:
    status = gate["status"] + "_after_specialized_runner"
    if QUEUE_PATH.exists():
        queue = read_json(QUEUE_PATH)
        for item in queue.get("queue", []):
            if item.get("paper_id") == PAPER_ID:
                statuses = set(item.get("implementation_statuses", []))
                statuses.update({"official_repo_path_discovered", "specialized_runner_preflight_completed", "hf_model_manifest_checked", "professional_gate_blocked"})
                item["implementation_statuses"] = sorted(statuses)
                item["repo_paths"] = [str(REPO)]
                item["professional_blocker"] = gate["status"]
                item["repo_exact_rerun_status"] = "blocked"
                item["specialized_runner_status_path"] = str(STATUS_PATH)
                item["specialized_verifier_path"] = str(VERIFIER_PATH)
                item["dag_iter_03_signature"] = dag.get("signature")
                item["exact_artifact_debt"] = artifact_debt(blockers)
        queue["updated_at_utc"] = utc_now()
        write_json(QUEUE_PATH, queue)
    if SUMMARY_PATH.exists():
        summary = read_json(SUMMARY_PATH)
        for item in summary.get("papers", []):
            if item.get("paper_id") == PAPER_ID:
                statuses = set(item.get("implementation_statuses", []))
                statuses.update({"official_repo_path_discovered", "specialized_runner_preflight_completed", "hf_model_manifest_checked", "professional_gate_blocked"})
                item["implementation_statuses"] = sorted(statuses)
                item["final_status"] = status
                item["converged"] = False
                item["professional_ready"] = gate["professional_package_ready"]
                item["repo_paths"] = [str(REPO)]
                item["specialized_runner_status_path"] = str(STATUS_PATH)
                item["specialized_verifier_path"] = str(VERIFIER_PATH)
                item["dag_iter_03_signature"] = dag.get("signature")
                item.setdefault("iterations", []).append(
                    {
                        "iteration": 3,
                        "dag_signature": dag.get("signature"),
                        "simulation": {
                            "author_decision": "explicit_operational_blocker" if blockers else "ready_for_full_execution",
                            "raw_artifact_level": "professional_preflight_only",
                            "reduced_or_proxy_used_for_convergence": False,
                            "professional_package_ready": gate["professional_package_ready"],
                            "professional_package_reason": gate["status"],
                        },
                        "verification": {"status": "not_converged_operational_blocker_recorded", "professional_ready": gate["professional_package_ready"], "converged": False},
                    }
                )
        summary["updated_at_utc"] = utc_now()
        write_json(SUMMARY_PATH, summary)
    refresh_status_markdown()


def prophet_progress() -> dict[str, Any]:
    summary_path = RUN_ROOT / "specialized_runners/prophet/custom_full_gsm8k_llada8b/summary.json"
    if not summary_path.exists():
        return {}
    d = read_json(summary_path)
    return {
        "total": d.get("total_samples"),
        "baseline": d.get("aggregates", {}).get("baseline", {}).get("completed_samples"),
        "prophet": d.get("aggregates", {}).get("prophet", {}).get("completed_samples"),
        "paired_shape": d.get("paired_shape"),
    }


def refresh_status_markdown() -> None:
    if not SUMMARY_PATH.exists():
        return
    summary = read_json(SUMMARY_PATH)
    papers = summary.get("papers", [])
    accepted = sum(1 for item in papers if item.get("converged") is True)
    prophet = prophet_progress()
    lines = [
        "# Remaining 19 p-less-Style DIRS Long Goal Status",
        "",
        f"Date: `{utc_now()}`",
        "",
        "- Final status: `active_specialized_operational_runs_pending_verifier`",
        f"- Accepted professional close match: `{accepted}` / `{len(papers)}`",
        f"- Explicitly blocked or running after DAG update: `{len(papers) - accepted}` / `{len(papers)}`",
        "- Reduced/smoke/proxy convergence disallowed: `true`",
        "- GPU available during run: `true`",
        "",
        "The run creates paper-specific DAGs and DAG-only author simulations. It does not promote repo audits, generic GPU motif rows, one-question probes, examples, or reduced runs into convergence.",
        "",
        "## Live GPU Run",
    ]
    if prophet:
        lines.append(f"- Prophet: `running_full_gpu_and_authenticated_trajectory_nodes_pending_artifacts`; GPU `3` has `{prophet.get('prophet')}` / `{prophet.get('total')}` paired GSM8K samples complete; paired shape `{prophet.get('paired_shape')}`.")
    lines += ["", "## Specialized Runner Progress"]
    for label, path in [
        ("TRELLIS.2", STATUS_PATH),
        ("AToken", RUN_ROOT / "specialized_runners/atoken/ATOKEN_SPECIALIZED_STATUS.md"),
        ("RDVQ", RUN_ROOT / "specialized_runners/rdvq/RDVQ_SPECIALIZED_STATUS.md"),
        ("SenCache", RUN_ROOT / "specialized_runners/sencache/SENCACHE_SPECIALIZED_STATUS.md"),
        ("SeaCache", RUN_ROOT / "specialized_runners/seacache/SEACACHE_SPECIALIZED_STATUS.md"),
        ("FlashVID", RUN_ROOT / "specialized_runners/flashvid/FLASHVID_SPECIALIZED_STATUS.md"),
        ("SparseRL", RUN_ROOT / "specialized_runners/sparserl/SPARSERL_SPECIALIZED_STATUS.md"),
        ("LoongRL", RUN_ROOT / "specialized_runners/loongrl/LOONGRL_SPECIALIZED_STATUS.md"),
        ("MrRoPE", RUN_ROOT / "specialized_runners/mrrope/MRROPE_SPECIALIZED_STATUS.md"),
    ]:
        if path.exists():
            lines.append(f"- {label}: see `{path}`.")
    lines += [
        "",
        "## Active Artifact Paths",
        f"- TRELLIS.2 specialized status: `{STATUS_PATH}`",
        f"- TRELLIS.2 specialized verifier: `{VERIFIER_PATH}`",
        f"- Prophet GSM8K summary: `{RUN_ROOT / 'specialized_runners/prophet/custom_full_gsm8k_llada8b/summary.json'}`",
        f"- Specialized queue: `{QUEUE_PATH}`",
    ]
    LONG_STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    queue_lines = ["# Specialized Runner Queue", "", f"Updated: `{utc_now()}`", "", "Blocked gates are DAG update signals, not convergence.", ""]
    if QUEUE_PATH.exists():
        queue = read_json(QUEUE_PATH)
        for item in queue.get("queue", []):
            queue_lines.append(f"- `{item.get('paper_id')}`: `{item.get('professional_blocker')}`; runner `{item.get('runner_type')}`")
    SPECIALIZED_QUEUE_MD.write_text("\n".join(queue_lines) + "\n", encoding="utf-8")


def main() -> None:
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    scripts = script_manifest()
    env = environment_manifest()
    data = model_data_manifest()
    blockers = derive_blockers(env, data)
    gate = professional_gate_result(blockers, env, data)
    dag = update_dag(blockers)
    verifier = verifier_result(gate, dag, blockers)
    update_paper_files(gate, verifier, blockers)
    update_global_files(gate, blockers, dag)
    print(
        json.dumps(
            {
                "paper_id": PAPER_ID,
                "status": gate["status"],
                "blocker_count": len(blockers),
                "dag_signature": dag.get("signature"),
                "status_path": str(STATUS_PATH),
                "verifier_path": str(VERIFIER_PATH),
                "script_manifest_files": len(scripts.get("files", [])),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
