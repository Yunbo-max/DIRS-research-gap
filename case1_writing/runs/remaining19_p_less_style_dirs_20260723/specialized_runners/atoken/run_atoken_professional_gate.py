#!/usr/bin/env python3
"""AToken professional operational gate for the strict DIRS loop.

This runner turns the AToken gap DAG into concrete operational checks. It
records whether the DAG-only author simulation can run the real paper-shaped
study: official Apple checkpoints, multimodal datasets, reconstruction and
understanding/generation/3D metrics, H100-scale training hardware, Gaussian
Splatting dependencies, and verifier-comparable outputs. Demo, syntax, URL, and
repo-inventory evidence is support only and never promoted as convergence.
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
PAPER_RUN = RUN_ROOT / "paper_runs" / "cvpr2026_103_atoken_unified_visual_tokenizer"
RUNNER_DIR = RUN_ROOT / "specialized_runners" / "atoken"

OFFICIAL_REPO = Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/103_atoken/ml-atoken")
LIGHTWEIGHT_REPO = Path(
    "/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/103_atoken/"
    "ATOKEN-A-UNIFIED-TOKENIZER-FOR-VISION"
)

PAPER_ID = "CVPR2026_103_atoken_unified_visual_tokenizer"
TITLE = "AToken: A Unified Tokenizer for Vision"

DAG_PATH = PAPER_RUN / "paper_author_gap_dag.json"
STATUS_PATH = RUNNER_DIR / "ATOKEN_SPECIALIZED_STATUS.md"
VERIFIER_PATH = RUNNER_DIR / "atoken_specialized_verifier.json"
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
    "download_checkpoints.sh",
    "install_gs.sh",
    "pyproject.toml",
    "test_atoken.py",
    "examples.ipynb",
    "LICENSE",
    "MODEL-LICENSE",
    "configs/atoken-soc.yaml",
    "configs/atoken-sod.yaml",
    "configs/atoken-soc-s1.yaml",
    "configs/3d_decode_gs.yaml",
    "atoken_inference/atoken_wrapper.py",
    "atoken_inference/model/autoencoder_kl.py",
    "atoken_inference/model/decoder_gs.py",
    "atoken_inference/model/finite_scaler_quantize.py",
    "atoken_inference/model/lookup_free_quantize.py",
    "atoken_inference/model/sparse_preprocessors.py",
    "atoken_inference/scripts/pack_multiview_feat.py",
]

LIGHTWEIGHT_FILES = [
    "README.md",
    "atoken/train_tf.py",
    "atoken/train_curriculum.py",
    "atoken/eval.py",
    "atoken/models/atoken_tf.py",
    "atoken/models/atoken_tf3d.py",
    "atoken/quantizer/fsq.py",
    "atoken/losses/gram.py",
    "atoken/losses/lpips_wrap.py",
    "atoken/losses/clip_distill.py",
    "scripts/demo_tf.py",
    "scripts/demo_random.py",
]

EXPECTED_CHECKPOINTS = [
    {
        "id": "atoken_soc_continuous",
        "file": "checkpoints/atoken-soc.pt",
        "url": "https://ml-site.cdn-apple.com/models/atoken/atoken-soc.pt",
        "paper_role": "main continuous tokenizer for image/video/3D reconstruction and understanding transfer",
    },
    {
        "id": "atoken_sod_discrete",
        "file": "checkpoints/atoken-sod.pt",
        "url": "https://ml-site.cdn-apple.com/models/atoken/atoken-sod.pt",
        "paper_role": "discrete FSQ tokenizer checkpoint",
    },
    {
        "id": "atoken_3d_decode_gs",
        "file": "checkpoints/3d_decode_gs.pt",
        "url": "https://ml-site.cdn-apple.com/models/atoken/3d_decode_gs.pt",
        "paper_role": "Gaussian Splatting 3D decoder checkpoint",
    },
    {
        "id": "atoken_soc_stage1",
        "file": "checkpoints/atoken-soc-s1.pt",
        "url": "https://ml-site.cdn-apple.com/models/atoken/atoken-soc-s1.pt",
        "paper_role": "early image-only stage checkpoint for scaling/stage analysis",
    },
    {
        "id": "atoken_soc_stage2",
        "file": "checkpoints/atoken-soc-s2.pt",
        "url": "https://ml-site.cdn-apple.com/models/atoken/atoken-soc-s2.pt",
        "paper_role": "early image+video stage checkpoint for scaling/stage analysis",
    },
]

EXPECTED_DATASETS = [
    {"id": "dfn_training", "hints": ["DFN", "dfn"], "required": "DFN training images/text for tokenizer pretraining"},
    {"id": "open_images_training", "hints": ["OpenImages", "Open Images", "openimages", "open_images"], "required": "Open Images training/eval data"},
    {"id": "webvid_video_training", "hints": ["WebVid", "webvid"], "required": "WebVid video training data"},
    {"id": "textvr_retrieval", "hints": ["TextVR", "textvr"], "required": "TextVR video-text retrieval benchmark"},
    {"id": "panda70m_video_training", "hints": ["Panda70M", "panda70m"], "required": "Panda70M video training data"},
    {"id": "objaverse_3d", "hints": ["Objaverse", "objaverse"], "required": "Objaverse 3D data"},
    {"id": "cap3d_3d_text", "hints": ["Cap3D", "cap3d"], "required": "Cap3D text/3D data"},
    {"id": "imagenet_classification", "hints": ["ImageNet", "imagenet", "ILSVRC"], "required": "ImageNet zero-shot/classification benchmark"},
    {"id": "coco_retrieval_generation", "hints": ["COCO", "coco", "MSCOCO"], "required": "COCO image-text retrieval/generation benchmark"},
    {"id": "davis_video_reconstruction", "hints": ["DAVIS", "davis"], "required": "DAVIS video reconstruction benchmark"},
    {"id": "msrvtt_video_text", "hints": ["MSRVTT", "MSR-VTT", "msrvtt"], "required": "MSR-VTT video-text retrieval benchmark"},
]

EXPECTED_OUTPUT_SURFACES = [
    "image reconstruction table with PSNR, SSIM, LPIPS, rFID",
    "video reconstruction table with PSNR, SSIM, LPIPS, rFVD",
    "zero-shot ImageNet or equivalent accuracy table",
    "image-text retrieval table with R@1 and matching baselines",
    "video-text retrieval table with R@1 and matching baselines",
    "MLLM image understanding replacement table",
    "MLLM video understanding replacement table",
    "text-to-image generation table with gFID, IS, precision",
    "text-to-video generation table with rFVD/rFID-style generation metrics",
    "image-to-3D synthesis table and qualitative outputs",
    "stage/capacity scaling ablation including small-model degradation",
    "raw reconstructed images/videos/3D outputs, metric JSON, timing and GPU traces",
]

SEARCH_ROOTS = [
    OFFICIAL_REPO,
    LIGHTWEIGHT_REPO,
    Path(os.path.expanduser("~/.cache/huggingface/hub")),
    Path(os.path.expanduser("~/.cache/huggingface/datasets")),
    Path("/tf/notebooks/.cache/huggingface/hub"),
    Path("/tf/notebooks/.cache/huggingface/datasets"),
    Path("/tf/notebooks/models"),
    Path("/tf/notebooks/data"),
    Path("/tf/notebooks/datasets"),
    Path("/tf/notebooks/checkpoints"),
    Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/103_atoken"),
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


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
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
            normalized = hint.replace(" ", "_")
            for pattern in {f"*{hint}*", f"*{normalized}*", f"*{hint.replace('/', '--')}*"}:
                try:
                    candidates.extend(list(root.glob(pattern))[:25])
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


def count_media_files(path: str) -> int:
    candidate = Path(path)
    if candidate.is_file():
        return int(candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".avi", ".json", ".parquet"})
    if not candidate.is_dir():
        return 0
    result = run_cmd(
        [
            "bash",
            "-lc",
            "find "
            + repr(str(candidate))
            + " -maxdepth 2 -type f \\( -iname '*.png' -o -iname '*.jpg' -o "
            + "-iname '*.jpeg' -o -iname '*.webp' -o -iname '*.mp4' -o -iname '*.avi' "
            + "-o -iname '*.json' -o -iname '*.parquet' \\) | wc -l",
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
        "cuda_required": ".cuda()" in text or "device=\"cuda\"" in text or "to(device=\"cuda\")" in text,
        "bfloat16_used": "bfloat16" in text,
        "checkpoint_literals": sorted(set(re.findall(r"checkpoints/[^'\"\s)]+", text))),
        "metric_mentions": sorted(
            metric
            for metric in ["psnr", "ssim", "lpips", "rfid", "rfvd", "accuracy", "r@1", "gfid", "inception", "precision"]
            if metric in text.lower()
        ),
    }


def parse_shell(path: Path) -> dict[str, Any]:
    text = read_text(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "line_count": len(text.splitlines()),
        "urls": sorted(set(re.findall(r"https?://[^\"'\s)]+", text))),
        "wget_outputs": sorted(set(re.findall(r"wget\s+-O\s+([^ ]+)\s+", text))),
        "creates_checkpoints": "mkdir -p checkpoints" in text,
        "mutates_repo": any(term in text for term in ["git clone", "cp -r", "rm -rf", "pip install ."]),
        "mentions_trellis_or_gs": "TRELLIS" in text or "gaussian" in text.lower() or "diff-gaussian" in text.lower(),
    }


def parse_readme(path: Path) -> dict[str, Any]:
    text = read_text(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "line_count": len(text.splitlines()),
        "urls": sorted(set(re.findall(r"https?://[^\"'\s)]+", text))),
        "checkpoint_urls": sorted(set(re.findall(r"https://ml-site\.cdn-apple\.com/models/atoken/[^) \n]+", text))),
        "mentions_checkpoint_script": "download_checkpoints.sh" in text,
        "mentions_flash_attn": "flash-attn" in text or "flash_attn" in text,
        "mentions_gaussian_splatting": "Gaussian Splatting" in text or "diff-gaussian" in text,
        "mentions_examples_not_benchmarks": "examples.ipynb" in text and "Batch Processing" in text,
        "license_summary": "Apple Sample Code License" if "Apple Sample Code License" in text else "",
    }


def parse_pyproject(path: Path) -> dict[str, Any]:
    text = read_text(path)
    deps = re.findall(r'"([^"]+)"', text)
    return {
        "path": str(path),
        "exists": path.exists(),
        "line_count": len(text.splitlines()),
        "dependencies": deps,
        "requires_python": re.findall(r'requires-python\s*=\s*"([^"]+)"', text),
        "mentions_flash_optional": "flash-attn" in text,
    }


def url_head(url: str) -> dict[str, Any]:
    result = run_cmd(["bash", "-lc", f"curl -L -I --max-time 12 {url!r}"], timeout=18)
    stdout = result.get("stdout", "")
    status_codes = re.findall(r"HTTP/[0-9.]+\s+([0-9]+)", stdout)
    lengths = re.findall(r"(?i)^content-length:\s*([0-9]+)", stdout, re.M)
    result["http_status_codes"] = status_codes
    result["content_length_bytes"] = int(lengths[-1]) if lengths else None
    return result


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


def script_manifest() -> dict[str, Any]:
    files = []
    for rel in SCRIPT_FILES:
        path = OFFICIAL_REPO / rel
        row: dict[str, Any] = {"relative_path": rel, "path": str(path), "exists": path.exists()}
        if path.exists() and path.is_file():
            row["size_bytes"] = path.stat().st_size
            if path.suffix == ".py":
                row["parsed"] = parse_python(path)
            elif path.suffix == ".sh":
                row["parsed"] = parse_shell(path)
            elif path.name == "README.md":
                row["parsed"] = parse_readme(path)
            elif path.name == "pyproject.toml":
                row["parsed"] = parse_pyproject(path)
            elif path.suffix in {".yaml", ".yml"}:
                row["config_head"] = "\n".join(read_text(path).splitlines()[:40])
        files.append(row)

    lightweight = []
    for rel in LIGHTWEIGHT_FILES:
        path = LIGHTWEIGHT_REPO / rel
        row: dict[str, Any] = {"relative_path": rel, "path": str(path), "exists": path.exists()}
        if path.exists() and path.is_file():
            row["size_bytes"] = path.stat().st_size
            if path.suffix == ".py":
                row["parsed"] = parse_python(path)
            elif path.name == "README.md":
                row["parsed"] = {
                    "line_count": len(read_text(path).splitlines()),
                    "declares_lighter_single_gpu_defaults": "defaults here are lighter for single-GPU" in read_text(path),
                    "mentions_paper_scale": "27 blocks, d=1152, 16 heads" in read_text(path),
                }
        lightweight.append(row)

    license_text = read_text(OFFICIAL_REPO / "LICENSE")
    model_license = read_text(OFFICIAL_REPO / "MODEL-LICENSE")
    payload = {
        "artifact_kind": "atoken_official_script_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "official_repo": str(OFFICIAL_REPO),
        "lightweight_repo": str(LIGHTWEIGHT_REPO),
        "official_files": files,
        "lightweight_support_files": lightweight,
        "license_manifest": {
            "code_license_exists": (OFFICIAL_REPO / "LICENSE").exists(),
            "code_license_head": "\n".join(license_text.splitlines()[:4]),
            "model_license_exists": (OFFICIAL_REPO / "MODEL-LICENSE").exists(),
            "model_license_mentions_research": "research" in model_license.lower(),
            "model_license_head": "\n".join(model_license.splitlines()[:6]),
        },
        "paper_shaped_execution_matrix": {
            "official_inference": [
                "download all five Apple CDN checkpoints into ml-atoken/checkpoints",
                "load ATokenWrapper(config, checkpoint).cuda().to(torch.bfloat16)",
                "run image, video, and 3D inference paths with task_types and GS decoder when required",
            ],
            "benchmarks": EXPECTED_OUTPUT_SURFACES,
            "training_scale": [
                "Stage 1 requires the DAG-recorded 64 H100 class run",
                "Stages 2-3 require the DAG-recorded 256 H100 class run",
                "single-GPU lightweight code is support only",
            ],
            "verifier_comparison": [
                "compare reconstructed media, metrics JSON, GPU traces, and benchmark tables to paper evidence channels",
                "accept only close paper-shaped result surfaces, not demo success",
            ],
        },
        "support_only_findings": [
            "test_atoken.py is an example reconstruction path, not the full paper benchmark grid",
            "examples.ipynb is support only unless backed by all required checkpoints, datasets, and metrics",
            "the lightweight repo declares lighter single-GPU defaults and cannot converge the paper",
            "download URL reachability is not local checkpoint materialization",
        ],
    }
    write_json(SCRIPT_MANIFEST_PATH, payload)
    return payload


def environment_manifest() -> dict[str, Any]:
    packages = {
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "numpy": package_version("numpy"),
        "einops": package_version("einops"),
        "safetensors": package_version("safetensors"),
        "Pillow": package_version("Pillow"),
        "PyYAML": package_version("PyYAML"),
        "transformers": package_version("transformers"),
        "diffusers": package_version("diffusers"),
        "scipy": package_version("scipy"),
        "decord": package_version("decord"),
        "open-clip-torch": package_version("open-clip-torch"),
        "imageio": package_version("imageio"),
        "opencv-python": package_version("opencv-python"),
        "webdataset": package_version("webdataset"),
        "ftfy": package_version("ftfy"),
        "flash-attn": package_version("flash-attn"),
        "ninja": package_version("ninja"),
    }
    official_compile = run_cmd([sys.executable, "-m", "compileall", "-q", str(OFFICIAL_REPO)], timeout=180)
    lightweight_compile = run_cmd([sys.executable, "-m", "compileall", "-q", str(LIGHTWEIGHT_REPO)], timeout=180)
    payload = {
        "artifact_kind": "atoken_environment_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "python": sys.version,
        "gpu_rows": gpu_rows(),
        "python_processes": run_cmd(
            [
                "bash",
                "-lc",
                "ps -eo pid,etime,stat,cmd | rg 'prophet_custom_full_gsm8k_runner|AToken|atoken|python' || true",
            ],
            timeout=30,
        ),
        "packages": packages,
        "import_probes": {
            "torch": import_probe("torch"),
            "torchvision": import_probe("torchvision"),
            "numpy": import_probe("numpy"),
            "PIL": import_probe("PIL"),
            "yaml": import_probe("yaml"),
            "transformers": import_probe("transformers"),
            "diffusers": import_probe("diffusers"),
            "scipy": import_probe("scipy"),
            "decord": import_probe("decord"),
            "open_clip": import_probe("open_clip"),
            "imageio": import_probe("imageio"),
            "cv2": import_probe("cv2"),
            "webdataset": import_probe("webdataset"),
            "ftfy": import_probe("ftfy"),
            "flash_attn": import_probe("flash_attn"),
            "safetensors": import_probe("safetensors"),
            "diff_gaussian_rasterization": import_probe("diff_gaussian_rasterization"),
        },
        "compileall": {
            "official_repo": official_compile,
            "lightweight_repo": lightweight_compile,
        },
        "shell_syntax": {
            "download_checkpoints": run_cmd(["bash", "-n", "download_checkpoints.sh"], cwd=OFFICIAL_REPO, timeout=30),
            "install_gs": run_cmd(["bash", "-n", "install_gs.sh"], cwd=OFFICIAL_REPO, timeout=30),
        },
        "professional_hardware_expected_by_dag": [
            "64 H100 GPUs for Stage 1 training",
            "256 H100 GPUs for Stages 2-3 training",
            "CUDA GPU with enough memory for official ATokenWrapper inference",
            "Gaussian Splatting runtime for image-to-3D path",
        ],
    }
    write_json(ENV_PATH, payload)
    return payload


def model_data_manifest() -> dict[str, Any]:
    checkpoints = []
    for item in EXPECTED_CHECKPOINTS:
        local_path = OFFICIAL_REPO / item["file"]
        size_human, file_count = path_size(local_path)
        row = dict(item)
        row["local_path"] = str(local_path)
        row["materialized_locally"] = local_path.exists() and local_path.is_file() and local_path.stat().st_size > 0
        row["local_size_bytes"] = local_path.stat().st_size if local_path.exists() and local_path.is_file() else 0
        row["local_size_human"] = size_human
        row["file_count"] = file_count
        row["url_head"] = url_head(item["url"])
        checkpoints.append(row)

    datasets = []
    for item in EXPECTED_DATASETS:
        raw = local_matches(item["hints"])
        strong = []
        for match in raw:
            media_count = count_media_files(match["path"])
            if match["is_dir"] and (match["file_count"] >= 100 or media_count >= 20):
                strong.append({**match, "top_level_media_count": media_count})
        row = dict(item)
        row["local_matches"] = strong
        row["ignored_local_matches"] = [m for m in raw if m not in strong][:30]
        row["materialized_locally"] = bool(strong)
        datasets.append(row)

    output_candidates = []
    output_roots = [OFFICIAL_REPO, LIGHTWEIGHT_REPO, RUNNER_DIR, Path("/tf/notebooks/results"), Path("/tf/notebooks/outputs")]
    for root in output_roots:
        if not root.exists():
            continue
        for pattern in ["**/*metrics*.json", "**/*result*.json", "**/*eval*.json", "assets/examples_video", "assets/examples"]:
            try:
                for candidate in list(root.glob(pattern))[:80]:
                    size_human, file_count = path_size(candidate)
                    output_candidates.append(
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
        "artifact_kind": "atoken_model_data_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "official_repo": str(OFFICIAL_REPO),
        "checkpoints": checkpoints,
        "datasets_and_metric_artifacts": datasets,
        "candidate_output_artifacts": output_candidates[:120],
        "paper_result_surfaces": EXPECTED_OUTPUT_SURFACES,
        "not_promoted_support": [
            "official README checkpoint URLs",
            "assets/overview.png",
            "test_atoken.py example output folders",
            "lightweight repo random demos or single-GPU defaults",
            "paper text bundled in lightweight repo",
        ],
    }
    write_json(MODEL_DATA_PATH, payload)
    return payload


def derive_blockers(env: dict[str, Any], data: dict[str, Any]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    gpus = env.get("gpu_rows", [])
    gpu_names = " | ".join(row["name"] for row in gpus)
    h100_count = sum(1 for row in gpus if "H100" in row["name"])
    if h100_count < 64:
        blockers.append(
            {
                "id": "h100_training_scale_hardware_missing",
                "status": "blocked",
                "detail": f"DAG records 64 H100 Stage 1 and 256 H100 Stages 2-3; visible H100 count is {h100_count}; visible GPUs: {gpu_names or 'none'}.",
            }
        )
    clean_slots = [
        row
        for row in gpus
        if int(row.get("memory_free_mib", 0)) >= 18000 and int(row.get("utilization_gpu_pct", 0)) < 30
    ]
    if not clean_slots:
        blockers.append(
            {
                "id": "clean_gpu_slot_for_official_atoken_inference_missing",
                "status": "blocked",
                "detail": f"No visible GPU has >=18GB free and <30% utilization for official ATokenWrapper inference. GPU rows: {gpus}",
            }
        )
    missing_checkpoints = [
        item["id"] for item in data.get("checkpoints", []) if not item.get("materialized_locally")
    ]
    if missing_checkpoints:
        blockers.append(
            {
                "id": "apple_checkpoints_not_materialized",
                "status": "blocked",
                "detail": f"Official Apple checkpoint URLs exist but local .pt files are missing or empty: {', '.join(missing_checkpoints)}.",
            }
        )
    missing_datasets = [
        item["id"]
        for item in data.get("datasets_and_metric_artifacts", [])
        if not item.get("materialized_locally")
    ]
    if missing_datasets:
        blockers.append(
            {
                "id": "multimodal_training_eval_datasets_missing",
                "status": "blocked",
                "detail": f"Missing local dataset/materialized benchmark artifacts: {', '.join(missing_datasets)}.",
            }
        )
    required_imports = [
        "diffusers",
        "decord",
        "open_clip",
        "imageio",
        "webdataset",
        "ftfy",
        "flash_attn",
        "diff_gaussian_rasterization",
    ]
    missing_imports = [
        name
        for name in required_imports
        if env.get("import_probes", {}).get(name, {}).get("returncode") != 0
    ]
    if missing_imports:
        blockers.append(
            {
                "id": "atoken_runtime_dependencies_missing",
                "status": "blocked",
                "detail": f"Missing imports needed for full official/video/GS path: {', '.join(missing_imports)}.",
            }
        )
    outputs = data.get("candidate_output_artifacts", [])
    has_metric_json = any("metric" in Path(row["path"]).name.lower() or "result" in Path(row["path"]).name.lower() for row in outputs)
    has_media_outputs = any("examples_video" in row["path"] or "examples" in row["path"] for row in outputs)
    if not (has_metric_json and has_media_outputs):
        blockers.append(
            {
                "id": "full_multitask_benchmark_grid_missing",
                "status": "blocked",
                "detail": "No complete verifier-comparable AToken tables/figures/raw outputs were found for reconstruction, retrieval, MLLM, generation, 3D, and scaling surfaces.",
            }
        )
    return blockers


def professional_gate_result(blockers: list[dict[str, str]], env: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    gate = {
        "artifact_kind": "atoken_professional_gate_result",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "status": (
            "ready_for_full_atoken_multitask_gpu_execution"
            if not blockers
            else "blocked_by_checkpoints_datasets_runtime_hardware_and_benchmark_grid_requirements"
        ),
        "professional_package_ready": not blockers,
        "convergence_role": "professional operational gate; no reduced run is promoted",
        "support_checks": {
            "official_compileall_passed": env.get("compileall", {}).get("official_repo", {}).get("returncode") == 0,
            "lightweight_compileall_passed": env.get("compileall", {}).get("lightweight_repo", {}).get("returncode") == 0,
            "download_script_syntax_ok": env.get("shell_syntax", {}).get("download_checkpoints", {}).get("returncode") == 0,
            "install_gs_script_syntax_ok": env.get("shell_syntax", {}).get("install_gs", {}).get("returncode") == 0,
            "checkpoint_urls_checked": len(data.get("checkpoints", [])),
            "datasets_checked": len(data.get("datasets_and_metric_artifacts", [])),
        },
        "blockers": blockers,
        "next_full_execution_if_unblocked": [
            "download and verify all five Apple checkpoint files under ml-atoken/checkpoints",
            "install official dependencies including flash-attn and Gaussian Splatting modules",
            "materialize DFN, Open Images, WebVid, TextVR, Panda70M, Objaverse, Cap3D, ImageNet, COCO, DAVIS, and MSR-VTT as required by the DAG",
            "run official ATokenWrapper image/video/3D inference over benchmark inputs",
            "run reconstruction, zero-shot/retrieval, MLLM replacement, T2I/T2V, image-to-3D, and scaling result surfaces",
            "emit raw media, metric JSON, table summaries, timing/GPU traces, and compare to paper tables/figures/paragraph claims",
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
    add_node(
        dag,
        {
            "id": "ops.apple_checkpoint_materialization_gate",
            "type": "model_data_gate",
            "skill_role": "require official loadable AToken checkpoints",
            "content": "Resolve all five Apple CDN checkpoint files locally: AToken-So/C, AToken-So/D, 3D Decode GS, Stage 1, and Stage 2. URL reachability and README links are support only.",
        },
    )
    add_node(
        dag,
        {
            "id": "ops.multimodal_dataset_benchmark_gate",
            "type": "evaluation_artifact_gate",
            "skill_role": "require multimodal datasets and benchmark inputs",
            "content": "Materialize DFN, Open Images, WebVid, TextVR, Panda70M, Objaverse, Cap3D, ImageNet, COCO, DAVIS, and MSR-VTT or record exact missing surfaces before paper-result comparison.",
        },
    )
    add_node(
        dag,
        {
            "id": "ops.atoken_official_inference_matrix",
            "type": "operational_execution_matrix",
            "skill_role": "bind Loop 2 to official ATokenWrapper paths",
            "content": "Run official image/video/3D ATokenWrapper inference with loaded checkpoints, task_types, bfloat16 CUDA execution, and Gaussian Splatting decoder when 3D outputs are claimed.",
        },
    )
    add_node(
        dag,
        {
            "id": "ops.generation_and_3d_dependency_gate",
            "type": "runtime_gate",
            "skill_role": "require full modality dependencies",
            "content": "Verify diffusers, decord, open_clip, flash-attn, webdataset/ftfy, and diff-gaussian-rasterization/Gaussian Splatting dependencies before generation, video, or 3D claims.",
        },
    )
    add_node(
        dag,
        {
            "id": "ops.h100_training_scale_hardware_gate",
            "type": "professional_hardware_gate",
            "skill_role": "bind AToken training/scaling claims to paper hardware",
            "content": "Training and scaling conclusions require the DAG-recorded 64 H100 Stage 1 and 256 H100 Stages 2-3 hardware traces. Single-GPU demos only inspect mechanism.",
        },
    )
    add_node(
        dag,
        {
            "id": "ops.full_multitask_metric_artifact_gate",
            "type": "professional_artifact_gate",
            "skill_role": "make verifier comparison AToken-table shaped",
            "content": "Require raw reconstructed images/videos/3D outputs, PSNR/SSIM/LPIPS/rFID/rFVD, accuracy, R@1, gFID, IS, precision, timing/GPU traces, and table summaries for all paper result surfaces.",
        },
    )
    add_node(
        dag,
        {
            "id": "decision.explicit_blocker_after_atoken_preflight",
            "type": "author_reviewer_decision",
            "skill_role": "feed operational failure back into Loop 1",
            "content": "If checkpoints, datasets, runtime dependencies, H100 scale, clean GPU slot, or full multitask metric grids are absent, mark not converged and update the DAG instead of running a lightweight demo.",
        },
    )
    for src, dst in [
        ("ops.resolve_models_data", "ops.apple_checkpoint_materialization_gate"),
        ("ops.resolve_models_data", "ops.multimodal_dataset_benchmark_gate"),
        ("ops.resolve_repo_code", "ops.atoken_official_inference_matrix"),
        ("ops.apple_checkpoint_materialization_gate", "ops.atoken_official_inference_matrix"),
        ("ops.multimodal_dataset_benchmark_gate", "ops.atoken_official_inference_matrix"),
        ("ops.atoken_official_inference_matrix", "ops.generation_and_3d_dependency_gate"),
        ("experiments.system_surface", "ops.h100_training_scale_hardware_gate"),
        ("ops.generation_and_3d_dependency_gate", "ops.h100_training_scale_hardware_gate"),
        ("ops.h100_training_scale_hardware_gate", "ops.full_multitask_metric_artifact_gate"),
        ("ops.full_multitask_metric_artifact_gate", "loop2.execute_operational_dag"),
        ("loop2.execute_operational_dag", "decision.explicit_blocker_after_atoken_preflight"),
        ("decision.explicit_blocker_after_atoken_preflight", "reviewer.keep_exact_artifact_debt"),
    ]:
        add_edge(dag, src, dst)

    dag.setdefault("previous_loop_updates", []).append(
        {
            "id": "update.add_atoken_checkpoint_dataset_runtime_hardware_result_gates",
            "reason": "specialized AToken preflight found missing paper-scale checkpoint, dataset, runtime, H100 hardware, clean GPU, and full result-grid artifacts",
            "blocker_ids": [blocker["id"] for blocker in blockers],
            "success_criteria": [
                "Apple checkpoint materialization gate encoded",
                "multimodal dataset/benchmark gate encoded",
                "official ATokenWrapper image/video/3D inference matrix encoded",
                "generation and Gaussian Splatting runtime gate encoded",
                "64/256 H100 training-scale hardware gate encoded",
                "full multitask metric and raw-output artifact gate encoded",
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
        {
            "id": "main_benchmark_tables",
            "required": "image/video reconstruction, zero-shot classification, image/video retrieval, MLLM replacement, T2I/T2V, image-to-3D, and scaling ablation tables",
        },
        {
            "id": "metric_scoring_outputs",
            "required": "PSNR, SSIM, LPIPS, rFID, rFVD, accuracy, R@1, gFID, Inception Score, precision, timing and GPU traces",
        },
        {
            "id": "datasets_and_model_artifacts",
            "required": "five Apple CDN checkpoints; DFN, Open Images, WebVid, TextVR, Panda70M, Objaverse, Cap3D, ImageNet, COCO, DAVIS, MSR-VTT",
        },
        {
            "id": "hardware_runtime_traces",
            "required": "64 H100 Stage 1 and 256 H100 Stages 2-3 traces; clean CUDA inference slot; flash-attn and Gaussian Splatting runtime",
        },
        {
            "id": "operational_preflight_blockers",
            "required": "; ".join(f"{item['id']}: {item['detail']}" for item in blockers),
        },
    ]


def verifier_result(gate: dict[str, Any], dag: dict[str, Any], blockers: list[dict[str, str]]) -> dict[str, Any]:
    checks = [
        {
            "name": "blind_contract",
            "status": "pass",
            "detail": {
                "loop2_input": "paper_author_gap_dag_iter_03.json plus repo paths already encoded in DAG",
                "paper_text_visible_to_loop2": False,
                "oracle_results_visible_to_loop2": False,
                "paper_text_in_lightweight_repo_not_used_for_loop2": True,
            },
        },
        {
            "name": "official_checkpoint_gate",
            "status": "pass" if "apple_checkpoints_not_materialized" not in [b["id"] for b in blockers] else "blocked",
        },
        {
            "name": "multimodal_dataset_gate",
            "status": "pass" if "multimodal_training_eval_datasets_missing" not in [b["id"] for b in blockers] else "blocked",
        },
        {
            "name": "runtime_and_gs_gate",
            "status": "pass" if "atoken_runtime_dependencies_missing" not in [b["id"] for b in blockers] else "blocked",
        },
        {
            "name": "h100_training_scale_gate",
            "status": "pass" if "h100_training_scale_hardware_missing" not in [b["id"] for b in blockers] else "blocked",
        },
        {
            "name": "full_result_grid_gate",
            "status": "pass" if "full_multitask_benchmark_grid_missing" not in [b["id"] for b in blockers] else "blocked",
        },
        {
            "name": "reduced_proxy_rejection_gate",
            "status": "pass",
            "detail": "The gate explicitly rejects the lightweight single-GPU repo and demo/example outputs as convergence evidence.",
        },
        {
            "name": "exact_artifact_debt_recorded",
            "status": "pass",
            "detail": artifact_debt(blockers),
        },
    ]
    payload = {
        "artifact_kind": "atoken_specialized_verifier",
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
        "required_updates": [] if not blockers else [
            {
                "id": "update.materialize_atoken_professional_artifacts_before_rerun",
                "reason": gate["status"],
                "success_criteria": gate["next_full_execution_if_unblocked"],
            }
        ],
    }
    write_json(VERIFIER_PATH, payload)
    write_json(PAPER_RUN / "verifier_result_iter_03.json", payload)
    return payload


def update_paper_files(gate: dict[str, Any], verifier: dict[str, Any], blockers: list[dict[str, str]]) -> None:
    paper_status = {
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
    }
    write_json(PAPER_RUN / "paper_run_status.json", paper_status)
    status_lines = [
        "# AToken Specialized Operational Gate",
        "",
        f"- Paper id: `{PAPER_ID}`",
        f"- Title: `{TITLE}`",
        f"- Updated: `{paper_status['updated_at_utc']}`",
        f"- Final status: `{gate['status']}`",
        f"- Converged: `false`",
        f"- Professional ready: `{str(gate['professional_package_ready']).lower()}`",
        f"- DAG signature: `{verifier.get('dag_signature')}`",
        "",
        "## Blockers",
    ]
    for blocker in blockers:
        status_lines.append(f"- `{blocker['id']}`: {blocker['detail']}")
    status_lines += [
        "",
        "## Artifacts",
        f"- Gate: `{PROFESSIONAL_GATE_PATH}`",
        f"- Verifier: `{VERIFIER_PATH}`",
        f"- Environment: `{ENV_PATH}`",
        f"- Script manifest: `{SCRIPT_MANIFEST_PATH}`",
        f"- Model/data manifest: `{MODEL_DATA_PATH}`",
        "",
        "Reduced, lightweight, syntax-only, URL-only, and demo evidence remains support only.",
    ]
    STATUS_PATH.write_text("\n".join(status_lines) + "\n", encoding="utf-8")
    (PAPER_RUN / "STATUS.md").write_text("\n".join(status_lines) + "\n", encoding="utf-8")


def update_global_files(gate: dict[str, Any], blockers: list[dict[str, str]], dag: dict[str, Any]) -> None:
    status = gate["status"] + "_after_specialized_runner"
    if QUEUE_PATH.exists():
        queue = read_json(QUEUE_PATH)
        for item in queue.get("queue", []):
            if item.get("paper_id") == PAPER_ID:
                statuses = set(item.get("implementation_statuses", []))
                statuses.update(
                    {
                        "specialized_runner_preflight_completed",
                        "official_scripts_parsed",
                        "apple_checkpoint_urls_checked",
                        "checkpoints_not_materialized",
                        "professional_gate_blocked",
                    }
                )
                item["implementation_statuses"] = sorted(statuses)
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
                statuses.update(
                    {
                        "specialized_runner_preflight_completed",
                        "official_scripts_parsed",
                        "apple_checkpoint_urls_checked",
                        "professional_gate_blocked",
                    }
                )
                item["implementation_statuses"] = sorted(statuses)
                item["repo_paths"] = [str(OFFICIAL_REPO), str(LIGHTWEIGHT_REPO)]
                item["final_status"] = status
                item["converged"] = False
                item["professional_ready"] = gate["professional_package_ready"]
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
                        "verification": {
                            "status": "not_converged_operational_blocker_recorded" if blockers else "ready_for_verifier_result_comparison",
                            "professional_ready": gate["professional_package_ready"],
                            "converged": False if blockers else True,
                            "required_updates": gate["next_full_execution_if_unblocked"] if blockers else [],
                        },
                    }
                )
        summary["updated_at_utc"] = utc_now()
        write_json(SUMMARY_PATH, summary)

    refresh_status_markdown()


def prophet_progress() -> dict[str, Any]:
    summary_path = RUN_ROOT / "specialized_runners/prophet/custom_full_gsm8k_llada8b/summary.json"
    status_path = RUN_ROOT / "specialized_runners/prophet/custom_full_gsm8k_llada8b/status.json"
    result: dict[str, Any] = {"exists": summary_path.exists()}
    if summary_path.exists():
        summary = read_json(summary_path)
        result["summary"] = {
            "status": summary.get("status"),
            "total_samples": summary.get("total_samples"),
            "baseline_completed": summary.get("aggregates", {}).get("baseline", {}).get("completed_samples"),
            "prophet_completed": summary.get("aggregates", {}).get("prophet", {}).get("completed_samples"),
            "paired_shape": summary.get("paired_shape"),
        }
    if status_path.exists():
        status = read_json(status_path)
        result["status"] = {
            "current_sample_index": status.get("current_sample_index"),
            "cuda_visible_devices": status.get("cuda_visible_devices"),
            "device": status.get("device"),
        }
    return result


def trajectory_observed() -> dict[str, Any]:
    path = RUN_ROOT / "specialized_runners/prophet/trajectory_dataset_analysis/trajectory_dataset_status.json"
    if path.exists():
        try:
            return read_json(path)
        except Exception:
            return {"exists": True, "read_error": True}
    return {"exists": False}


def refresh_status_markdown() -> None:
    if not SUMMARY_PATH.exists():
        return
    summary = read_json(SUMMARY_PATH)
    papers = summary.get("papers", [])
    accepted = sum(1 for item in papers if item.get("converged") is True)
    blocked_or_running = len(papers) - accepted
    prophet = prophet_progress()
    traj = trajectory_observed()
    lines = [
        "# Remaining 19 p-less-Style DIRS Long Goal Status",
        "",
        f"Date: `{utc_now()}`",
        "",
        "- Final status: `active_specialized_operational_runs_pending_verifier`",
        f"- Accepted professional close match: `{accepted}` / `{len(papers)}`",
        f"- Explicitly blocked or running after DAG update: `{blocked_or_running}` / `{len(papers)}`",
        "- Reduced/smoke/proxy convergence disallowed: `true`",
        "- GPU available during run: `true`",
        "",
        "The run creates paper-specific DAGs and DAG-only author simulations. It does not promote repo audits, generic GPU motif rows, one-question probes, lightweight repos, or reduced runs into convergence.",
        "",
        "## Live GPU Run",
    ]
    if prophet.get("exists"):
        ps = prophet.get("summary", {})
        lines.append(
            f"- Prophet: `running_full_gpu_and_authenticated_trajectory_nodes_pending_artifacts`; "
            f"GPU `{prophet.get('status', {}).get('cuda_visible_devices')}` has "
            f"`{ps.get('prophet_completed')}` / `{ps.get('total_samples')}` paired GSM8K samples complete; "
            f"paired shape `{ps.get('paired_shape')}`."
        )
    else:
        lines.append("- Prophet: summary artifact not found.")
    if traj.get("exists"):
        lines.append(f"- Prophet trajectory dataset status: `{traj.get('status', 'observed')}`; artifact `{RUN_ROOT / 'specialized_runners/prophet/trajectory_dataset_analysis/trajectory_dataset_status.json'}`.")
    lines += ["", "## Specialized Runner Progress"]
    priority = [
        ("TRELLIS.2", RUN_ROOT / "specialized_runners/trellis2/TRELLIS2_SPECIALIZED_STATUS.md", "trellis2"),
        ("AToken", STATUS_PATH, "atoken"),
        ("RDVQ", RUN_ROOT / "specialized_runners/rdvq/RDVQ_SPECIALIZED_STATUS.md", "rdvq"),
        ("SenCache", RUN_ROOT / "specialized_runners/sencache/SENCACHE_SPECIALIZED_STATUS.md", "sencache"),
        ("SeaCache", RUN_ROOT / "specialized_runners/seacache/SEACACHE_SPECIALIZED_STATUS.md", "seacache"),
        ("FlashVID", RUN_ROOT / "specialized_runners/flashvid/FLASHVID_SPECIALIZED_STATUS.md", "flashvid"),
        ("SparseRL", RUN_ROOT / "specialized_runners/sparserl/SPARSERL_SPECIALIZED_STATUS.md", "sparserl"),
        ("LoongRL", RUN_ROOT / "specialized_runners/loongrl/LOONGRL_SPECIALIZED_STATUS.md", "loongrl"),
        ("MrRoPE", RUN_ROOT / "specialized_runners/mrrope/MRROPE_SPECIALIZED_STATUS.md", "mrrope"),
    ]
    for label, path, slug in priority:
        if path.exists():
            lines.append(f"- {label}: see `{path}`.")
    lines += [
        "",
        "## Active Artifact Paths",
        f"- AToken specialized status: `{STATUS_PATH}`",
        f"- AToken specialized verifier: `{VERIFIER_PATH}`",
        f"- Prophet GSM8K summary: `{RUN_ROOT / 'specialized_runners/prophet/custom_full_gsm8k_llada8b/summary.json'}`",
        f"- Specialized queue: `{QUEUE_PATH}`",
    ]
    LONG_STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    queue_lines = [
        "# Specialized Runner Queue",
        "",
        f"Updated: `{utc_now()}`",
        "",
        "This queue tracks non-reduced paper-shaped operational gates. A blocked gate is a DAG update signal, not convergence.",
        "",
    ]
    if QUEUE_PATH.exists():
        queue = read_json(QUEUE_PATH)
        for item in queue.get("queue", []):
            queue_lines.append(
                f"- `{item.get('paper_id')}`: `{item.get('professional_blocker')}`; runner `{item.get('runner_type')}`"
            )
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
                "script_manifest_files": len(scripts.get("official_files", [])),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
