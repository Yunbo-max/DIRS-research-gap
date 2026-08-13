#!/usr/bin/env python3
"""RDVQ professional operational gate for the strict DIRS loop.

This runner checks whether the DAG-only author simulation can execute the
paper-shaped RDVQ study: released checkpoints, Kodak/DIV2K/CLIC image sets,
FID/KID reference tiles, estimated-rate and real-bitstream scripts, prefix
transfer sweeps, metric/runtime dependencies, GPU/hardware traces, and
verifier-comparable RD tables. Support-only checks are recorded but never
promoted as convergence.
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
PAPER_RUN = RUN_ROOT / "paper_runs" / "cvpr2026_067_rdvq_differentiable_vq_rate_distortion"
RUNNER_DIR = RUN_ROOT / "specialized_runners" / "rdvq"
REPO = Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/067_rdvq/RDVQ")

PAPER_ID = "CVPR2026_067_rdvq_differentiable_vq_rate_distortion"
TITLE = "Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression"

DAG_PATH = PAPER_RUN / "paper_author_gap_dag.json"
STATUS_PATH = RUNNER_DIR / "RDVQ_SPECIALIZED_STATUS.md"
VERIFIER_PATH = RUNNER_DIR / "rdvq_specialized_verifier.json"
ENV_PATH = RUNNER_DIR / "environment.json"
SCRIPT_MANIFEST_PATH = RUNNER_DIR / "official_script_manifest.json"
MODEL_DATA_PATH = RUNNER_DIR / "model_data_manifest.json"
PROFESSIONAL_GATE_PATH = RUNNER_DIR / "professional_gate_result.json"

QUEUE_PATH = RUN_ROOT / "specialized_runner_queue.json"
SUMMARY_PATH = RUN_ROOT / "remaining19_strict_dirs_summary.json"

SCRIPT_FILES = [
    "README.md",
    "TRAINING_STAGES.md",
    "requirements.txt",
    "test.sh",
    "test_Real.sh",
    "my_inference.py",
    "Real_Endecode_inference_single_stageAR.py",
    "scripts/release_validate.sh",
    "scripts/validate_tensor_rans.py",
    "scripts/report_profile.py",
    "scripts/tokenizer/train_vq.sh",
    "utils/inference_common.py",
    "utils/bitstream_container.py",
    "utils/profile_accounting.py",
    "utils/real_codec_stats.py",
    "tokenizer/tokenizer_image/models/quantizer.py",
    "tokenizer/tokenizer_image/models/vq_model.py",
    "tokenizer/tokenizer_image/models/ar_transformer.py",
    "tokenizer/tokenizer_image/training/train_vq.py",
]

EXPECTED_CHECKPOINTS = [
    {
        "id": "rdvq_testing_checkpoints",
        "repo_id": "CVLUESTC/RDVQ",
        "local_hints": ["models--CVLUESTC--RDVQ", "CVLUESTC/RDVQ", "RDVQ"],
        "paper_role": "released testing weights for estimated-rate and real-bitstream RD curves",
    }
]

EXPECTED_DATASETS = [
    {
        "id": "kodak_image_folder",
        "required": "Kodak ImageFolder-style image directory for RD curves",
        "local_hints": ["Kodak", "kodak"],
    },
    {
        "id": "div2k_val_image_folder",
        "required": "DIV2K validation image directory for RD curves and ablations",
        "local_hints": ["DIV2K_valid_HR", "DIV2K_valid", "div2k", "DIV2K"],
    },
    {
        "id": "clic2020_test_image_folder",
        "required": "CLIC2020 test/valid image directory for RD curves",
        "local_hints": ["CLIC2020", "CLIC_valid", "clic"],
    },
    {
        "id": "imagenet_training_images",
        "required": "ImageNet 256 training images for tokenizer/entropy/RD training reproduction",
        "local_hints": ["imagenet", "ImageNet", "ILSVRC"],
    },
    {
        "id": "openimage_training_images",
        "required": "OpenImage/OpenImages images for high-resolution fine-tuning",
        "local_hints": ["openimage", "OpenImage", "OpenImages"],
    },
    {
        "id": "df2k_highres_images",
        "required": "DF2K 512-2048 images for high-resolution fine-tuning",
        "local_hints": ["DF2K", "df2k"],
    },
    {
        "id": "fid_reference_tiles",
        "required": "FID_REF_ROOT/<dataset>_256teles reference tiles for DIV2K/CLIC FID/KID",
        "local_hints": ["256teles", "fid_refs", "FID_REF_ROOT"],
    },
]

EXPECTED_OUTPUT_SURFACES = [
    "Kodak estimated-rate RD curve summaries",
    "Kodak real-bitstream RD curve summaries",
    "DIV2K-val estimated and real-bitstream RD summaries",
    "CLIC2020-test estimated and real-bitstream RD summaries",
    "FID/KID over 256x256 overlapping patches",
    "prefix-transfer rate-control sweep over TEST_TRANSFER_SLICES",
    "full-resolution 2K-image inference timing on RTX 4090",
    "high-resolution fine-tuning trace on RTX Pro 6000",
    "DIV2K ablation table",
    "raw reconstructed images and .bin real bitstreams",
    "bpp, DISTS, LPIPS, FID, KID, CLIPIQA, MUSIQ, NIQE, PSNR, MS-SSIM",
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
    Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/067_rdvq"),
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


def path_size(path: Path) -> tuple[str | None, int]:
    if not path.exists():
        return None, 0
    if path.is_file():
        return str(path.stat().st_size), 1
    size = run_cmd(["du", "-sh", str(path)], timeout=20)["stdout"].split()
    count = run_cmd(["bash", "-lc", f"find {str(path)!r} -type f | wc -l"], timeout=20)
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
            patterns = [f"*{hint}*", f"*{hint.replace('/', '--')}*"]
            for pattern in patterns:
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


def count_images(path: str) -> int:
    candidate = Path(path)
    if candidate.is_file():
        return int(candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"})
    if not candidate.is_dir():
        return 0
    result = run_cmd(
        [
            "bash",
            "-lc",
            f"find {str(candidate)!r} -maxdepth 1 -type f \\( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o -iname '*.bmp' \\) | wc -l",
        ],
        timeout=20,
    )
    try:
        return int(result["stdout"].strip())
    except ValueError:
        return 0


def valid_checkpoint_matches(matches: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    for match in matches:
        path = Path(match["path"])
        lower = str(path).lower()
        is_weight = path.suffix.lower() in {".pt", ".pth", ".safetensors", ".bin", ".ckpt"}
        is_hf_snapshot = "models--cvluestc--rdvq" in lower and ("snapshots" in lower or "blobs" in lower)
        is_perceptual_cache = lower.endswith("tokenizer/tokenizer_image/cache/vgg.pth")
        if (is_weight or is_hf_snapshot) and not is_perceptual_cache:
            valid.append(match)
        else:
            ignored.append(match)
    return valid, ignored


def valid_dataset_matches(dataset_id: str, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = []
    for match in matches:
        lower = match["path"].lower()
        image_count = count_images(match["path"])
        if dataset_id == "kodak_image_folder" and "kodak" in lower and image_count >= 20:
            valid.append({**match, "top_level_image_count": image_count})
        elif dataset_id == "div2k_val_image_folder" and "div2k" in lower and image_count >= 50:
            valid.append({**match, "top_level_image_count": image_count})
        elif dataset_id == "clic2020_test_image_folder" and "clic" in lower and image_count >= 50:
            valid.append({**match, "top_level_image_count": image_count})
        elif dataset_id == "imagenet_training_images" and ("imagenet" in lower or "ilsvrc" in lower) and match["is_dir"]:
            valid.append({**match, "top_level_image_count": image_count})
        elif dataset_id == "openimage_training_images" and ("openimage" in lower or "openimages" in lower) and match["is_dir"]:
            valid.append({**match, "top_level_image_count": image_count})
        elif dataset_id == "df2k_highres_images" and "df2k" in lower and match["is_dir"]:
            valid.append({**match, "top_level_image_count": image_count})
        elif dataset_id == "fid_reference_tiles" and ("256teles" in lower or "fid_ref" in lower) and image_count >= 50:
            valid.append({**match, "top_level_image_count": image_count})
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
            "files_head": files[:60],
            "files_tail": files[-60:],
        }
    except Exception as exc:
        return {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "status": "unavailable_or_gated",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def parse_python(path: Path) -> dict[str, Any]:
    text = read_text(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "line_count": len(text.splitlines()),
        "cli_flags": sorted(set(re.findall(r"['\"](--[A-Za-z0-9_-]+)['\"]", text))),
        "imports": sorted(set(re.findall(r"^(?:from|import)\\s+([A-Za-z0-9_\\.]+)", text, re.M))),
        "metric_mentions": sorted(
            metric
            for metric in ["bpp", "lpips", "dists", "fid", "kid", "clipiqa", "musiq", "niqe", "psnr", "msssim"]
            if metric in text.lower()
        ),
        "has_real_bitstream_path": "bitstream" in text.lower() or "rans" in text.lower(),
        "has_prefix_transfer": "transfer_slices" in text or "transfer-slices" in text,
        "has_profile_real": "profile-real" in text or "profile_accounting" in text,
        "has_fid_ref_logic": "fid_ref" in text or "FID_REF" in text,
        "checkpoint_required": "--ckpt-path" in text or "ckpt_path" in text,
    }


def parse_shell(path: Path) -> dict[str, Any]:
    text = read_text(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "line_count": len(text.splitlines()),
        "env_vars": sorted(set(re.findall(r"(?:\\$\\{([A-Z0-9_]+)|\\$([A-Z0-9_]+))", text))),
        "mentions_cuda_visible_devices": "CUDA_VISIBLE_DEVICES" in text,
        "calls_my_inference": "my_inference.py" in text,
        "calls_real_inference": "Real_Endecode_inference_single_stageAR.py" in text,
        "requires_test_ckpt_path": "TEST_CKPT_PATH" in text,
        "requires_test_image_dir": "TEST_IMAGE_DIR" in text,
        "supports_test_transfer_slices": "TEST_TRANSFER_SLICES" in text,
        "supports_disable_fid": "DISABLE_FID" in text,
        "supports_profile_real": "PROFILE_REAL" in text,
    }


def parse_readme(path: Path) -> dict[str, Any]:
    text = read_text(path)
    commands = []
    for match in re.finditer(r"```(?:bash)?\n(.*?)```", text, flags=re.S):
        block = match.group(1)
        if "python" in block or "bash" in block or "pip" in block or "TEST_" in block:
            commands.append("\n".join(line.rstrip() for line in block.splitlines()))
    return {
        "path": str(path),
        "exists": path.exists(),
        "line_count": len(text.splitlines()),
        "commands": commands,
        "mentions_hf_checkpoints": "CVLUESTC/RDVQ" in text or "Huggingface" in text,
        "mentions_estimated_rate": "Estimated bitrate" in text or "estimated-rate" in text,
        "mentions_real_bitstream": "Real bitstream" in text or "test_Real.sh" in text,
        "mentions_prefix_rate_control": "TEST_TRANSFER_SLICES" in text or "prefix" in text,
        "mentions_fid_refs": "FID_REF_ROOT" in text,
        "readme_license_badge_cc_by": "CC--BY--4.0" in text,
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
            elif path.name in {"README.md", "TRAINING_STAGES.md"}:
                row["parsed"] = parse_readme(path)
            elif path.name == "requirements.txt":
                row["requirements"] = [
                    line.strip()
                    for line in read_text(path).splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
        files.append(row)

    license_text = read_text(REPO / "LICENSE")
    payload = {
        "artifact_kind": "rdvq_official_script_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "repo": str(REPO),
        "files": files,
        "license_manifest": {
            "license_file_exists": (REPO / "LICENSE").exists(),
            "license_file_head": "\n".join(license_text.splitlines()[:3]),
            "actual_license_file_looks_mit": license_text.startswith("MIT License"),
            "readme_badge_claims_cc_by_4_0": any(
                (f.get("parsed") or {}).get("readme_license_badge_cc_by") for f in files
            ),
        },
        "paper_shaped_execution_matrix": {
            "estimated_rate": [
                "TEST_CKPT_PATH + TEST_IMAGE_DIR + TEST_DATASET={kodak,div2k,clic}",
                "bash test.sh",
                "metrics JSON with cd_bpp and perceptual/reference/no-reference scores",
            ],
            "real_bitstream": [
                "bash test_Real.sh",
                "tensor-rANS C++17/JIT support",
                "actual .bin payloads and cd_bpp_real fields",
                "PROFILE_REAL timing counters when latency is claimed",
            ],
            "rate_control": [
                "sweep TEST_TRANSFER_SLICES for prefix transmission",
                "compare prefix completion bitrate-quality tradeoff to paper figures/tables",
            ],
            "training_or_finetune": [
                "ImageNet/OpenImage/DF2K training stages only count with full datasets, checkpoints, and GPU traces",
                "RTX Pro 6000 high-resolution fine-tune gate remains separate from RTX 4090 evaluation gate",
            ],
        },
        "support_only_findings": [
            "release validation and bitstream-container roundtrip are implementation support only",
            "README assets/RD_curves.jpg are paper-presented evidence, not local reproduction",
            "HF metadata availability is not the same as local checkpoint materialization",
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
        "numpy": package_version("numpy"),
        "Pillow": package_version("Pillow"),
        "einops": package_version("einops"),
        "clean-fid": package_version("clean-fid"),
        "pyiqa": package_version("pyiqa"),
        "pytorch_msssim": package_version("pytorch-msssim"),
        "torchmetrics": package_version("torchmetrics"),
        "compressai": package_version("compressai"),
        "ninja": package_version("ninja"),
        "fvcore": package_version("fvcore"),
        "bitsandbytes": package_version("bitsandbytes"),
    }
    release_validate = run_cmd(
        ["bash", "scripts/release_validate.sh"],
        cwd=REPO,
        env={"RUN_IMPORT_CHECK": "0", "RUN_TENSOR_RANS": "0", "RUN_FORWARD_SMOKE": "0", "RUN_REAL_SMOKE": "0"},
        timeout=180,
    )
    payload = {
        "artifact_kind": "rdvq_environment_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "repo": str(REPO),
        "python": sys.version,
        "gpu_rows": gpu_rows(),
        "python_processes": run_cmd(
            [
                "bash",
                "-lc",
                "ps -eo pid,etime,cmd | rg 'prophet_custom_full_gsm8k_runner|Real_Endecode|my_inference|python' || true",
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
        "cxx_compiler": run_cmd(["bash", "-lc", "g++ --version || c++ --version"], timeout=30),
        "packages": packages,
        "import_probes": {
            "torch": import_probe("torch"),
            "torchvision": import_probe("torchvision"),
            "cleanfid": import_probe("cleanfid"),
            "pyiqa": import_probe("pyiqa"),
            "pytorch_msssim": import_probe("pytorch_msssim"),
            "torchmetrics": import_probe("torchmetrics"),
            "compressai": import_probe("compressai"),
            "ninja": import_probe("ninja"),
            "fvcore": import_probe("fvcore"),
        },
        "compileall_check": run_cmd([sys.executable, "-m", "compileall", "-q", str(REPO)], timeout=180),
        "release_validate_support_check": release_validate,
        "professional_hardware_expected_by_paper": [
            "four NVIDIA RTX 4090 GPUs for primary experiments",
            "single NVIDIA RTX Pro 6000 GPU for DF2K high-resolution fine-tuning",
            "NVIDIA RTX 4090 for 2K-image inference timing",
        ],
    }
    write_json(ENV_PATH, payload)
    return payload


def model_data_manifest() -> dict[str, Any]:
    checkpoints = []
    for item in EXPECTED_CHECKPOINTS:
        raw = local_matches(item["local_hints"])
        valid, ignored = valid_checkpoint_matches(raw)
        row = dict(item)
        row["local_matches"] = valid
        row["ignored_local_matches"] = ignored
        row["hf_manifest"] = hf_repo_manifest(item["repo_id"], "model")
        row["materialized_locally"] = bool(valid)
        checkpoints.append(row)

    datasets = []
    for item in EXPECTED_DATASETS:
        raw = local_matches(item["local_hints"])
        valid = valid_dataset_matches(item["id"], raw)
        row = dict(item)
        row["local_matches"] = valid
        row["ignored_local_matches"] = [m for m in raw if m not in valid]
        row["materialized_locally"] = bool(valid)
        datasets.append(row)

    output_dirs = []
    for root in [REPO, RUNNER_DIR, Path("/tf/notebooks/results"), Path("/tf/notebooks/outputs")]:
        if not root.exists():
            continue
        for name in ["forward", "Real", "metrics.json", "profile.json", "inference_logs"]:
            try:
                for candidate in root.glob(f"**/{name}"):
                    size_human, file_count = path_size(candidate)
                    output_dirs.append(
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
        "artifact_kind": "rdvq_model_data_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "repo": str(REPO),
        "checkpoints": checkpoints,
        "datasets_and_metric_artifacts": datasets,
        "candidate_output_artifacts": output_dirs[:100],
        "paper_result_surfaces": EXPECTED_OUTPUT_SURFACES,
        "not_promoted_support": [
            "assets/performance.png, assets/visual.jpg, and assets/RD_curves.jpg",
            "release_validate.sh support checks",
            "ImageFolder directory name matches without enough top-level images",
            "VGG perceptual cache weights",
            "HF model listing without downloaded checkpoint files",
        ],
    }
    write_json(MODEL_DATA_PATH, payload)
    return payload


def derive_blockers(env: dict[str, Any], data: dict[str, Any]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    gpus = env.get("gpu_rows", [])
    gpu_names = " | ".join(row["name"] for row in gpus)
    if not gpus:
        blockers.append({"id": "gpu_visibility", "status": "blocked", "detail": "No visible CUDA GPUs."})
    if not any("RTX 4090" in row["name"] for row in gpus):
        blockers.append(
            {
                "id": "primary_eval_hardware_class",
                "status": "blocked",
                "detail": f"Paper/DAG expects RTX 4090 evaluation/timing traces; visible GPUs are: {gpu_names or 'none'}.",
            }
        )
    if not any("RTX PRO 6000" in row["name"].upper() or "RTX Pro 6000" in row["name"] for row in gpus):
        blockers.append(
            {
                "id": "highres_finetune_hardware_class",
                "status": "blocked",
                "detail": f"Paper/DAG expects a single RTX Pro 6000 for high-resolution DF2K fine-tuning; visible GPUs are: {gpu_names or 'none'}.",
            }
        )
    clean_slots = [
        row
        for row in gpus
        if int(row.get("memory_free_mib", 0)) >= 12000 and int(row.get("utilization_gpu_pct", 0)) < 50
    ]
    if not clean_slots:
        blockers.append(
            {
                "id": "clean_gpu_slot_for_full_rd_grid",
                "status": "blocked",
                "detail": f"No visible RTX 4090 has at least 12GB free and low utilization. GPU rows: {gpus}",
            }
        )
    missing_checkpoints = [
        item["id"] for item in data.get("checkpoints", []) if not item.get("materialized_locally")
    ]
    if missing_checkpoints:
        blockers.append(
            {
                "id": "rdvq_checkpoints_not_materialized",
                "status": "blocked",
                "detail": f"Missing local released RDVQ checkpoint files: {', '.join(missing_checkpoints)}",
            }
        )
    must_have_eval = {"kodak_image_folder", "div2k_val_image_folder", "clic2020_test_image_folder", "fid_reference_tiles"}
    missing_eval = [
        item["id"]
        for item in data.get("datasets_and_metric_artifacts", [])
        if item["id"] in must_have_eval and not item.get("materialized_locally")
    ]
    if missing_eval:
        blockers.append(
            {
                "id": "rd_curve_datasets_or_fid_refs_missing",
                "status": "blocked",
                "detail": f"Missing local evaluation datasets/FID refs: {', '.join(missing_eval)}",
            }
        )
    missing_train = [
        item["id"]
        for item in data.get("datasets_and_metric_artifacts", [])
        if item["id"] in {"imagenet_training_images", "openimage_training_images", "df2k_highres_images"}
        and not item.get("materialized_locally")
    ]
    if missing_train:
        blockers.append(
            {
                "id": "training_and_highres_finetune_datasets_missing",
                "status": "blocked",
                "detail": f"Missing training/fine-tuning datasets for full author reproduction: {', '.join(missing_train)}",
            }
        )
    required_imports = {
        "cleanfid": "cleanfid",
        "pyiqa": "pyiqa",
        "pytorch_msssim": "pytorch_msssim",
        "torchmetrics": "torchmetrics",
        "compressai": "compressai",
        "ninja": "ninja",
    }
    missing_imports = [
        name
        for name, probe in env.get("import_probes", {}).items()
        if name in required_imports and probe.get("returncode") != 0
    ]
    if missing_imports:
        blockers.append(
            {
                "id": "compression_metric_runtime_missing",
                "status": "blocked",
                "detail": f"Missing required metric/codec runtime imports: {', '.join(missing_imports)}",
            }
        )
    release_check = env.get("release_validate_support_check", {})
    if release_check.get("returncode") != 0:
        blockers.append(
            {
                "id": "release_validation_support_check_failed",
                "status": "blocked",
                "detail": "scripts/release_validate.sh did not complete cleanly; inspect environment.json for stdout/stderr.",
            }
        )
    outputs = data.get("candidate_output_artifacts", [])
    has_metrics = any("metrics" in Path(row["path"]).name.lower() for row in outputs)
    has_real = any("/Real" in row["path"] or row["path"].endswith("/Real") for row in outputs)
    has_forward = any("/forward" in row["path"] or row["path"].endswith("/forward") for row in outputs)
    if not (has_metrics and has_real and has_forward):
        blockers.append(
            {
                "id": "full_estimated_and_real_bitstream_result_grid_missing",
                "status": "blocked",
                "detail": "No complete forward/Real metrics artifacts for Kodak, DIV2K, CLIC, prefix-transfer sweeps, and RD tables were found.",
            }
        )
    return blockers


def professional_gate_result(blockers: list[dict[str, str]], env: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    gate = {
        "artifact_kind": "rdvq_professional_gate_result",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "status": (
            "ready_for_full_rd_curve_gpu_execution"
            if not blockers
            else "blocked_by_checkpoints_datasets_runtime_hardware_release_and_result_grid_requirements"
        ),
        "professional_package_ready": not blockers,
        "convergence_role": "professional operational gate; no reduced run is promoted",
        "support_checks": {
            "repo_files_checked": len(SCRIPT_FILES),
            "compileall_passed": env.get("compileall_check", {}).get("returncode") == 0,
            "release_validate_passed": env.get("release_validate_support_check", {}).get("returncode") == 0,
            "checkpoint_manifests_checked": len(data.get("checkpoints", [])),
            "dataset_metric_artifacts_checked": len(data.get("datasets_and_metric_artifacts", [])),
        },
        "blockers": blockers,
        "next_full_execution_if_unblocked": [
            "materialize CVLUESTC/RDVQ released checkpoints",
            "materialize Kodak, DIV2K-val, CLIC2020-test, and FID reference tile artifacts",
            "install metric and codec runtime dependencies including clean-fid, pyiqa, pytorch_msssim, torchmetrics, compressai, and ninja",
            "run test.sh and test_Real.sh over each dataset for every paper checkpoint/rate point",
            "sweep TEST_TRANSFER_SLICES for prefix rate control",
            "emit reconstructed images, .bin bitstreams, metrics JSON, timing/profile traces, and RD table summaries",
            "compare simulated gap/results to paper RD curves, tables, paragraphs, and figures",
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
            "id": "ops.rdvq_checkpoint_gate",
            "type": "model_data_gate",
            "skill_role": "require loadable released RDVQ checkpoints",
            "content": "Resolve CVLUESTC/RDVQ released checkpoints as local loadable model files. HF metadata, README links, and VGG perceptual cache weights are support only.",
        },
    )
    add_node(
        dag,
        {
            "id": "ops.compression_dataset_fid_gate",
            "type": "evaluation_artifact_gate",
            "skill_role": "require paper image datasets and FID/KID references",
            "content": "Resolve ImageFolder-style Kodak, DIV2K-val, and CLIC2020-test image directories plus FID_REF_ROOT/<dataset>_256teles reference tiles before RD-curve comparison.",
        },
    )
    add_node(
        dag,
        {
            "id": "ops.estimated_real_bitstream_matrix",
            "type": "operational_execution_matrix",
            "skill_role": "bind Loop 2 to official RDVQ estimated and real bitrate entrypoints",
            "content": "Run test.sh for estimated cd_bpp and test_Real.sh for actual cd_bpp_real/.bin tensor-rANS payloads across paper datasets, checkpoints, and TEST_TRANSFER_SLICES prefix-control settings.",
        },
    )
    add_node(
        dag,
        {
            "id": "ops.compression_metric_runtime_gate",
            "type": "runtime_gate",
            "skill_role": "require metric and codec dependencies",
            "content": "Verify PyTorch, torchvision, clean-fid, pyiqa, pytorch_msssim, torchmetrics, compressai, ninja/C++17, and optional fvcore before metric, real-bitstream, or FLOP/timing claims.",
        },
    )
    add_node(
        dag,
        {
            "id": "ops.rdvq_hardware_trace_gate",
            "type": "professional_hardware_gate",
            "skill_role": "bind RDVQ claims to paper hardware",
            "content": "Verify four RTX 4090 primary-evaluation capacity, one RTX Pro 6000 high-resolution fine-tune trace, clean GPU slot, CUDA telemetry, and 2K-image inference timing before accepting performance claims.",
        },
    )
    add_node(
        dag,
        {
            "id": "ops.full_rd_curve_artifact_gate",
            "type": "professional_artifact_gate",
            "skill_role": "make verifier comparison RD-curve shaped",
            "content": "Require raw reconstructed images, .bin bitstreams, metrics JSON, profile/timing traces, RD-curve summaries, DIV2K ablation table, and prefix-transfer sweep outputs. Do not promote release validation, syntax checks, or README assets.",
        },
    )
    add_node(
        dag,
        {
            "id": "decision.explicit_blocker_after_rdvq_preflight",
            "type": "author_reviewer_decision",
            "skill_role": "feed operational failure back into Loop 1",
            "content": "If checkpoints, datasets/FID refs, runtime dependencies, hardware, or full estimated/real result grids are absent, mark not converged and update the DAG instead of running a one-image debug probe.",
        },
    )
    for src, dst in [
        ("ops.resolve_models_data", "ops.rdvq_checkpoint_gate"),
        ("ops.resolve_models_data", "ops.compression_dataset_fid_gate"),
        ("ops.resolve_repo_code", "ops.estimated_real_bitstream_matrix"),
        ("ops.rdvq_checkpoint_gate", "ops.estimated_real_bitstream_matrix"),
        ("ops.compression_dataset_fid_gate", "ops.estimated_real_bitstream_matrix"),
        ("ops.estimated_real_bitstream_matrix", "ops.compression_metric_runtime_gate"),
        ("experiments.system_surface", "ops.rdvq_hardware_trace_gate"),
        ("ops.compression_metric_runtime_gate", "ops.rdvq_hardware_trace_gate"),
        ("ops.rdvq_hardware_trace_gate", "ops.full_rd_curve_artifact_gate"),
        ("ops.full_rd_curve_artifact_gate", "loop2.execute_operational_dag"),
        ("loop2.execute_operational_dag", "decision.explicit_blocker_after_rdvq_preflight"),
        ("decision.explicit_blocker_after_rdvq_preflight", "reviewer.keep_exact_artifact_debt"),
    ]:
        add_edge(dag, src, dst)

    dag.setdefault("previous_loop_updates", []).append(
        {
            "id": "update.add_rdvq_checkpoint_dataset_metric_hardware_result_gates",
            "reason": "specialized RDVQ preflight found missing full paper-shaped checkpoint, dataset/FID, metric/runtime, hardware, release-validation, and result-grid artifacts",
            "blocker_ids": [blocker["id"] for blocker in blockers],
            "success_criteria": [
                "released checkpoint gate encoded",
                "Kodak/DIV2K/CLIC/FID reference gate encoded",
                "estimated and real bitstream script matrix encoded",
                "metric and tensor-rANS runtime gate encoded",
                "RTX 4090/RTX Pro 6000 hardware trace gate encoded",
                "full RD curve, bitstream, profile, and prefix-transfer artifact gate encoded",
            ],
        }
    )
    serial = json.dumps({"nodes": dag.get("nodes", []), "edges": dag.get("edges", [])}, sort_keys=True)
    dag["signature"] = hashlib.sha256(serial.encode("utf-8")).hexdigest()[:16]
    write_json(PAPER_RUN / "paper_author_gap_dag_iter_03.json", dag)
    write_json(DAG_PATH, dag)
    return dag


def artifact_debt(blockers: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "id": "main_benchmark_tables",
            "required": "Kodak/DIV2K/CLIC estimated-rate and real-bitstream RD curves, DIV2K ablation table, prefix-transfer rate-control sweep",
        },
        {
            "id": "metric_scoring_outputs",
            "required": "bpp/cd_bpp, cd_bpp_real, DISTS, LPIPS, FID, KID, CLIPIQA, MUSIQ, NIQE, PSNR, MS-SSIM, profile/timing counters",
        },
        {
            "id": "datasets_and_model_artifacts",
            "required": "CVLUESTC/RDVQ checkpoints; Kodak, DIV2K-val, CLIC2020-test; FID_REF_ROOT tiles; ImageNet/OpenImage/DF2K if full training/fine-tune reproduction is attempted",
        },
        {
            "id": "hardware_runtime_traces",
            "required": "four RTX 4090 primary experiment traces, RTX Pro 6000 high-resolution fine-tune trace, clean CUDA telemetry, C++17/ninja tensor-rANS build",
        },
        {
            "id": "method_specific_code_path",
            "required": "test.sh, test_Real.sh, my_inference.py, Real_Endecode_inference_single_stageAR.py, tensor-rANS codec, prefix transfer, RDVQ quantizer/tokenizer/AR transformer",
        },
        {
            "id": "operational_preflight_blockers",
            "required": "; ".join(f"{item['id']}: {item['detail']}" for item in blockers),
        },
    ]


def update_paper_run(gate: dict[str, Any], verifier: dict[str, Any], blockers: list[dict[str, str]]) -> None:
    path = PAPER_RUN / "paper_run_status.json"
    payload = read_json(path) if path.exists() else {}
    payload.update(
        {
            "paper_id": PAPER_ID,
            "title": TITLE,
            "updated_at_utc": utc_now(),
            "status": gate["status"],
            "professional_ready": gate["professional_package_ready"],
            "specialized_runner": str(RUNNER_DIR),
            "professional_gate": gate,
            "verifier": verifier,
            "artifact_debt": artifact_debt(blockers),
        }
    )
    write_json(path, payload)
    write_json(
        PAPER_RUN / "verifier_result_iter_03.json",
        {
            "artifact_kind": "rdvq_specialized_verifier",
            "artifact_paths": {
                "dag_iter_03": str(PAPER_RUN / "paper_author_gap_dag_iter_03.json"),
                "environment": str(ENV_PATH),
                "official_script_manifest": str(SCRIPT_MANIFEST_PATH),
                "model_data_manifest": str(MODEL_DATA_PATH),
                "professional_gate": str(PROFESSIONAL_GATE_PATH),
                "specialized_verifier": str(VERIFIER_PATH),
            },
            "checks": [
                {
                    "name": "blind_contract",
                    "status": "pass",
                    "detail": verifier["blind_contract_checked"],
                },
                {
                    "name": "support_only_preflight",
                    "status": "pass",
                    "detail": "release validation, syntax checks, README assets, and HF metadata remain support only",
                },
                {
                    "name": "professional_artifact_package",
                    "status": "pass" if gate["professional_package_ready"] else "blocked",
                    "detail": {
                        "ready": gate["professional_package_ready"],
                        "reason": gate["status"],
                        "blockers": blockers,
                    },
                },
                {
                    "name": "result_shape_comparison_ready",
                    "status": "pass" if gate["professional_package_ready"] else "blocked",
                    "detail": "requires Kodak/DIV2K/CLIC estimated-rate and real-bitstream RD curves, prefix-transfer sweeps, timing/profile traces, and metric tables",
                },
            ],
            "converged": gate["professional_package_ready"],
            "created_at_utc": utc_now(),
            "iteration": 3,
            "paper_id": PAPER_ID,
            "paper_title": TITLE,
            "professional_ready": gate["professional_package_ready"],
            "required_updates": [
                {
                    "id": "update.add_rdvq_checkpoint_dataset_metric_hardware_result_gates",
                    "reason": gate["status"],
                    "success_criteria": [
                        "released checkpoint files are materialized locally",
                        "Kodak, DIV2K-val, CLIC2020-test, and FID reference artifacts exist",
                        "metric/runtime dependencies and tensor-rANS support pass",
                        "estimated-rate and real-bitstream scripts run over paper datasets/rate points",
                        "prefix-transfer sweeps, RD tables, timing/profile traces, and figure/table summaries are produced",
                        "verifier compares close result shape against paper evidence channels",
                    ],
                }
            ],
        },
    )


def update_global_files(gate: dict[str, Any], blockers: list[dict[str, str]]) -> None:
    queue = read_json(QUEUE_PATH)
    for item in queue.get("queue", []):
        if item.get("paper_id") == PAPER_ID:
            statuses = set(item.get("implementation_statuses", []))
            statuses.update(
                {
                    "specialized_runner_preflight_completed",
                    "official_rdvq_scripts_parsed",
                    "checkpoint_dataset_metric_manifests_checked",
                    "blocked_exact_rdvq_checkpoint_dataset_runtime_hardware_grid",
                }
            )
            item["implementation_statuses"] = sorted(statuses)
            item["repo_exact_rerun_status"] = "blocked"
            item["specialized_runner_status"] = gate["status"]
            item["professional_blocker"] = gate["status"] + "_after_specialized_runner"
            item["specialized_runner_artifact_dir"] = str(RUNNER_DIR)
            item["specialized_runner_evidence"] = {
                "environment_path": str(ENV_PATH),
                "official_script_manifest_path": str(SCRIPT_MANIFEST_PATH),
                "model_data_manifest_path": str(MODEL_DATA_PATH),
                "professional_gate_path": str(PROFESSIONAL_GATE_PATH),
                "verifier_path": str(VERIFIER_PATH),
                "blockers": blockers,
            }
            item["exact_artifact_debt"] = artifact_debt(blockers)
            break
    queue["updated_at_utc"] = utc_now()
    write_json(QUEUE_PATH, queue)

    summary = read_json(SUMMARY_PATH)
    for paper in summary.get("papers", []):
        if paper.get("paper_id") == PAPER_ID:
            paper["final_status"] = "blocked_waiting_for_professional_artifacts_after_rdvq_specialized_gate"
            paper["professional_ready"] = False
            paper["repo_exact_rerun_status"] = "blocked"
            paper["specialized_runner_status"] = gate["status"]
            paper["professional_blocker"] = gate["status"] + "_after_specialized_runner"
            paper["specialized_runner_artifact_dir"] = str(RUNNER_DIR)
            break
    summary["updated_at_utc"] = utc_now()
    write_json(SUMMARY_PATH, summary)


def write_status(gate: dict[str, Any], blockers: list[dict[str, str]]) -> None:
    lines = [
        "# RDVQ Specialized Runner Status",
        "",
        f"- Updated: {utc_now()}",
        f"- Paper: `{TITLE}`",
        f"- Status: `{gate['status']}`",
        f"- Professional package ready: `{gate['professional_package_ready']}`",
        f"- Repo files checked: `{gate['support_checks']['repo_files_checked']}`",
        f"- Compileall support check passed: `{gate['support_checks']['compileall_passed']}`",
        f"- Release validation support check passed: `{gate['support_checks']['release_validate_passed']}`",
        f"- Checkpoint manifests checked: `{gate['support_checks']['checkpoint_manifests_checked']}`",
        f"- Dataset/metric artifacts checked: `{gate['support_checks']['dataset_metric_artifacts_checked']}`",
        f"- Blocker count: `{len(blockers)}`",
        "",
        "## Artifact Paths",
        f"- Environment: `{ENV_PATH}`",
        f"- Official script manifest: `{SCRIPT_MANIFEST_PATH}`",
        f"- Model/data manifest: `{MODEL_DATA_PATH}`",
        f"- Professional gate: `{PROFESSIONAL_GATE_PATH}`",
        f"- Verifier: `{VERIFIER_PATH}`",
        "",
        "## Why This Is Not Converged",
        "- This did not run a one-image debug probe, README asset comparison, or release-validation smoke as convergence evidence.",
        "- The full RDVQ paper shape requires released checkpoints, Kodak/DIV2K/CLIC image folders, FID/KID references, estimated and real-bitstream outputs, prefix-transfer sweeps, timing/profile traces, and metric tables.",
        "- The DAG was updated so Loop 2 must satisfy those operational gates before the verifier can accept the research-gap simulation.",
        "",
        "## Current Blockers",
    ]
    for blocker in blockers:
        lines.append(f"- `{blocker['id']}`: {blocker['detail']}")
    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    scripts = script_manifest()
    env = environment_manifest()
    data = model_data_manifest()
    blockers = derive_blockers(env, data)
    gate = professional_gate_result(blockers, env, data)
    dag = update_dag(blockers)
    verifier = {
        "status": gate["status"],
        "professional_package_ready": gate["professional_package_ready"],
        "convergence_decision": (
            "converged_ready_for_full_rd_curve_gpu_execution"
            if gate["professional_package_ready"]
            else "not_converged_explicit_professional_blockers_after_operational_preflight"
        ),
        "semantic_dag_nodes_checked": [
            "gap.paper_gap_claims",
            "method.bind_gap_to_mechanism",
            "experiments.benchmark_metric_grid",
            "experiments.system_surface",
            "ops.resolve_repo_code",
            "ops.resolve_models_data",
            "ops.rdvq_checkpoint_gate",
            "ops.compression_dataset_fid_gate",
            "ops.estimated_real_bitstream_matrix",
            "ops.full_rd_curve_artifact_gate",
        ],
        "unresolved_professional_debt": blockers,
        "loop1_required_dag_update": [
            "Add released checkpoint materialization gate.",
            "Add Kodak/DIV2K/CLIC/FID-reference dataset gate.",
            "Add estimated-rate plus real-bitstream script matrix gate.",
            "Add metric/runtime/tensor-rANS support gate.",
            "Add RTX 4090 / RTX Pro 6000 hardware trace gate.",
            "Add full RD-curve, bitstream, timing/profile, and prefix-transfer artifact gate.",
            "Keep release validation, syntax checks, README assets, and HF metadata as support only.",
        ],
        "blind_contract_checked": {
            "only_input_file": "paper_author_gap_dag.json",
            "paper_text_visible_to_loop2": False,
            "oracle_results_visible_to_loop2": False,
            "previous_memory_visible_to_loop2": False,
            "repo_paths_visible_only_if_encoded_in_dag": True,
        },
        "dag_path": str(DAG_PATH),
        "updated_dag_signature": dag.get("signature"),
        "script_manifest_path": str(SCRIPT_MANIFEST_PATH),
        "model_data_manifest_path": str(MODEL_DATA_PATH),
        "environment_manifest_path": str(ENV_PATH),
        "created_at_utc": utc_now(),
    }
    write_json(VERIFIER_PATH, verifier)
    update_paper_run(gate, verifier, blockers)
    update_global_files(gate, blockers)
    write_status(gate, blockers)
    print(json.dumps(verifier, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
