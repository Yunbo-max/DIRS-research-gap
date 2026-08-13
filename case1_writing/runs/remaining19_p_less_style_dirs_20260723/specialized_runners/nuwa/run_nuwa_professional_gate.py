#!/usr/bin/env python3
"""NuWa professional operational gate for the strict DIRS loop.

This turns the NuWa gap DAG into concrete author-simulation requirements:
resolve the official repo, validate the method code, materialize class-specific
datasets/checkpoints, run the full pruning/evaluation grid, and emit
verifier-comparable accuracy/compute/runtime/cost artifacts. Syntax/import
checks and repo inventory are support only; they never converge the paper.
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
PAPER_RUN = RUN_ROOT / "paper_runs" / "cvpr2026_016_nuwa_class_specific_vit_pruning"
RUNNER_DIR = RUN_ROOT / "specialized_runners" / "nuwa"
REPO = Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/016_nuwa/NuWa")

PAPER_ID = "CVPR2026_016_nuwa_class_specific_vit_pruning"
TITLE = "NuWa: Deriving Lightweight Class-Specific Vision Transformers for Edge Devices"

DAG_PATH = PAPER_RUN / "paper_author_gap_dag.json"
QUEUE_PATH = RUN_ROOT / "specialized_runner_queue.json"
SUMMARY_PATH = RUN_ROOT / "remaining19_strict_dirs_summary.json"
LONG_STATUS_PATH = RUN_ROOT / "LONGGOAL_STATUS.md"
SPECIALIZED_QUEUE_MD = RUN_ROOT / "SPECIALIZED_RUNNER_QUEUE.md"

STATUS_PATH = RUNNER_DIR / "NUWA_SPECIALIZED_STATUS.md"
VERIFIER_PATH = RUNNER_DIR / "nuwa_specialized_verifier.json"
PROFESSIONAL_GATE_PATH = RUNNER_DIR / "professional_gate_result.json"
ENV_PATH = RUNNER_DIR / "environment.json"
SCRIPT_MANIFEST_PATH = RUNNER_DIR / "official_script_manifest.json"
MODEL_DATA_PATH = RUNNER_DIR / "model_data_manifest.json"

SCRIPT_FILES = [
    "README.md",
    "main.py",
    "method/nuwa.py",
    "method/get_anchor_model.py",
    "method/get_clibration_data.py",
    "method/pruning.py",
    "method/utils.py",
    "dataset/utils.py",
    "dataset/imagenet.py",
    "dataset/cifar10.py",
    "dataset/cifar100.py",
    "dataset/coco.py",
    "model/utils.py",
    "model/vit.py",
    "model/swin.py",
    "engine/eval.py",
    "engine/train.py",
    "engine/utils.py",
    "utils/utils.py",
]

EXPECTED_DATASETS = [
    {
        "id": "imagenet_1k_train_val",
        "hints": ["imagenet", "ILSVRC2012", "ImageNet"],
        "repo_expected_roots": ["/root/autodl-pvt/nuwa/data/dataset/imagenet"],
        "required": "ImageNet-1K train/val with class-specific subtask labels",
    },
    {
        "id": "cifar10",
        "hints": ["CIFAR10", "cifar10"],
        "repo_expected_roots": ["/root/autodl-pvt/nuwa/data/dataset/CIFAR10"],
        "required": "CIFAR-10 class-specific derivation benchmark",
    },
    {
        "id": "cifar100",
        "hints": ["CIFAR100", "cifar100"],
        "repo_expected_roots": ["/root/autodl-pvt/nuwa/data/dataset/CIFAR100"],
        "required": "CIFAR-100 class-specific derivation benchmark",
    },
    {
        "id": "coco2017_detection_segmentation",
        "hints": ["coco", "COCO2017", "annotations"],
        "repo_expected_roots": ["/root/autodl-pub/datasets/coco"],
        "required": "COCO detection/segmentation transfer benchmark",
    },
    {
        "id": "class_specific_subtasks",
        "hints": ["sub_task", "T1-10", "subtask"],
        "repo_expected_roots": ["/root/autodl-pvt/nuwa/data/sub_task"],
        "required": "task files mapping ImageNet/CIFAR/COCO classes to edge-device subtasks",
    },
    {
        "id": "calibration_features",
        "hints": ["activation", "feature", "calibration", "clibration"],
        "repo_expected_roots": ["/root/autodl-pvt/nuwa/data/activation", "/root/autodl-pvt/nuwa/data/feature"],
        "required": "activation/calibration feature tensors used by self-knowledge purification and SVD pruning",
    },
]

EXPECTED_CHECKPOINTS = [
    {
        "id": "timm_deit_pretrained_weights",
        "hints": ["deit_base_patch16_224", "deit_small_patch16_224", "deit_tiny_patch16_224"],
        "repo_expected_paths": [],
        "required": "timm pretrained DeiT weights for ImageNet experiments",
    },
    {
        "id": "vit_large_pretrained_weights",
        "hints": ["vit_large_patch16_224", "vit_large"],
        "repo_expected_paths": [],
        "required": "ViT-Large pretrained weights for scaling/ablation if paper grid requires it",
    },
    {
        "id": "cifar_finetuned_vit_weights",
        "hints": ["deit_base(cifar10).pt", "deit_small(cifar10).pt", "deit_base(cifar100).pt", "vit_large(cifar100).pt"],
        "repo_expected_paths": [
            "/root/autodl-pvt/nuwa/data/param/cifar10",
            "/root/autodl-pvt/nuwa/data/param/cifar100",
        ],
        "required": "CIFAR-specific fine-tuned checkpoints loaded by model/utils.py",
    },
    {
        "id": "mask_rcnn_swin_coco_checkpoint",
        "hints": ["mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth"],
        "repo_expected_paths": [
            "/root/autodl-pvt/nuwa/model/weights/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth"
        ],
        "required": "Mask-RCNN Swin-T COCO checkpoint for detection/segmentation transfer",
    },
    {
        "id": "mmdetection_swin_config",
        "hints": ["mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py"],
        "repo_expected_paths": ["/root/autodl-pvt/nuwa/mmdetection/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py"],
        "required": "mmdetection config used by model/swin.py",
    },
]

EXPECTED_OUTPUT_SURFACES = [
    "class-specific ImageNet accuracy/pruning-rate/GFLOPs tables",
    "training-free versus training-dependent pruning comparisons",
    "Magnitude/Wanda/Numerical/X-Pruner/DC-ViT/RECAP/MDP baseline comparison",
    "derivation-cost table with GPU hours/AWS cost simulation",
    "edge-device latency/throughput/memory table on RTX 4090 and Jetson Orin NX",
    "component ablation for self-knowledge purification, MHA SVD, MLP closed-form pruning, head focus, target architecture",
    "class-specific accuracy-pruning sweeps",
    "large deployment cost simulation",
    "COCO detection/segmentation transfer table with mAP",
    "raw stdout, JSON metrics, checkpoints/masks, timing traces, memory traces, and device traces",
]

SEARCH_ROOTS = [
    REPO,
    Path("/root/autodl-pvt/nuwa"),
    Path("/root/autodl-pub/datasets"),
    Path("/tf/notebooks/data"),
    Path("/tf/notebooks/datasets"),
    Path("/tf/notebooks/checkpoints"),
    Path("/tf/notebooks/models"),
    Path(os.path.expanduser("~/.cache/torch")),
    Path(os.path.expanduser("~/.cache/huggingface")),
    Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/supplemental/CVPR2026_016_nuwa"),
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


def local_matches(hints: list[str], explicit_paths: list[str] | None = None) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for explicit in explicit_paths or []:
        p = Path(explicit)
        if p.exists():
            seen.add(str(p.resolve()))
            size_human, file_count = path_size(p)
            matches.append({"path": str(p), "is_dir": p.is_dir(), "file_count": file_count, "size_human": size_human})
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for hint in hints:
            candidates = []
            exact = root / hint
            if exact.exists():
                candidates.append(exact)
            for pattern in {f"*{hint}*", f"**/*{hint}*"}:
                try:
                    candidates.extend(list(root.glob(pattern))[:40])
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
                matches.append({"path": str(candidate), "is_dir": candidate.is_dir(), "file_count": file_count, "size_human": size_human})
    return matches[:80]


def parse_python(path: Path) -> dict[str, Any]:
    text = read_text(path)
    syntax = run_cmd([sys.executable, "-m", "py_compile", str(path)], timeout=45)
    return {
        "path": str(path),
        "exists": path.exists(),
        "line_count": len(text.splitlines()),
        "syntax_ok": syntax["returncode"] == 0,
        "syntax_stderr": syntax["stderr"],
        "cli_flags": sorted(set(re.findall(r"['\"](--[A-Za-z0-9_-]+)['\"]", text))),
        "imports": sorted(set(re.findall(r"^(?:from|import)\s+([A-Za-z0-9_\.]+)", text, re.M))),
        "hardcoded_paths": sorted(set(re.findall(r"/(?:root|share|tf)/[^'\"\s),]+", text))),
        "torch_loads": sorted(set(re.findall(r"torch\.load\(([^)\n]+)", text))),
        "metric_mentions": sorted(
            term
            for term in ["accuracy", "gflops", "mparam", "latency", "throughput", "memory", "mAP", "bbox_mAP", "segm_mAP"]
            if term.lower() in text.lower()
        ),
    }


def git_info() -> dict[str, Any]:
    safe = str(REPO)
    return {
        "repo": str(REPO),
        "exists": REPO.exists(),
        "remote": run_cmd(["git", "-C", safe, "-c", f"safe.directory={safe}", "remote", "-v"], timeout=30),
        "head": run_cmd(["git", "-C", safe, "-c", f"safe.directory={safe}", "rev-parse", "HEAD"], timeout=30),
    }


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
        path = REPO / rel
        row: dict[str, Any] = {"relative_path": rel, "path": str(path), "exists": path.exists()}
        if path.exists() and path.is_file():
            row["size_bytes"] = path.stat().st_size
            if path.suffix == ".py":
                row["parsed"] = parse_python(path)
            elif path.name == "README.md":
                text = read_text(path)
                row["parsed"] = {
                    "line_count": len(text.splitlines()),
                    "content": text,
                    "minimal": len(text.splitlines()) <= 5,
                }
        files.append(row)
    payload = {
        "artifact_kind": "nuwa_official_script_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "repo": str(REPO),
        "git": git_info(),
        "files": files,
        "paper_shaped_execution_matrix": {
            "entrypoint": "python main.py with model_name/dataset_name/task_name/task_type/pruning_rate/output_dir/baseline_dir",
            "required_model_families": ["DeiT Base/Small/Tiny", "ViT Large", "Mask-RCNN Swin-T"],
            "required_datasets": [item["id"] for item in EXPECTED_DATASETS],
            "required_outputs": EXPECTED_OUTPUT_SURFACES,
            "verifier_comparison": [
                "compare top-1 accuracy/mAP deltas, pruning rate, GFLOPs, params, latency, throughput, memory, GPU hours, and AWS cost against paper tables/figures/paragraphs",
                "accept only close paper-shaped result surfaces; syntax or dry-run support cannot converge",
            ],
        },
        "support_only_findings": [
            "README is a minimal title-only file and does not document the full benchmark grid",
            "repo inventory and compile/import checks are support only",
            "CIFAR auto-download would still be reduced if used alone; ImageNet/COCO/class-specific task grid is required",
        ],
    }
    write_json(SCRIPT_MANIFEST_PATH, payload)
    return payload


def environment_manifest() -> dict[str, Any]:
    packages = {
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "timm": package_version("timm"),
        "numpy": package_version("numpy"),
        "tqdm": package_version("tqdm"),
        "wandb": package_version("wandb"),
        "transformers": package_version("transformers"),
        "mmcv": package_version("mmcv"),
        "mmdet": package_version("mmdet"),
        "mmengine": package_version("mmengine"),
        "fvcore": package_version("fvcore"),
        "ptflops": package_version("ptflops"),
        "pycocotools": package_version("pycocotools"),
    }
    payload = {
        "artifact_kind": "nuwa_environment_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "python": sys.version,
        "gpu_rows": gpu_rows(),
        "python_processes": run_cmd(
            ["bash", "-lc", "ps -eo pid,etime,stat,cmd | rg 'prophet_custom_full_gsm8k_runner|nuwa|NuWa|python' || true"],
            timeout=30,
        ),
        "packages": packages,
        "import_probes": {
            "torch": import_probe("torch"),
            "torchvision": import_probe("torchvision"),
            "timm": import_probe("timm"),
            "numpy": import_probe("numpy"),
            "tqdm": import_probe("tqdm"),
            "wandb": import_probe("wandb"),
            "transformers": import_probe("transformers"),
            "mmcv": import_probe("mmcv"),
            "mmdet": import_probe("mmdet"),
            "mmengine": import_probe("mmengine"),
            "fvcore": import_probe("fvcore"),
            "ptflops": import_probe("ptflops"),
            "pycocotools": import_probe("pycocotools"),
        },
        "compileall": {
            "repo": run_cmd([sys.executable, "-m", "compileall", "-q", str(REPO)], timeout=180),
        },
        "professional_hardware_expected_by_dag": [
            "NVIDIA RTX 4090 runtime traces",
            "NVIDIA Jetson Orin NX real-device profiling",
            "CUDA GPU for full ImageNet/CIFAR/COCO pruning/evaluation grid",
        ],
    }
    write_json(ENV_PATH, payload)
    return payload


def model_data_manifest() -> dict[str, Any]:
    datasets = []
    for item in EXPECTED_DATASETS:
        explicit = item.get("repo_expected_roots", [])
        matches = local_matches(item["hints"], explicit)
        strong = [m for m in matches if (m["is_dir"] and m["file_count"] >= 100) or (not m["is_dir"] and m["file_count"] == 1)]
        row = dict(item)
        row["local_matches"] = strong[:20]
        row["ignored_local_matches"] = [m for m in matches if m not in strong][:20]
        row["materialized_locally"] = bool(strong)
        datasets.append(row)

    checkpoints = []
    for item in EXPECTED_CHECKPOINTS:
        explicit = item.get("repo_expected_paths", [])
        matches = local_matches(item["hints"], explicit)
        strong = [m for m in matches if not m["is_dir"] or m["file_count"] >= 1]
        row = dict(item)
        row["local_matches"] = strong[:20]
        row["ignored_local_matches"] = [m for m in matches if m not in strong][:20]
        row["materialized_locally"] = bool(strong)
        checkpoints.append(row)

    supplemental = []
    for p in [
        Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/pdf/CVPR2026_016_nuwa_supplemental.zip"),
        Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/pdf/CVPR2026_016_nuwa_arxiv.pdf"),
        Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/pdf/CVPR2026_016_nuwa_cvf.pdf"),
    ]:
        size_human, file_count = path_size(p)
        supplemental.append({"path": str(p), "exists": p.exists(), "size_human": size_human, "file_count": file_count})

    result_candidates = []
    for root in [REPO, RUNNER_DIR, Path("/tf/notebooks/results"), Path("/tf/notebooks/outputs")]:
        if not root.exists():
            continue
        for pattern in ["**/*result*.json", "**/*metric*.json", "**/*performance*.txt", "**/*mask*.pt", "**/*cluster*.pt"]:
            try:
                for candidate in list(root.glob(pattern))[:80]:
                    size_human, file_count = path_size(candidate)
                    result_candidates.append(
                        {"path": str(candidate), "is_dir": candidate.is_dir(), "file_count": file_count, "size_human": size_human}
                    )
            except OSError:
                pass

    payload = {
        "artifact_kind": "nuwa_model_data_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "datasets": datasets,
        "checkpoints": checkpoints,
        "supplemental_oracle_side_artifacts": supplemental,
        "verifier_comparable_result_candidates": result_candidates[:120],
        "paper_shaped_outputs_required": EXPECTED_OUTPUT_SURFACES,
    }
    write_json(MODEL_DATA_PATH, payload)
    return payload


def status_from_artifacts(env: dict[str, Any], manifest: dict[str, Any], model_data: dict[str, Any]) -> dict[str, Any]:
    blockers = []
    gpu = env["gpu_rows"]
    gpu_names = " | ".join(row["name"] for row in gpu)
    clean_4090 = [
        row for row in gpu
        if "RTX 4090" in row["name"] and row["memory_free_mib"] >= 12000 and row["utilization_gpu_pct"] < 30
    ]
    if not clean_4090:
        blockers.append(
            {
                "id": "clean_rtx4090_for_runtime_trace_missing",
                "status": "blocked",
                "detail": f"NuWa needs RTX 4090 runtime traces; no clean 4090 has >=12GB free and <30% util. GPUs: {gpu}.",
            }
        )
    if "Jetson" not in gpu_names and "Orin" not in gpu_names:
        blockers.append(
            {
                "id": "jetson_orin_nx_device_missing",
                "status": "blocked",
                "detail": f"Paper/DAG expects Jetson Orin NX edge profiling; visible devices are {gpu_names}.",
            }
        )

    syntax_bad = [
        f["relative_path"]
        for f in manifest["files"]
        if f.get("parsed", {}).get("syntax_ok") is False
    ]
    if syntax_bad:
        blockers.append(
            {
                "id": "official_repo_python_syntax_errors",
                "status": "blocked",
                "detail": "Official NuWa repo has syntax errors before execution: " + ", ".join(syntax_bad),
            }
        )

    missing_imports = [
        name for name, result in env["import_probes"].items()
        if result["returncode"] != 0 and name in {"mmcv", "mmdet", "mmengine", "fvcore", "ptflops", "pycocotools"}
    ]
    if missing_imports:
        blockers.append(
            {
                "id": "nuwa_runtime_dependencies_missing",
                "status": "blocked",
                "detail": "Missing imports needed for full recognition/detection/segmentation/runtime path: " + ", ".join(missing_imports),
            }
        )

    missing_datasets = [d["id"] for d in model_data["datasets"] if not d["materialized_locally"]]
    if missing_datasets:
        blockers.append(
            {
                "id": "nuwa_datasets_and_task_artifacts_missing",
                "status": "blocked",
                "detail": "Missing local dataset/task/calibration artifacts: " + ", ".join(missing_datasets),
            }
        )

    missing_checkpoints = [c["id"] for c in model_data["checkpoints"] if not c["materialized_locally"]]
    if missing_checkpoints:
        blockers.append(
            {
                "id": "nuwa_checkpoints_or_configs_missing",
                "status": "blocked",
                "detail": "Missing local model/checkpoint/config artifacts: " + ", ".join(missing_checkpoints),
            }
        )

    result_candidates = model_data["verifier_comparable_result_candidates"]
    if len(result_candidates) < 8:
        blockers.append(
            {
                "id": "full_pruning_benchmark_grid_missing",
                "status": "blocked",
                "detail": "No complete verifier-comparable NuWa tables/figures/raw outputs were found for accuracy/pruning/compute/runtime/cost surfaces.",
            }
        )

    status = (
        "ready_for_full_paper_grid_execution_not_converged"
        if not blockers
        else "blocked_by_syntax_datasets_checkpoints_runtime_hardware_and_result_grid_requirements"
    )
    payload = {
        "artifact_kind": "nuwa_professional_gate_result",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "status": status,
        "professional_package_ready": not blockers,
        "convergence_role": "professional operational gate; no reduced run is promoted",
        "blockers": blockers,
        "support_checks": {
            "repo_discovered_and_encoded": REPO.exists(),
            "script_files_checked": len(manifest["files"]),
            "compileall_repo_passed": env["compileall"]["repo"]["returncode"] == 0,
            "datasets_checked": len(model_data["datasets"]),
            "checkpoints_checked": len(model_data["checkpoints"]),
        },
        "next_full_execution_if_unblocked": [
            "fix or replace the official source snapshot so every NuWa method file compiles",
            "install mmcv/mmdet/mmengine/fvcore/ptflops/pycocotools runtime stack",
            "materialize ImageNet-1K, CIFAR-10, CIFAR-100, COCO2017, class-specific task files, and calibration tensors",
            "materialize timm pretrained weights, CIFAR fine-tuned checkpoints, Mask-RCNN Swin checkpoint/config",
            "run python main.py across model/dataset/task/pruning-rate grid including NuWa and baselines",
            "record top-1/mAP, pruning rate, GFLOPs, params, latency, throughput, memory, GPU hours, AWS cost, RTX 4090 trace, Jetson Orin NX trace",
            "compare result shapes to paper tables, figures, and paragraph claims before convergence",
        ],
    }
    write_json(PROFESSIONAL_GATE_PATH, payload)
    return payload


def ensure_node(dag: dict[str, Any], node: dict[str, Any]) -> None:
    nodes = dag.setdefault("nodes", [])
    for existing in nodes:
        if existing.get("id") == node["id"]:
            existing.update(node)
            return
    nodes.append(node)


def ensure_edge(dag: dict[str, Any], source: str, target: str) -> None:
    edges = dag.setdefault("edges", [])
    edge = [source, target]
    if edge not in edges:
        edges.append(edge)


def update_dag(gate: dict[str, Any]) -> dict[str, Any]:
    dag = read_json(DAG_PATH)
    for node in dag.get("nodes", []):
        if node.get("id") == "ops.resolve_repo_code":
            node["repo_paths"] = [str(REPO)]
            node["content"] = (
                "repos=/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/016_nuwa/NuWa; "
                "code_artifacts=main_py; method_nuwa_py; method_get_anchor_model_py; "
                "method_pruning_py; method_get_clibration_data_py; dataset_imagenet_py; "
                "dataset_coco_py; model_vit_py; model_swin_py"
            )

    new_nodes = [
        (
            "ops.nuwa_repo_path_resolution_gate",
            "Resolve the official NuWa repo path and parse main.py/method/dataset/model/engine entrypoints before Loop 2 execution.",
            "operational_dependency",
        ),
        (
            "ops.nuwa_source_compile_gate",
            "Require every official NuWa method file to compile; current snapshot has f-string syntax errors in get_anchor_model.py, get_clibration_data.py, and method/utils.py.",
            "operational_dependency",
        ),
        (
            "ops.nuwa_runtime_dependency_gate",
            "Require mmcv, mmdet, mmengine, fvcore, ptflops, pycocotools, timm, torch, and torchvision for full recognition plus COCO transfer.",
            "operational_dependency",
        ),
        (
            "ops.nuwa_dataset_task_calibration_gate",
            "Materialize ImageNet-1K, CIFAR-10/100, COCO2017, class-specific sub_task files, and activation/calibration feature tensors.",
            "operational_dependency",
        ),
        (
            "ops.nuwa_checkpoint_config_gate",
            "Materialize timm DeiT/ViT weights, CIFAR fine-tuned checkpoints, Mask-RCNN Swin-T checkpoint, and mmdetection config.",
            "operational_dependency",
        ),
        (
            "ops.nuwa_full_pruning_runtime_matrix",
            "Run NuWa and baselines across model/dataset/task/pruning-rate grid and record accuracy/mAP, pruning rate, GFLOPs, params, latency, throughput, memory, GPU hours, AWS cost, RTX 4090 and Jetson Orin NX traces.",
            "operational_execution",
        ),
        (
            "decision.explicit_blocker_after_nuwa_preflight",
            "If any source, dependency, dataset, checkpoint, device, or result-grid gate is missing, block and feed exact requirements back into Loop 1; never converge from syntax/import/repo evidence.",
            "author_reviewer_decision",
        ),
    ]
    for node_id, content, typ in new_nodes:
        ensure_node(dag, {"id": node_id, "content": content, "type": typ, "skill_role": "paper-specific operational gate"})

    for source, target in [
        ("ops.resolve_repo_code", "ops.nuwa_repo_path_resolution_gate"),
        ("ops.nuwa_repo_path_resolution_gate", "ops.nuwa_source_compile_gate"),
        ("ops.nuwa_source_compile_gate", "ops.nuwa_runtime_dependency_gate"),
        ("ops.nuwa_runtime_dependency_gate", "ops.nuwa_dataset_task_calibration_gate"),
        ("ops.nuwa_dataset_task_calibration_gate", "ops.nuwa_checkpoint_config_gate"),
        ("ops.nuwa_checkpoint_config_gate", "ops.nuwa_full_pruning_runtime_matrix"),
        ("ops.nuwa_full_pruning_runtime_matrix", "reviewer.require_professional_artifact_package"),
        ("ops.nuwa_full_pruning_runtime_matrix", "reviewer.compare_result_shapes"),
        ("reviewer.keep_exact_artifact_debt", "decision.explicit_blocker_after_nuwa_preflight"),
    ]:
        ensure_edge(dag, source, target)

    dag.setdefault("previous_loop_updates", []).append(
        {
            "iteration": 3,
            "created_at_utc": utc_now(),
            "source": "nuwa_specialized_professional_gate",
            "status": gate["status"],
            "blocker_ids": [b["id"] for b in gate["blockers"]],
            "repo_paths": [str(REPO)],
            "converged": False,
        }
    )
    signature_src = json.dumps(dag.get("nodes", []), sort_keys=True) + json.dumps(dag.get("edges", []), sort_keys=True)
    dag["signature"] = hashlib.sha256(signature_src.encode("utf-8")).hexdigest()[:16]
    write_json(DAG_PATH, dag)
    write_json(PAPER_RUN / "paper_author_gap_dag_iter_03.json", dag)
    return dag


def specialized_verifier(gate: dict[str, Any], dag: dict[str, Any]) -> dict[str, Any]:
    checks = [
        {"name": "blind_contract", "status": "pass", "detail": dag.get("blind_contract", {})},
        {
            "name": "repo_path_encoded",
            "status": "pass" if str(REPO) in json.dumps(dag) else "fail",
            "detail": [str(REPO)],
        },
        {
            "name": "reduced_proxy_rejection_gate",
            "status": "pass",
            "detail": "repo/syntax/import/data checks are support only and cannot converge",
        },
        {
            "name": "professional_artifact_package",
            "status": "pass" if gate["professional_package_ready"] else "blocked",
            "detail": {
                "ready": gate["professional_package_ready"],
                "reason": gate["status"],
                "blockers": gate["blockers"],
            },
        },
        {
            "name": "result_shape_comparison_ready",
            "status": "blocked",
            "detail": "requires full NuWa pruning/runtime/cost grid outputs before comparing to paper evidence channels",
        },
    ]
    payload = {
        "artifact_kind": "nuwa_specialized_verifier",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "iteration": 3,
        "converged": False,
        "professional_ready": gate["professional_package_ready"],
        "checks": checks,
        "required_updates": [
            {
                "id": "update.nuwa_operational_gates_from_preflight",
                "reason": gate["status"],
                "success_criteria": [
                    "repo path encoded in DAG",
                    "syntax/source gate present",
                    "runtime dependency gate present",
                    "dataset/task/calibration gate present",
                    "checkpoint/config gate present",
                    "full pruning/runtime/cost result grid gate present",
                ],
            }
        ],
        "artifact_paths": {
            "professional_gate": str(PROFESSIONAL_GATE_PATH),
            "environment": str(ENV_PATH),
            "script_manifest": str(SCRIPT_MANIFEST_PATH),
            "model_data_manifest": str(MODEL_DATA_PATH),
            "dag_iter_03": str(PAPER_RUN / "paper_author_gap_dag_iter_03.json"),
        },
    }
    write_json(VERIFIER_PATH, payload)
    write_json(PAPER_RUN / "verifier_result_iter_03.json", payload)
    return payload


def update_queue_and_summary(gate: dict[str, Any], verifier: dict[str, Any], dag: dict[str, Any]) -> None:
    queue_obj = read_json(QUEUE_PATH)
    items = queue_obj.get("queue", queue_obj if isinstance(queue_obj, list) else [])
    for item in items:
        if item.get("paper_id") == PAPER_ID:
            item["priority"] = "high"
            item["professional_blocker"] = gate["status"]
            item["repo_exact_rerun_status"] = "code_present_blocked_by_source_runtime_data_devices"
            item["repo_paths"] = [str(REPO)]
            item["specialized_runner_status"] = gate["status"]
            item["specialized_runner_artifact_dir"] = str(RUNNER_DIR)
            statuses = item.setdefault("implementation_statuses", [])
            for status in [
                "official_repo_discovered_and_encoded",
                "official_scripts_parsed",
                "source_compile_preflight_completed",
                "blocked_by_source_runtime_datasets_checkpoints_and_devices",
            ]:
                if status not in statuses:
                    statuses.append(status)
            item["specialized_runner_evidence"] = {
                "blockers": gate["blockers"],
                "verifier_path": str(VERIFIER_PATH),
                "environment_path": str(ENV_PATH),
                "official_script_manifest_path": str(SCRIPT_MANIFEST_PATH),
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
            for status in [
                "official_repo_discovered_and_encoded",
                "official_scripts_parsed",
                "source_compile_preflight_completed",
                "blocked_by_source_runtime_datasets_checkpoints_and_devices",
            ]:
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
                        "raw_artifact_level": "repo_source_dependency_data_device_preflight_only",
                        "blocker_ids": [b["id"] for b in gate["blockers"]],
                    },
                    "verification": verifier,
                }
            ]
            break
    summary["updated_at_utc"] = utc_now()
    summary["final_status"] = "running_professional_two_loop_not_converged"
    write_json(SUMMARY_PATH, summary)


def status_markdown(gate: dict[str, Any], verifier: dict[str, Any], dag: dict[str, Any]) -> None:
    lines = [
        f"# NuWa Specialized Professional Gate",
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
        f"- Environment: `{ENV_PATH}`",
        f"- Script manifest: `{SCRIPT_MANIFEST_PATH}`",
        f"- Model/data manifest: `{MODEL_DATA_PATH}`",
        "",
        "## Verifier Checks",
        "",
    ]
    for check in verifier["checks"]:
        lines.append(f"- `{check['name']}`: `{check['status']}`")
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_status_markdown() -> None:
    summary = read_json(SUMMARY_PATH)
    queue = read_json(QUEUE_PATH)
    papers = summary.get("papers", [])
    accepted = sum(1 for p in papers if p.get("converged"))
    blocked = sum(1 for p in papers if not p.get("converged"))
    high = [p for p in papers if p.get("specialized_runner_status") or p.get("repo_paths")]
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
    for paper in high[:24]:
        status = (
            paper.get("specialized_runner_status")
            or paper.get("professional_blocker")
            or paper.get("final_status")
            or "unknown"
        )
        lines.append(
            f"- `{paper.get('paper_id')}`: `{status}` "
            f"repo_paths={paper.get('repo_paths', [])}"
        )
    prophet_status = RUN_ROOT / "specialized_runners/prophet/custom_full_gsm8k_llada8b/status.json"
    if prophet_status.exists():
        ps = read_json(prophet_status)
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

    queue_items = queue.get("queue", queue if isinstance(queue, list) else [])
    qlines = ["# Specialized Runner Queue", ""]
    for item in queue_items:
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
    gate = status_from_artifacts(env, manifest, model_data)
    dag = update_dag(gate)
    verifier = specialized_verifier(gate, dag)
    update_queue_and_summary(gate, verifier, dag)
    status_markdown(gate, verifier, dag)
    refresh_status_markdown()
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
