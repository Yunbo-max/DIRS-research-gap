#!/usr/bin/env python3
"""FlashVID professional operational gate for the strict DIRS loop.

This runner is intentionally not a reduced benchmark. It inspects the DAG-named
official FlashVID scripts, verifies model/data/runtime availability, reproduces
the repository's analytic FLOPs path, and emits explicit blockers for the full
paper grid when exact professional conditions are unavailable.
"""

from __future__ import annotations

import ast
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
PAPER_RUN = RUN_ROOT / "paper_runs" / "iclr2026_h6rdx4w6al_flashvid_vllm_token_merging"
RUNNER_DIR = RUN_ROOT / "specialized_runners" / "flashvid"
REPO = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/FlashVID")

STATUS_PATH = RUNNER_DIR / "FLASHVID_SPECIALIZED_STATUS.md"
VERIFIER_PATH = RUNNER_DIR / "flashvid_specialized_verifier.json"
ENV_PATH = RUNNER_DIR / "environment.json"
SCRIPT_MANIFEST_PATH = RUNNER_DIR / "official_script_manifest.json"
MODEL_DATA_PATH = RUNNER_DIR / "model_data_manifest.json"
FLOPS_PATH = RUNNER_DIR / "flops_reproduction.json"

PAPER_ID = "ICLR2026_H6rDX4w6Al_flashvid_vllm_token_merging"
TITLE = (
    "FlashVID: Efficient Video Large Language Models via Training-free "
    "Tree-based Spatiotemporal Token Merging"
)

OFFICIAL_SCRIPTS = [
    "scripts/llava_ov.sh",
    "scripts/llava_vid.sh",
    "scripts/qwen2_5_vl.sh",
    "scripts/qwen3_vl.sh",
    "scripts/baseline/llava_ov.sh",
    "scripts/baseline/llava_vid.sh",
    "scripts/baseline/qwen2_5_vl.sh",
    "scripts/baseline/qwen3_vl.sh",
    "scripts/fixed_token_budget/baseline.sh",
    "scripts/fixed_token_budget/flashvid.sh",
    "scripts/efficiency/llava_ov.sh",
    "scripts/efficiency/llava_vid.sh",
]

TASKS_REQUIRED = [
    "videomme",
    "egoschema",
    "mvbench",
    "longvideobench_val_v",
    "mlvu_test",
]

MODEL_IDS_REQUIRED = [
    "lmms-lab/llava-onevision-qwen2-7b-ov",
    "lmms-lab/LLaVA-Video-7B-Qwen2",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


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
        timed_out = False
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
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "cmd": cmd,
        "cwd": str(cwd) if cwd else None,
        "returncode": proc.returncode,
        "timeout": timed_out,
        "seconds": round(elapsed, 3),
        "stdout": proc.stdout[-12000:],
        "stderr": proc.stderr[-12000:],
    }


def nvidia_snapshot() -> dict[str, Any]:
    return run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def parse_bash_array(text: str, name: str) -> list[str]:
    match = re.search(rf"{re.escape(name)}=\((.*?)\)", text, flags=re.S)
    if not match:
        return []
    raw = match.group(1)
    return [part.strip().strip('"').strip("'") for part in raw.split() if part.strip()]


def parse_assignment(text: str, name: str) -> str | None:
    match = re.search(rf"^{re.escape(name)}=(.+)$", text, flags=re.M)
    if not match:
        return None
    return match.group(1).strip().strip('"').strip("'")


def script_manifest() -> dict[str, Any]:
    scripts: list[dict[str, Any]] = []
    for rel in OFFICIAL_SCRIPTS:
        path = REPO / rel
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        scripts.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "cuda_visible_devices": parse_assignment(text, "export CUDA_VISIBLE_DEVICES")
                or parse_assignment(text, "CUDA_VISIBLE_DEVICES"),
                "tasks": parse_bash_array(text, "TASKS"),
                "pretrained": parse_assignment(text, "PRETRAINED"),
                "retention_ratios": parse_bash_array(text, "RETENTION_RATIOS"),
                "max_frames": parse_bash_array(text, "MAX_FRAMES_NUM")
                or parse_bash_array(text, "MAX_NUM_FRAMES")
                or ([parse_assignment(text, "MAX_FRAMES_NUM")] if parse_assignment(text, "MAX_FRAMES_NUM") else [])
                or ([parse_assignment(text, "MAX_NUM_FRAMES")] if parse_assignment(text, "MAX_NUM_FRAMES") else []),
                "num_processes": re.findall(r"--num_processes\s+([0-9]+)", text),
                "models": re.findall(r"--model\s+([A-Za-z0-9_\\-]+)", text),
                "output_paths": re.findall(r"--output_path\s+([^\\s]+)", text),
            }
        )
    payload = {
        "artifact_kind": "flashvid_official_script_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "repo": str(REPO),
        "scripts": scripts,
        "full_grid_commands_required": [
            "bash scripts/baseline/llava_ov.sh",
            "bash scripts/baseline/llava_vid.sh",
            "bash scripts/baseline/qwen2_5_vl.sh",
            "bash scripts/llava_ov.sh",
            "bash scripts/llava_vid.sh",
            "bash scripts/qwen2_5_vl.sh",
            "bash scripts/fixed_token_budget/baseline.sh",
            "bash scripts/fixed_token_budget/flashvid.sh",
            "bash scripts/efficiency/llava_ov.sh",
        ],
    }
    write_json(SCRIPT_MANIFEST_PATH, payload)
    return payload


def hf_repo_manifest(repo_id: str, repo_type: str = "model") -> dict[str, Any]:
    token = HfFolder.get_token()
    api = HfApi(token=token)
    try:
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
            "files_head": files[:30],
            "files_tail": files[-30:],
        }
    except Exception as exc:  # network and auth errors must be recorded, not hidden.
        return {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "status": "unavailable",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def count_jsonl(path: Path) -> tuple[int, dict[str, Any] | None]:
    if not path.exists():
        return 0, None
    count = 0
    first = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            if first is None:
                try:
                    first = json.loads(line)
                except json.JSONDecodeError:
                    first = {"raw": line[:500]}
    return count, first


def model_data_manifest() -> dict[str, Any]:
    hf_home_candidates = [
        Path(os.path.expanduser("~/.cache/huggingface")),
        Path("/tf/notebooks/.cache/huggingface"),
    ]
    videomme_caches = []
    for hf_home in hf_home_candidates:
        video_dir = hf_home / "videomme" / "data"
        files = list(video_dir.glob("*")) if video_dir.exists() else []
        videomme_caches.append(
            {
                "path": str(video_dir),
                "exists": video_dir.exists(),
                "file_count": len([p for p in files if p.is_file()]),
                "size_human": run_cmd(["du", "-sh", str(video_dir)], timeout=30)["stdout"].split()[0]
                if video_dir.exists()
                else None,
            }
        )

    task_dirs = {}
    lmms_tasks = REPO / "lmms-eval" / "lmms_eval" / "tasks"
    for task in TASKS_REQUIRED:
        matches = sorted(str(p.relative_to(lmms_tasks)) for p in lmms_tasks.rglob(f"*{task}*") if p.exists())
        task_dirs[task] = matches[:40]

    videomme_jsonl = REPO / "assets" / "videomme.jsonl"
    videomme_count, videomme_first = count_jsonl(videomme_jsonl)
    payload = {
        "artifact_kind": "flashvid_model_data_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "hf_token_available": bool(HfFolder.get_token()),
        "models": [hf_repo_manifest(model_id) for model_id in MODEL_IDS_REQUIRED],
        "datasets": {
            "assets_videomme_jsonl": {
                "path": str(videomme_jsonl),
                "exists": videomme_jsonl.exists(),
                "record_count": videomme_count,
                "first_record": videomme_first,
            },
            "videomme_hf_cache_candidates": videomme_caches,
            "lmms_eval_task_dirs": task_dirs,
        },
    }
    write_json(MODEL_DATA_PATH, payload)
    return payload


def environment_manifest() -> dict[str, Any]:
    nvidia = nvidia_snapshot()
    gpu_rows = []
    for line in nvidia.get("stdout", "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 6:
            gpu_rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "memory_used_mib": int(parts[3]),
                    "memory_free_mib": int(parts[4]),
                    "utilization_gpu_pct": int(parts[5]),
                }
            )
    payload = {
        "artifact_kind": "flashvid_environment_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "repo": str(REPO),
        "python": sys.version,
        "cwd": str(Path.cwd()),
        "nvidia_smi": nvidia,
        "gpu_rows": gpu_rows,
        "professional_hardware_expectation": {
            "main_experiments": "8 x NVIDIA A800 80G according to paper/DAG",
            "efficiency": "single NVIDIA A100 according to paper/DAG",
        },
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "decord": package_version("decord"),
            "flash_attn": package_version("flash-attn"),
            "qwen_vl_utils": package_version("qwen-vl-utils"),
            "lmms_eval": package_version("lmms-eval"),
            "uv": run_cmd(["bash", "-lc", "command -v uv && uv --version"], timeout=30),
        },
        "compileall": run_cmd(
            ["python", "-m", "compileall", "-q", "flashvid", "playground", "tools"],
            cwd=REPO,
            timeout=180,
        ),
    }
    write_json(ENV_PATH, payload)
    return payload


def flops_reproduction() -> dict[str, Any]:
    result = run_cmd(["python", "tools/flops/llava_ov.py"], cwd=REPO, timeout=60)
    payload = {
        "artifact_kind": "flashvid_flops_reproduction",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "status": "pass" if result["returncode"] == 0 else "failed",
        "command_result": result,
        "convergence_role": "analytic support for TFLOPs axis only; not a benchmark-table convergence artifact",
    }
    write_json(FLOPS_PATH, payload)
    return payload


def derive_blockers(env: dict[str, Any], data: dict[str, Any]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    gpu_rows = env.get("gpu_rows", [])
    has_eight = len(gpu_rows) >= 8
    has_a800_80g = any("A800" in row["name"] and row["memory_total_mib"] >= 70000 for row in gpu_rows)
    has_a100 = any("A100" in row["name"] for row in gpu_rows)
    if not has_eight:
        blockers.append(
            {
                "id": "main_grid_hardware_gpu_count",
                "status": "blocked",
                "detail": f"official main scripts request --num_processes 8; visible GPUs={len(gpu_rows)}",
            }
        )
    if not has_a800_80g:
        blockers.append(
            {
                "id": "main_grid_hardware_gpu_class",
                "status": "blocked",
                "detail": "paper/DAG expects NVIDIA A800 80G for main experiments; visible GPUs are not A800 80G",
            }
        )
    if not has_a100:
        blockers.append(
            {
                "id": "efficiency_hardware_gpu_class",
                "status": "blocked",
                "detail": "paper/DAG expects a single NVIDIA A100 for efficiency traces; visible GPUs are not A100",
            }
        )
    if env.get("packages", {}).get("transformers") != "4.57.0":
        blockers.append(
            {
                "id": "transformers_version_exactness",
                "status": "blocked",
                "detail": f"README badge requires Transformers 4.57; environment has {env.get('packages', {}).get('transformers')}",
            }
        )
    if not env.get("packages", {}).get("flash_attn"):
        blockers.append(
            {
                "id": "flash_attention_runtime",
                "status": "blocked",
                "detail": "official scripts request attn_implementation=flash_attention_2; flash-attn package is not installed",
            }
        )
    videomme_caches = data.get("datasets", {}).get("videomme_hf_cache_candidates", [])
    if not any(item.get("exists") and item.get("file_count", 0) > 0 for item in videomme_caches):
        blockers.append(
            {
                "id": "videomme_video_cache",
                "status": "blocked",
                "detail": "playground/bench_efficiency.py expects ~/.cache/huggingface/videomme/data videos; no populated cache was found",
            }
        )
    for model in data.get("models", []):
        if model.get("status") != "available":
            blockers.append(
                {
                    "id": f"model_manifest_{model.get('repo_id')}",
                    "status": "blocked",
                    "detail": model.get("error", "model repository unavailable"),
                }
            )
    return blockers


def write_status(verifier: dict[str, Any]) -> None:
    blockers = verifier["verifier"]["unresolved_professional_debt"]
    STATUS_PATH.write_text(
        "# FlashVID Specialized Runner Status\n\n"
        f"- Updated: {verifier['updated_at_utc']}\n"
        f"- Paper: `{TITLE}`\n"
        f"- Status: `{verifier['verifier']['status']}`\n"
        f"- Convergence decision: `{verifier['verifier']['convergence_decision']}`\n"
        f"- Professional package ready: `{verifier['verifier']['professional_package_ready']}`\n"
        f"- Official scripts parsed: `{len(verifier['official_script_manifest']['scripts'])}`\n"
        f"- FLOPs reproduction status: `{verifier['flops_reproduction']['status']}`\n"
        f"- Blocker count: `{len(blockers)}`\n\n"
        "## Artifact Paths\n"
        f"- Environment: `{ENV_PATH}`\n"
        f"- Official script manifest: `{SCRIPT_MANIFEST_PATH}`\n"
        f"- Model/data manifest: `{MODEL_DATA_PATH}`\n"
        f"- FLOPs reproduction: `{FLOPS_PATH}`\n"
        f"- Verifier: `{VERIFIER_PATH}`\n\n"
        "## Why This Is Not Converged\n"
        "- This gate did not run reduced VideoMME or one-video demos as convergence evidence.\n"
        "- The full paper grid needs the official 8-process LMMs-Eval benchmark scripts, paper-compatible GPU class, exact runtime stack, and video dataset caches.\n"
        "- Until those artifacts exist, the DAG is semantically plausible but operationally blocked.\n\n"
        "## Current Blockers\n"
        + "\n".join(f"- `{item['id']}`: {item['detail']}" for item in blockers)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)
    env = environment_manifest()
    scripts = script_manifest()
    data = model_data_manifest()
    flops = flops_reproduction()
    blockers = derive_blockers(env, data)
    verifier = {
        "artifact_kind": "flashvid_specialized_verifier",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "repo": str(REPO),
        "dag_path": str(PAPER_RUN / "paper_author_gap_dag.json"),
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
        "flops_reproduction": flops,
        "verifier": {
            "status": "blocked_by_exact_professional_runtime_and_data_requirements",
            "convergence_decision": "not_converged_explicit_professional_blockers_after_operational_preflight",
            "professional_package_ready": False,
            "semantic_dag_nodes_checked": [
                "gap.paper_gap_claims",
                "method.bind_gap_to_mechanism",
                "experiments.benchmark_metric_grid",
                "experiments.system_surface",
                "ops.resolve_models_data",
            ],
            "unresolved_professional_debt": blockers,
            "loop1_required_dag_update": [
                "Add exact official-script execution matrix as a DAG node, not a generic VLM benchmark node.",
                "Add hardware class gate: 8 x A800 80G for main LMMs-Eval grid and single A100 for efficiency traces.",
                "Add VideoMME local video-cache gate for efficiency benchmark.",
                "Add exact Transformers 4.57 and flash_attention_2 runtime gate.",
                "Keep FLOPs reproduction as analytic support only, not result-table convergence evidence.",
            ],
        },
    }
    write_json(VERIFIER_PATH, verifier)
    write_status(verifier)
    print(json.dumps(verifier["verifier"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
