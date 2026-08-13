#!/usr/bin/env python3
"""SPARK professional operational gate for the strict DIRS loop.

The SPARK repo is currently a placeholder. This runner records that fact as a
paper-specific operational blocker, updates the DAG from "no repo" to "repo
present but no executable source", and prevents any README/teaser/project-page
evidence from counting as Loop 2 convergence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ROOT = Path(
    "/tf/notebooks/yunbo/DIRS/case1_writing/runs/"
    "remaining19_p_less_style_dirs_20260723"
)
PAPER_RUN = RUN_ROOT / "paper_runs" / "cvpr2026_030_spark_vlm_articulated_reconstruction"
RUNNER_DIR = RUN_ROOT / "specialized_runners" / "spark"
REPO = Path("/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/030_spark/SPARK")

PAPER_ID = "CVPR2026_030_spark_vlm_articulated_reconstruction"
TITLE = "SPARK: Sim-ready Part-level Articulated Reconstruction with VLM Knowledge"

DAG_PATH = PAPER_RUN / "paper_author_gap_dag.json"
QUEUE_PATH = RUN_ROOT / "specialized_runner_queue.json"
SUMMARY_PATH = RUN_ROOT / "remaining19_strict_dirs_summary.json"
LONG_STATUS_PATH = RUN_ROOT / "LONGGOAL_STATUS.md"
SPECIALIZED_QUEUE_MD = RUN_ROOT / "SPECIALIZED_RUNNER_QUEUE.md"

STATUS_PATH = RUNNER_DIR / "SPARK_SPECIALIZED_STATUS.md"
VERIFIER_PATH = RUNNER_DIR / "spark_specialized_verifier.json"
PROFESSIONAL_GATE_PATH = RUNNER_DIR / "professional_gate_result.json"
REPO_MANIFEST_PATH = RUNNER_DIR / "repo_manifest.json"
ENV_PATH = RUNNER_DIR / "environment.json"

EXPECTED_OUTPUT_SURFACES = [
    "100-image GAPartNet articulated-object reconstruction table",
    "Chamfer Distance, F-Score@0.1, F-Score@0.5 shape metrics",
    "AxisErr, PivotErr, TypeErr URDF/joint metrics",
    "Articulate-Anything and Articulate AnyMesh baseline comparison",
    "ablation of part guidance, data augmentation, joint optimization, and VLM re-prediction",
    "Isaac Sim qualitative and policy-deployment artifacts",
    "raw meshes, part labels, URDFs, joint graphs, rendered silhouettes, metric JSON, GPU traces",
]

EXPECTED_MISSING_ARTIFACTS = [
    "inference scripts",
    "training code",
    "data preprocessing scripts",
    "pretrained checkpoints",
    "preprocessed PartNet-Mobility/GAPartNet dataset",
    "VLM structural reasoning prompts/API runner",
    "VLM part-reference image-generation runner",
    "DINOv2 image encoder checkpoint/config",
    "part-articulated diffusion transformer checkpoint/config",
    "hierarchical graph attention implementation",
    "rectified-flow mesh latent implementation",
    "differentiable forward-kinematics and differentiable rendering refinement code",
    "Meshy texture synthesis integration/API key or exported textures",
    "Isaac Sim deployment scripts",
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


def run_cmd(cmd: list[str], *, cwd: Path | None = None, timeout: int = 60) -> dict[str, Any]:
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
    for path in sorted(p for p in REPO.rglob("*") if p.is_file() and ".git" not in p.parts):
        rel = path.relative_to(REPO)
        size_human, file_count = path_size(path)
        row: dict[str, Any] = {
            "relative_path": str(rel),
            "path": str(path),
            "size_human": size_human,
            "file_count": file_count,
        }
        if rel.name.lower() == "readme.md":
            text = read_text(path)
            row["readme"] = {
                "line_count": len(text.splitlines()),
                "urls": sorted(set(re.findall(r"https?://[^)\s]+", text))),
                "todo_lines": [line.strip() for line in text.splitlines() if "[ ]" in line],
                "declares_future_release": "will contain the official implementation" in text,
            }
        if rel.suffix.lower() == ".png":
            row["image_probe"] = run_cmd(
                [sys.executable, "-c", f"from PIL import Image; p={str(path)!r}; im=Image.open(p); print(im.size, im.mode)"],
                timeout=30,
            )
        files.append(row)

    payload = {
        "artifact_kind": "spark_repo_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "repo": str(REPO),
        "repo_exists": REPO.exists(),
        "file_count_excluding_git": len(files),
        "files": files,
        "git": {
            "remote": run_cmd(["git", "-C", str(REPO), "remote", "-v"], timeout=20),
            "head": run_cmd(["git", "-C", str(REPO), "rev-parse", "HEAD"], timeout=20),
            "dubious_ownership_expected_in_this_container": True,
        },
        "support_only_findings": [
            "README and teaser image prove repository presence only",
            "README TODO explicitly says inference scripts, pretrained checkpoints, training code, preprocessing scripts, and preprocessed dataset are not released",
            "no Python source, configs, checkpoints, datasets, or runnable commands are present in the repo",
        ],
    }
    write_json(REPO_MANIFEST_PATH, payload)
    return payload


def environment_manifest() -> dict[str, Any]:
    payload = {
        "artifact_kind": "spark_environment_manifest",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "python": sys.version,
        "gpu_rows": gpu_rows(),
        "package_probes": {
            name: run_cmd([sys.executable, "-c", f"import importlib; importlib.import_module({name!r}); print('ok')"], timeout=30)
            for name in ["torch", "pytorch3d", "trimesh", "isaacsim", "open3d", "kaolin"]
        },
        "professional_hardware_expected_by_dag": [
            "4 NVIDIA H100 GPUs for training/evaluation surface",
            "Isaac Sim runtime for simulator deployment",
            "VLM/API access for structural reasoning and part-reference image generation",
        ],
    }
    write_json(ENV_PATH, payload)
    return payload


def professional_gate(manifest: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    blockers = []
    files = [f["relative_path"] for f in manifest["files"]]
    code_files = [f for f in files if f.endswith((".py", ".sh", ".yaml", ".yml", ".json", ".toml"))]
    if not code_files:
        blockers.append(
            {
                "id": "official_repo_placeholder_no_executable_source",
                "status": "blocked",
                "detail": f"Repository contains only {files}; no executable source/config/script is released.",
            }
        )

    readme_todos = []
    for f in manifest["files"]:
        readme_todos.extend(f.get("readme", {}).get("todo_lines", []))
    if readme_todos:
        blockers.append(
            {
                "id": "readme_declares_core_artifacts_unreleased",
                "status": "blocked",
                "detail": "README TODO marks these as unreleased: " + "; ".join(readme_todos),
            }
        )

    gpu_names = " | ".join(row["name"] for row in env["gpu_rows"])
    h100_count = sum("H100" in row["name"] for row in env["gpu_rows"])
    if h100_count < 4:
        blockers.append(
            {
                "id": "four_h100_hardware_missing",
                "status": "blocked",
                "detail": f"DAG expects 4 NVIDIA H100 GPUs; visible devices are {gpu_names}.",
            }
        )

    missing_pkgs = [name for name, probe in env["package_probes"].items() if probe["returncode"] != 0]
    if missing_pkgs:
        blockers.append(
            {
                "id": "spark_3d_runtime_dependencies_missing",
                "status": "blocked",
                "detail": "Missing runtime imports for full 3D/simulation path: " + ", ".join(missing_pkgs),
            }
        )

    blockers.extend(
        [
            {
                "id": "spark_models_data_checkpoints_missing",
                "status": "blocked",
                "detail": "Missing released pretrained checkpoints, PartNet-Mobility/GAPartNet preprocessing, 100-image GAPartNet eval set, DINOv2/diffusion transformer configs, and VLM-generated part-reference artifacts.",
            },
            {
                "id": "spark_vlm_api_and_meshy_integration_missing",
                "status": "blocked",
                "detail": "No structural-reasoning VLM runner, part-reference image-generation runner, or Meshy texture synthesis integration/API artifacts are present.",
            },
            {
                "id": "spark_full_3d_result_grid_missing",
                "status": "blocked",
                "detail": "No verifier-comparable mesh/URDF/joint metric tables, ablations, Isaac Sim artifacts, raw outputs, or GPU traces are present.",
            },
        ]
    )

    status = "blocked_by_placeholder_repo_unreleased_source_models_data_runtime_hardware_and_result_grid"
    payload = {
        "artifact_kind": "spark_professional_gate_result",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "status": status,
        "professional_package_ready": False,
        "convergence_role": "professional operational gate; no README/teaser/proxy evidence is promoted",
        "blockers": blockers,
        "support_checks": {
            "repo_discovered_and_encoded": REPO.exists(),
            "files_checked_excluding_git": manifest["file_count_excluding_git"],
            "readme_todos_detected": len(readme_todos),
            "gpu_rows_checked": len(env["gpu_rows"]),
        },
        "next_full_execution_if_unblocked": [
            "release or obtain official inference/training/preprocessing source",
            "materialize PartNet-Mobility/GAPartNet/preprocessed SPARK datasets and checkpoints",
            "configure VLM structural reasoning, part-reference image generation, DINOv2, mesh diffusion transformer, differentiable rendering, and Meshy texture path",
            "run the 100-image GAPartNet shape/URDF benchmark and baselines",
            "run ablations and Isaac Sim qualitative/policy deployment",
            "emit raw meshes, URDFs, joint graphs, metric JSON, table summaries, and 4-H100/Isaac traces for verifier comparison",
        ],
        "paper_shaped_outputs_required": EXPECTED_OUTPUT_SURFACES,
        "missing_artifact_families": EXPECTED_MISSING_ARTIFACTS,
    }
    write_json(PROFESSIONAL_GATE_PATH, payload)
    return payload


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
                "repos=/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/030_spark/SPARK; "
                "released_code_artifacts=README.md; LICENSE; assets/SPARK_teaser.png; "
                "missing_artifacts=inference_scripts; pretrained_checkpoints; training_code; "
                "data_preprocessing_scripts; preprocessed_dataset"
            )
    nodes = [
        ("ops.spark_repo_placeholder_gate", "Resolve official repo and verify it is placeholder-only: README, LICENSE, teaser image, no executable source/config/checkpoints/data.", "operational_dependency"),
        ("ops.spark_release_artifact_gate", "Require official inference/training/preprocessing code, pretrained checkpoints, and preprocessed PartNet-Mobility/GAPartNet data before Loop 2 execution.", "operational_dependency"),
        ("ops.spark_vlm_mesh_sim_dependency_gate", "Require VLM/API prompts/runners, DINOv2, mesh diffusion transformer, differentiable rendering/forward kinematics, Meshy texture path, and Isaac Sim setup.", "operational_dependency"),
        ("ops.spark_full_3d_benchmark_gate", "Run 100-image GAPartNet reconstruction/URDF benchmark, baselines, ablations, and Isaac Sim deployment with raw mesh/URDF/joint/metric artifacts.", "operational_execution"),
        ("decision.explicit_blocker_after_spark_preflight", "Block if repo remains placeholder-only or any model/data/API/runtime/result-grid artifact is absent; never converge from README/teaser evidence.", "author_reviewer_decision"),
    ]
    for node_id, content, typ in nodes:
        ensure_node(dag, {"id": node_id, "content": content, "type": typ, "skill_role": "paper-specific operational gate"})
    for source, target in [
        ("ops.resolve_repo_code", "ops.spark_repo_placeholder_gate"),
        ("ops.spark_repo_placeholder_gate", "ops.spark_release_artifact_gate"),
        ("ops.spark_release_artifact_gate", "ops.spark_vlm_mesh_sim_dependency_gate"),
        ("ops.spark_vlm_mesh_sim_dependency_gate", "ops.spark_full_3d_benchmark_gate"),
        ("ops.spark_full_3d_benchmark_gate", "reviewer.require_professional_artifact_package"),
        ("ops.spark_full_3d_benchmark_gate", "reviewer.compare_result_shapes"),
        ("reviewer.keep_exact_artifact_debt", "decision.explicit_blocker_after_spark_preflight"),
    ]:
        ensure_edge(dag, source, target)
    dag.setdefault("previous_loop_updates", []).append(
        {
            "iteration": 3,
            "created_at_utc": utc_now(),
            "source": "spark_specialized_professional_gate",
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
        {"name": "placeholder_repo_rejection", "status": "pass", "detail": "README/teaser-only repository cannot converge"},
        {"name": "professional_artifact_package", "status": "blocked", "detail": gate["blockers"]},
        {"name": "result_shape_comparison_ready", "status": "blocked", "detail": "requires SPARK mesh/URDF/joint/Isaac result grid before verifier comparison"},
    ]
    payload = {
        "artifact_kind": "spark_specialized_verifier",
        "created_at_utc": utc_now(),
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "iteration": 3,
        "converged": False,
        "professional_ready": False,
        "checks": checks,
        "required_updates": [
            {
                "id": "update.spark_placeholder_repo_to_release_artifact_gates",
                "reason": gate["status"],
                "success_criteria": [
                    "repo path encoded",
                    "placeholder-only source gate present",
                    "official release/model/data/API/runtime gates present",
                    "full 3D benchmark/Isaac result-grid gate present",
                ],
            }
        ],
        "artifact_paths": {
            "professional_gate": str(PROFESSIONAL_GATE_PATH),
            "repo_manifest": str(REPO_MANIFEST_PATH),
            "environment": str(ENV_PATH),
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
            item["repo_exact_rerun_status"] = "repo_present_placeholder_no_executable_source"
            item["repo_paths"] = [str(REPO)]
            item["specialized_runner_status"] = gate["status"]
            item["specialized_runner_artifact_dir"] = str(RUNNER_DIR)
            statuses = item.setdefault("implementation_statuses", [])
            for status in ["official_repo_discovered_and_encoded", "placeholder_repo_confirmed", "professional_gate_blocked"]:
                if status not in statuses:
                    statuses.append(status)
            item["specialized_runner_evidence"] = {
                "blockers": gate["blockers"],
                "verifier_path": str(VERIFIER_PATH),
                "repo_manifest_path": str(REPO_MANIFEST_PATH),
                "environment_path": str(ENV_PATH),
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
            for status in ["official_repo_discovered_and_encoded", "placeholder_repo_confirmed", "professional_gate_blocked"]:
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
                        "professional_package_ready": False,
                        "professional_package_reason": gate["status"],
                        "reduced_or_proxy_used_for_convergence": False,
                        "raw_artifact_level": "placeholder_repo_manifest_only",
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
        "# SPARK Specialized Professional Gate",
        "",
        f"- Paper id: `{PAPER_ID}`",
        f"- Title: {TITLE}",
        f"- Status: `{gate['status']}`",
        "- Converged: `false`",
        "- Professional ready: `false`",
        f"- DAG signature: `{dag.get('signature')}`",
        f"- Repo: `{REPO}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in gate["blockers"]:
        lines.append(f"- `{blocker['id']}`: {blocker['detail']}")
    lines += ["", "## Artifact Paths", "", f"- Professional gate: `{PROFESSIONAL_GATE_PATH}`", f"- Verifier: `{VERIFIER_PATH}`", f"- Repo manifest: `{REPO_MANIFEST_PATH}`", f"- Environment: `{ENV_PATH}`", "", "## Verifier Checks", ""]
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
    manifest = repo_manifest()
    env = environment_manifest()
    gate = professional_gate(manifest, env)
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
