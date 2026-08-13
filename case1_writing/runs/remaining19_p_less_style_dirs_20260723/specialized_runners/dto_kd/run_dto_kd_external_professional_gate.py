#!/usr/bin/env python3
"""External-source professional gate for the DTO-KD paper.

DTO-KD has a clear paper-shaped experimental contract but no official code repo
in the local source pass. This runner prevents a false convergence path: an
unrelated KD repo, a small CIFAR-only sketch, or a local RTX 4090 debug run
cannot replace the paper's four-H100, ImageNet/CIFAR/COCO, teacher/student
distillation grid. The output updates Loop 1 with precise operational debt.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ROOT = Path(
    "/tf/notebooks/yunbo/DIRS/case1_writing/runs/"
    "remaining19_p_less_style_dirs_20260723"
)
SOURCE_ROOT = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h")
QUEUE_PATH = RUN_ROOT / "specialized_runner_queue.json"
SUMMARY_PATH = RUN_ROOT / "remaining19_strict_dirs_summary.json"

CFG: dict[str, Any] = {
    "short": "dto_kd",
    "paper_id": "ICLR2026_QMItTyQW92_dto_kd_dynamic_tradeoff_distillation",
    "paper_run": "iclr2026_qmittyqw92_dto_kd_dynamic_tradeoff_distillation",
    "title": "DTO-KD: Dynamic Trade-off Optimization for Effective Knowledge Distillation",
    "runner_type": "distillation_tradeoff_runner",
    "status": "blocked_by_no_official_source_repo_h100_data_checkpoints_and_full_kd_training_grid",
    "source_files": {
        "status_file": SOURCE_ROOT / "status/ICLR2026_QMItTyQW92_source_status.txt",
        "chip": SOURCE_ROOT / "chips/ICLR2026_QMItTyQW92_dto_kd_dynamic_tradeoff_distillation.chip.json",
        "paper_text": SOURCE_ROOT / "text/ICLR2026_QMItTyQW92_openreview.txt",
        "supplement_text": SOURCE_ROOT / "text/ICLR2026_QMItTyQW92_supplementary.txt",
        "forum_html": SOURCE_ROOT / "text/ICLR2026_QMItTyQW92_openreview_forum.html",
    },
    "candidate_repos_checked": [
        "/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/pgm_text_distill_sdtt",
        "/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/attn-dynamics-basis",
        "/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/CoCo",
    ],
    "data_candidate_paths": {
        "imagenet_1k": [
            "/tf/notebooks/datasets/imagenet",
            "/tf/notebooks/data/imagenet",
            "/datasets/imagenet",
            "/data/imagenet",
        ],
        "cifar100": [
            "/tf/notebooks/datasets/cifar100",
            "/tf/notebooks/data/cifar100",
            "/datasets/cifar100",
            "/data/cifar100",
        ],
        "coco2017": [
            "/tf/notebooks/datasets/coco2017",
            "/tf/notebooks/data/coco2017",
            "/datasets/coco2017",
            "/data/coco2017",
        ],
    },
    "checkpoint_candidate_paths": {
        "regnety_160_teacher": ["checkpoints/regnety_160", "weights/regnety_160"],
        "deit_tiny_student": ["checkpoints/deit_tiny", "weights/deit_tiny"],
        "deit_small_student": ["checkpoints/deit_small", "weights/deit_small"],
        "vidt_base_teacher": ["checkpoints/vidt_base", "weights/vidt_base"],
        "vidt_nano_tiny_small_students": ["checkpoints/vidt_students", "weights/vidt_students"],
    },
    "paper_table_targets": {
        "table_1_imagenet_top1": {
            "DTO-KD_Ti": 79.7,
            "DTO-KD_S": 83.1,
            "protocol": "300 epochs ImageNet-1K, RegNetY-160 teacher, DeiT Tiny/Small students",
        },
        "table_2_cifar100_top1": {
            "DTO-KD_values": [72.35, 75.68, 76.40, 70.90, 77.95, 78.22],
            "protocol": "six CIFAR-100 homogeneous/heterogeneous teacher-student pairs",
        },
        "table_3_coco_detection": {
            "DTO-KD_nano_AP": 43.7,
            "DTO-KD_tiny_AP": 47.4,
            "DTO-KD_small_AP": 49.6,
            "protocol": "50 epochs COCO2017 ViDT distillation from ViDT-base teacher",
        },
        "table_4_ablation": {
            "full_projector_dto_gradclip_AP_AP50_AP75": [43.7, 63.1, 46.8],
            "baseline_AP_AP50_AP75": [41.0, 59.2, 42.8],
        },
        "table_5_teacher_scale": {
            "claim": "DTO-KD is best for nano/tiny students with both ViDT-small and ViDT-base teachers",
        },
        "figures_3_4": {
            "required": "dynamic pi trajectories and TIDE classification/localization error analysis",
        },
    },
    "paper_shaped_outputs_required": [
        "official or faithfully reimplemented DTO-KD training code from Algorithm 1 and Equations 8-18",
        "four-H100 training logs and environment traces",
        "ImageNet-1K 300-epoch RegNetY-160 to DeiT-Ti/DeiT-S distillation outputs",
        "CIFAR-100 six-pair KD grid against FitNet/RKD/PKT/KD/OFD/CRD/DIST/ReviewKD/DKD/ReviewKD++ baselines",
        "COCO2017 ViDT-nano/tiny/small 50-epoch distillation outputs against ViDT, Token-Matching, and VkD",
        "component ablation outputs for projector optimization, DTO, and gradient clipping",
        "teacher-scale robustness outputs and Figure 3/4 dynamic-pi/error-analysis data",
    ],
}


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


def run_cmd(cmd: list[str], *, timeout: int = 60, cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": proc.returncode,
            "timeout": False,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "timeout": True,
            "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }


def path_size(path: Path) -> tuple[str | None, int]:
    if not path.exists():
        return None, 0
    if path.is_file():
        return str(path.stat().st_size), 1
    size = run_cmd(["du", "-sh", str(path)], timeout=20)["stdout"].split()
    files = run_cmd(["find", str(path), "-type", "f"], timeout=30)["stdout"].splitlines()
    return (size[0] if size else None), len(files)


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


def source_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    source_files = {}
    for name, path in cfg["source_files"].items():
        text = read_text(path)
        source_files[name] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "line_count": len(text.splitlines()) if text else 0,
            "contains_code_url_absent_note": "No official GitHub" in text or "code_url\": null" in text,
        }
    candidate_repos = []
    for repo_s in cfg["candidate_repos_checked"]:
        repo = Path(repo_s)
        readme = read_text(repo / "README.md")
        candidate_repos.append(
            {
                "path": str(repo),
                "exists": repo.exists(),
                "readme_title_head": "\n".join(readme.splitlines()[:8]),
                "rejected_as_exact_dto_kd_repo": "DTO-KD" not in readme and "Dynamic Trade-off Optimization" not in readme,
            }
        )
    payload = {
        "artifact_kind": "dto_kd_source_manifest",
        "created_at_utc": utc_now(),
        "paper_id": cfg["paper_id"],
        "paper_title": cfg["title"],
        "source_files": source_files,
        "candidate_repos_checked": candidate_repos,
        "official_source_repo_found": False,
        "support_only_warning": "Paper PDF/supplement/status facts can repair the DAG but cannot substitute for runnable source and generated training outputs.",
    }
    write_json(Path(cfg["runner_dir"]) / "source_manifest.json", payload)
    return payload


def artifact_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    data = {}
    for name, paths in cfg["data_candidate_paths"].items():
        rows = []
        for path_s in paths:
            path = Path(path_s)
            size_human, file_count = path_size(path)
            rows.append(
                {
                    "path": path_s,
                    "exists": path.exists(),
                    "is_dir": path.is_dir() if path.exists() else False,
                    "size_human": size_human,
                    "file_count": file_count,
                }
            )
        data[name] = rows
    checkpoints = {}
    for name, rels in cfg["checkpoint_candidate_paths"].items():
        rows = []
        for rel in rels:
            path = RUN_ROOT / "external_artifacts" / "dto_kd" / rel
            size_human, file_count = path_size(path)
            rows.append(
                {
                    "path": str(path),
                    "exists": path.exists(),
                    "is_dir": path.is_dir() if path.exists() else False,
                    "size_human": size_human,
                    "file_count": file_count,
                }
            )
        checkpoints[name] = rows
    result_root = RUN_ROOT / "specialized_runners" / "dto_kd" / "results"
    result_candidates = []
    for pattern in ["**/*.json", "**/*.csv", "**/*.jsonl", "**/*.pt", "**/*.pth"]:
        for path in result_root.glob(pattern):
            size_human, file_count = path_size(path)
            result_candidates.append({"path": str(path), "is_dir": path.is_dir(), "size_human": size_human, "file_count": file_count})
            if len(result_candidates) >= 200:
                break
    payload = {
        "artifact_kind": "dto_kd_artifact_manifest",
        "created_at_utc": utc_now(),
        "paper_id": cfg["paper_id"],
        "paper_title": cfg["title"],
        "dataset_candidates": data,
        "checkpoint_candidates": checkpoints,
        "result_candidates": result_candidates,
        "paper_table_targets": cfg["paper_table_targets"],
        "paper_shaped_outputs_required": cfg["paper_shaped_outputs_required"],
    }
    write_json(Path(cfg["runner_dir"]) / "model_data_manifest.json", payload)
    return payload


def professional_gate(cfg: dict[str, Any], source: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    gpu = gpu_rows()
    gpu_names = " | ".join(row["name"] for row in gpu)
    h100_count = sum(1 for row in gpu if "H100" in row["name"])
    missing_datasets = [
        name
        for name, rows in artifacts["dataset_candidates"].items()
        if not any(row["exists"] and row["file_count"] > 0 for row in rows)
    ]
    missing_checkpoints = [
        name
        for name, rows in artifacts["checkpoint_candidates"].items()
        if not any(row["exists"] and row["file_count"] > 0 for row in rows)
    ]
    blockers = [
        {
            "id": "dto_kd_official_source_repository_missing",
            "status": "blocked",
            "detail": "OpenReview metadata, paper, supplement, status file, and local candidate repo scan do not provide an official DTO-KD source repo.",
        },
        {
            "id": "dto_kd_four_h100_hardware_missing",
            "status": "blocked",
            "detail": f"Paper reports all experiments on four NVIDIA H100 GPUs; visible GPUs are {gpu_names}; H100 count={h100_count}.",
        },
        {
            "id": "dto_kd_required_datasets_missing",
            "status": "blocked",
            "detail": "Missing or unmaterialized dataset candidates: " + ", ".join(missing_datasets),
        },
        {
            "id": "dto_kd_teacher_student_checkpoints_missing",
            "status": "blocked",
            "detail": "Missing teacher/student checkpoints or initialization artifacts: " + ", ".join(missing_checkpoints),
        },
        {
            "id": "dto_kd_full_training_grid_missing",
            "status": "blocked",
            "detail": "No ImageNet 300-epoch, CIFAR-100 six-pair, COCO 50-epoch, ablation, teacher-scale, or Figure 3/4 raw outputs exist under the runner result root.",
        },
        {
            "id": "dto_kd_result_shape_verifier_waiting_for_tables_figures",
            "status": "blocked",
            "detail": "Verifier needs outputs comparable to Table 1, Table 2, Table 3, Table 4, Table 5, Figure 3, and Figure 4 before accepting close result shape.",
        },
        {
            "id": "dto_kd_reimplementation_required_but_unvalidated",
            "status": "blocked",
            "detail": "Without official code, a faithful implementation from Algorithm 1 and Equations 8-18 must be created and validated before any experiment can count.",
        },
    ]
    gate = {
        "artifact_kind": "dto_kd_professional_gate_result",
        "created_at_utc": utc_now(),
        "paper_id": cfg["paper_id"],
        "paper_title": cfg["title"],
        "status": cfg["status"],
        "professional_package_ready": False,
        "converged": False,
        "convergence_role": "external-source blocker and paper-contract DAG repair; no unrelated repo or reduced GPU proxy is promoted",
        "gpu_rows": gpu,
        "blockers": blockers,
        "support_checks": {
            "source_files_checked": source["source_files"],
            "official_source_repo_found": source["official_source_repo_found"],
            "candidate_repo_count": len(source["candidate_repos_checked"]),
            "h100_count": h100_count,
            "result_candidate_count": len(artifacts["result_candidates"]),
        },
        "next_full_execution_if_unblocked": [
            "obtain official source or implement DTO-KD faithfully from Algorithm 1 and Equations 8-18",
            "materialize ImageNet-1K, CIFAR-100, COCO2017, RegNetY-160, DeiT Tiny/Small, and ViDT teacher/student checkpoints",
            "run four-H100 paper-shaped training grids, not a local 4090 debug proxy",
            "emit raw per-run logs, seeds, metric JSON/CSV, GPU traces, and table/figure summaries",
            "verify close result shape against Table 1-5 and Figure 3-4 targets",
        ],
        "paper_table_targets": cfg["paper_table_targets"],
        "paper_shaped_outputs_required": cfg["paper_shaped_outputs_required"],
    }
    write_json(Path(cfg["runner_dir"]) / "professional_gate_result.json", gate)
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


def update_dag(cfg: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    paper_dir = RUN_ROOT / "paper_runs" / cfg["paper_run"]
    dag_path = paper_dir / "paper_author_gap_dag.json"
    dag = read_json(dag_path)
    for node in dag.get("nodes", []):
        if node.get("id") == "ops.resolve_repo_code":
            node["repo_paths"] = []
            node["content"] = (
                "repos=official source repository not found; code_artifacts=Algorithm 1; "
                "Equations 8-18; supplementary theorem proof; requires faithful reimplementation "
                "or official release before Loop 2 execution can converge"
            )
    dag_nodes = [
        ("source_availability_gate", "Confirm no official DTO-KD code repo in OpenReview metadata, paper, supplement, status file, or local candidate repos; reject unrelated KD repos."),
        ("h100_hardware_gate", "Require four NVIDIA H100 traces for the paper-shaped ImageNet/CIFAR/COCO training grid."),
        ("dataset_checkpoint_gate", "Resolve ImageNet-1K, CIFAR-100, COCO2017, RegNetY-160, DeiT Tiny/Small, ViDT-base, and ViDT nano/tiny/small artifacts."),
        ("faithful_reimplementation_gate", "If official code remains absent, implement Algorithm 1 and Equations 8-18 exactly enough to run classification/detection objectives."),
        ("full_training_grid", "Run ImageNet 300-epoch, CIFAR-100 six-pair, COCO 50-epoch, component ablation, teacher-scale robustness, and dynamic-pi/error-analysis protocols."),
        ("metric_table_verifier_gate", "Compare operational outputs against Table 1-5 and Figure 3-4; close shape can pass only from generated artifacts."),
    ]
    for suffix, content in dag_nodes:
        ensure_node(
            dag,
            {
                "id": f"ops.dto_kd_{suffix}",
                "content": content,
                "type": "operational_execution" if suffix == "full_training_grid" else "operational_dependency",
                "skill_role": "paper-specific external-source gate",
            },
        )
    decision_id = "decision.explicit_blocker_after_dto_kd_preflight"
    ensure_node(
        dag,
        {
            "id": decision_id,
            "content": "Block if source, H100 hardware, datasets, teacher/student checkpoints, full training outputs, or table/figure metric summaries are missing; feed exact debt into Loop 1.",
            "type": "author_reviewer_decision",
            "skill_role": "paper-specific external-source gate",
        },
    )
    chain = [
        "ops.resolve_repo_code",
        "ops.dto_kd_source_availability_gate",
        "ops.dto_kd_h100_hardware_gate",
        "ops.dto_kd_dataset_checkpoint_gate",
        "ops.dto_kd_faithful_reimplementation_gate",
        "ops.dto_kd_full_training_grid",
        "ops.dto_kd_metric_table_verifier_gate",
        "reviewer.require_professional_artifact_package",
    ]
    for src, dst in zip(chain, chain[1:]):
        ensure_edge(dag, src, dst)
    ensure_edge(dag, "ops.dto_kd_full_training_grid", "reviewer.compare_result_shapes")
    ensure_edge(dag, "reviewer.keep_exact_artifact_debt", decision_id)
    dag.setdefault("previous_loop_updates", []).append(
        {
            "iteration": 3,
            "created_at_utc": utc_now(),
            "source": "dto_kd_external_professional_gate",
            "status": gate["status"],
            "blocker_ids": [b["id"] for b in gate["blockers"]],
            "converged": False,
        }
    )
    sig_src = json.dumps(dag.get("nodes", []), sort_keys=True) + json.dumps(dag.get("edges", []), sort_keys=True)
    dag["signature"] = hashlib.sha256(sig_src.encode("utf-8")).hexdigest()[:16]
    write_json(dag_path, dag)
    write_json(paper_dir / "paper_author_gap_dag_iter_03.json", dag)
    return dag


def verifier(cfg: dict[str, Any], gate: dict[str, Any], dag: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "artifact_kind": "dto_kd_specialized_verifier",
        "created_at_utc": utc_now(),
        "paper_id": cfg["paper_id"],
        "paper_title": cfg["title"],
        "iteration": 3,
        "converged": False,
        "professional_ready": False,
        "checks": [
            {"name": "blind_contract", "status": "pass", "detail": dag.get("blind_contract", {})},
            {"name": "official_repo_source", "status": "blocked", "detail": "No official source repository found."},
            {"name": "gpu_requirement_interpretation", "status": "blocked", "detail": "Paper requires four H100; local RTX 4090 run would be reduced/proxy."},
            {"name": "professional_artifact_package", "status": "blocked", "detail": gate["blockers"]},
            {"name": "result_shape_comparison_ready", "status": "blocked", "detail": "requires full outputs for Table 1-5 and Figure 3-4 comparison."},
        ],
        "required_updates": [
            {
                "id": "update.dto_kd_external_source_and_full_training_contract",
                "reason": gate["status"],
                "success_criteria": [
                    "official repo or faithful implementation exists",
                    "four-H100/data/checkpoint artifacts materialized",
                    "full ImageNet/CIFAR/COCO training outputs generated",
                    "verifier compares operational results to paper table and figure targets",
                ],
            }
        ],
        "artifact_paths": {
            "professional_gate": str(Path(cfg["runner_dir"]) / "professional_gate_result.json"),
            "source_manifest": str(Path(cfg["runner_dir"]) / "source_manifest.json"),
            "model_data_manifest": str(Path(cfg["runner_dir"]) / "model_data_manifest.json"),
            "dag_iter_03": str(RUN_ROOT / "paper_runs" / cfg["paper_run"] / "paper_author_gap_dag_iter_03.json"),
        },
    }
    write_json(Path(cfg["runner_dir"]) / "dto_kd_specialized_verifier.json", payload)
    write_json(RUN_ROOT / "paper_runs" / cfg["paper_run"] / "verifier_result_iter_03.json", payload)
    return payload


def update_queue_summary(cfg: dict[str, Any], gate: dict[str, Any], verify: dict[str, Any], dag: dict[str, Any]) -> None:
    queue_obj = read_json(QUEUE_PATH)
    for item in queue_obj.get("queue", []):
        if item.get("paper_id") == cfg["paper_id"]:
            item["priority"] = "high"
            item["professional_blocker"] = gate["status"]
            item["repo_exact_rerun_status"] = "blocked_no_official_repo_requires_faithful_reimplementation_and_full_h100_grid"
            item["repo_paths"] = []
            item["specialized_runner_status"] = gate["status"]
            item["specialized_runner_artifact_dir"] = str(Path(cfg["runner_dir"]))
            statuses = item.setdefault("implementation_statuses", [])
            for status in [
                "official_source_repo_not_found_verified",
                "h100_hardware_required_current_4090_proxy_rejected",
                "professional_gate_blocked",
                "faithful_reimplementation_required_before_execution",
            ]:
                if status not in statuses:
                    statuses.append(status)
            item["specialized_runner_evidence"] = {
                "blockers": gate["blockers"],
                "verifier_path": str(Path(cfg["runner_dir"]) / "dto_kd_specialized_verifier.json"),
                "source_manifest_path": str(Path(cfg["runner_dir"]) / "source_manifest.json"),
                "model_data_manifest_path": str(Path(cfg["runner_dir"]) / "model_data_manifest.json"),
            }
            break
    write_json(QUEUE_PATH, queue_obj)

    summary = read_json(SUMMARY_PATH)
    for paper in summary.get("papers", []):
        if paper.get("paper_id") == cfg["paper_id"]:
            paper["final_status"] = "blocked_waiting_for_professional_artifacts_after_dag_update"
            paper["converged"] = False
            paper["repo_paths"] = []
            paper["specialized_runner_status"] = gate["status"]
            paper["professional_blocker"] = gate["status"]
            paper["specialized_runner_artifact_dir"] = str(Path(cfg["runner_dir"]))
            statuses = paper.setdefault("implementation_statuses", [])
            for status in [
                "official_source_repo_not_found_verified",
                "h100_hardware_required_current_4090_proxy_rejected",
                "professional_gate_blocked",
                "faithful_reimplementation_required_before_execution",
            ]:
                if status not in statuses:
                    statuses.append(status)
            paper["iterations"] = paper.get("iterations", []) + [
                {
                    "iteration": 3,
                    "dag_signature": dag.get("signature"),
                    "simulation": {
                        "paper_id": cfg["paper_id"],
                        "paper_title": cfg["title"],
                        "created_at_utc": gate["created_at_utc"],
                        "input_contract": dag.get("blind_contract", {}),
                        "paper_text_seen": False,
                        "previous_memory_seen": False,
                        "oracle_results_seen": False,
                        "author_decision": "explicit_operational_blocker",
                        "professional_package_ready": False,
                        "professional_package_reason": gate["status"],
                        "reduced_or_proxy_used_for_convergence": False,
                        "raw_artifact_level": "source_absence_hardware_data_checkpoint_preflight_only",
                        "gpu_used": False,
                        "gpu_use_reason": "paper requires four H100 and source/data/checkpoints; local RTX 4090 proxy rejected",
                        "blocker_ids": [b["id"] for b in gate["blockers"]],
                    },
                    "verification": verify,
                }
            ]
            break
    summary["updated_at_utc"] = utc_now()
    summary["final_status"] = "running_professional_two_loop_not_converged"
    write_json(SUMMARY_PATH, summary)


def write_status(cfg: dict[str, Any], gate: dict[str, Any], verify: dict[str, Any], dag: dict[str, Any]) -> None:
    status_path = Path(cfg["runner_dir"]) / "DTO_KD_SPECIALIZED_STATUS.md"
    lines = [
        "# DTO-KD External Professional Gate",
        "",
        f"- Paper id: `{cfg['paper_id']}`",
        f"- Title: {cfg['title']}",
        f"- Status: `{gate['status']}`",
        "- Converged: `false`",
        "- Professional ready: `false`",
        "- GPU used: `false`; local RTX 4090 proxy is rejected because the paper requires four H100 plus source/data/checkpoints.",
        f"- DAG signature: `{dag.get('signature')}`",
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
        f"- Professional gate: `{Path(cfg['runner_dir']) / 'professional_gate_result.json'}`",
        f"- Verifier: `{Path(cfg['runner_dir']) / 'dto_kd_specialized_verifier.json'}`",
        f"- Source manifest: `{Path(cfg['runner_dir']) / 'source_manifest.json'}`",
        f"- Model/data manifest: `{Path(cfg['runner_dir']) / 'model_data_manifest.json'}`",
        "",
        "## Verifier Checks",
        "",
    ]
    for check in verify["checks"]:
        lines.append(f"- `{check['name']}`: `{check['status']}`")
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cfg = dict(CFG)
    cfg["runner_dir"] = str(RUN_ROOT / "specialized_runners" / cfg["short"])
    Path(cfg["runner_dir"]).mkdir(parents=True, exist_ok=True)
    source = source_manifest(cfg)
    artifacts = artifact_manifest(cfg)
    gate = professional_gate(cfg, source, artifacts)
    dag = update_dag(cfg, gate)
    verify = verifier(cfg, gate, dag)
    update_queue_summary(cfg, gate, verify, dag)
    write_status(cfg, gate, verify, dag)
    refresh = run_cmd([sys.executable, str(RUN_ROOT / "refresh_longgoal_status.py")], cwd=RUN_ROOT, timeout=120)
    print(
        json.dumps(
            {
                "paper_id": cfg["paper_id"],
                "status": gate["status"],
                "converged": False,
                "blocker_count": len(gate["blockers"]),
                "blocker_ids": [b["id"] for b in gate["blockers"]],
                "dag_signature": dag.get("signature"),
                "status_path": str(Path(cfg["runner_dir"]) / "DTO_KD_SPECIALIZED_STATUS.md"),
                "refresh_returncode": refresh["returncode"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
