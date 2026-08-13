#!/usr/bin/env python3
"""Refresh strict DIRS long-goal status from queue plus summary state."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = RUN_ROOT / "remaining19_strict_dirs_summary.json"
QUEUE_PATH = RUN_ROOT / "specialized_runner_queue.json"
LONG_STATUS_PATH = RUN_ROOT / "LONGGOAL_STATUS.md"
SPECIALIZED_QUEUE_MD = RUN_ROOT / "SPECIALIZED_RUNNER_QUEUE.md"
PROPHET_STATUS_PATH = RUN_ROOT / "specialized_runners/prophet/custom_full_gsm8k_llada8b/status.json"
PROPHET_SUMMARY_PATH = RUN_ROOT / "specialized_runners/prophet/custom_full_gsm8k_llada8b/summary.json"
PROPHET_ROWS_PATH = RUN_ROOT / "specialized_runners/prophet/custom_full_gsm8k_llada8b/per_sample_results.jsonl"
PROPHET_TRAJECTORY_STATUS_PATH = RUN_ROOT / "specialized_runners/prophet/trajectory_dataset_analysis/trajectory_dataset_status.json"
PROPHET_LIVE_VERIFIER_PATH = RUN_ROOT / "paper_runs/iclr2026_g88nt4ietg_prophet_dlm_early_commit_decoding/verifier_result_iter_07_live.json"
PROPHET_REFRESH_SCRIPT = RUN_ROOT / "specialized_runners/prophet/refresh_prophet_live_verifier.py"
PROPHET_INTEGRITY_REPORT_PATH = RUN_ROOT / "specialized_runners/prophet/prophet_live_integrity_report.json"
PROPHET_INTEGRITY_STATUS_PATH = RUN_ROOT / "specialized_runners/prophet/PROPHET_LIVE_INTEGRITY_STATUS.md"
PROPHET_PAPER_COMPARISON_REPORT_PATH = RUN_ROOT / "specialized_runners/prophet/prophet_paper_result_comparison.json"
PROPHET_PAPER_COMPARISON_STATUS_PATH = RUN_ROOT / "specialized_runners/prophet/PROPHET_PAPER_RESULT_COMPARISON_STATUS.md"
PROPHET_MULTIBENCHMARK_MANIFEST_PATH = RUN_ROOT / "specialized_runners/prophet/multibenchmark_table1_full/multibenchmark_grid_campaign.json"
PROPHET_TABLE2_MANIFEST_PATH = RUN_ROOT / "specialized_runners/prophet/table2_acceleration_combinations/table2_acceleration_campaign.json"
PROPHET_DREAM_MANIFEST_PATH = RUN_ROOT / "specialized_runners/prophet/dream7b_table1_axis/dream7b_axis_campaign.json"
PROPHET_SOURCE_PARITY_AUDIT_REPORT_PATH = RUN_ROOT / "specialized_runners/prophet/source_parity_blocker_audit.json"
PROPHET_SOURCE_PARITY_AUDIT_STATUS_PATH = RUN_ROOT / "specialized_runners/prophet/SOURCE_PARITY_BLOCKER_AUDIT.md"
PROPHET_GSM8K_RISK_AUDIT_REPORT_PATH = RUN_ROOT / "specialized_runners/prophet/gsm8k_live_shape_risk_audit.json"
PROPHET_GSM8K_RISK_AUDIT_STATUS_PATH = RUN_ROOT / "specialized_runners/prophet/GSM8K_LIVE_SHAPE_RISK_AUDIT.md"
PROPHET_GSM8K_PROTOCOL_AUDIT_REPORT_PATH = RUN_ROOT / "specialized_runners/prophet/gsm8k_protocol_parity_audit.json"
PROPHET_GSM8K_PROTOCOL_AUDIT_STATUS_PATH = RUN_ROOT / "specialized_runners/prophet/GSM8K_PROTOCOL_PARITY_AUDIT.md"
PROPHET_PAPER_ID = "ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding"
LOOP1_REPAIR_REPORT_PATH = RUN_ROOT / "loop1_dag_repair_audit_20260723.json"
COMPLETION_AUDIT_PATH = RUN_ROOT / "strict_dirs_completion_audit_20260723.json"
COMPLETION_AUDIT_SCRIPT = RUN_ROOT / "strict_dirs_completion_auditor.py"
STALENESS_AUDIT_PATH = RUN_ROOT / "explicit_blocker_staleness_audit_20260723.json"
STALENESS_AUDIT_STATUS_PATH = RUN_ROOT / "EXPLICIT_BLOCKER_STALENESS_AUDIT_20260723.md"
STALENESS_AUDIT_SCRIPT = RUN_ROOT / "explicit_blocker_staleness_audit.py"
GPU_RECHECK_REPORT_PATH = RUN_ROOT / "gpu_recheck_dispatcher_report.json"
GPU_RECHECK_STATE_PATH = RUN_ROOT / "gpu_recheck_dispatcher_state.json"
GPU_RECHECK_STATUS_PATH = RUN_ROOT / "GPU_RECHECK_DISPATCHER_STATUS.md"
GATE_SWEEP_REPORT_PATH = RUN_ROOT / "professional_gate_sweep_report.json"
GATE_SWEEP_STATE_PATH = RUN_ROOT / "professional_gate_sweep_state.json"
GATE_SWEEP_STATUS_PATH = RUN_ROOT / "PROFESSIONAL_GATE_SWEEP_STATUS.md"
MULTI_GPU_SCHEDULER_REPORT_PATH = RUN_ROOT / "multi_gpu_professional_scheduler_report.json"
MULTI_GPU_SCHEDULER_STATE_PATH = RUN_ROOT / "multi_gpu_professional_scheduler_state.json"
MULTI_GPU_SCHEDULER_STATUS_PATH = RUN_ROOT / "MULTI_GPU_PROFESSIONAL_SCHEDULER_STATUS.md"
SUPERVISOR_STATE_PATH = RUN_ROOT / "strict_dirs_goal_supervisor_state.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def merge_queue_into_summary(summary: dict[str, Any], queue_items: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {item.get("paper_id"): item for item in queue_items}
    for paper in summary.get("papers", []):
        item = by_id.get(paper.get("paper_id"))
        if not item:
            continue
        for key in [
            "repo_paths",
            "repo_exact_rerun_status",
            "professional_blocker",
            "specialized_runner_status",
            "specialized_runner_artifact_dir",
            "specialized_runner_evidence",
        ]:
            value = item.get(key)
            if value:
                paper[key] = value
        if item.get("implementation_statuses"):
            statuses = paper.setdefault("implementation_statuses", [])
            for status in item["implementation_statuses"]:
                if status not in statuses:
                    statuses.append(status)
    summary["updated_at_utc"] = utc_now()
    summary["final_status"] = "running_professional_two_loop_not_converged"
    return summary


def refresh_prophet_live_verifier_artifact() -> None:
    if not PROPHET_REFRESH_SCRIPT.exists() or not PROPHET_STATUS_PATH.exists():
        return
    subprocess.run(
        ["python", str(PROPHET_REFRESH_SCRIPT)],
        cwd=str(RUN_ROOT),
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def refresh_completion_audit_artifact() -> None:
    if not COMPLETION_AUDIT_SCRIPT.exists():
        return
    subprocess.run(
        ["python", str(COMPLETION_AUDIT_SCRIPT)],
        cwd=str(RUN_ROOT),
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def refresh_staleness_audit_artifact() -> None:
    if not STALENESS_AUDIT_SCRIPT.exists() or not COMPLETION_AUDIT_PATH.exists():
        return
    subprocess.run(
        ["python", str(STALENESS_AUDIT_SCRIPT)],
        cwd=str(RUN_ROOT),
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def attach_staleness_audit(summary: dict[str, Any]) -> dict[str, Any]:
    if not STALENESS_AUDIT_PATH.exists():
        return summary
    audit = read_json(STALENESS_AUDIT_PATH)
    counts = audit.get("counts", {})
    papers = audit.get("papers", [])
    summary["explicit_blocker_staleness_audit"] = {
        "status": "present",
        "report_path": str(STALENESS_AUDIT_PATH),
        "markdown_path": str(STALENESS_AUDIT_STATUS_PATH),
        "updated_at_utc": audit.get("created_at_utc"),
        "counts": counts,
        "gpu_recheck_paper_ids": [
            paper.get("paper_id")
            for paper in papers
            if paper.get("status") == "explicit_blocker_valid_recheck_after_gpu_release"
        ],
        "weak_evidence_paper_ids": [
            paper.get("paper_id")
            for paper in papers
            if paper.get("weak_evidence")
        ],
        "policy": audit.get("policy"),
    }
    return summary


def attach_gpu_recheck_dispatch(summary: dict[str, Any]) -> dict[str, Any]:
    if not GPU_RECHECK_REPORT_PATH.exists():
        return summary
    report = read_json(GPU_RECHECK_REPORT_PATH)
    state = read_json(GPU_RECHECK_STATE_PATH) if GPU_RECHECK_STATE_PATH.exists() else {}
    summary["gpu_recheck_dispatcher"] = {
        "status": report.get("status"),
        "report_path": str(GPU_RECHECK_REPORT_PATH),
        "state_path": str(GPU_RECHECK_STATE_PATH),
        "markdown_path": str(GPU_RECHECK_STATUS_PATH),
        "updated_at_utc": report.get("created_at_utc"),
        "pending_before_count": report.get("pending_before_count"),
        "recheck_executed_count": len(report.get("recheck_results", []) or []),
        "attempted_paper_ids": state.get("attempted_paper_ids", []),
        "policy": report.get("policy"),
    }
    return summary


def attach_professional_gate_sweep(summary: dict[str, Any]) -> dict[str, Any]:
    if not GATE_SWEEP_REPORT_PATH.exists():
        return summary
    report = read_json(GATE_SWEEP_REPORT_PATH)
    state = read_json(GATE_SWEEP_STATE_PATH) if GATE_SWEEP_STATE_PATH.exists() else {}
    summary["professional_gate_sweep"] = {
        "status": report.get("status"),
        "scope": report.get("scope"),
        "report_path": str(GATE_SWEEP_REPORT_PATH),
        "state_path": str(GATE_SWEEP_STATE_PATH),
        "markdown_path": str(GATE_SWEEP_STATUS_PATH),
        "updated_at_utc": report.get("created_at_utc"),
        "selected_before_cap_count": report.get("selected_before_cap_count"),
        "executed_count": len(report.get("results", []) or []),
        "attempted_paper_ids": state.get("attempted_paper_ids", []),
        "policy": report.get("policy"),
    }
    return summary


def attach_multi_gpu_scheduler(summary: dict[str, Any]) -> dict[str, Any]:
    if not MULTI_GPU_SCHEDULER_REPORT_PATH.exists():
        return summary
    report = read_json(MULTI_GPU_SCHEDULER_REPORT_PATH)
    state = read_json(MULTI_GPU_SCHEDULER_STATE_PATH) if MULTI_GPU_SCHEDULER_STATE_PATH.exists() else {}
    launch_history = state.get("launch_history") or report.get("launch_history") or []
    summary["multi_gpu_professional_scheduler"] = {
        "status": report.get("status"),
        "paper_id": report.get("paper_id"),
        "paper_title": report.get("paper_title"),
        "report_path": str(MULTI_GPU_SCHEDULER_REPORT_PATH),
        "state_path": str(MULTI_GPU_SCHEDULER_STATE_PATH),
        "markdown_path": str(MULTI_GPU_SCHEDULER_STATUS_PATH),
        "updated_at_utc": report.get("created_at_utc"),
        "gpu_pool": report.get("policy", {}).get("gpu_pool", []),
        "launch_count": sum(1 for item in report.get("launches", []) if item.get("launched")),
        "active_gpu_claim_count": len(report.get("final_active_gpu_claims", []) or []),
        "launch_history_count": len(launch_history),
        "recent_launch_history": launch_history[-6:],
        "policy": report.get("policy"),
    }
    return summary


def refresh_active_prophet_run(summary: dict[str, Any]) -> dict[str, Any]:
    if not PROPHET_STATUS_PATH.exists():
        return summary
    status = read_json(PROPHET_STATUS_PATH)
    running_summary = read_json(PROPHET_SUMMARY_PATH) if PROPHET_SUMMARY_PATH.exists() else {}
    trajectory = read_json(PROPHET_TRAJECTORY_STATUS_PATH) if PROPHET_TRAJECTORY_STATUS_PATH.exists() else {}
    live_verifier = read_json(PROPHET_LIVE_VERIFIER_PATH) if PROPHET_LIVE_VERIFIER_PATH.exists() else {}
    live_runner = live_verifier.get("live_full_gsm8k_runner", {})
    live_trajectory = live_verifier.get("trajectory_dataset_analysis", {})
    live_integrity = live_verifier.get("live_integrity", {})
    paper_comparison = live_verifier.get("paper_result_comparison", {})
    ablation_grid = live_verifier.get("ablation_grid_campaign", {})
    ablation_integrity = live_verifier.get("ablation_grid_integrity", {})
    multibenchmark_grid = live_verifier.get("multibenchmark_grid_campaign", {})
    table2_campaign = live_verifier.get("table2_acceleration_campaign", {})
    dream_campaign = live_verifier.get("dream7b_axis_campaign", {})
    source_parity = live_verifier.get("source_parity_blocker_audit", {})
    risk_audit = live_verifier.get("gsm8k_live_shape_risk_audit", {})
    protocol_audit = live_verifier.get("gsm8k_protocol_parity_audit", {})
    rows = PROPHET_ROWS_PATH.read_text(encoding="utf-8", errors="replace").splitlines() if PROPHET_ROWS_PATH.exists() else []
    aggregates = running_summary.get("aggregates", {})
    paired_completed = (
        live_runner.get("paired_completed_samples")
        or aggregates.get("baseline", {}).get("completed_samples")
        or aggregates.get("prophet", {}).get("completed_samples")
        or status.get("completed_sample_indices")
    )
    active = {
        "paper_id": PROPHET_PAPER_ID,
        "title": "Diffusion Language Models Know the Answer Before Decoding",
        "status": live_verifier.get("status") or "running_full_custom_gsm8k_on_gpu3_and_authenticated_trajectory_downloader",
        "pid": live_runner.get("pid") or status.get("pid"),
        "gpu": "physical GPU 3",
        "cuda_visible_devices": str(live_runner.get("cuda_visible_devices") or status.get("cuda_visible_devices") or status.get("gpu") or "3"),
        "full_split_requested": True,
        "paired_completed_samples": paired_completed,
        "total_samples": live_runner.get("total_samples") or status.get("total_samples") or running_summary.get("total_samples"),
        "jsonl_rows": live_runner.get("jsonl_rows") or len(rows),
        "rows_path": str(PROPHET_ROWS_PATH),
        "summary_path": str(PROPHET_SUMMARY_PATH),
        "running_summary": live_runner.get("running_summary") or running_summary,
        "live_verifier_path": str(PROPHET_LIVE_VERIFIER_PATH),
        "integrity_report_path": live_integrity.get("report_path") or str(PROPHET_INTEGRITY_REPORT_PATH),
        "integrity_status": live_integrity.get("status"),
        "integrity_gate_status": live_integrity.get("gate_status"),
        "paper_comparison_report_path": paper_comparison.get("report_path") or str(PROPHET_PAPER_COMPARISON_REPORT_PATH),
        "paper_comparison_status": paper_comparison.get("status"),
        "paper_comparison_gate_status": paper_comparison.get("gate_status"),
        "ablation_grid_status": ablation_grid.get("status"),
        "ablation_grid_manifest_path": ablation_grid.get("manifest_path"),
        "multibenchmark_grid_status": multibenchmark_grid.get("status"),
        "multibenchmark_grid_manifest_path": multibenchmark_grid.get("manifest_path"),
        "table2_acceleration_status": table2_campaign.get("status"),
        "table2_acceleration_manifest_path": table2_campaign.get("manifest_path"),
        "dream7b_axis_status": dream_campaign.get("status"),
        "dream7b_axis_manifest_path": dream_campaign.get("manifest_path"),
        "source_parity_audit_status": source_parity.get("status"),
        "source_parity_audit_report_path": source_parity.get("report_path") or str(PROPHET_SOURCE_PARITY_AUDIT_REPORT_PATH),
        "source_parity_audit_status_path": source_parity.get("status_path") or str(PROPHET_SOURCE_PARITY_AUDIT_STATUS_PATH),
        "gsm8k_live_shape_risk_audit_status": risk_audit.get("status"),
        "gsm8k_live_shape_risk_audit_report_path": risk_audit.get("report_path") or str(PROPHET_GSM8K_RISK_AUDIT_REPORT_PATH),
        "gsm8k_live_shape_risk_audit_status_path": risk_audit.get("status_path") or str(PROPHET_GSM8K_RISK_AUDIT_STATUS_PATH),
        "gsm8k_live_shape_risk_audit_loop2_author_can_read": risk_audit.get("loop2_author_can_read"),
        "gsm8k_protocol_parity_audit_status": protocol_audit.get("status"),
        "gsm8k_protocol_parity_audit_report_path": protocol_audit.get("report_path") or str(PROPHET_GSM8K_PROTOCOL_AUDIT_REPORT_PATH),
        "gsm8k_protocol_parity_audit_status_path": protocol_audit.get("status_path") or str(PROPHET_GSM8K_PROTOCOL_AUDIT_STATUS_PATH),
        "gsm8k_protocol_parity_audit_loop2_author_can_read": protocol_audit.get("loop2_author_can_read"),
        "trajectory_status_path": str(PROPHET_TRAJECTORY_STATUS_PATH),
        "trajectory_authenticated": bool(trajectory.get("authenticated") or trajectory.get("token_authenticated")),
        "trajectory_pid": trajectory.get("pid"),
        "trajectory_status": live_trajectory.get("status") or trajectory.get("status"),
        "trajectory_settings_completed": live_trajectory.get("settings_completed") or trajectory.get("settings_completed"),
        "trajectory_total_settings": live_trajectory.get("total_settings") or trajectory.get("total_settings"),
        "trajectory_observed_file_count": trajectory.get("file_count_downloaded") or trajectory.get("observed_file_count"),
        "trajectory_observed_size": trajectory.get("size_human") or trajectory.get("observed_size_human"),
        "updated_at_utc": live_verifier.get("created_at_utc") or status.get("updated_at_utc") or utc_now(),
    }
    runs = [run for run in summary.get("active_specialized_runs", []) if run.get("paper_id") != active["paper_id"]]
    if status.get("status") in {"running", "running_or_partial"}:
        runs.append(active)
    summary["active_specialized_runs"] = runs
    return summary


def build_prophet_live_queue_fields() -> dict[str, Any] | None:
    if not PROPHET_LIVE_VERIFIER_PATH.exists():
        return None
    live_verifier = read_json(PROPHET_LIVE_VERIFIER_PATH)
    live_runner = live_verifier.get("live_full_gsm8k_runner", {})
    live_trajectory = live_verifier.get("trajectory_dataset_analysis", {})
    live_integrity = live_verifier.get("live_integrity", {})
    paper_comparison = live_verifier.get("paper_result_comparison", {})
    ablation_grid = live_verifier.get("ablation_grid_campaign", {})
    ablation_integrity = live_verifier.get("ablation_grid_integrity", {})
    multibenchmark_grid = live_verifier.get("multibenchmark_grid_campaign", {})
    table2_campaign = live_verifier.get("table2_acceleration_campaign", {})
    dream_campaign = live_verifier.get("dream7b_axis_campaign", {})
    source_parity = live_verifier.get("source_parity_blocker_audit", {})
    risk_audit = live_verifier.get("gsm8k_live_shape_risk_audit", {})
    protocol_audit = live_verifier.get("gsm8k_protocol_parity_audit", {})
    status = live_verifier.get("status") or "running_full_custom_gsm8k_on_gpu3_and_trajectory_analysis"
    return {
        "specialized_runner_status": status,
        "professional_blocker": "running_full_custom_gsm8k_on_gpu3_and_trajectory_analysis_pending_full_artifacts",
        "specialized_runner_evidence": {
            "custom_full_gsm8k": {
                "pid": live_runner.get("pid"),
                "cuda_visible_devices": live_runner.get("cuda_visible_devices"),
                "model_id": live_runner.get("model_id"),
                "full_split_requested": live_runner.get("full_split_requested"),
                "full_gsm8k_complete": live_runner.get("full_gsm8k_complete"),
                "paired_completed_samples": live_runner.get("paired_completed_samples"),
                "total_samples": live_runner.get("total_samples"),
                "jsonl_rows": live_runner.get("jsonl_rows"),
                "rows_path": live_runner.get("rows_path"),
                "status_path": live_runner.get("status_path"),
                "summary_path": live_runner.get("summary_path"),
                "running_summary": live_runner.get("running_summary"),
                "updated_at_utc": live_runner.get("updated_at_utc"),
            },
            "live_integrity": {
                "status": live_integrity.get("status"),
                "gate_status": live_integrity.get("gate_status"),
                "reasons": live_integrity.get("reasons"),
                "row_count": live_integrity.get("row_count"),
                "paired_completed_samples_from_rows": live_integrity.get("paired_completed_samples_from_rows"),
                "incomplete_sample_count": live_integrity.get("incomplete_sample_count"),
                "duplicate_pair_count": live_integrity.get("duplicate_pair_count"),
                "json_parse_error_count": live_integrity.get("json_parse_error_count"),
                "summary_consistency": live_integrity.get("summary_consistency"),
                "report_path": live_integrity.get("report_path") or str(PROPHET_INTEGRITY_REPORT_PATH),
                "status_path": live_integrity.get("status_path") or str(PROPHET_INTEGRITY_STATUS_PATH),
                "updated_at_utc": live_integrity.get("updated_at_utc"),
            },
            "paper_result_comparison": {
                "status": paper_comparison.get("status"),
                "gate_status": paper_comparison.get("gate_status"),
                "blockers": paper_comparison.get("blockers"),
                "primary_gsm8k_status": paper_comparison.get("primary_gsm8k_status"),
                "trajectory_status": paper_comparison.get("trajectory_status"),
                "report_path": paper_comparison.get("report_path") or str(PROPHET_PAPER_COMPARISON_REPORT_PATH),
                "status_path": paper_comparison.get("status_path") or str(PROPHET_PAPER_COMPARISON_STATUS_PATH),
                "updated_at_utc": paper_comparison.get("updated_at_utc"),
            },
            "ablation_grid_campaign": {
                "status": ablation_grid.get("status"),
                "manifest_path": ablation_grid.get("manifest_path"),
                "status_path": ablation_grid.get("status_path"),
                "runnable_config_count": ablation_grid.get("runnable_config_count"),
                "completed_config_count": ablation_grid.get("completed_config_count"),
                "running_config_count": ablation_grid.get("running_config_count"),
                "pending_config_count": ablation_grid.get("pending_config_count"),
                "blocked_configs": ablation_grid.get("blocked_configs"),
                "updated_at_utc": ablation_grid.get("updated_at_utc"),
            },
            "ablation_grid_integrity": {
                "status": ablation_integrity.get("status"),
                "report_path": ablation_integrity.get("report_path"),
                "status_path": ablation_integrity.get("status_path"),
                "blocked_config_ids": ablation_integrity.get("blocked_config_ids"),
                "manifest_blocked_config_ids": ablation_integrity.get("manifest_blocked_config_ids"),
                "manifest_blocked_configs": ablation_integrity.get("manifest_blocked_configs"),
                "running_config_ids": ablation_integrity.get("running_config_ids"),
                "complete_config_ids": ablation_integrity.get("complete_config_ids"),
                "pending_config_ids": ablation_integrity.get("pending_config_ids"),
                "updated_at_utc": ablation_integrity.get("updated_at_utc"),
            },
            "multibenchmark_grid_campaign": {
                "status": multibenchmark_grid.get("status"),
                "manifest_path": multibenchmark_grid.get("manifest_path"),
                "status_path": multibenchmark_grid.get("status_path"),
                "runnable_config_count": multibenchmark_grid.get("runnable_config_count"),
                "linked_existing_artifact_count": multibenchmark_grid.get("linked_existing_artifact_count"),
                "completed_config_count": multibenchmark_grid.get("completed_config_count"),
                "running_config_count": multibenchmark_grid.get("running_config_count"),
                "pending_config_count": multibenchmark_grid.get("pending_config_count"),
                "blocked_configs": multibenchmark_grid.get("blocked_configs"),
                "updated_at_utc": multibenchmark_grid.get("updated_at_utc"),
            },
            "table2_acceleration_campaign": {
                "status": table2_campaign.get("status"),
                "manifest_path": table2_campaign.get("manifest_path"),
                "status_path": table2_campaign.get("status_path"),
                "linked_existing_artifact_count": table2_campaign.get("linked_existing_artifact_count"),
                "linked_existing_complete_count": table2_campaign.get("linked_existing_complete_count"),
                "runnable_config_count": table2_campaign.get("runnable_config_count"),
                "blocked_configs": table2_campaign.get("blocked_configs"),
                "updated_at_utc": table2_campaign.get("updated_at_utc"),
            },
            "dream7b_axis_campaign": {
                "status": dream_campaign.get("status"),
                "manifest_path": dream_campaign.get("manifest_path"),
                "status_path": dream_campaign.get("status_path"),
                "runnable_config_count": dream_campaign.get("runnable_config_count"),
                "blocked_configs": dream_campaign.get("blocked_configs"),
                "updated_at_utc": dream_campaign.get("updated_at_utc"),
            },
            "source_parity_blocker_audit": {
                "status": source_parity.get("status"),
                "report_path": source_parity.get("report_path") or str(PROPHET_SOURCE_PARITY_AUDIT_REPORT_PATH),
                "status_path": source_parity.get("status_path") or str(PROPHET_SOURCE_PARITY_AUDIT_STATUS_PATH),
                "blocker_ids": source_parity.get("blocker_ids"),
                "runnable_new_node_count": source_parity.get("runnable_new_node_count"),
                "can_converge_from_this_audit_alone": source_parity.get("can_converge_from_this_audit_alone"),
                "remote_matches_local": source_parity.get("remote_matches_local"),
                "updated_at_utc": source_parity.get("updated_at_utc"),
            },
            "gsm8k_live_shape_risk_audit": {
                "status": risk_audit.get("status"),
                "report_path": risk_audit.get("report_path") or str(PROPHET_GSM8K_RISK_AUDIT_REPORT_PATH),
                "status_path": risk_audit.get("status_path") or str(PROPHET_GSM8K_RISK_AUDIT_STATUS_PATH),
                "loop2_author_can_read": risk_audit.get("loop2_author_can_read"),
                "paper_oracle_target_values_included": risk_audit.get("paper_oracle_target_values_included"),
                "can_converge_from_this_audit_alone": risk_audit.get("can_converge_from_this_audit_alone"),
                "do_not_stop_before_full_split": risk_audit.get("do_not_stop_before_full_split"),
                "failing_metrics": risk_audit.get("failing_metrics"),
                "repair_axis_ids": risk_audit.get("repair_axis_ids"),
                "updated_at_utc": risk_audit.get("updated_at_utc"),
            },
            "gsm8k_protocol_parity_audit": {
                "status": protocol_audit.get("status"),
                "report_path": protocol_audit.get("report_path") or str(PROPHET_GSM8K_PROTOCOL_AUDIT_REPORT_PATH),
                "status_path": protocol_audit.get("status_path") or str(PROPHET_GSM8K_PROTOCOL_AUDIT_STATUS_PATH),
                "loop2_author_can_read": protocol_audit.get("loop2_author_can_read"),
                "paper_oracle_target_values_included": protocol_audit.get("paper_oracle_target_values_included"),
                "can_converge_from_this_audit_alone": protocol_audit.get("can_converge_from_this_audit_alone"),
                "finding_statuses": protocol_audit.get("finding_statuses"),
                "covered_repair_nodes": protocol_audit.get("covered_repair_nodes"),
                "updated_at_utc": protocol_audit.get("updated_at_utc"),
            },
            "trajectory_dataset": {
                "status": live_trajectory.get("status"),
                "settings_completed": live_trajectory.get("settings_completed"),
                "total_settings": live_trajectory.get("total_settings"),
                "current_setting": live_trajectory.get("current_setting"),
                "rows_written": live_trajectory.get("rows_written"),
                "status_path": live_trajectory.get("status_path"),
                "summary_path": live_trajectory.get("summary_path"),
                "trajectory_complete": live_trajectory.get("trajectory_complete"),
                "updated_at_utc": live_trajectory.get("updated_at_utc"),
            },
            "live_verifier": {
                "path": str(PROPHET_LIVE_VERIFIER_PATH),
                "converged": live_verifier.get("converged"),
                "professional_package_ready": live_verifier.get("professional_package_ready"),
                "support_only_until": live_verifier.get("support_only_until", []),
                "created_at_utc": live_verifier.get("created_at_utc"),
            },
        },
        "implementation_statuses": [
            "code_inspected",
            "trajectory_dataset_available",
            "custom_full_gsm8k_on_gpu3_running",
            "live_jsonl_integrity_checking",
            "full_released_trajectory_dataset_analyzing",
            "full_ablation_grid_campaign_manifest_ready",
            "full_table1_multibenchmark_campaign_manifest_ready",
            "table2_acceleration_combination_campaign_manifest_ready",
            "dream7b_axis_campaign_manifest_ready",
            "source_parity_blocker_audit_ready",
            "gsm8k_live_shape_risk_audit_ready",
            "gsm8k_protocol_parity_audit_ready",
            "live_verifier_refreshing_from_gpu_artifacts",
        ],
        "next_actions": [
            "let full GSM8K baseline/Prophet paired split reach 1319/1319 samples",
            "complete all configured trajectory analysis settings",
            "resolve or explicitly block multi-benchmark grid debt",
            "produce or explicitly block static step and block-length ablations",
            "produce or explicitly block the Dream-7B axis",
            "rerun verifier against paper tables, figures, paragraphs, and appendix evidence",
        ],
    }


def apply_prophet_live_to_queue(queue_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    live_fields = build_prophet_live_queue_fields()
    if not live_fields:
        return queue_items
    for item in queue_items:
        if item.get("paper_id") != PROPHET_PAPER_ID:
            continue
        item["specialized_runner_status"] = live_fields["specialized_runner_status"]
        item["professional_blocker"] = live_fields["professional_blocker"]
        evidence = item.setdefault("specialized_runner_evidence", {})
        evidence.update(live_fields["specialized_runner_evidence"])
        statuses = item.setdefault("implementation_statuses", [])
        for status in live_fields["implementation_statuses"]:
            if status not in statuses:
                statuses.append(status)
        item["next_actions"] = live_fields["next_actions"]
    return queue_items


def render_status(summary: dict[str, Any]) -> None:
    papers = summary.get("papers", [])
    accepted = sum(1 for paper in papers if paper.get("converged"))
    not_converged = len(papers) - accepted
    visible = sorted(
        papers,
        key=lambda paper: (
            0 if paper.get("specialized_runner_status") else 1,
            paper.get("paper_id", ""),
        ),
    )
    lines = [
        "# Remaining 19 Strict DIRS Long Goal Status",
        "",
        f"- Updated: `{utc_now()}`",
        f"- Final status: `{summary.get('final_status')}`",
        f"- Accepted/converged papers: `{accepted}`",
        f"- Not yet converged papers: `{not_converged}`",
        "- Policy: no reduced/small/proxy/syntax-only evidence can converge a paper.",
        "",
        "## Active / Specialized Runs",
        "",
    ]
    for paper in visible:
        status = paper.get("specialized_runner_status") or paper.get("professional_blocker") or paper.get("final_status") or "unknown"
        lines.append(f"- `{paper.get('paper_id')}`: `{status}` repo_paths={paper.get('repo_paths', [])}")
    if PROPHET_LIVE_VERIFIER_PATH.exists():
        pv = read_json(PROPHET_LIVE_VERIFIER_PATH)
        live = pv.get("live_full_gsm8k_runner", {})
        integrity = pv.get("live_integrity", {})
        comparison = pv.get("paper_result_comparison", {})
        traj = pv.get("trajectory_dataset_analysis", {})
        ablation_integrity = pv.get("ablation_grid_integrity", {})
        multibench = pv.get("multibenchmark_grid_campaign", {})
        table2 = pv.get("table2_acceleration_campaign", {})
        dream = pv.get("dream7b_axis_campaign", {})
        source_parity = pv.get("source_parity_blocker_audit", {})
        risk_audit = pv.get("gsm8k_live_shape_risk_audit", {})
        protocol_audit = pv.get("gsm8k_protocol_parity_audit", {})
        lines += [
            "",
            "## Prophet Live GPU Run",
            "",
            f"- Status: `{pv.get('status')}`",
            f"- Samples: `{live.get('paired_completed_samples')}/{live.get('total_samples')}`",
            f"- JSONL rows: `{live.get('jsonl_rows')}`",
            f"- JSONL integrity: `{integrity.get('status')}` gate=`{integrity.get('gate_status')}`",
            f"- Integrity report: `{integrity.get('report_path')}`",
            f"- Paper comparison: `{comparison.get('status')}` gate=`{comparison.get('gate_status')}`",
            f"- Paper comparison report: `{comparison.get('report_path')}`",
            f"- Trajectory comparison: `{comparison.get('trajectory_status')}`",
            f"- Multi-benchmark grid: `{multibench.get('status')}`",
            f"- Multi-benchmark manifest: `{multibench.get('manifest_path')}`",
            f"- Table 2 acceleration: `{table2.get('status')}`",
            f"- Table 2 manifest: `{table2.get('manifest_path')}`",
            f"- Dream-7B axis: `{dream.get('status')}`",
            f"- Dream-7B manifest: `{dream.get('manifest_path')}`",
            f"- Source parity blocker audit: `{source_parity.get('status')}`",
            f"- Source parity audit report: `{source_parity.get('report_path')}`",
            f"- GSM8K live shape risk audit: `{risk_audit.get('status')}` failing_metrics=`{risk_audit.get('failing_metrics')}` loop2_visible=`{risk_audit.get('loop2_author_can_read')}`",
            f"- GSM8K risk audit report: `{risk_audit.get('report_path')}`",
            f"- GSM8K protocol parity audit: `{protocol_audit.get('status')}` findings=`{protocol_audit.get('finding_statuses')}` loop2_visible=`{protocol_audit.get('loop2_author_can_read')}`",
            f"- GSM8K protocol audit report: `{protocol_audit.get('report_path')}`",
            f"- Ablation grid: `{pv.get('ablation_grid_campaign', {}).get('status')}`",
            f"- Ablation manifest: `{pv.get('ablation_grid_campaign', {}).get('manifest_path')}`",
            f"- Ablation integrity: `{ablation_integrity.get('status')}` running=`{len(ablation_integrity.get('running_config_ids', []) or [])}` complete=`{len(ablation_integrity.get('complete_config_ids', []) or [])}` integrity_blocked=`{len(ablation_integrity.get('blocked_config_ids', []) or [])}` manifest_blocked=`{len(ablation_integrity.get('manifest_blocked_config_ids', []) or [])}`",
            f"- Ablation integrity report: `{ablation_integrity.get('report_path')}`",
            f"- GPU: `{live.get('cuda_visible_devices')}`",
            f"- Trajectory: `{traj.get('settings_completed')}/{traj.get('total_settings')}` status=`{traj.get('status')}`",
            f"- Trajectory rows: `{traj.get('rows_written')}`",
            f"- Updated: `{pv.get('created_at_utc')}`",
        ]
    elif PROPHET_STATUS_PATH.exists():
        ps = read_json(PROPHET_STATUS_PATH)
        lines += [
            "",
            "## Prophet Live GPU Run",
            "",
            f"- Status: `{ps.get('status')}`",
            f"- Samples: `{ps.get('completed_sample_indices')}/{ps.get('total_samples')}`",
            f"- GPU: `{ps.get('cuda_visible_devices')}`",
            f"- Updated: `{ps.get('updated_at_utc')}`",
        ]
    if LOOP1_REPAIR_REPORT_PATH.exists():
        repair = read_json(LOOP1_REPAIR_REPORT_PATH)
        lines += [
            "",
            "## Loop 1 DAG Repair",
            "",
            f"- Status: `latest_verifier_feedback_encoded_for_{len(repair.get('papers', []))}_papers`",
            f"- Report: `{LOOP1_REPAIR_REPORT_PATH}`",
            f"- Policy: `{repair.get('policy', {}).get('loop2_input')}`; oracle_values_exposed_to_loop2=`{repair.get('policy', {}).get('oracle_values_exposed_to_loop2')}`",
            f"- Updated: `{repair.get('created_at_utc')}`",
        ]
    if COMPLETION_AUDIT_PATH.exists():
        audit = read_json(COMPLETION_AUDIT_PATH)
        counts = audit.get("counts", {})
        lines += [
            "",
            "## Completion Audit",
            "",
            f"- Goal complete: `{audit.get('goal_complete')}`",
            f"- Accepted: `{counts.get('accepted')}`",
            f"- Explicitly blocked: `{counts.get('explicitly_blocked')}`",
            f"- Running: `{counts.get('running')}`",
            f"- Unresolved: `{counts.get('unresolved')}`",
            f"- Report: `{COMPLETION_AUDIT_PATH}`",
            f"- Updated: `{audit.get('created_at_utc')}`",
        ]
    staleness = summary.get("explicit_blocker_staleness_audit", {})
    if staleness:
        counts = staleness.get("counts", {})
        weak = staleness.get("weak_evidence_paper_ids") or []
        gpu_recheck = staleness.get("gpu_recheck_paper_ids") or []
        lines += [
            "",
            "## Explicit Blocker Staleness Audit",
            "",
            f"- Evidence-bound explicit blockers: `{counts.get('explicit_blocker_evidence_bound')}`",
            f"- Evidence-bound after GPU recheck: `{counts.get('explicit_blocker_evidence_bound_after_gpu_recheck')}`",
            f"- Recheck after active GPU release: `{counts.get('explicit_blocker_valid_recheck_after_gpu_release')}`",
            f"- Needs Loop 1 repair: `{counts.get('needs_loop1_repair')}`",
            f"- Running, not explicit blocker: `{counts.get('running_not_explicit_blocker')}`",
            f"- Weak blocker evidence papers: `{len(weak)}`",
            f"- GPU recheck papers: `{len(gpu_recheck)}`",
            f"- Report: `{staleness.get('report_path')}`",
            f"- Markdown: `{staleness.get('markdown_path')}`",
            f"- Updated: `{staleness.get('updated_at_utc')}`",
        ]
    gpu_recheck = summary.get("gpu_recheck_dispatcher", {})
    if gpu_recheck:
        lines += [
            "",
            "## GPU Recheck Dispatcher",
            "",
            f"- Status: `{gpu_recheck.get('status')}`",
            f"- Pending before latest dispatch: `{gpu_recheck.get('pending_before_count')}`",
            f"- Rechecks executed in latest dispatch: `{gpu_recheck.get('recheck_executed_count')}`",
            f"- Attempted papers: `{len(gpu_recheck.get('attempted_paper_ids') or [])}`",
            f"- Report: `{gpu_recheck.get('report_path')}`",
            f"- State: `{gpu_recheck.get('state_path')}`",
            f"- Markdown: `{gpu_recheck.get('markdown_path')}`",
            f"- Updated: `{gpu_recheck.get('updated_at_utc')}`",
        ]
    gate_sweep = summary.get("professional_gate_sweep", {})
    if gate_sweep:
        lines += [
            "",
            "## Professional Gate Sweep",
            "",
            f"- Status: `{gate_sweep.get('status')}`",
            f"- Scope: `{gate_sweep.get('scope')}`",
            f"- Selected before cap: `{gate_sweep.get('selected_before_cap_count')}`",
            f"- Executed in latest sweep: `{gate_sweep.get('executed_count')}`",
            f"- Attempted papers: `{len(gate_sweep.get('attempted_paper_ids') or [])}`",
            f"- Report: `{gate_sweep.get('report_path')}`",
            f"- State: `{gate_sweep.get('state_path')}`",
            f"- Markdown: `{gate_sweep.get('markdown_path')}`",
            f"- Updated: `{gate_sweep.get('updated_at_utc')}`",
        ]
    scheduler = summary.get("multi_gpu_professional_scheduler", {})
    if scheduler:
        lines += [
            "",
            "## Multi-GPU Professional Scheduler",
            "",
            f"- Status: `{scheduler.get('status')}`",
            f"- Paper: `{scheduler.get('paper_title')}`",
            f"- GPU pool: `{','.join(scheduler.get('gpu_pool') or [])}`",
            f"- Launches in latest tick: `{scheduler.get('launch_count')}`",
            f"- Cumulative launch history: `{scheduler.get('launch_history_count')}`",
            f"- Active Prophet GPU claims: `{scheduler.get('active_gpu_claim_count')}`",
            f"- Report: `{scheduler.get('report_path')}`",
            f"- State: `{scheduler.get('state_path')}`",
            f"- Markdown: `{scheduler.get('markdown_path')}`",
            f"- Updated: `{scheduler.get('updated_at_utc')}`",
        ]
    if SUPERVISOR_STATE_PATH.exists():
        supervisor = read_json(SUPERVISOR_STATE_PATH)
        prophet = supervisor.get("prophet", {})
        lines += [
            "",
            "## Goal Supervisor",
            "",
            f"- Status: `{supervisor.get('supervisor_status')}`",
            f"- Progress since previous tick: `{supervisor.get('progress_since_previous_tick')}`",
            f"- Last progress: `{supervisor.get('last_progress_at')}`",
            f"- Prophet samples: `{prophet.get('paired_completed_samples')}/{prophet.get('total_samples')}`",
            f"- Prophet PID alive: `{prophet.get('pid_alive')}`",
            f"- Updated: `{supervisor.get('updated_at_utc')}`",
        ]
    LONG_STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_queue(queue_items: list[dict[str, Any]]) -> None:
    lines = ["# Specialized Runner Queue", ""]
    for item in queue_items:
        status = item.get("specialized_runner_status") or item.get("professional_blocker") or "unknown"
        lines.append(
            f"- `{item.get('paper_id')}` | priority=`{item.get('priority')}` | "
            f"status=`{status}` | runner=`{item.get('runner_type')}` | repos={item.get('repo_paths', [])}"
        )
    SPECIALIZED_QUEUE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    refresh_prophet_live_verifier_artifact()
    refresh_completion_audit_artifact()
    refresh_staleness_audit_artifact()
    summary = read_json(SUMMARY_PATH)
    queue_obj = read_json(QUEUE_PATH)
    queue_items = queue_obj.get("queue", queue_obj if isinstance(queue_obj, list) else [])
    queue_items = apply_prophet_live_to_queue(queue_items)
    summary = merge_queue_into_summary(summary, queue_items)
    summary = refresh_active_prophet_run(summary)
    summary = attach_staleness_audit(summary)
    summary = attach_gpu_recheck_dispatch(summary)
    summary = attach_professional_gate_sweep(summary)
    summary = attach_multi_gpu_scheduler(summary)
    write_json(SUMMARY_PATH, summary)
    write_json(QUEUE_PATH, queue_obj)
    render_status(summary)
    render_queue(queue_items)
    print(
        json.dumps(
            {
                "summary": str(SUMMARY_PATH),
                "status": str(LONG_STATUS_PATH),
                "queue_markdown": str(SPECIALIZED_QUEUE_MD),
                "paper_count": len(summary.get("papers", [])),
                "queue_count": len(queue_items),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
