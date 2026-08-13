#!/usr/bin/env python3
"""Refresh live Prophet verifier artifacts while the full GPU run is active.

The custom GSM8K runner is a real non-reduced operational node, but it is not a
paper-level convergence artifact until the full split and companion evidence
channels are complete. This utility publishes the current live state without
promoting partial results.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ROOT = Path("/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723")
PAPER_RUN = RUN_ROOT / "paper_runs/iclr2026_g88nt4ietg_prophet_dlm_early_commit_decoding"
RUNNER_DIR = RUN_ROOT / "specialized_runners/prophet"
CUSTOM_DIR = RUNNER_DIR / "custom_full_gsm8k_llada8b"
TRAJ_DIR = RUNNER_DIR / "trajectory_dataset_analysis"

STATUS_PATH = CUSTOM_DIR / "status.json"
SUMMARY_PATH = CUSTOM_DIR / "summary.json"
ROWS_PATH = CUSTOM_DIR / "per_sample_results.jsonl"
TRAJ_STATUS_PATH = TRAJ_DIR / "trajectory_dataset_status.json"
TRAJ_SUMMARY_PATH = TRAJ_DIR / "trajectory_analysis_summary.json"
TRAJ_ROWS_PATH = TRAJ_DIR / "trajectory_first_emergence_rows.jsonl"
TRAJ_SEMANTICS_AUDIT_PATH = TRAJ_DIR / "trajectory_metric_semantics_audit.json"

SPECIALIZED_VERIFIER_PATH = RUNNER_DIR / "prophet_specialized_verifier.json"
SPECIALIZED_STATUS_PATH = RUNNER_DIR / "PROPHET_SPECIALIZED_STATUS.md"
INTEGRITY_SCRIPT_PATH = RUNNER_DIR / "prophet_live_integrity_check.py"
INTEGRITY_REPORT_PATH = RUNNER_DIR / "prophet_live_integrity_report.json"
INTEGRITY_STATUS_PATH = RUNNER_DIR / "PROPHET_LIVE_INTEGRITY_STATUS.md"
PAPER_COMPARISON_SCRIPT_PATH = RUNNER_DIR / "prophet_paper_result_comparator.py"
PAPER_COMPARISON_REPORT_PATH = RUNNER_DIR / "prophet_paper_result_comparison.json"
PAPER_COMPARISON_STATUS_PATH = RUNNER_DIR / "PROPHET_PAPER_RESULT_COMPARISON_STATUS.md"
ABLATION_MANIFEST_PATH = RUNNER_DIR / "ablation_grid_full_gsm8k/ablation_grid_campaign.json"
ABLATION_STATUS_PATH = RUNNER_DIR / "ablation_grid_full_gsm8k/ABLATION_GRID_STATUS.md"
TABLE1_THRESHOLD_MANIFEST_PATH = RUNNER_DIR / "table1_threshold_repair_full_gsm8k/table1_threshold_repair_campaign.json"
TABLE1_THRESHOLD_STATUS_PATH = RUNNER_DIR / "table1_threshold_repair_full_gsm8k/TABLE1_THRESHOLD_REPAIR_STATUS.md"
PROTOCOL_REPAIR_MANIFEST_PATH = RUNNER_DIR / "protocol_repair_full_gsm8k/protocol_repair_campaign.json"
PROTOCOL_REPAIR_STATUS_PATH = RUNNER_DIR / "protocol_repair_full_gsm8k/PROTOCOL_REPAIR_STATUS.md"
ABLATION_INTEGRITY_SCRIPT_PATH = RUNNER_DIR / "prophet_ablation_grid_integrity_check.py"
ABLATION_INTEGRITY_REPORT_PATH = RUNNER_DIR / "ablation_grid_full_gsm8k/ablation_grid_integrity_report.json"
ABLATION_INTEGRITY_STATUS_PATH = RUNNER_DIR / "ablation_grid_full_gsm8k/ABLATION_GRID_INTEGRITY_STATUS.md"
MULTIBENCHMARK_MANIFEST_PATH = RUNNER_DIR / "multibenchmark_table1_full/multibenchmark_grid_campaign.json"
MULTIBENCHMARK_STATUS_PATH = RUNNER_DIR / "multibenchmark_table1_full/MULTIBENCHMARK_GRID_STATUS.md"
TABLE2_MANIFEST_PATH = RUNNER_DIR / "table2_acceleration_combinations/table2_acceleration_campaign.json"
TABLE2_STATUS_PATH = RUNNER_DIR / "table2_acceleration_combinations/TABLE2_ACCELERATION_STATUS.md"
DREAM_MANIFEST_PATH = RUNNER_DIR / "dream7b_table1_axis/dream7b_axis_campaign.json"
DREAM_STATUS_PATH = RUNNER_DIR / "dream7b_table1_axis/DREAM7B_AXIS_STATUS.md"
SOURCE_PARITY_AUDIT_SCRIPT_PATH = RUNNER_DIR / "prophet_source_parity_blocker_audit.py"
SOURCE_PARITY_AUDIT_REPORT_PATH = RUNNER_DIR / "source_parity_blocker_audit.json"
SOURCE_PARITY_AUDIT_STATUS_PATH = RUNNER_DIR / "SOURCE_PARITY_BLOCKER_AUDIT.md"
GSM8K_RISK_AUDIT_SCRIPT_PATH = RUNNER_DIR / "prophet_gsm8k_live_shape_risk_audit.py"
GSM8K_RISK_AUDIT_REPORT_PATH = RUNNER_DIR / "gsm8k_live_shape_risk_audit.json"
GSM8K_RISK_AUDIT_STATUS_PATH = RUNNER_DIR / "GSM8K_LIVE_SHAPE_RISK_AUDIT.md"
GSM8K_PROTOCOL_AUDIT_SCRIPT_PATH = RUNNER_DIR / "prophet_gsm8k_protocol_parity_audit.py"
GSM8K_PROTOCOL_AUDIT_REPORT_PATH = RUNNER_DIR / "gsm8k_protocol_parity_audit.json"
GSM8K_PROTOCOL_AUDIT_STATUS_PATH = RUNNER_DIR / "GSM8K_PROTOCOL_PARITY_AUDIT.md"
GSM8K_TABLE1_SELECTION_AUDIT_REPORT_PATH = RUNNER_DIR / "gsm8k_table1_protocol_selection_audit.json"
GSM8K_TABLE1_SELECTION_AUDIT_STATUS_PATH = RUNNER_DIR / "GSM8K_TABLE1_PROTOCOL_SELECTION_AUDIT.md"
PAPER_LIVE_VERIFIER_PATH = PAPER_RUN / "verifier_result_iter_07_live.json"
PAPER_OPERATIONAL_ARTIFACTS_PATH = PAPER_RUN / "operational_artifacts.json"

PAPER_ID = "ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding"
TITLE = "Diffusion Language Models Know the Answer Before Decoding"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))


def process_alive(pid: Any) -> bool | None:
    if pid in (None, "", 0):
        return None
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def refresh_integrity_report() -> None:
    if not INTEGRITY_SCRIPT_PATH.exists() or not ROWS_PATH.exists():
        return
    subprocess.run(
        ["python", str(INTEGRITY_SCRIPT_PATH)],
        cwd=str(RUNNER_DIR),
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def refresh_paper_comparison_report() -> None:
    if not PAPER_COMPARISON_SCRIPT_PATH.exists() or not SUMMARY_PATH.exists():
        return
    subprocess.run(
        ["python", str(PAPER_COMPARISON_SCRIPT_PATH)],
        cwd=str(RUNNER_DIR),
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def refresh_ablation_integrity_report() -> None:
    if not ABLATION_INTEGRITY_SCRIPT_PATH.exists() or not ABLATION_MANIFEST_PATH.exists():
        return
    subprocess.run(
        ["python", str(ABLATION_INTEGRITY_SCRIPT_PATH)],
        cwd=str(RUNNER_DIR),
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def refresh_source_parity_audit_report() -> None:
    if not SOURCE_PARITY_AUDIT_SCRIPT_PATH.exists():
        return
    subprocess.run(
        ["python", str(SOURCE_PARITY_AUDIT_SCRIPT_PATH)],
        cwd=str(RUNNER_DIR),
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=90,
        check=False,
    )


def refresh_gsm8k_risk_audit_report() -> None:
    if not GSM8K_RISK_AUDIT_SCRIPT_PATH.exists() or not PAPER_COMPARISON_REPORT_PATH.exists():
        return
    subprocess.run(
        ["python", str(GSM8K_RISK_AUDIT_SCRIPT_PATH)],
        cwd=str(RUNNER_DIR),
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def refresh_gsm8k_protocol_audit_report() -> None:
    if not GSM8K_PROTOCOL_AUDIT_SCRIPT_PATH.exists():
        return
    subprocess.run(
        ["python", str(GSM8K_PROTOCOL_AUDIT_SCRIPT_PATH)],
        cwd=str(RUNNER_DIR),
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


def estimate_remaining_seconds(summary: dict[str, Any], paired_done: int, total: int) -> float | None:
    if paired_done <= 0 or paired_done >= total:
        return 0.0
    aggregates = summary.get("aggregates", {})
    baseline = aggregates.get("baseline", {}).get("mean_seconds")
    prophet = aggregates.get("prophet", {}).get("mean_seconds")
    if baseline is None or prophet is None:
        return None
    return float(total - paired_done) * float(baseline + prophet)


def status_from_state(
    runner_status: str | None,
    runner_alive: bool | None,
    full_gsm8k_complete: bool,
    trajectory_complete: bool,
) -> tuple[str, str]:
    if runner_status == "failed":
        return (
            "blocked_by_custom_full_gsm8k_runner_failure",
            "not_converged_runner_failed_before_paper_shaped_artifacts",
        )
    if not full_gsm8k_complete:
        if runner_alive is False:
            return (
                "blocked_by_custom_full_gsm8k_runner_stopped_before_completion",
                "not_converged_runner_stopped_before_full_split",
            )
        return (
            "running_full_custom_gsm8k_on_gpu3_and_trajectory_analysis",
            "not_yet_converged_running_real_operational_nodes",
        )
    if not trajectory_complete:
        return (
            "running_full_gsm8k_complete_waiting_for_trajectory_analysis",
            "not_yet_converged_waiting_for_companion_trajectory_artifacts",
        )
    return (
        "blocked_by_table1_table2_ablation_and_dream_axis_debt_after_full_gsm8k_trajectory",
        "not_converged_explicit_professional_debt_after_real_operational_nodes",
    )


def integrity_gate_status(integrity: dict[str, Any]) -> str:
    status = integrity.get("status")
    if status == "blocked_integrity_failure":
        return "blocked"
    if isinstance(status, str) and status.startswith("pass_"):
        return "pass"
    return "missing"


def paper_comparison_gate_status(comparison: dict[str, Any]) -> str:
    status = comparison.get("status")
    if status == "pass_all_paper_result_targets":
        return "pass"
    if isinstance(status, str) and status.startswith("blocked_"):
        return "blocked"
    if isinstance(status, str) and status.startswith("running_"):
        return "running"
    return "missing"


def build_payload() -> dict[str, Any]:
    now = utc_now()
    status = read_json(STATUS_PATH, {})
    summary = read_json(SUMMARY_PATH, {})
    trajectory = read_json(TRAJ_STATUS_PATH, {})
    trajectory_summary = read_json(TRAJ_SUMMARY_PATH, {})
    trajectory_semantics_audit = read_json(TRAJ_SEMANTICS_AUDIT_PATH, {})
    integrity = read_json(INTEGRITY_REPORT_PATH, {})
    paper_comparison = read_json(PAPER_COMPARISON_REPORT_PATH, {})
    ablation_integrity = read_json(ABLATION_INTEGRITY_REPORT_PATH, {})
    protocol_repair_campaign = read_json(PROTOCOL_REPAIR_MANIFEST_PATH, {})
    ablation_campaign = read_json(ABLATION_MANIFEST_PATH, {})
    table1_threshold_campaign = read_json(TABLE1_THRESHOLD_MANIFEST_PATH, {})
    multibenchmark_campaign = read_json(MULTIBENCHMARK_MANIFEST_PATH, {})
    table2_campaign = read_json(TABLE2_MANIFEST_PATH, {})
    dream_campaign = read_json(DREAM_MANIFEST_PATH, {})
    source_parity_audit = read_json(SOURCE_PARITY_AUDIT_REPORT_PATH, {})
    gsm8k_risk_audit = read_json(GSM8K_RISK_AUDIT_REPORT_PATH, {})
    gsm8k_protocol_audit = read_json(GSM8K_PROTOCOL_AUDIT_REPORT_PATH, {})
    gsm8k_table1_selection_audit = read_json(GSM8K_TABLE1_SELECTION_AUDIT_REPORT_PATH, {})
    integrity_status = integrity_gate_status(integrity)
    paper_comparison_status = paper_comparison_gate_status(paper_comparison)
    rows = line_count(ROWS_PATH)
    trajectory_status_rows = int(trajectory.get("rows_written") or 0) if isinstance(trajectory, dict) else 0
    trajectory_live_rows = line_count(TRAJ_ROWS_PATH)
    trajectory_observed_rows = max(trajectory_status_rows, trajectory_live_rows)
    trajectory_live = dict(trajectory)
    if trajectory_observed_rows:
        trajectory_live["rows_written"] = trajectory_observed_rows
        trajectory_live["rows_path"] = str(TRAJ_ROWS_PATH)
        trajectory_live["observed_rows_updated_at_utc"] = now
        if trajectory_live_rows > trajectory_status_rows:
            trajectory_live["status_note"] = "rows file is ahead of the analyzer status JSON; analyzer is still running."
    trajectory_summary_settings = trajectory_summary.get("settings", []) if isinstance(trajectory_summary, dict) else []
    trajectory_professional_gate = (
        trajectory_summary.get("professional_gate", {}) if isinstance(trajectory_summary, dict) else {}
    )
    if trajectory_summary_settings:
        trajectory_live.setdefault("settings_completed", len(trajectory_summary_settings))
        trajectory_live.setdefault("total_settings", len(trajectory_summary_settings))
        trajectory_live.setdefault("summary_path", str(TRAJ_SUMMARY_PATH))
    aggregates = summary.get("aggregates", {})
    baseline_done = int(aggregates.get("baseline", {}).get("completed_samples") or 0)
    prophet_done = int(aggregates.get("prophet", {}).get("completed_samples") or 0)
    paired_done = min(baseline_done, prophet_done) if baseline_done and prophet_done else int(status.get("completed_sample_indices") or 0)
    total = int(status.get("total_samples") or summary.get("total_samples") or 1319)
    full_gsm8k_complete = paired_done >= total and total > 0
    trajectory_status_done = trajectory_live.get("status") in {"complete", "completed", "analyzed"}
    trajectory_count_done = int(trajectory_live.get("settings_completed") or 0) >= int(
        trajectory_live.get("total_settings") or 1
    )
    trajectory_summary_done = bool(
        trajectory_summary_settings
        and trajectory_professional_gate.get("all_expected_counts_present") is not False
    )
    trajectory_complete = bool(trajectory_live and trajectory_status_done and (trajectory_count_done or trajectory_summary_done))
    if trajectory_complete and trajectory_live.get("status_note"):
        trajectory_live["status_note"] = (
            "trajectory analysis complete; rows file has the newest observed count."
        )
    runner_alive = process_alive(status.get("pid"))
    status_label, convergence_decision = status_from_state(
        status.get("status"),
        runner_alive,
        full_gsm8k_complete,
        trajectory_complete,
    )
    if integrity_status == "blocked":
        status_label = "blocked_by_live_jsonl_integrity_failure"
        convergence_decision = "not_converged_live_jsonl_integrity_failure"
    eta_seconds = estimate_remaining_seconds(summary, paired_done, total)
    support_only_until = []
    if not full_gsm8k_complete:
        support_only_until.append("full GSM8K baseline/Prophet paired split reaches 1319/1319 samples")
    if not trajectory_complete:
        support_only_until.append("trajectory analysis completes all configured settings")
    support_only_until.extend(
        [
            "GSM8K protocol repair rerun is produced or explicitly blocked after full-run result-shape mismatch",
            "Table 1 threshold-dynamics repair candidates are produced or explicitly blocked after primary shape mismatch",
            "multi-benchmark grid debt is resolved or explicitly blocked",
            "Table 2 acceleration-combination debt is resolved or explicitly blocked",
            "static step and block-length ablations are produced or explicitly blocked",
            "Dream-7B axis is produced or explicitly blocked",
        ]
    )
    protocol_repair_statuses = protocol_repair_campaign.get("config_statuses", {})
    protocol_repair_merged = protocol_repair_campaign.get("merged_artifact", {})
    protocol_repair_done = [
        item
        for item in protocol_repair_statuses.values()
        if item.get("summary_status") == "completed" or item.get("status") == "completed"
    ]
    protocol_repair_pending = [
        item
        for item in protocol_repair_statuses.values()
        if item.get("status") not in {"completed", "running"}
    ]
    protocol_repair_running = [
        item
        for item in protocol_repair_statuses.values()
        if item.get("status") == "running"
    ]
    protocol_repair_status = (
        "missing_manifest"
        if not protocol_repair_campaign
        else (
            "completed_full_protocol_repair_merged_artifact"
            if protocol_repair_merged.get("status") == "completed"
            else (
            "completed_full_protocol_repair_rerun"
            if protocol_repair_statuses and len(protocol_repair_done) == len(protocol_repair_statuses)
            else ("running_full_protocol_repair_rerun" if protocol_repair_running else "ready_waiting_for_gpu_capacity")
            )
        )
    )
    ablation_statuses = ablation_campaign.get("config_statuses", {})
    ablation_done = [
        item
        for item in ablation_statuses.values()
        if item.get("summary_status") == "completed" or item.get("status") == "completed"
    ]
    ablation_pending = [
        item
        for item in ablation_statuses.values()
        if item.get("status") not in {"completed", "running"}
    ]
    ablation_running = [
        item
        for item in ablation_statuses.values()
        if item.get("status") == "running"
    ]
    ablation_grid_status = (
        "missing_manifest"
        if not ablation_campaign
        else (
            "completed_full_ablation_grid"
            if ablation_statuses and len(ablation_done) == len(ablation_statuses)
            else ("running_full_ablation_grid" if ablation_running else "ready_waiting_for_gpu_capacity")
        )
    )
    table1_threshold_statuses = table1_threshold_campaign.get("config_statuses", {})
    table1_threshold_done = [
        item
        for item in table1_threshold_statuses.values()
        if item.get("summary_status") == "completed" or item.get("status") == "completed"
    ]
    table1_threshold_pending = [
        item
        for item in table1_threshold_statuses.values()
        if item.get("status") not in {"completed", "running"}
    ]
    table1_threshold_running = [
        item
        for item in table1_threshold_statuses.values()
        if item.get("status") == "running"
    ]
    table1_threshold_status = (
        "missing_manifest"
        if not table1_threshold_campaign
        else (
            "completed_full_table1_threshold_repair_candidates"
            if table1_threshold_statuses and len(table1_threshold_done) == len(table1_threshold_statuses)
            else (
                "running_full_table1_threshold_repair_candidates"
                if table1_threshold_running
                else "ready_waiting_for_gpu_capacity"
            )
        )
    )
    multibenchmark_statuses = multibenchmark_campaign.get("config_statuses", {})
    multibenchmark_done = [
        item
        for item in multibenchmark_statuses.values()
        if item.get("status") == "completed_or_has_results_pending_verifier"
    ]
    multibenchmark_running = [
        item
        for item in multibenchmark_statuses.values()
        if item.get("status") == "running"
    ]
    multibenchmark_pending = [
        item
        for item in multibenchmark_statuses.values()
        if item.get("status") in {"pending", "stopped_without_results"}
    ]
    multibenchmark_grid_status = (
        "missing_manifest"
        if not multibenchmark_campaign
        else (
            "completed_full_multibenchmark_lmeval_artifacts_pending_parity_verifier"
            if multibenchmark_statuses and len(multibenchmark_done) == len(multibenchmark_statuses)
            else (
                "running_full_multibenchmark_lmeval_grid"
                if multibenchmark_running
                else "ready_waiting_for_gpu_capacity_and_prompt_scorer_parity_resolution"
            )
        )
    )
    table2_linked = table2_campaign.get("linked_existing_artifacts", [])
    table2_linked_complete = [
        item for item in table2_linked if item.get("full_split_complete")
    ]
    table2_blocked = table2_campaign.get("blocked_configs", [])
    table2_status = (
        "missing_manifest"
        if not table2_campaign
        else (
            "linked_baseline_prophet_complete_but_external_rows_blocked"
            if table2_linked and len(table2_linked_complete) == len(table2_linked) and table2_blocked
            else (
                "running_waiting_for_full_gsm8k_linked_rows_and_external_artifacts"
                if table2_linked and len(table2_linked_complete) < len(table2_linked)
                else "blocked_by_missing_external_sdtt_fastdllm_artifacts"
            )
        )
    )
    dream_blocked = dream_campaign.get("blocked_configs", [])
    dream_status = (
        "missing_manifest"
        if not dream_campaign
        else (
            "explicit_dream7b_axis_blockers_recorded"
            if dream_blocked and not dream_campaign.get("runnable_configs")
            else "ready_waiting_for_dream7b_gpu_execution"
        )
    )
    trajectory_comparison_status = paper_comparison.get("trajectory_comparison", {}).get("status")
    trajectory_audit_diagnoses = [
        item.get("diagnosis")
        for item in trajectory_semantics_audit.get("setting_audits", [])
        if item.get("diagnosis")
    ]
    trajectory_debt_status = (
        "running"
        if not trajectory_complete
        else (
            "complete"
            if trajectory_comparison_status == "pass_close_to_paper_trajectory_shape"
            else (trajectory_audit_diagnoses[0] if trajectory_audit_diagnoses else trajectory_comparison_status or "complete")
        )
    )
    required_updates: list[dict[str, Any]] = []
    if trajectory_comparison_status == "blocked_trajectory_result_shape_mismatch":
        failed_settings = [
            check.get("setting_id")
            for check in paper_comparison.get("trajectory_comparison", {}).get("checks", [])
            if check.get("status") == "fail"
        ]
        required_updates.append(
            {
                "id": "trajectory_metric_semantics_repair",
                "reason": (
                    "Completed trajectory artifacts do not match the paper-shape verifier for one or more "
                    "first-emergence settings. Loop 2 must treat this as a simulation/DAG mismatch, not as a "
                    "missing artifact."
                ),
                "success_criteria": [
                    "Add a DAG node that audits first-emergence metric semantics without reading paper oracle values.",
                    "Check denominator choice, correct-row filtering, 25%/50% step thresholds, constraint-token handling, and block-length mapping.",
                    "Name the failed simulation setting IDs only, then rerun or re-score the trajectory artifact from DAG instructions.",
                    "Verifier comparison must move from blocked trajectory mismatch to pass or to a more specific operational blocker.",
                ],
                "failed_simulation_setting_ids": failed_settings,
            }
        )
    risk_status = gsm8k_risk_audit.get("status")
    if risk_status == "postcompletion_shape_mismatch_requires_loop1_dag_repair":
        risk_snapshot = gsm8k_risk_audit.get("comparison_snapshot_without_oracle_targets", {})
        risk_axes = [
            item.get("id")
            for item in gsm8k_risk_audit.get("possible_dag_repair_axes", [])
            if item.get("id")
        ]
        protocol_findings = {
            item.get("id"): item.get("status")
            for item in gsm8k_protocol_audit.get("findings", [])
            if item.get("id")
        }
        required_updates.append(
            {
                "id": "gsm8k_postcompletion_protocol_shape_repair",
                "artifact_family": "full_gsm8k_baseline_prophet_result_shape",
                "reason": (
                    "The completed full-split GSM8K operational artifact does not match the paper-shape "
                    "verifier. Loop 2 must treat this as a protocol/runner/scoring repair problem, not as "
                    "convergence evidence and not as a generic paper-reading mismatch."
                ),
                "success_criteria": [
                    "Add or activate DAG-only nodes for prompt template parity, suffix constraints, answer-region start/length, evaluation harness parity, and generated-answer extraction parity.",
                    "Do not expose paper target table values or paper text to Loop 2; only encode non-oracle repair axes and repo/config artifact requirements.",
                    "Run or rescore the full GSM8K artifact through the repaired protocol path, or record an explicit source/artifact blocker if the exact path is unavailable.",
                    "Verifier must compare final accuracy, step, speed, and result-direction shape against the paper evidence channels after the repaired artifact exists.",
                ],
                "failed_metric_ids": risk_snapshot.get("failing_metrics", []),
                "repair_axis_ids": risk_axes,
                "protocol_finding_statuses": protocol_findings,
                "risk_audit_status": risk_status,
                "report_path": str(GSM8K_RISK_AUDIT_REPORT_PATH),
                "oracle_values_exposed_to_loop2": False,
            }
        )
    verifier = {
        "artifact_kind": "prophet_live_specialized_verifier",
        "created_at_utc": now,
        "paper_id": PAPER_ID,
        "paper_title": TITLE,
        "dag_path": str(PAPER_RUN / "paper_author_gap_dag.json"),
        "dag_iter": "iter_07_live",
        "converged": False,
        "professional_package_ready": False,
        "convergence_decision": convergence_decision,
        "status": status_label,
        "required_updates": required_updates,
        "blind_contract_checked": {
            "only_input_file": "paper_author_gap_dag.json",
            "paper_text_visible_to_loop2": False,
            "oracle_results_visible_to_loop2": False,
            "previous_memory_visible_to_loop2": False,
            "repo_paths_visible_only_if_encoded_in_dag": True,
        },
        "live_full_gsm8k_runner": {
            "status_path": str(STATUS_PATH),
            "summary_path": str(SUMMARY_PATH),
            "rows_path": str(ROWS_PATH),
            "pid": status.get("pid"),
            "pid_alive": runner_alive,
            "cuda_visible_devices": status.get("cuda_visible_devices"),
            "model_id": status.get("model_id"),
            "dataset": status.get("dataset") or summary.get("dataset"),
            "paired_completed_samples": paired_done,
            "total_samples": total,
            "jsonl_rows": rows,
            "estimated_remaining_seconds": eta_seconds,
            "full_split_requested": bool(status.get("full_split_requested", True)),
            "full_gsm8k_complete": full_gsm8k_complete,
            "running_summary": summary,
            "updated_at_utc": status.get("updated_at_utc") or summary.get("created_at_utc"),
        },
        "live_integrity": {
            "report_path": str(INTEGRITY_REPORT_PATH),
            "status_path": str(INTEGRITY_STATUS_PATH),
            "status": integrity.get("status"),
            "gate_status": integrity_status,
            "reasons": integrity.get("reasons", []),
            "row_count": integrity.get("row_summary", {}).get("row_count"),
            "paired_completed_samples_from_rows": integrity.get("row_summary", {}).get("paired_completed_samples_from_rows"),
            "incomplete_sample_count": len(integrity.get("row_summary", {}).get("incomplete_samples", [])),
            "duplicate_pair_count": len(integrity.get("row_summary", {}).get("duplicate_pairs", [])),
            "json_parse_error_count": len(integrity.get("json_parse_errors", [])),
            "summary_consistency": integrity.get("summary_consistency", {}).get("status"),
            "updated_at_utc": integrity.get("created_at_utc"),
        },
        "paper_result_comparison": {
            "report_path": str(PAPER_COMPARISON_REPORT_PATH),
            "status_path": str(PAPER_COMPARISON_STATUS_PATH),
            "status": paper_comparison.get("status"),
            "gate_status": paper_comparison_status,
            "blockers": paper_comparison.get("blockers", []),
            "primary_gsm8k_status": paper_comparison.get("primary_gsm8k_comparison", {}).get("status"),
            "trajectory_status": paper_comparison.get("trajectory_comparison", {}).get("status"),
            "explicit_debt_statuses": {
                key: value.get("status")
                for key, value in paper_comparison.get("explicit_debt_comparison", {}).items()
            },
            "visibility_contract": paper_comparison.get("visibility_contract", {}),
            "updated_at_utc": paper_comparison.get("created_at_utc"),
        },
        "gsm8k_protocol_repair_campaign": {
            "manifest_path": str(PROTOCOL_REPAIR_MANIFEST_PATH),
            "status_path": str(PROTOCOL_REPAIR_STATUS_PATH),
            "status": protocol_repair_status,
            "merged_artifact": protocol_repair_merged,
            "runnable_config_count": len(protocol_repair_campaign.get("runnable_configs", [])),
            "completed_config_count": len(protocol_repair_done),
            "running_config_count": len(protocol_repair_running),
            "pending_config_count": len(protocol_repair_pending),
            "blocked_configs": protocol_repair_campaign.get("blocked_configs", []),
            "launch_result": protocol_repair_campaign.get("launch_result"),
            "updated_at_utc": protocol_repair_campaign.get("created_at_utc"),
            "oracle_values_exposed_to_loop2": False,
        },
        "trajectory_dataset_analysis": {
            "status_path": str(TRAJ_STATUS_PATH),
            "summary_path": str(TRAJ_SUMMARY_PATH),
            "rows_path": str(TRAJ_ROWS_PATH),
            "status": trajectory_live.get("status"),
            "settings_completed": trajectory_live.get("settings_completed"),
            "total_settings": trajectory_live.get("total_settings"),
            "rows_written": trajectory_live.get("rows_written"),
            "current_setting": trajectory_live.get("current_setting"),
            "trajectory_complete": trajectory_complete,
            "updated_at_utc": trajectory_live.get("updated_at_utc"),
            "observed_rows_updated_at_utc": trajectory_live.get("observed_rows_updated_at_utc"),
            "status_note": trajectory_live.get("status_note"),
        },
        "trajectory_metric_semantics_audit": {
            "audit_path": str(TRAJ_SEMANTICS_AUDIT_PATH),
            "status": trajectory_semantics_audit.get("status") or "missing",
            "oracle_values_exposed_to_loop2": trajectory_semantics_audit.get("oracle_values_exposed_to_loop2"),
            "failed_simulation_setting_ids": trajectory_semantics_audit.get("failed_simulation_setting_ids_from_dag", []),
            "diagnoses": trajectory_audit_diagnoses,
            "updated_at_utc": trajectory_semantics_audit.get("created_at_utc"),
        },
        "ablation_grid_campaign": {
            "manifest_path": str(ABLATION_MANIFEST_PATH),
            "status_path": str(ABLATION_STATUS_PATH),
            "status": ablation_grid_status,
            "runnable_config_count": len(ablation_campaign.get("runnable_configs", [])),
            "completed_config_count": len(ablation_done),
            "running_config_count": len(ablation_running),
            "pending_config_count": len(ablation_pending),
            "blocked_configs": ablation_campaign.get("blocked_configs", []),
            "launch_result": ablation_campaign.get("launch_result"),
            "updated_at_utc": ablation_campaign.get("created_at_utc"),
        },
        "ablation_grid_integrity": {
            "report_path": str(ABLATION_INTEGRITY_REPORT_PATH),
            "status_path": str(ABLATION_INTEGRITY_STATUS_PATH),
            "status": ablation_integrity.get("status") or "missing",
            "reasons": ablation_integrity.get("reasons", []),
            "blocked_config_ids": ablation_integrity.get("blocked_config_ids", []),
            "manifest_blocked_config_ids": ablation_integrity.get("manifest_blocked_config_ids", []),
            "manifest_blocked_configs": ablation_integrity.get("manifest_blocked_configs", []),
            "running_config_ids": ablation_integrity.get("running_config_ids", []),
            "complete_config_ids": ablation_integrity.get("complete_config_ids", []),
            "pending_config_ids": ablation_integrity.get("pending_config_ids", []),
            "updated_at_utc": ablation_integrity.get("created_at_utc"),
        },
        "table1_threshold_repair_campaign": {
            "manifest_path": str(TABLE1_THRESHOLD_MANIFEST_PATH),
            "status_path": str(TABLE1_THRESHOLD_STATUS_PATH),
            "status": table1_threshold_status,
            "baseline_full_split_ready": table1_threshold_campaign.get("baseline_status", {}).get(
                "full_split_ready"
            ),
            "runnable_config_count": len(table1_threshold_campaign.get("runnable_configs", [])),
            "completed_config_count": len(table1_threshold_done),
            "running_config_count": len(table1_threshold_running),
            "pending_config_count": len(table1_threshold_pending),
            "blocked_configs": table1_threshold_campaign.get("blocked_configs", []),
            "launch_result": table1_threshold_campaign.get("launch_result"),
            "fixed_protocol": table1_threshold_campaign.get("fixed_protocol", {}),
            "updated_at_utc": table1_threshold_campaign.get("created_at_utc"),
            "oracle_values_exposed_to_loop2": False,
        },
        "multibenchmark_grid_campaign": {
            "manifest_path": str(MULTIBENCHMARK_MANIFEST_PATH),
            "status_path": str(MULTIBENCHMARK_STATUS_PATH),
            "status": multibenchmark_grid_status,
            "runnable_config_count": len(multibenchmark_campaign.get("runnable_configs", [])),
            "linked_existing_artifact_count": len(multibenchmark_campaign.get("linked_existing_artifacts", [])),
            "completed_config_count": len(multibenchmark_done),
            "running_config_count": len(multibenchmark_running),
            "pending_config_count": len(multibenchmark_pending),
            "blocked_configs": multibenchmark_campaign.get("blocked_configs", []),
            "launch_result": multibenchmark_campaign.get("launch_result"),
            "updated_at_utc": multibenchmark_campaign.get("created_at_utc"),
        },
        "table2_acceleration_campaign": {
            "manifest_path": str(TABLE2_MANIFEST_PATH),
            "status_path": str(TABLE2_STATUS_PATH),
            "status": table2_status,
            "linked_existing_artifact_count": len(table2_linked),
            "linked_existing_complete_count": len(table2_linked_complete),
            "runnable_config_count": len(table2_campaign.get("runnable_configs", [])),
            "blocked_configs": table2_blocked,
            "launch_result": table2_campaign.get("launch_result"),
            "updated_at_utc": table2_campaign.get("created_at_utc"),
        },
        "dream7b_axis_campaign": {
            "manifest_path": str(DREAM_MANIFEST_PATH),
            "status_path": str(DREAM_STATUS_PATH),
            "status": dream_status,
            "runnable_config_count": len(dream_campaign.get("runnable_configs", [])),
            "blocked_configs": dream_blocked,
            "launch_result": dream_campaign.get("launch_result"),
            "updated_at_utc": dream_campaign.get("created_at_utc"),
        },
        "source_parity_blocker_audit": {
            "report_path": str(SOURCE_PARITY_AUDIT_REPORT_PATH),
            "status_path": str(SOURCE_PARITY_AUDIT_STATUS_PATH),
            "status": source_parity_audit.get("status") or "missing",
            "blocker_ids": [
                item.get("id") for item in source_parity_audit.get("blocker_audits", [])
            ],
            "runnable_new_node_count": source_parity_audit.get("counts", {}).get("runnable_new_node_count"),
            "can_converge_from_this_audit_alone": source_parity_audit.get("verifier_implication", {}).get(
                "can_converge_from_this_audit_alone"
            ),
            "remote_matches_local": source_parity_audit.get("repository_snapshot", {}).get("remote_matches_local"),
            "updated_at_utc": source_parity_audit.get("created_at_utc"),
        },
        "gsm8k_live_shape_risk_audit": {
            "report_path": str(GSM8K_RISK_AUDIT_REPORT_PATH),
            "status_path": str(GSM8K_RISK_AUDIT_STATUS_PATH),
            "status": gsm8k_risk_audit.get("status") or "missing",
            "loop2_author_can_read": gsm8k_risk_audit.get("visibility_contract", {}).get("loop2_author_can_read"),
            "paper_oracle_target_values_included": gsm8k_risk_audit.get("visibility_contract", {}).get(
                "paper_oracle_target_values_included"
            ),
            "can_converge_from_this_audit_alone": gsm8k_risk_audit.get("risk_interpretation", {}).get(
                "can_converge_from_this_audit_alone"
            ),
            "do_not_stop_before_full_split": gsm8k_risk_audit.get("risk_interpretation", {}).get(
                "do_not_stop_before_full_split"
            ),
            "failing_metrics": gsm8k_risk_audit.get("comparison_snapshot_without_oracle_targets", {}).get(
                "failing_metrics", []
            ),
            "repair_axis_ids": [
                item.get("id") for item in gsm8k_risk_audit.get("possible_dag_repair_axes", [])
            ],
            "updated_at_utc": gsm8k_risk_audit.get("created_at_utc"),
        },
        "gsm8k_protocol_parity_audit": {
            "report_path": str(GSM8K_PROTOCOL_AUDIT_REPORT_PATH),
            "status_path": str(GSM8K_PROTOCOL_AUDIT_STATUS_PATH),
            "status": gsm8k_protocol_audit.get("status") or "missing",
            "loop2_author_can_read": gsm8k_protocol_audit.get("visibility_contract", {}).get(
                "loop2_author_can_read"
            ),
            "paper_oracle_target_values_included": gsm8k_protocol_audit.get("visibility_contract", {}).get(
                "paper_oracle_target_values_included"
            ),
            "can_converge_from_this_audit_alone": gsm8k_protocol_audit.get("verifier_implication", {}).get(
                "can_converge_from_this_audit_alone"
            ),
            "finding_statuses": {
                item.get("id"): item.get("status")
                for item in gsm8k_protocol_audit.get("findings", [])
                if item.get("id")
            },
            "covered_repair_nodes": gsm8k_protocol_audit.get("dag_coverage", {}).get(
                "covered_repair_nodes", []
            ),
            "updated_at_utc": gsm8k_protocol_audit.get("created_at_utc"),
        },
        "gsm8k_table1_protocol_selection_audit": {
            "report_path": str(GSM8K_TABLE1_SELECTION_AUDIT_REPORT_PATH),
            "status_path": str(GSM8K_TABLE1_SELECTION_AUDIT_STATUS_PATH),
            "status": gsm8k_table1_selection_audit.get("status") or "missing",
            "loop2_author_can_read": gsm8k_table1_selection_audit.get("visibility_contract", {}).get(
                "loop2_author_can_read"
            ),
            "paper_oracle_target_values_included": gsm8k_table1_selection_audit.get(
                "visibility_contract", {}
            ).get("paper_oracle_target_values_included"),
            "oracle_values_exposed_to_loop2": gsm8k_table1_selection_audit.get(
                "visibility_contract", {}
            ).get("oracle_values_exposed_to_loop2"),
            "primary_status_after_selection_repair": gsm8k_table1_selection_audit.get(
                "non_oracle_result_interpretation", {}
            ).get("primary_status_after_selection_repair"),
            "failed_metric_ids": gsm8k_table1_selection_audit.get(
                "non_oracle_result_interpretation", {}
            ).get("failed_metric_ids", []),
            "recommended_dag_update": gsm8k_table1_selection_audit.get("recommended_dag_update", {}),
            "updated_at_utc": gsm8k_table1_selection_audit.get("created_at_utc"),
        },
        "checks": [
            {
                "name": "blind_contract",
                "status": "pass",
                "detail": "Loop 2 input is the encoded DAG and repo paths, not paper text or prior memory.",
            },
            {
                "name": "real_gpu_run_active",
                "status": (
                    "pass"
                    if runner_alive and not full_gsm8k_complete
                    else ("complete" if full_gsm8k_complete else "blocked")
                ),
                "detail": (
                    f"physical GPU {status.get('cuda_visible_devices')} runner pid={status.get('pid')} "
                    f"alive={runner_alive} paired={paired_done}/{total}"
                ),
            },
            {
                "name": "live_jsonl_integrity",
                "status": integrity_status,
                "detail": {
                    "status": integrity.get("status"),
                    "reasons": integrity.get("reasons", []),
                    "row_summary": integrity.get("row_summary", {}),
                    "summary_consistency": integrity.get("summary_consistency", {}),
                    "report_path": str(INTEGRITY_REPORT_PATH),
                },
            },
            {
                "name": "full_gsm8k_result_shape_ready",
                "status": "pass" if full_gsm8k_complete else "running",
                "detail": summary.get("paired_shape", {}),
            },
            {
                "name": "paper_result_target_comparison",
                "status": paper_comparison_status,
                "detail": {
                    "status": paper_comparison.get("status"),
                    "blockers": paper_comparison.get("blockers", []),
                    "primary_gsm8k_comparison": paper_comparison.get("primary_gsm8k_comparison", {}),
                    "trajectory_comparison": paper_comparison.get("trajectory_comparison", {}),
                    "explicit_debt_comparison": paper_comparison.get("explicit_debt_comparison", {}),
                    "report_path": str(PAPER_COMPARISON_REPORT_PATH),
                },
            },
            {
                "name": "gsm8k_protocol_repair_campaign_ready",
                "status": "pass" if protocol_repair_status.startswith("completed_") else ("running" if protocol_repair_campaign else "missing"),
                "detail": {
                    "status": protocol_repair_status,
                    "manifest_path": str(PROTOCOL_REPAIR_MANIFEST_PATH),
                    "merged_artifact": protocol_repair_merged,
                    "runnable_config_count": len(protocol_repair_campaign.get("runnable_configs", [])),
                    "completed_config_count": len(protocol_repair_done),
                    "pending_config_count": len(protocol_repair_pending),
                    "oracle_values_exposed_to_loop2": False,
                },
            },
            {
                "name": "trajectory_analysis_ready",
                "status": "pass" if trajectory_complete else "running",
                "detail": trajectory_live,
            },
            {
                "name": "trajectory_metric_semantics_audit_ready",
                "status": "pass" if trajectory_semantics_audit.get("status") == "completed_metric_semantics_audit" else "missing",
                "detail": {
                    "status": trajectory_semantics_audit.get("status"),
                    "audit_path": str(TRAJ_SEMANTICS_AUDIT_PATH),
                    "failed_simulation_setting_ids": trajectory_semantics_audit.get("failed_simulation_setting_ids_from_dag", []),
                    "diagnoses": trajectory_audit_diagnoses,
                    "oracle_values_exposed_to_loop2": trajectory_semantics_audit.get("oracle_values_exposed_to_loop2"),
                },
            },
            {
                "name": "ablation_grid_campaign_ready",
                "status": "running" if ablation_campaign else "missing",
                "detail": {
                    "status": ablation_grid_status,
                    "manifest_path": str(ABLATION_MANIFEST_PATH),
                    "runnable_config_count": len(ablation_campaign.get("runnable_configs", [])),
                    "completed_config_count": len(ablation_done),
                    "pending_config_count": len(ablation_pending),
                    "blocked_config_ids": [
                        item.get("id") for item in ablation_campaign.get("blocked_configs", [])
                    ],
                },
            },
            {
                "name": "ablation_grid_integrity",
                "status": (
                    "blocked"
                    if str(ablation_integrity.get("status", "")).startswith("blocked")
                    else ("pass" if ablation_integrity else "missing")
                ),
                "detail": {
                    "status": ablation_integrity.get("status"),
                    "report_path": str(ABLATION_INTEGRITY_REPORT_PATH),
                    "blocked_config_ids": ablation_integrity.get("blocked_config_ids", []),
                    "manifest_blocked_config_ids": ablation_integrity.get("manifest_blocked_config_ids", []),
                    "running_config_ids": ablation_integrity.get("running_config_ids", []),
                    "complete_config_ids": ablation_integrity.get("complete_config_ids", []),
                    "pending_config_ids": ablation_integrity.get("pending_config_ids", []),
                },
            },
            {
                "name": "table1_threshold_repair_campaign_ready",
                "status": "running" if table1_threshold_campaign else "missing",
                "detail": {
                    "status": table1_threshold_status,
                    "manifest_path": str(TABLE1_THRESHOLD_MANIFEST_PATH),
                    "baseline_full_split_ready": table1_threshold_campaign.get(
                        "baseline_status", {}
                    ).get("full_split_ready"),
                    "runnable_config_count": len(table1_threshold_campaign.get("runnable_configs", [])),
                    "completed_config_count": len(table1_threshold_done),
                    "pending_config_count": len(table1_threshold_pending),
                    "blocked_config_ids": [
                        item.get("id") for item in table1_threshold_campaign.get("blocked_configs", [])
                    ],
                    "oracle_values_exposed_to_loop2": False,
                },
            },
            {
                "name": "multibenchmark_grid_campaign_ready",
                "status": "running" if multibenchmark_campaign else "missing",
                "detail": {
                    "status": multibenchmark_grid_status,
                    "manifest_path": str(MULTIBENCHMARK_MANIFEST_PATH),
                    "runnable_config_count": len(multibenchmark_campaign.get("runnable_configs", [])),
                    "linked_existing_artifact_count": len(multibenchmark_campaign.get("linked_existing_artifacts", [])),
                    "completed_config_count": len(multibenchmark_done),
                    "pending_config_count": len(multibenchmark_pending),
                    "blocked_config_ids": [
                        item.get("id") for item in multibenchmark_campaign.get("blocked_configs", [])
                    ],
                },
            },
            {
                "name": "table2_acceleration_campaign_ready",
                "status": "running" if table2_campaign else "missing",
                "detail": {
                    "status": table2_status,
                    "manifest_path": str(TABLE2_MANIFEST_PATH),
                    "linked_existing_artifact_count": len(table2_linked),
                    "linked_existing_complete_count": len(table2_linked_complete),
                    "runnable_config_count": len(table2_campaign.get("runnable_configs", [])),
                    "blocked_config_ids": [
                        item.get("id") for item in table2_campaign.get("blocked_configs", [])
                    ],
                },
            },
            {
                "name": "dream7b_axis_campaign_ready",
                "status": "blocked" if dream_campaign else "missing",
                "detail": {
                    "status": dream_status,
                    "manifest_path": str(DREAM_MANIFEST_PATH),
                    "runnable_config_count": len(dream_campaign.get("runnable_configs", [])),
                    "blocked_config_ids": [
                        item.get("id") for item in dream_campaign.get("blocked_configs", [])
                    ],
                },
            },
            {
                "name": "source_parity_blocker_audit_ready",
                "status": (
                    "pass"
                    if source_parity_audit.get("status") == "evidence_bound_source_parity_blockers_ready"
                    else "missing"
                ),
                "detail": {
                    "status": source_parity_audit.get("status"),
                    "report_path": str(SOURCE_PARITY_AUDIT_REPORT_PATH),
                    "status_path": str(SOURCE_PARITY_AUDIT_STATUS_PATH),
                    "blocker_ids": [
                        item.get("id") for item in source_parity_audit.get("blocker_audits", [])
                    ],
                    "can_converge_from_this_audit_alone": source_parity_audit.get("verifier_implication", {}).get(
                        "can_converge_from_this_audit_alone"
                    ),
                },
            },
            {
                "name": "gsm8k_live_shape_risk_audit_ready",
                "status": "pass" if gsm8k_risk_audit else "missing",
                "detail": {
                    "status": gsm8k_risk_audit.get("status"),
                    "report_path": str(GSM8K_RISK_AUDIT_REPORT_PATH),
                    "status_path": str(GSM8K_RISK_AUDIT_STATUS_PATH),
                    "loop2_author_can_read": gsm8k_risk_audit.get("visibility_contract", {}).get(
                        "loop2_author_can_read"
                    ),
                    "can_converge_from_this_audit_alone": gsm8k_risk_audit.get("risk_interpretation", {}).get(
                        "can_converge_from_this_audit_alone"
                    ),
                    "failing_metrics": gsm8k_risk_audit.get(
                        "comparison_snapshot_without_oracle_targets", {}
                    ).get("failing_metrics", []),
                    "repair_axis_ids": [
                        item.get("id") for item in gsm8k_risk_audit.get("possible_dag_repair_axes", [])
                    ],
                },
            },
            {
                "name": "gsm8k_protocol_parity_audit_ready",
                "status": "pass" if gsm8k_protocol_audit else "missing",
                "detail": {
                    "status": gsm8k_protocol_audit.get("status"),
                    "report_path": str(GSM8K_PROTOCOL_AUDIT_REPORT_PATH),
                    "status_path": str(GSM8K_PROTOCOL_AUDIT_STATUS_PATH),
                    "loop2_author_can_read": gsm8k_protocol_audit.get("visibility_contract", {}).get(
                        "loop2_author_can_read"
                    ),
                    "can_converge_from_this_audit_alone": gsm8k_protocol_audit.get(
                        "verifier_implication", {}
                    ).get("can_converge_from_this_audit_alone"),
                    "finding_statuses": {
                        item.get("id"): item.get("status")
                        for item in gsm8k_protocol_audit.get("findings", [])
                        if item.get("id")
                    },
                    "covered_repair_nodes": gsm8k_protocol_audit.get("dag_coverage", {}).get(
                        "covered_repair_nodes", []
                    ),
                },
            },
            {
                "name": "gsm8k_table1_protocol_selection_audit_ready",
                "status": "pass" if gsm8k_table1_selection_audit else "missing",
                "detail": {
                    "status": gsm8k_table1_selection_audit.get("status"),
                    "report_path": str(GSM8K_TABLE1_SELECTION_AUDIT_REPORT_PATH),
                    "status_path": str(GSM8K_TABLE1_SELECTION_AUDIT_STATUS_PATH),
                    "loop2_author_can_read": gsm8k_table1_selection_audit.get(
                        "visibility_contract", {}
                    ).get("loop2_author_can_read"),
                    "paper_oracle_target_values_included": gsm8k_table1_selection_audit.get(
                        "visibility_contract", {}
                    ).get("paper_oracle_target_values_included"),
                    "primary_status_after_selection_repair": gsm8k_table1_selection_audit.get(
                        "non_oracle_result_interpretation", {}
                    ).get("primary_status_after_selection_repair"),
                    "failed_metric_ids": gsm8k_table1_selection_audit.get(
                        "non_oracle_result_interpretation", {}
                    ).get("failed_metric_ids", []),
                    "recommended_dag_update": gsm8k_table1_selection_audit.get(
                        "recommended_dag_update", {}
                    ),
                },
            },
            {
                "name": "reduced_proxy_rejection_gate",
                "status": "pass",
                "detail": "partial live metrics are monitoring evidence only and cannot converge the paper.",
            },
            {
                "name": "result_shape_comparison_ready",
                "status": "blocked",
                "detail": support_only_until,
            },
        ],
        "support_only_until": support_only_until,
        "unresolved_professional_debt": [
            {
                "id": "full_step_vs_prophet_gsm8k",
                "status": "running" if not full_gsm8k_complete else "complete",
                "needed": "Full GSM8K zero-shot baseline and Prophet results with accuracy, steps, speed, and raw rows.",
            },
            {
                "id": "trajectory_analysis_all_settings",
                "status": trajectory_debt_status,
                "needed": "DLM-Decoding-Analysis trajectory evidence for configured settings.",
            },
            {
                "id": "gsm8k_protocol_repair",
                "status": protocol_repair_status,
                "needed": "Full GSM8K prompt/constraint/scoring repair rerun after postcompletion result-shape mismatch.",
            },
            {
                "id": "table1_threshold_dynamics_repair",
                "status": table1_threshold_status,
                "needed": "Full-split Prophet-only Table 1 threshold candidates paired to the completed full-step baseline after primary shape mismatch.",
            },
            {
                "id": "multi_benchmark_grid",
                "status": multibenchmark_grid_status,
                "needed": "MMLU, ARC-C, HellaSwag, TruthfulQA, WinoGrande, PIQA, GPQA, HumanEval, MBPP and planning settings.",
            },
            {
                "id": "table2_acceleration_combinations",
                "status": table2_status,
                "needed": "SDTT, SDTT+Prophet, Fast-dLLM, and Fast-dLLM+Prophet GSM8K comparison.",
            },
            {
                "id": "static_step_and_block_length_ablations",
                "status": ablation_grid_status,
                "needed": "Static-step budget and block-length ablations.",
            },
            {
                "id": "dream_model_axis",
                "status": dream_status,
                "needed": "Dream-7B axis or explicit code/data blocker.",
            },
        ],
    }
    return verifier


def write_status(verifier: dict[str, Any]) -> None:
    live = verifier["live_full_gsm8k_runner"]
    integrity = verifier.get("live_integrity", {})
    comparison = verifier.get("paper_result_comparison", {})
    traj = verifier["trajectory_dataset_analysis"]
    ablation = verifier.get("ablation_grid_campaign", {})
    ablation_integrity = verifier.get("ablation_grid_integrity", {})
    table1_threshold = verifier.get("table1_threshold_repair_campaign", {})
    multibench = verifier.get("multibenchmark_grid_campaign", {})
    table2 = verifier.get("table2_acceleration_campaign", {})
    dream = verifier.get("dream7b_axis_campaign", {})
    source_parity = verifier.get("source_parity_blocker_audit", {})
    risk_audit = verifier.get("gsm8k_live_shape_risk_audit", {})
    protocol_audit = verifier.get("gsm8k_protocol_parity_audit", {})
    table1_selection_audit = verifier.get("gsm8k_table1_protocol_selection_audit", {})
    shape = live.get("running_summary", {}).get("paired_shape", {})
    lines = [
        "# Prophet Specialized Runner Status",
        "",
        f"- Updated: `{verifier['created_at_utc']}`",
        f"- Paper: `{TITLE}`",
        f"- Status: `{verifier['status']}`",
        f"- Full GSM8K paired samples: `{live['paired_completed_samples']}` / `{live['total_samples']}`",
        f"- JSONL rows: `{live['jsonl_rows']}`",
        f"- GPU: `{live.get('cuda_visible_devices')}`",
        f"- Runner PID: `{live.get('pid')}`",
        f"- JSONL integrity: `{integrity.get('status')}` gate=`{integrity.get('gate_status')}` rows=`{integrity.get('row_count')}`",
        f"- Integrity report: `{integrity.get('report_path')}`",
        f"- Paper comparison: `{comparison.get('status')}` gate=`{comparison.get('gate_status')}`",
        f"- Paper comparison report: `{comparison.get('report_path')}`",
        f"- Trajectory settings: `{traj.get('settings_completed')}` / `{traj.get('total_settings')}` status=`{traj.get('status')}`",
        f"- Trajectory rows: `{traj.get('rows_written')}`",
        f"- Ablation grid: `{ablation.get('status')}` runnable=`{ablation.get('runnable_config_count')}` completed=`{ablation.get('completed_config_count')}`",
        f"- Ablation manifest: `{ablation.get('manifest_path')}`",
        f"- Ablation integrity: `{ablation_integrity.get('status')}` running=`{len(ablation_integrity.get('running_config_ids', []) or [])}` complete=`{len(ablation_integrity.get('complete_config_ids', []) or [])}` integrity_blocked=`{len(ablation_integrity.get('blocked_config_ids', []) or [])}` manifest_blocked=`{len(ablation_integrity.get('manifest_blocked_config_ids', []) or [])}`",
        f"- Ablation integrity report: `{ablation_integrity.get('report_path')}`",
        f"- Table 1 threshold repair: `{table1_threshold.get('status')}` baseline_ready=`{table1_threshold.get('baseline_full_split_ready')}` runnable=`{table1_threshold.get('runnable_config_count')}` completed=`{table1_threshold.get('completed_config_count')}` running=`{table1_threshold.get('running_config_count')}`",
        f"- Table 1 threshold manifest: `{table1_threshold.get('manifest_path')}`",
        f"- Multi-benchmark grid: `{multibench.get('status')}` runnable=`{multibench.get('runnable_config_count')}` completed=`{multibench.get('completed_config_count')}`",
        f"- Multi-benchmark manifest: `{multibench.get('manifest_path')}`",
        f"- Table 2 acceleration campaign: `{table2.get('status')}` linked_complete=`{table2.get('linked_existing_complete_count')}` blockers=`{len(table2.get('blocked_configs', []) or [])}`",
        f"- Table 2 manifest: `{table2.get('manifest_path')}`",
        f"- Dream-7B axis: `{dream.get('status')}` blockers=`{len(dream.get('blocked_configs', []) or [])}`",
        f"- Dream-7B manifest: `{dream.get('manifest_path')}`",
        f"- Source parity audit: `{source_parity.get('status')}` blockers=`{len(source_parity.get('blocker_ids', []) or [])}`",
        f"- Source parity report: `{source_parity.get('report_path')}`",
        f"- GSM8K live shape risk audit: `{risk_audit.get('status')}` failing_metrics=`{risk_audit.get('failing_metrics')}` loop2_visible=`{risk_audit.get('loop2_author_can_read')}`",
        f"- GSM8K risk audit report: `{risk_audit.get('report_path')}`",
        f"- GSM8K protocol parity audit: `{protocol_audit.get('status')}` findings=`{protocol_audit.get('finding_statuses')}` loop2_visible=`{protocol_audit.get('loop2_author_can_read')}`",
        f"- GSM8K protocol audit report: `{protocol_audit.get('report_path')}`",
        f"- GSM8K Table 1 protocol-selection audit: `{table1_selection_audit.get('status')}` primary_after_repair=`{table1_selection_audit.get('primary_status_after_selection_repair')}` failed_metrics=`{table1_selection_audit.get('failed_metric_ids')}` loop2_visible=`{table1_selection_audit.get('loop2_author_can_read')}`",
        f"- GSM8K Table 1 protocol-selection audit report: `{table1_selection_audit.get('report_path')}`",
        f"- Mean step reduction: `{shape.get('mean_step_reduction')}`",
        f"- Mean seconds speedup: `{shape.get('speedup_mean_seconds')}`",
        "- Convergence: `pending_full_artifacts_and_verifier_comparison`",
        "",
        "This is a full-split GPU run, not reduced convergence evidence. It remains active until the verifier can compare final artifact shape to the paper evidence channels.",
    ]
    SPECIALIZED_STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_paper_operational_artifacts(verifier: dict[str, Any]) -> None:
    existing = read_json(PAPER_OPERATIONAL_ARTIFACTS_PATH, {})
    integrity = verifier.get("live_integrity", {})
    comparison = verifier.get("paper_result_comparison", {})
    existing["prophet_live_verifier"] = {
        "path": str(PAPER_LIVE_VERIFIER_PATH),
        "specialized_verifier_path": str(SPECIALIZED_VERIFIER_PATH),
        "status": verifier["status"],
        "converged": verifier["converged"],
        "paired_completed_samples": verifier["live_full_gsm8k_runner"]["paired_completed_samples"],
        "total_samples": verifier["live_full_gsm8k_runner"]["total_samples"],
        "integrity_report_path": integrity.get("report_path"),
        "integrity_status": integrity.get("status"),
        "integrity_gate_status": integrity.get("gate_status"),
        "paper_comparison_report_path": comparison.get("report_path"),
        "paper_comparison_status": comparison.get("status"),
        "paper_comparison_gate_status": comparison.get("gate_status"),
        "ablation_grid_manifest_path": verifier.get("ablation_grid_campaign", {}).get("manifest_path"),
        "ablation_grid_status": verifier.get("ablation_grid_campaign", {}).get("status"),
        "ablation_grid_integrity_report_path": verifier.get("ablation_grid_integrity", {}).get("report_path"),
        "ablation_grid_integrity_status": verifier.get("ablation_grid_integrity", {}).get("status"),
        "table1_threshold_repair_manifest_path": verifier.get("table1_threshold_repair_campaign", {}).get(
            "manifest_path"
        ),
        "table1_threshold_repair_status": verifier.get("table1_threshold_repair_campaign", {}).get("status"),
        "table1_threshold_repair_baseline_full_split_ready": verifier.get(
            "table1_threshold_repair_campaign", {}
        ).get("baseline_full_split_ready"),
        "multibenchmark_grid_manifest_path": verifier.get("multibenchmark_grid_campaign", {}).get("manifest_path"),
        "multibenchmark_grid_status": verifier.get("multibenchmark_grid_campaign", {}).get("status"),
        "table2_acceleration_manifest_path": verifier.get("table2_acceleration_campaign", {}).get("manifest_path"),
        "table2_acceleration_status": verifier.get("table2_acceleration_campaign", {}).get("status"),
        "dream7b_axis_manifest_path": verifier.get("dream7b_axis_campaign", {}).get("manifest_path"),
        "dream7b_axis_status": verifier.get("dream7b_axis_campaign", {}).get("status"),
        "source_parity_audit_report_path": verifier.get("source_parity_blocker_audit", {}).get("report_path"),
        "source_parity_audit_status": verifier.get("source_parity_blocker_audit", {}).get("status"),
        "gsm8k_live_shape_risk_audit_report_path": verifier.get("gsm8k_live_shape_risk_audit", {}).get("report_path"),
        "gsm8k_live_shape_risk_audit_status": verifier.get("gsm8k_live_shape_risk_audit", {}).get("status"),
        "gsm8k_live_shape_risk_audit_loop2_author_can_read": verifier.get("gsm8k_live_shape_risk_audit", {}).get(
            "loop2_author_can_read"
        ),
        "gsm8k_protocol_parity_audit_report_path": verifier.get("gsm8k_protocol_parity_audit", {}).get("report_path"),
        "gsm8k_protocol_parity_audit_status": verifier.get("gsm8k_protocol_parity_audit", {}).get("status"),
        "gsm8k_protocol_parity_audit_loop2_author_can_read": verifier.get("gsm8k_protocol_parity_audit", {}).get(
            "loop2_author_can_read"
        ),
        "gsm8k_table1_protocol_selection_audit_report_path": verifier.get(
            "gsm8k_table1_protocol_selection_audit", {}
        ).get("report_path"),
        "gsm8k_table1_protocol_selection_audit_status": verifier.get(
            "gsm8k_table1_protocol_selection_audit", {}
        ).get("status"),
        "gsm8k_table1_protocol_selection_audit_loop2_author_can_read": verifier.get(
            "gsm8k_table1_protocol_selection_audit", {}
        ).get("loop2_author_can_read"),
        "gsm8k_table1_protocol_selection_audit_primary_status_after_selection_repair": verifier.get(
            "gsm8k_table1_protocol_selection_audit", {}
        ).get("primary_status_after_selection_repair"),
        "trajectory_settings_completed": verifier["trajectory_dataset_analysis"].get("settings_completed"),
        "trajectory_total_settings": verifier["trajectory_dataset_analysis"].get("total_settings"),
        "trajectory_rows_written": verifier["trajectory_dataset_analysis"].get("rows_written"),
        "updated_at_utc": verifier["created_at_utc"],
    }
    write_json(PAPER_OPERATIONAL_ARTIFACTS_PATH, existing)


def main() -> None:
    refresh_integrity_report()
    refresh_ablation_integrity_report()
    refresh_source_parity_audit_report()
    refresh_paper_comparison_report()
    refresh_gsm8k_risk_audit_report()
    refresh_gsm8k_protocol_audit_report()
    verifier = build_payload()
    write_json(SPECIALIZED_VERIFIER_PATH, verifier)
    write_json(PAPER_LIVE_VERIFIER_PATH, verifier)
    write_status(verifier)
    update_paper_operational_artifacts(verifier)
    print(json.dumps(verifier, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
