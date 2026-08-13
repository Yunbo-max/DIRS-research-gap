#!/usr/bin/env python3
"""Regression checks for strict Prophet Loop 1/Loop 2 repair behavior.

These checks keep a completed-but-mismatched operational result from being
accepted as convergence, and keep verifier oracle values out of the DAG-only
Loop 2 contract.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


RUN_ROOT = Path("/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723")
RUNNER_DIR = RUN_ROOT / "specialized_runners/prophet"
REFRESH_PATH = RUNNER_DIR / "refresh_prophet_live_verifier.py"
SYNC_PATH = RUN_ROOT / "sync_verifier_blockers_to_dags.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_postcompletion_mismatch_creates_repair_update() -> dict[str, object]:
    refresh = load_module("refresh_prophet_live_verifier_regression", REFRESH_PATH)

    synthetic_status = {
        "status": "completed",
        "completed_sample_indices": 1319,
        "total_samples": 1319,
        "pid": None,
    }
    synthetic_summary = {
        "total_samples": 1319,
        "aggregates": {
            "baseline": {"completed_samples": 1319, "mean_seconds": 1.0},
            "prophet": {"completed_samples": 1319, "mean_seconds": 0.8},
        },
    }
    synthetic_trajectory = {
        "status": "complete",
        "settings_completed": 1,
        "total_settings": 1,
    }
    synthetic_trajectory_summary = {
        "settings": [{"id": "llada8b_gsm8k"}],
        "professional_gate": {"all_expected_counts_present": True},
    }
    synthetic_paper_comparison = {
        "status": "running_paper_result_comparison",
        "trajectory_comparison": {"status": "pass_close_to_paper_trajectory_shape"},
    }
    synthetic_risk_audit = {
        "status": "postcompletion_shape_mismatch_requires_loop1_dag_repair",
        "comparison_snapshot_without_oracle_targets": {
            "failing_metrics": ["accuracy_delta", "prophet_avg_steps", "step_speedup"],
        },
        "possible_dag_repair_axes": [
            {"id": "prompt_template_parity"},
            {"id": "suffix_constraint_semantics"},
            {"id": "generated_answer_extractor_parity"},
        ],
    }
    synthetic_protocol_audit = {
        "findings": [
            {"id": "prompt_template_parity", "status": "partial_unproven_equivalence"},
            {"id": "generated_answer_extractor_parity", "status": "custom_extractor_not_exact"},
        ],
    }

    payload_by_path = {
        refresh.STATUS_PATH: synthetic_status,
        refresh.SUMMARY_PATH: synthetic_summary,
        refresh.TRAJ_STATUS_PATH: synthetic_trajectory,
        refresh.TRAJ_SUMMARY_PATH: synthetic_trajectory_summary,
        refresh.PAPER_COMPARISON_REPORT_PATH: synthetic_paper_comparison,
        refresh.GSM8K_RISK_AUDIT_REPORT_PATH: synthetic_risk_audit,
        refresh.GSM8K_PROTOCOL_AUDIT_REPORT_PATH: synthetic_protocol_audit,
    }

    def fake_read_json(path: Path, default: object = None) -> object:
        return payload_by_path.get(path, default if default is not None else {})

    def fake_line_count(path: Path) -> int:
        if path == refresh.ROWS_PATH:
            return 1319 * 2
        if path == refresh.TRAJ_ROWS_PATH:
            return 1
        return 0

    refresh.read_json = fake_read_json
    refresh.line_count = fake_line_count
    refresh.process_alive = lambda pid: None

    payload = refresh.build_payload()
    updates = {
        update.get("id"): update
        for update in payload.get("required_updates", [])
        if isinstance(update, dict)
    }
    update = updates.get("gsm8k_postcompletion_protocol_shape_repair")
    assert update, "postcompletion mismatch did not create GSM8K repair update"
    assert update["oracle_values_exposed_to_loop2"] is False
    assert update["artifact_family"] == "full_gsm8k_baseline_prophet_result_shape"
    assert "accuracy_delta" in update["failed_metric_ids"]
    assert "prompt_template_parity" in update["repair_axis_ids"]
    assert update["protocol_finding_statuses"]["generated_answer_extractor_parity"] == "custom_extractor_not_exact"

    return {
        "required_update_id": update["id"],
        "failed_metric_count": len(update["failed_metric_ids"]),
        "repair_axis_count": len(update["repair_axis_ids"]),
        "oracle_values_exposed_to_loop2": update["oracle_values_exposed_to_loop2"],
    }


def assert_sync_preserves_false_oracle_flag() -> dict[str, object]:
    sync = load_module("sync_verifier_blockers_to_dags_regression", SYNC_PATH)
    verifier = {
        "status": "not_converged",
        "converged": False,
        "required_updates": [
            {
                "id": "gsm8k_postcompletion_protocol_shape_repair",
                "artifact_family": "full_gsm8k_baseline_prophet_result_shape",
                "reason": "completed full split has wrong result shape",
                "success_criteria": ["repair protocol and rerun full GSM8K"],
                "failed_metric_ids": ["accuracy_delta"],
                "repair_axis_ids": ["prompt_template_parity"],
                "protocol_finding_statuses": {"prompt_template_parity": "partial"},
                "risk_audit_status": "postcompletion_shape_mismatch_requires_loop1_dag_repair",
                "report_path": "/tmp/gsm8k_live_shape_risk_audit.json",
                "oracle_values_exposed_to_loop2": False,
            }
        ],
    }

    feedback = sync.collect_verifier_feedback(verifier)
    update = feedback["required_updates"][0]
    assert update["oracle_values_exposed_to_loop2"] is False
    assert update["repair_axis_ids"] == ["prompt_template_parity"]
    assert update["protocol_finding_statuses"]["prompt_template_parity"] == "partial"

    return {
        "required_update_id": update["id"],
        "oracle_values_exposed_to_loop2": update["oracle_values_exposed_to_loop2"],
        "repair_axis_ids": update["repair_axis_ids"],
    }


def main() -> None:
    summary = {
        "postcompletion_required_update": assert_postcompletion_mismatch_creates_repair_update(),
        "sync_required_update_preservation": assert_sync_preserves_false_oracle_flag(),
    }
    print(json.dumps({"status": "pass", "checks": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
