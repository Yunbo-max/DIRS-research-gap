#!/usr/bin/env python3
"""Verifier-only live risk audit for Prophet GSM8K result shape.

This file is deliberately not a Loop 2 input. It watches the real full-split
GPU run and prepares DAG repair axes if the final artifact shape does not match
the paper-level verifier. It must never promote partial rows to convergence.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ROOT = Path("/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723")
RUNNER_DIR = RUN_ROOT / "specialized_runners/prophet"
PAPER_RUN = RUN_ROOT / "paper_runs/iclr2026_g88nt4ietg_prophet_dlm_early_commit_decoding"

STATUS_PATH = RUNNER_DIR / "custom_full_gsm8k_llada8b/status.json"
SUMMARY_PATH = RUNNER_DIR / "custom_full_gsm8k_llada8b/summary.json"
COMPARISON_PATH = RUNNER_DIR / "prophet_paper_result_comparison.json"
DAG_PATH = PAPER_RUN / "paper_author_gap_dag.json"

AUDIT_PATH = RUNNER_DIR / "gsm8k_live_shape_risk_audit.json"
AUDIT_STATUS_PATH = RUNNER_DIR / "GSM8K_LIVE_SHAPE_RISK_AUDIT.md"


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


def metric_check_statuses(primary: dict[str, Any]) -> dict[str, str]:
    checks = primary.get("checks", {})
    statuses: dict[str, str] = {}
    for name, check in checks.items():
        statuses[name] = str(check.get("status") or "missing")
    return statuses


def observed_metrics(primary: dict[str, Any]) -> dict[str, Any]:
    observed = primary.get("observed", {})
    allowed = [
        "paired_completed_samples",
        "total_samples",
        "baseline_accuracy_pct",
        "prophet_accuracy_pct",
        "accuracy_delta_pct_points",
        "baseline_avg_steps",
        "prophet_avg_steps",
        "step_speedup",
        "wallclock_speedup",
    ]
    return {key: observed.get(key) for key in allowed if key in observed}


def repair_axes_from_statuses(statuses: dict[str, str]) -> list[dict[str, Any]]:
    axes = []
    failed = {name for name, status in statuses.items() if status == "fail"}
    if {"prophet_avg_steps", "step_speedup"} & failed:
        axes.extend(
            [
                {
                    "id": "prompt_template_parity",
                    "reason": "The answer-emergence and exit point can shift if the simulation prompt is not the paper's exact evaluation prompt.",
                    "dag_node_candidate": "protocol.extract_exact_prompt_template_without_oracle_values",
                },
                {
                    "id": "suffix_constraint_semantics",
                    "reason": "Prophet monitors a final-answer region; a different suffix or constrained-token layout can delay or advance exits.",
                    "dag_node_candidate": "runner.bind_suffix_constraints_and_answer_region",
                },
                {
                    "id": "answer_region_start_and_length",
                    "reason": "Step savings depend on the exact answer-region token span used by the early-commit rule.",
                    "dag_node_candidate": "runner.audit_answer_region_start_length_from_release_code",
                },
                {
                    "id": "simple_evals_vs_lm_eval_protocol",
                    "reason": "The paper describes simple-evals-style scoring, while the release exposes an lm-eval integration and custom runner glue.",
                    "dag_node_candidate": "protocol.align_evaluation_harness_before_claiming_table1_shape",
                },
            ]
        )
    if {"accuracy_delta", "prophet_accuracy"} & failed:
        axes.extend(
            [
                {
                    "id": "generated_answer_extractor_parity",
                    "reason": "A scoring/extraction mismatch can change the direction of the accuracy delta without changing the underlying samples.",
                    "dag_node_candidate": "verifier.bind_generated_answer_extractor_semantics",
                },
                {
                    "id": "released_eval_harness_vs_custom_runner_semantics",
                    "reason": "The custom full-split runner must be checked against the released eval path before a result-shape mismatch is attributed to the idea.",
                    "dag_node_candidate": "runner.compare_custom_runner_to_released_eval_llada_path",
                },
            ]
        )
    if not axes:
        axes.append(
            {
                "id": "monitor_until_full_split",
                "reason": "No repair axis is activated before the full split produces final comparable artifacts.",
                "dag_node_candidate": "verifier.wait_for_full_split_before_repair",
            }
        )
    deduped: list[dict[str, Any]] = []
    seen = set()
    for axis in axes:
        axis_id = axis["id"]
        if axis_id in seen:
            continue
        seen.add(axis_id)
        deduped.append(axis)
    return deduped


def build_payload() -> dict[str, Any]:
    now = utc_now()
    status = read_json(STATUS_PATH, {})
    summary = read_json(SUMMARY_PATH, {})
    comparison = read_json(COMPARISON_PATH, {})
    primary = comparison.get("primary_gsm8k_comparison", {})
    aggregates = summary.get("aggregates", {})
    baseline_done = int(aggregates.get("baseline", {}).get("completed_samples") or 0)
    prophet_done = int(aggregates.get("prophet", {}).get("completed_samples") or 0)
    paired_done = int(primary.get("observed", {}).get("paired_completed_samples") or min(baseline_done, prophet_done))
    total = int(primary.get("observed", {}).get("total_samples") or summary.get("total_samples") or status.get("total_samples") or 1319)
    full_split_complete = bool(primary.get("complete") or (total > 0 and paired_done >= total))
    progress = float(paired_done / total) if total else 0.0
    statuses = metric_check_statuses(primary)
    failing_metrics = sorted(name for name, metric_status in statuses.items() if metric_status == "fail")
    monitoring_threshold_reached = progress >= 0.75

    if full_split_complete and failing_metrics:
        audit_status = "postcompletion_shape_mismatch_requires_loop1_dag_repair"
        recommended_action = "create_loop1_dag_repair_nodes_then_rerun_or_rescore"
        do_not_stop_before_full_split = False
    elif full_split_complete:
        audit_status = "postcompletion_shape_close_no_risk_repair"
        recommended_action = "continue_remaining_professional_debt_gates"
        do_not_stop_before_full_split = False
    elif monitoring_threshold_reached and failing_metrics:
        audit_status = "monitoring_only_likely_postcompletion_shape_mismatch"
        recommended_action = "wait_for_full_split_then_activate_repair_axes_if_final_comparison_still_fails"
        do_not_stop_before_full_split = True
    else:
        audit_status = "monitoring_only_waiting_for_full_split"
        recommended_action = "wait_for_full_split_before_result_shape_judgment"
        do_not_stop_before_full_split = True

    repair_axes = repair_axes_from_statuses(statuses)
    dag_snapshot = read_json(DAG_PATH, {})
    node_ids = [
        node.get("id")
        for node in dag_snapshot.get("nodes", [])
        if isinstance(node, dict) and node.get("id")
    ]
    existing_repair_related_nodes = [
        node_id
        for node_id in node_ids
        if any(term in node_id for term in ["prompt", "answer", "harness", "protocol", "source_parity"])
    ]

    return {
        "artifact_kind": "prophet_gsm8k_live_shape_risk_audit",
        "created_at_utc": now,
        "status": audit_status,
        "paper_id": "ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding",
        "paper_title": "Diffusion Language Models Know the Answer Before Decoding",
        "visibility_contract": {
            "loop2_author_can_read": False,
            "verifier_can_read": True,
            "paper_oracle_target_values_included": False,
            "oracle_values_exposed_to_loop2": False,
        },
        "live_progress": {
            "paired_completed_samples": paired_done,
            "total_samples": total,
            "progress_fraction": progress,
            "full_split_complete": full_split_complete,
            "status_path": str(STATUS_PATH),
            "summary_path": str(SUMMARY_PATH),
        },
        "comparison_snapshot_without_oracle_targets": {
            "comparison_status": comparison.get("status"),
            "primary_gsm8k_status": primary.get("status"),
            "metric_statuses": statuses,
            "failing_metrics": failing_metrics,
            "observed_metrics": observed_metrics(primary),
            "comparison_report_path": str(COMPARISON_PATH),
        },
        "risk_interpretation": {
            "can_converge_from_this_audit_alone": False,
            "partial_rows_are_convergence_evidence": False,
            "monitoring_threshold_reached": monitoring_threshold_reached,
            "do_not_stop_before_full_split": do_not_stop_before_full_split,
            "recommended_action": recommended_action,
        },
        "possible_dag_repair_axes": repair_axes,
        "current_dag_repair_related_nodes": existing_repair_related_nodes,
        "status_path": str(AUDIT_STATUS_PATH),
        "report_path": str(AUDIT_PATH),
    }


def write_status(payload: dict[str, Any]) -> None:
    progress = payload["live_progress"]
    snapshot = payload["comparison_snapshot_without_oracle_targets"]
    risk = payload["risk_interpretation"]
    axes = payload["possible_dag_repair_axes"]
    lines = [
        "# GSM8K Live Shape Risk Audit",
        "",
        f"- Updated: `{payload['created_at_utc']}`",
        f"- Status: `{payload['status']}`",
        f"- Samples: `{progress.get('paired_completed_samples')}/{progress.get('total_samples')}`",
        f"- Full split complete: `{progress.get('full_split_complete')}`",
        f"- Comparison status: `{snapshot.get('comparison_status')}`",
        f"- Primary GSM8K status: `{snapshot.get('primary_gsm8k_status')}`",
        f"- Failing metrics: `{snapshot.get('failing_metrics')}`",
        f"- Can converge from this audit alone: `{risk.get('can_converge_from_this_audit_alone')}`",
        f"- Do not stop before full split: `{risk.get('do_not_stop_before_full_split')}`",
        f"- Loop 2 can read this: `{payload['visibility_contract']['loop2_author_can_read']}`",
        f"- Report: `{payload['report_path']}`",
        "",
        "## Possible DAG Repair Axes",
        "",
    ]
    for axis in axes:
        lines.append(f"- `{axis['id']}`: {axis['reason']}")
    AUDIT_STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = build_payload()
    write_json(AUDIT_PATH, payload)
    write_status(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
