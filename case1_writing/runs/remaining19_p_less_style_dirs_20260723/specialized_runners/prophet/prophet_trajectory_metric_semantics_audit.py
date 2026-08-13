#!/usr/bin/env python3
"""Audit trajectory first-emergence metric semantics from DAG-visible artifacts.

This is a Loop 2 repair artifact: it reads the DAG repair node and the
simulation-produced trajectory rows, but it does not read paper text or target
oracle values. The verifier can then decide whether the mismatch is a metric
semantics issue, an implementation issue, or a true paper-shape failure.
"""

from __future__ import annotations

import json
import os
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ROOT = Path(__file__).resolve().parents[2]
PAPER_DAG = (
    RUN_ROOT
    / "paper_runs/iclr2026_g88nt4ietg_prophet_dlm_early_commit_decoding/paper_author_gap_dag.json"
)
TRAJECTORY_DIR = Path(__file__).resolve().parent / "trajectory_dataset_analysis"
ROWS_PATH = TRAJECTORY_DIR / "trajectory_first_emergence_rows.jsonl"
SUMMARY_PATH = TRAJECTORY_DIR / "trajectory_analysis_summary.json"
OUT_JSON = TRAJECTORY_DIR / "trajectory_metric_semantics_audit.json"
OUT_MD = TRAJECTORY_DIR / "TRAJECTORY_METRIC_SEMANTICS_AUDIT.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def dag_failed_setting_ids() -> list[str]:
    dag = read_json(PAPER_DAG, {})
    for node in dag.get("nodes", []):
        if node.get("id") == "loop1.required_update.trajectory_metric_semantics_repair":
            return list(node.get("failed_simulation_setting_ids", []))
    return []


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ROWS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(count: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return count / denom * 100.0


def setting_audit(rows: list[dict[str, Any]], setting_id: str) -> dict[str, Any]:
    setting_rows = [row for row in rows if row.get("setting_id") == setting_id]
    correct_rows = [row for row in setting_rows if row.get("correct")]
    step_rows = [row for row in correct_rows if row.get("first_emergence_step") is not None]
    steps = [int(row["first_emergence_step"]) for row in step_rows]
    total = len(setting_rows)
    correct = len(correct_rows)
    correct_source_counts = Counter(str(row.get("correct_source", "missing")) for row in setting_rows)
    stored_disagreement_count = sum(
        1
        for row in setting_rows
        if row.get("correct_source") == "author_notebook_gsm8k_recomputed_final_answer"
        and row.get("stored_correct") != row.get("recomputed_correct")
    )
    variants = {
        "correct_denominator": {
            "by_25pct_le": pct(
                sum(1 for row in correct_rows if row.get("first_emergence_pct") is not None and row["first_emergence_pct"] <= 25),
                correct,
            ),
            "by_25pct_lt": pct(
                sum(1 for row in correct_rows if row.get("first_emergence_pct") is not None and row["first_emergence_pct"] < 25),
                correct,
            ),
            "by_50pct_le": pct(
                sum(1 for row in correct_rows if row.get("first_emergence_pct") is not None and row["first_emergence_pct"] <= 50),
                correct,
            ),
            "by_50pct_lt": pct(
                sum(1 for row in correct_rows if row.get("first_emergence_pct") is not None and row["first_emergence_pct"] < 50),
                correct,
            ),
        },
        "all_rows_denominator": {
            "by_25pct_le": pct(
                sum(1 for row in setting_rows if row.get("first_emergence_pct") is not None and row["first_emergence_pct"] <= 25),
                total,
            ),
            "by_50pct_le": pct(
                sum(1 for row in setting_rows if row.get("first_emergence_pct") is not None and row["first_emergence_pct"] <= 50),
                total,
            ),
        },
        "step_zero_sensitivity_correct_denominator": {
            "step_eq_0": pct(sum(1 for row in correct_rows if row.get("first_emergence_step") == 0), correct),
            "step_le_1": pct(
                sum(
                    1
                    for row in correct_rows
                    if row.get("first_emergence_step") is not None and row["first_emergence_step"] <= 1
                ),
                correct,
            ),
            "step_gt0_le_25pct": pct(
                sum(
                    1
                    for row in correct_rows
                    if row.get("first_emergence_step") is not None and 0 < row["first_emergence_step"] <= 64
                ),
                correct,
            ),
            "step_gt0_le_50pct": pct(
                sum(
                    1
                    for row in correct_rows
                    if row.get("first_emergence_step") is not None and 0 < row["first_emergence_step"] <= 128
                ),
                correct,
            ),
        },
    }
    common_steps = Counter(steps).most_common(12)
    step_summary = {
        "min": min(steps) if steps else None,
        "median": statistics.median(steps) if steps else None,
        "mean": statistics.mean(steps) if steps else None,
        "p90": sorted(steps)[int(0.9 * len(steps)) - 1] if steps else None,
        "max": max(steps) if steps else None,
        "most_common": [{"step": step, "count": count} for step, count in common_steps],
    }
    step0_pct = variants["step_zero_sensitivity_correct_denominator"]["step_eq_0"] or 0.0
    uses_author_gsm8k_correctness = (
        correct_source_counts.get("author_notebook_gsm8k_recomputed_final_answer", 0) == total
        and total > 0
    )
    if uses_author_gsm8k_correctness:
        diagnosis = "resolved_by_author_notebook_gsm8k_correctness_semantics"
    elif "constraint" in setting_id and step0_pct > 10.0:
        diagnosis = "blocked_by_constraint_initial_state_semantics"
    else:
        diagnosis = "requires_metric_semantics_review"
    return {
        "setting_id": setting_id,
        "row_count": total,
        "correct_count": correct,
        "correct_rate_in_simulation_artifact": pct(correct, total),
        "correct_source_counts": dict(correct_source_counts),
        "stored_correct_disagreement_count": stored_disagreement_count,
        "variants": variants,
        "step_summary": step_summary,
        "diagnosis": diagnosis,
        "loop2_next_checks": [
            "For GSM8K, recompute correctness from pred_text vs gt_text using the released analysis/visualize.ipynb numeric extraction rule; do not trust stale .pt correct flags.",
            "Verify whether constrained trajectories include fixed/generated answer-region tokens at step 0.",
            "Re-score first emergence with and without step-0 matches for constrained settings.",
            "Confirm whether percentages should be computed over correct trajectories or all trajectories.",
            "Confirm whether threshold comparisons are strict or inclusive.",
            "Keep verifier paper targets hidden from this audit artifact.",
        ],
    }


def main() -> None:
    rows = load_rows()
    summary = read_json(SUMMARY_PATH, {})
    failed_ids = dag_failed_setting_ids()
    if not failed_ids:
        failed_ids = sorted({row.get("setting_id") for row in rows if row.get("setting_id")})
    payload = {
        "artifact_kind": "prophet_trajectory_metric_semantics_audit",
        "created_at_utc": utc_now(),
        "dag_path": str(PAPER_DAG),
        "rows_path": str(ROWS_PATH),
        "summary_path": str(SUMMARY_PATH),
        "oracle_values_exposed_to_loop2": False,
        "failed_simulation_setting_ids_from_dag": failed_ids,
        "professional_gate": summary.get("professional_gate", {}),
        "setting_audits": [setting_audit(rows, setting_id) for setting_id in failed_ids],
        "status": "completed_metric_semantics_audit",
    }
    write_json(OUT_JSON, payload)
    lines = [
        "# Trajectory Metric Semantics Audit",
        "",
        f"- Updated: `{payload['created_at_utc']}`",
        f"- Status: `{payload['status']}`",
        f"- Oracle values exposed to Loop 2: `{payload['oracle_values_exposed_to_loop2']}`",
        f"- Rows: `{payload['rows_path']}`",
        "",
    ]
    for audit in payload["setting_audits"]:
        variants = audit["variants"]
        step0 = variants["step_zero_sensitivity_correct_denominator"]
        lines.extend(
            [
                f"## {audit['setting_id']}",
                "",
                f"- Diagnosis: `{audit['diagnosis']}`",
                f"- Rows/correct: `{audit['row_count']}` / `{audit['correct_count']}`",
                f"- Correct source counts: `{audit['correct_source_counts']}`",
                f"- Stored-correct disagreements: `{audit['stored_correct_disagreement_count']}`",
                f"- Correct-denominator <=25/<=50: `{variants['correct_denominator']['by_25pct_le']}` / `{variants['correct_denominator']['by_50pct_le']}`",
                f"- All-row denominator <=25/<=50: `{variants['all_rows_denominator']['by_25pct_le']}` / `{variants['all_rows_denominator']['by_50pct_le']}`",
                f"- Step 0 / step <=1 over correct rows: `{step0['step_eq_0']}` / `{step0['step_le_1']}`",
                f"- Excluding step 0, <=25/<=50 over correct rows: `{step0['step_gt0_le_25pct']}` / `{step0['step_gt0_le_50pct']}`",
                "",
            ]
        )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"audit": str(OUT_JSON), "status": payload["status"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
