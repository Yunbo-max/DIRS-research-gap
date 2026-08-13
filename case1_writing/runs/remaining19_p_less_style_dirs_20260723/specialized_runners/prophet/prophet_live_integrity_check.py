#!/usr/bin/env python3
"""Integrity checks for the live Prophet full-GSM8K JSONL artifact."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNNER_DIR = Path(__file__).resolve().parent
CUSTOM_DIR = RUNNER_DIR / "custom_full_gsm8k_llada8b"
ROWS_PATH = CUSTOM_DIR / "per_sample_results.jsonl"
SUMMARY_PATH = CUSTOM_DIR / "summary.json"
STATUS_PATH = CUSTOM_DIR / "status.json"
REPORT_PATH = RUNNER_DIR / "prophet_live_integrity_report.json"
STATUS_MD = RUNNER_DIR / "PROPHET_LIVE_INTEGRITY_STATUS.md"

EXPECTED_VARIANTS = {"baseline", "prophet"}
REQUIRED_ROW_FIELDS = {
    "artifact_kind",
    "sample_index",
    "variant",
    "question",
    "gold_answer",
    "generated_text",
    "strict_exact_match",
    "flexible_exact_match",
    "seconds",
    "steps",
    "exit_info",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not ROWS_PATH.exists():
        return rows, [{"line_number": None, "error": "missing_rows_path"}]
    with ROWS_PATH.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append({"line_number": line_number, "error": str(exc), "line_prefix": line[:120]})
    return rows, errors


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    variant_counts = Counter(row.get("variant") for row in rows)
    pair_counts: dict[int, set[str]] = defaultdict(set)
    duplicate_counts = Counter()
    field_errors = []
    metric_errors = []
    index_errors = []

    for row_number, row in enumerate(rows, start=1):
        sample_index = row.get("sample_index")
        variant = row.get("variant")
        duplicate_counts[(sample_index, variant)] += 1
        if isinstance(sample_index, int):
            pair_counts[sample_index].add(variant)
            if sample_index < 0:
                index_errors.append({"row_number": row_number, "sample_index": sample_index, "error": "negative_sample_index"})
        else:
            index_errors.append({"row_number": row_number, "sample_index": sample_index, "error": "sample_index_not_int"})
        missing_fields = sorted(REQUIRED_ROW_FIELDS - set(row))
        if missing_fields:
            field_errors.append({"row_number": row_number, "missing_fields": missing_fields})
        if variant not in EXPECTED_VARIANTS:
            metric_errors.append({"row_number": row_number, "variant": variant, "error": "unexpected_variant"})
        for bool_key in ["strict_exact_match", "flexible_exact_match"]:
            if bool_key in row and not isinstance(row.get(bool_key), bool):
                metric_errors.append({"row_number": row_number, "field": bool_key, "error": "not_bool"})
        if "seconds" in row and not isinstance(row.get("seconds"), (int, float)):
            metric_errors.append({"row_number": row_number, "field": "seconds", "error": "not_numeric"})

    duplicates = [
        {"sample_index": sample_index, "variant": variant, "count": count}
        for (sample_index, variant), count in duplicate_counts.items()
        if count > 1
    ]
    complete_samples = sorted(idx for idx, variants in pair_counts.items() if variants == EXPECTED_VARIANTS)
    incomplete_samples = [
        {"sample_index": idx, "variants_present": sorted(variants)}
        for idx, variants in sorted(pair_counts.items())
        if variants != EXPECTED_VARIANTS
    ]
    return {
        "row_count": len(rows),
        "variant_counts": dict(sorted((str(k), v) for k, v in variant_counts.items())),
        "sample_count_observed": len(pair_counts),
        "paired_completed_samples_from_rows": len(complete_samples),
        "first_sample_index": min(pair_counts) if pair_counts else None,
        "last_sample_index": max(pair_counts) if pair_counts else None,
        "duplicate_pairs": duplicates,
        "incomplete_samples": incomplete_samples,
        "field_errors": field_errors[:50],
        "metric_errors": metric_errors[:50],
        "index_errors": index_errors[:50],
    }


def summary_consistency(row_summary: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    aggregates = summary.get("aggregates", {})
    deltas = {}
    statuses = []
    for variant in sorted(EXPECTED_VARIANTS):
        row_count = int(row_summary["variant_counts"].get(variant, 0))
        summary_count = int(aggregates.get(variant, {}).get("completed_samples") or 0)
        delta = row_count - summary_count
        deltas[variant] = {"row_count": row_count, "summary_count": summary_count, "delta": delta}
        if abs(delta) <= 1:
            statuses.append("pass_or_live_race")
        else:
            statuses.append("fail")
    return {
        "status": "pass_or_live_race" if "fail" not in statuses else "fail",
        "deltas": deltas,
        "summary_status": summary.get("status"),
        "summary_total_samples": summary.get("total_samples"),
    }


def classify(report: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = []
    if report["json_parse_errors"]:
        reasons.append("json_parse_errors")
    row_summary = report["row_summary"]
    if row_summary["duplicate_pairs"]:
        reasons.append("duplicate_sample_variant_pairs")
    if row_summary["field_errors"]:
        reasons.append("required_fields_missing")
    if row_summary["metric_errors"]:
        reasons.append("metric_schema_errors")
    if row_summary["index_errors"]:
        reasons.append("sample_index_errors")
    incomplete = row_summary["incomplete_samples"]
    if len(incomplete) > 1:
        reasons.append("more_than_one_incomplete_sample_pair")
    elif len(incomplete) == 1 and incomplete[0]["sample_index"] != row_summary["last_sample_index"]:
        reasons.append("incomplete_pair_not_latest_sample")
    if report["summary_consistency"]["status"] == "fail":
        reasons.append("summary_counts_diverge_from_rows")
    if reasons:
        return "blocked_integrity_failure", reasons
    if incomplete:
        return "pass_running_with_expected_inflight_pair", ["latest_sample_pair_in_progress"]
    total = int(report.get("status_json", {}).get("total_samples") or report["summary_json"].get("total_samples") or 0)
    if total and row_summary["paired_completed_samples_from_rows"] >= total:
        return "pass_complete_jsonl_integrity", []
    return "pass_running_paired_rows_integrity", []


def write_status_md(report: dict[str, Any]) -> None:
    row_summary = report["row_summary"]
    lines = [
        "# Prophet Live Integrity Status",
        "",
        f"- Updated: `{report['created_at_utc']}`",
        f"- Status: `{report['status']}`",
        f"- Rows: `{row_summary['row_count']}`",
        f"- Paired completed from rows: `{row_summary['paired_completed_samples_from_rows']}`",
        f"- Incomplete samples: `{len(row_summary['incomplete_samples'])}`",
        f"- Duplicate pairs: `{len(row_summary['duplicate_pairs'])}`",
        f"- JSON parse errors: `{len(report['json_parse_errors'])}`",
        f"- Summary consistency: `{report['summary_consistency']['status']}`",
        f"- Reasons: `{report['reasons']}`",
    ]
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows, parse_errors = load_rows()
    row_summary = summarize_rows(rows)
    summary = read_json(SUMMARY_PATH, {})
    status = read_json(STATUS_PATH, {})
    report: dict[str, Any] = {
        "artifact_kind": "prophet_live_integrity_report",
        "created_at_utc": utc_now(),
        "rows_path": str(ROWS_PATH),
        "summary_path": str(SUMMARY_PATH),
        "status_path": str(STATUS_PATH),
        "json_parse_errors": parse_errors,
        "row_summary": row_summary,
        "summary_consistency": summary_consistency(row_summary, summary),
        "summary_json": {
            "status": summary.get("status"),
            "total_samples": summary.get("total_samples"),
            "aggregates": summary.get("aggregates", {}),
            "paired_shape": summary.get("paired_shape", {}),
        },
        "status_json": {
            "status": status.get("status"),
            "pid": status.get("pid"),
            "cuda_visible_devices": status.get("cuda_visible_devices"),
            "current_sample_index": status.get("current_sample_index"),
            "completed_sample_indices": status.get("completed_sample_indices"),
            "total_samples": status.get("total_samples"),
            "updated_at_utc": status.get("updated_at_utc"),
        },
    }
    report["status"], report["reasons"] = classify(report)
    write_json(REPORT_PATH, report)
    write_status_md(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
