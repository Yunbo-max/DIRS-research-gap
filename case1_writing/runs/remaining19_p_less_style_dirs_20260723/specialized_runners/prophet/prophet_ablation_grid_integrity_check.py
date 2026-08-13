#!/usr/bin/env python3
"""Integrity checks for Prophet full-GSM8K ablation-grid artifacts."""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNNER_DIR = Path(__file__).resolve().parent
CAMPAIGN_DIR = RUNNER_DIR / "ablation_grid_full_gsm8k"
MANIFEST_PATH = CAMPAIGN_DIR / "ablation_grid_campaign.json"
REPORT_PATH = CAMPAIGN_DIR / "ablation_grid_integrity_report.json"
STATUS_MD = CAMPAIGN_DIR / "ABLATION_GRID_INTEGRITY_STATUS.md"

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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def process_alive(pid: Any) -> bool:
    if pid in (None, "", 0):
        return False
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def expected_variants(config: dict[str, Any]) -> set[str]:
    return {item.strip() for item in str(config.get("variants", "")).split(",") if item.strip()}


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not path.exists():
        return rows, []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append({"line_number": line_number, "error": str(exc), "line_prefix": line[:120]})
    return rows, errors


def summarize_rows(rows: list[dict[str, Any]], variants: set[str]) -> dict[str, Any]:
    variant_counts = Counter(row.get("variant") for row in rows)
    sample_variants: dict[int, set[str]] = defaultdict(set)
    duplicate_counts = Counter()
    field_errors: list[dict[str, Any]] = []
    metric_errors: list[dict[str, Any]] = []
    index_errors: list[dict[str, Any]] = []

    for row_number, row in enumerate(rows, start=1):
        sample_index = row.get("sample_index")
        variant = row.get("variant")
        duplicate_counts[(sample_index, variant)] += 1
        if isinstance(sample_index, int):
            sample_variants[sample_index].add(str(variant))
            if sample_index < 0:
                index_errors.append({"row_number": row_number, "sample_index": sample_index, "error": "negative_sample_index"})
        else:
            index_errors.append({"row_number": row_number, "sample_index": sample_index, "error": "sample_index_not_int"})
        missing = sorted(REQUIRED_ROW_FIELDS - set(row))
        if missing:
            field_errors.append({"row_number": row_number, "missing_fields": missing})
        if variant not in variants:
            metric_errors.append({"row_number": row_number, "variant": variant, "error": "unexpected_variant"})
        for key in ("strict_exact_match", "flexible_exact_match"):
            if key in row and not isinstance(row.get(key), bool):
                metric_errors.append({"row_number": row_number, "field": key, "error": "not_bool"})
        if "seconds" in row and not isinstance(row.get("seconds"), (int, float)):
            metric_errors.append({"row_number": row_number, "field": "seconds", "error": "not_numeric"})

    duplicates = [
        {"sample_index": sample_index, "variant": variant, "count": count}
        for (sample_index, variant), count in duplicate_counts.items()
        if count > 1
    ]
    complete_samples = sorted(idx for idx, got in sample_variants.items() if got == variants)
    incomplete_samples = [
        {"sample_index": idx, "variants_present": sorted(got)}
        for idx, got in sorted(sample_variants.items())
        if got != variants
    ]
    return {
        "row_count": len(rows),
        "variant_counts": dict(sorted((str(k), v) for k, v in variant_counts.items())),
        "sample_count_observed": len(sample_variants),
        "completed_samples_from_rows": len(complete_samples),
        "first_sample_index": min(sample_variants) if sample_variants else None,
        "last_sample_index": max(sample_variants) if sample_variants else None,
        "duplicate_sample_variant_pairs": duplicates,
        "incomplete_samples": incomplete_samples[:50],
        "field_errors": field_errors[:50],
        "metric_errors": metric_errors[:50],
        "index_errors": index_errors[:50],
    }


def summary_consistency(row_summary: dict[str, Any], summary: dict[str, Any], variants: set[str]) -> dict[str, Any]:
    aggregates = summary.get("aggregates", {})
    deltas = {}
    failures = []
    for variant in sorted(variants):
        row_count = int(row_summary["variant_counts"].get(variant, 0))
        summary_count = int(aggregates.get(variant, {}).get("completed_samples") or 0)
        delta = row_count - summary_count
        deltas[variant] = {"row_count": row_count, "summary_count": summary_count, "delta": delta}
        if abs(delta) > 1:
            failures.append(variant)
    return {
        "status": "pass_or_live_race" if not failures else "fail",
        "failed_variants": failures,
        "deltas": deltas,
        "summary_status": summary.get("status"),
        "summary_total_samples": summary.get("total_samples"),
    }


def classify_config(
    status_json: dict[str, Any],
    summary_json: dict[str, Any],
    row_summary: dict[str, Any],
    parse_errors: list[dict[str, Any]],
    consistency: dict[str, Any],
    variants: set[str],
) -> tuple[str, list[str]]:
    reasons = []
    if parse_errors:
        reasons.append("json_parse_errors")
    if row_summary["duplicate_sample_variant_pairs"]:
        reasons.append("duplicate_sample_variant_pairs")
    if row_summary["field_errors"]:
        reasons.append("required_fields_missing")
    if row_summary["metric_errors"]:
        reasons.append("metric_schema_errors")
    if row_summary["index_errors"]:
        reasons.append("sample_index_errors")
    incomplete = row_summary["incomplete_samples"]
    if len(incomplete) > 1:
        reasons.append("more_than_one_incomplete_sample")
    elif len(incomplete) == 1 and incomplete[0]["sample_index"] != row_summary["last_sample_index"]:
        reasons.append("incomplete_sample_not_latest")
    if consistency["status"] == "fail":
        reasons.append("summary_counts_diverge_from_rows")
    pid_alive = process_alive(status_json.get("pid"))
    raw_status = status_json.get("status")
    total = int(status_json.get("total_samples") or summary_json.get("total_samples") or 0)
    if reasons:
        return "blocked_integrity_failure", reasons
    if total and row_summary["completed_samples_from_rows"] >= total:
        return "pass_complete_jsonl_integrity", []
    if pid_alive and raw_status in {"launched", "starting", "loading_model", "running", "running_or_partial"}:
        if incomplete and len(variants) > 1:
            return "pass_running_with_expected_inflight_sample", ["latest_sample_in_progress"]
        return "pass_running_jsonl_integrity", []
    if row_summary["row_count"]:
        return "blocked_stopped_partial_artifact", ["process_not_alive_before_full_config_completion"]
    return "pending_no_rows_yet", []


def audit_config(config: dict[str, Any]) -> dict[str, Any]:
    out_dir = CAMPAIGN_DIR / config["id"]
    rows_path = out_dir / "per_sample_results.jsonl"
    summary_path = out_dir / "summary.json"
    status_path = out_dir / "status.json"
    variants = expected_variants(config)
    rows, parse_errors = load_rows(rows_path)
    row_summary = summarize_rows(rows, variants)
    summary_json = read_json(summary_path, {})
    status_json = read_json(status_path, {})
    consistency = summary_consistency(row_summary, summary_json, variants)
    status, reasons = classify_config(status_json, summary_json, row_summary, parse_errors, consistency, variants)
    return {
        "id": config["id"],
        "paper_role": config.get("paper_role"),
        "status": status,
        "reasons": reasons,
        "expected_variants": sorted(variants),
        "out_dir": str(out_dir),
        "rows_path": str(rows_path),
        "summary_path": str(summary_path),
        "status_path": str(status_path),
        "pid": status_json.get("pid"),
        "pid_alive": process_alive(status_json.get("pid")),
        "cuda_visible_devices": status_json.get("cuda_visible_devices") or status_json.get("gpu"),
        "completed_samples_status": status_json.get("completed_sample_indices"),
        "total_samples_status": status_json.get("total_samples"),
        "row_summary": row_summary,
        "json_parse_errors": parse_errors[:50],
        "summary_consistency": consistency,
        "updated_at_utc": status_json.get("updated_at_utc") or summary_json.get("created_at_utc"),
    }


def classify_grid(config_reports: list[dict[str, Any]]) -> tuple[str, list[str]]:
    blocked = [item for item in config_reports if str(item.get("status", "")).startswith("blocked")]
    running = [item for item in config_reports if str(item.get("status", "")).startswith("pass_running")]
    pending = [item for item in config_reports if item.get("status") == "pending_no_rows_yet"]
    complete = [item for item in config_reports if item.get("status") == "pass_complete_jsonl_integrity"]
    reasons = [item["id"] for item in blocked]
    if blocked:
        return "blocked_ablation_grid_integrity_failure", reasons
    if running:
        return "pass_running_ablation_grid_integrity", []
    if pending and complete:
        return "pass_partial_complete_with_pending_configs", []
    if pending:
        return "pending_no_ablation_rows_yet", []
    return "pass_complete_ablation_grid_integrity", []


def manifest_blocked_configs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    blocked = manifest.get("blocked_configs", [])
    return blocked if isinstance(blocked, list) else []


def write_status_md(report: dict[str, Any]) -> None:
    lines = [
        "# Prophet Ablation Grid Integrity",
        "",
        f"- Updated: `{report['created_at_utc']}`",
        f"- Status: `{report['status']}`",
        f"- Configs checked: `{len(report['configs'])}`",
        f"- Runnable config integrity blockers: `{len(report['blocked_config_ids'])}`",
        f"- Manifest source-parity blockers: `{len(report['manifest_blocked_config_ids'])}`",
        f"- Running configs: `{len(report['running_config_ids'])}`",
        f"- Complete configs: `{len(report['complete_config_ids'])}`",
        "",
        "## Manifest Source-Parity Blockers",
        "",
    ]
    for item in report["manifest_blocked_configs"]:
        lines.append(f"- `{item.get('id')}` status=`{item.get('status')}` role=`{item.get('paper_role')}`")
    lines += [
        "",
        "## Configs",
        "",
    ]
    for item in report["configs"]:
        row_summary = item["row_summary"]
        lines.append(
            f"- `{item['id']}` status=`{item['status']}` gpu=`{item.get('cuda_visible_devices')}` "
            f"pid=`{item.get('pid')}` rows=`{row_summary['row_count']}` "
            f"complete=`{row_summary['completed_samples_from_rows']}/{item.get('total_samples_status')}` "
            f"reasons=`{item.get('reasons')}`"
        )
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    manifest = read_json(MANIFEST_PATH, {})
    configs = manifest.get("runnable_configs", [])
    config_reports = [audit_config(config) for config in configs]
    status, reasons = classify_grid(config_reports)
    manifest_blockers = manifest_blocked_configs(manifest)
    report = {
        "artifact_kind": "prophet_ablation_grid_integrity_report",
        "created_at_utc": utc_now(),
        "manifest_path": str(MANIFEST_PATH),
        "campaign_dir": str(CAMPAIGN_DIR),
        "status": status,
        "reasons": reasons,
        "configs": config_reports,
        "blocked_config_ids": [item["id"] for item in config_reports if str(item.get("status", "")).startswith("blocked")],
        "manifest_blocked_configs": manifest_blockers,
        "manifest_blocked_config_ids": [item.get("id") for item in manifest_blockers],
        "running_config_ids": [item["id"] for item in config_reports if str(item.get("status", "")).startswith("pass_running")],
        "complete_config_ids": [item["id"] for item in config_reports if item.get("status") == "pass_complete_jsonl_integrity"],
        "pending_config_ids": [item["id"] for item in config_reports if item.get("status") == "pending_no_rows_yet"],
    }
    write_json(REPORT_PATH, report)
    write_status_md(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
