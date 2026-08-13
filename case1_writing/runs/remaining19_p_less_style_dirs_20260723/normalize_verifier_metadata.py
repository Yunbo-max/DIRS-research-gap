#!/usr/bin/env python3
"""Normalize paper-level verifier metadata for strict DIRS audits.

This does not change any convergence decision. It only fills missing verifier
identity fields so completion audits can distinguish concrete specialized
verifiers from generic JSON blobs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent
PAPER_RUNS = RUN_ROOT / "paper_runs"
REPORT_PATH = RUN_ROOT / "verifier_metadata_normalization_report.json"

ARTIFACT_KIND_BY_PAPER_ID = {
    "CVPR2026_052_seacache_spectral_evolution_cache": "seacache_specialized_verifier",
    "CVPR2026_053_sencache_sensitivity_aware_caching": "sencache_specialized_verifier",
    "ICLR2026_1J63FJYJKg_mrrope_mixed_radix_rope": "mrrope_specialized_verifier",
    "ICLR2026_H6rDX4w6Al_flashvid_vllm_token_merging": "flashvid_specialized_verifier",
    "ICLR2026_o29E01Q6bv_loongrl_long_context_reasoning": "loongrl_specialized_verifier",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    normalized = []
    unchanged = []
    errors = []
    now = utc_now()
    for path in sorted(PAPER_RUNS.glob("*/verifier_result_iter_03.json")):
        try:
            payload = read_json(path)
            paper_id = payload.get("paper_id")
            expected_kind = ARTIFACT_KIND_BY_PAPER_ID.get(paper_id)
            if payload.get("artifact_kind"):
                unchanged.append(
                    {
                        "path": str(path),
                        "paper_id": paper_id,
                        "artifact_kind": payload.get("artifact_kind"),
                        "reason": "already_present",
                    }
                )
                continue
            if not expected_kind:
                errors.append(
                    {
                        "path": str(path),
                        "paper_id": paper_id,
                        "reason": "missing_artifact_kind_and_no_mapping",
                    }
                )
                continue
            payload["artifact_kind"] = expected_kind
            payload.setdefault("metadata_normalization", {})
            payload["metadata_normalization"].update(
                {
                    "normalized_at_utc": now,
                    "normalizer": str(Path(__file__).resolve()),
                    "changed_fields": ["artifact_kind"],
                    "convergence_decision_unchanged": True,
                }
            )
            write_json(path, payload)
            normalized.append(
                {
                    "path": str(path),
                    "paper_id": paper_id,
                    "artifact_kind": expected_kind,
                }
            )
        except Exception as exc:  # pragma: no cover - audit utility
            errors.append({"path": str(path), "error": repr(exc)})

    report = {
        "artifact_kind": "verifier_metadata_normalization_report",
        "created_at_utc": now,
        "normalized_count": len(normalized),
        "unchanged_count": len(unchanged),
        "error_count": len(errors),
        "normalized": normalized,
        "unchanged": unchanged,
        "errors": errors,
    }
    write_json(REPORT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
