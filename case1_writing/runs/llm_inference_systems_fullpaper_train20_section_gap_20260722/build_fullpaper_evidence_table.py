#!/usr/bin/env python3
"""Build the 20-paper section evidence table for full-paper DIRS simulation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_TRACE = Path(
    "/tf/notebooks/yunbo/DIRS/case1_writing/runs/"
    "llm_inference_systems_abstract_train28_holdout_echo_20260720_clean_longgoal/"
    "training_trace.json"
)
ROOTS = [
    Path("/tf/notebooks/cvpr2026_oral_paper_memory_141"),
    Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h"),
    Path("/tf/notebooks/icml2026_oral_paper_memory_fresh_24h"),
]


def compact(value: object, max_items: int = 8) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [compact(item, max_items=max_items) for item in value[:max_items]]
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key, val in value.items():
            if key == "nodes" and isinstance(val, list):
                out[key] = [
                    {
                        "id": item.get("id"),
                        "kind": item.get("kind"),
                        "label": item.get("label") or item.get("name"),
                        "props": compact(item.get("props"), max_items=4),
                    }
                    if isinstance(item, dict)
                    else compact(item, max_items=4)
                    for item in val[:max_items]
                ]
            elif key == "events" and isinstance(val, list):
                out[key] = [
                    {
                        "id": item.get("id"),
                        "kind": item.get("kind"),
                        "claim": item.get("claim")
                        or item.get("delta")
                        or item.get("aggregator")
                        or item.get("outcome"),
                        "participants": item.get("participants"),
                    }
                    if isinstance(item, dict)
                    else compact(item, max_items=4)
                    for item in val[:max_items]
                ]
            elif key == "edges" and isinstance(val, list):
                out[key] = [
                    {
                        "source": item.get("source"),
                        "target": item.get("target"),
                        "relation": item.get("relation"),
                    }
                    if isinstance(item, dict)
                    else compact(item, max_items=4)
                    for item in val[:max_items]
                ]
            else:
                out[key] = compact(val, max_items=max_items)
        return out
    return str(value)


def find_artifacts(chip_id: str, meta: dict, cov: dict) -> dict[str, list[str]]:
    candidates: list[str] = []
    for key, value in meta.items():
        if isinstance(value, str) and ("local" in key or key.endswith("_pdf") or key.endswith("_text")):
            candidates.append(value)
    local_artifacts = cov.get("local_artifacts")
    if isinstance(local_artifacts, list):
        candidates.extend(item for item in local_artifacts if isinstance(item, str))

    short = "_".join(chip_id.split("_")[:2])
    for root in ROOTS:
        for subdir in ["text", "pdf", "pdfs", "status", "supplemental"]:
            directory = root / subdir
            if directory.exists():
                candidates.extend(str(path) for path in directory.glob(f"{short}*") if path.is_file())

    existing: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        path = Path(item)
        if path.exists() and str(path) not in seen:
            seen.add(str(path))
            existing.append(str(path))

    return {
        "text": [item for item in existing if Path(item).suffix.lower() in {".txt", ".md"}],
        "pdf": [item for item in existing if Path(item).suffix.lower() == ".pdf"],
        "status_or_json": [
            item
            for item in existing
            if Path(item).suffix.lower() == ".json" or "status" in item
        ],
    }


def result_summary(result: object) -> object:
    return compact(result, max_items=10)


def build(source_trace: Path, output: Path, count: int) -> dict:
    trace = json.loads(source_trace.read_text())
    chip_paths = [Path(item["chip_path"]) for item in trace[:count]]
    papers = []

    for index, chip_path in enumerate(chip_paths, 1):
        chip = json.loads(chip_path.read_text())
        meta = chip.get("chip_metadata", {})
        cov = chip.get("source_coverage", {})
        papers.append(
            {
                "index": index,
                "chip_id": chip.get("chip_id"),
                "title": meta.get("title"),
                "venue": meta.get("venue") or meta.get("source") or "",
                "chip_path": str(chip_path),
                "source_coverage": {
                    "abstract": cov.get("abstract"),
                    "introduction": cov.get("introduction"),
                    "related_work": cov.get("related_work"),
                    "method": cov.get("method"),
                    "experiments": cov.get("experiments"),
                    "results": cov.get("results"),
                    "limitations": cov.get("limitations"),
                    "appendix_or_supplement": cov.get("appendix_or_supplement")
                    if "appendix_or_supplement" in cov
                    else cov.get("appendix") or cov.get("supplement"),
                    "paper_pdf_evidence": cov.get("paper_pdf_read")
                    or cov.get("cvf_pdf_read")
                    or cov.get("paper_pdf")
                    or cov.get("openreview_pdf"),
                    "code_repo_inspected": cov.get("code_repo_inspected")
                    or cov.get("code_inspected")
                    or cov.get("github_code"),
                },
                "local_artifacts": find_artifacts(chip.get("chip_id"), meta, cov),
                "gap_evidence": compact(chip.get("problem_gap"), max_items=8),
                "method_evidence": compact(chip.get("method_mechanism"), max_items=10),
                "evaluation_evidence": compact(chip.get("evaluation_validation"), max_items=10),
                "experimental_setting_evidence": compact(chip.get("experimental_setting"), max_items=10),
                "result_and_limitation_evidence": result_summary(chip.get("result_outcome")),
                "footprint": compact(chip.get("footprint"), max_items=10),
            }
        )

    coverage_keys = [
        "introduction",
        "related_work",
        "method",
        "experiments",
        "results",
        "limitations",
        "appendix_or_supplement",
    ]
    report = {
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_run": str(source_trace.parent),
        "selection_rule": f"first {count} training chip paths from completed abstract longgoal training_trace.json; private holdout not read",
        "domain": "LLM Inference / Systems / Token Efficiency",
        "paper_count": len(papers),
        "coverage_counts": {
            key: sum(1 for paper in papers if paper["source_coverage"].get(key) is True)
            for key in coverage_keys
        },
        "local_artifact_counts": {
            "papers_with_local_text": sum(1 for paper in papers if paper["local_artifacts"]["text"]),
            "papers_with_local_pdf": sum(1 for paper in papers if paper["local_artifacts"]["pdf"]),
        },
        "papers": papers,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-trace", type=Path, default=DEFAULT_SOURCE_TRACE)
    parser.add_argument("--output", type=Path, default=RUN_DIR / "paper_section_evidence_table.json")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    report = build(args.source_trace, args.output, args.count)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paper_count": report["paper_count"],
                "coverage_counts": report["coverage_counts"],
                "local_artifact_counts": report["local_artifact_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
