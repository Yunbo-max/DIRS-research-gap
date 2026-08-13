#!/usr/bin/env python3
"""Validate the real-trace score join and prefix-predictiveness audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dataset = read_json(args.dataset)
    results = read_json(args.results)
    graph = read_json(args.graph)
    valid_paths = {
        record["path_id"]: tuple(record["nodes"])
        for record in graph["observed_paths"]
    }
    violations = []
    records = dataset["records"]
    chip_ids = [record["chip_id"] for record in records]
    if len(records) != 19 or len(set(chip_ids)) != 19:
        violations.append({"kind": "record_count_or_uniqueness_failure"})
    for record in records:
        path_id = record["path_id"]
        if (
            path_id not in valid_paths
            or tuple(record["selected_nodes"]) != valid_paths[path_id]
        ):
            violations.append(
                {
                    "kind": "invalid_connected_path_join",
                    "chip_id": record["chip_id"],
                }
            )
        if not 0.0 <= float(record["initial_score"]) <= 1.0:
            violations.append(
                {"kind": "invalid_initial_score", "chip_id": record["chip_id"]}
            )
        if not 0.0 <= float(record["final_score"]) <= 1.0:
            violations.append(
                {"kind": "invalid_final_score", "chip_id": record["chip_id"]}
            )
        expected_gain = record["final_score"] - record["initial_score"]
        if abs(expected_gain - record["repair_gain"]) > 1e-12:
            violations.append(
                {"kind": "repair_gain_mismatch", "chip_id": record["chip_id"]}
            )
    if results["data_summary"]["paper_count"] != len(records):
        violations.append({"kind": "report_dataset_count_mismatch"})
    if results["config"]["holdout_used"]:
        violations.append({"kind": "holdout_leakage_flag"})
    if results["adequacy_decision"]["enable_real_task_mcts_from_this_dataset"]:
        violations.append({"kind": "unsupported_enablement_decision"})
    if results["identifiability"]["same_paper_counterfactual_paths_per_case"] != 0:
        violations.append({"kind": "incorrect_counterfactual_count"})
    output = {
        "schema_version": "dirs.real_trace_prefix_audit_validation.v1",
        "record_count": len(records),
        "unique_chip_count": len(set(chip_ids)),
        "valid_graph_path_count": len(valid_paths),
        "violation_count": len(violations),
        "passed": not violations,
        "violations": violations,
    }
    write_json(args.output, output)
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
