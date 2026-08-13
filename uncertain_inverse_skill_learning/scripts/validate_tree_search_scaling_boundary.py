#!/usr/bin/env python3
"""Validate synthetic tree-search scaling result traces."""

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


def valid_path(path_id: str | None, depth: int) -> bool:
    return (
        isinstance(path_id, str)
        and len(path_id) == depth
        and set(path_id).issubset({"0", "1"})
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = read_json(args.results)
    violations = []
    audited_uniform_samples = 0
    audited_uct_simulations = 0
    audited_uct_decisions = 0
    max_ratio = max(results["config"]["budget_ratios"])

    for cell in results["cells"]:
        cell_id = cell["cell_id"]
        depth = int(cell["depth"])
        budget = int(cell["budget"])
        expected_path_count = 2**depth
        if cell["path_count"] != expected_path_count:
            violations.append(
                {"cell_id": cell_id, "kind": "path_count_depth_mismatch"}
            )
        for method, policy_result in cell["policy_results"].items():
            if policy_result.get("applicable") is False:
                if method != "sequential_halving" or budget >= expected_path_count:
                    violations.append(
                        {"cell_id": cell_id, "method": method, "kind": "bad_skip"}
                    )
            elif policy_result["episodes"] != results["config"]["episodes_per_cell"]:
                violations.append(
                    {
                        "cell_id": cell_id,
                        "method": method,
                        "kind": "wrong_episode_count",
                    }
                )

        audit = cell["audit_example"]
        for item in audit["best_paths"]:
            if not valid_path(item["path_id"], depth):
                violations.append(
                    {"cell_id": cell_id, "kind": "invalid_truth_path"}
                )
        for method in (
            "uniform_complete_path",
            "uct_mean_backup",
            "uct_max_backup",
        ):
            selected = audit[method]["selected"]
            if not valid_path(selected, depth):
                violations.append(
                    {
                        "cell_id": cell_id,
                        "method": method,
                        "kind": "invalid_selected_path",
                    }
                )
            if audit[method]["allocation_count"] != budget:
                violations.append(
                    {
                        "cell_id": cell_id,
                        "method": method,
                        "kind": "wrong_allocation_count",
                    }
                )

        halving = audit["sequential_halving"]
        if halving["applicable"]:
            if (
                not valid_path(halving["selected"], depth)
                or halving["allocation_count"] != budget
            ):
                violations.append(
                    {"cell_id": cell_id, "kind": "invalid_halving_audit"}
                )
        elif halving["selected"] is not None:
            violations.append(
                {"cell_id": cell_id, "kind": "skipped_halving_has_selection"}
            )

        should_save_trace = cell["budget_ratio"] == max_ratio
        uniform_trace = audit["uniform_complete_path"]["trace"]
        if should_save_trace:
            if len(uniform_trace) != budget:
                violations.append(
                    {"cell_id": cell_id, "kind": "wrong_uniform_trace_length"}
                )
            for sample in uniform_trace:
                audited_uniform_samples += 1
                if not valid_path(sample["path_id"], depth):
                    violations.append(
                        {"cell_id": cell_id, "kind": "invalid_uniform_path"}
                    )
                if not 0.0 <= float(sample["reward"]) <= 1.0:
                    violations.append(
                        {"cell_id": cell_id, "kind": "uniform_reward_out_of_range"}
                    )
        elif uniform_trace:
            violations.append(
                {"cell_id": cell_id, "kind": "unexpected_uniform_trace"}
            )

        for method in ("uct_mean_backup", "uct_max_backup"):
            trace = audit[method]["trace"]
            if should_save_trace and len(trace) != budget:
                violations.append(
                    {
                        "cell_id": cell_id,
                        "method": method,
                        "kind": "wrong_uct_trace_length",
                    }
                )
            if not should_save_trace and trace:
                violations.append(
                    {
                        "cell_id": cell_id,
                        "method": method,
                        "kind": "unexpected_uct_trace",
                    }
                )
            for simulation in trace:
                audited_uct_simulations += 1
                prefix: tuple[int, ...] = ()
                for decision in simulation["decisions"]:
                    audited_uct_decisions += 1
                    if (
                        tuple(decision["state_prefix"]) != prefix
                        or decision["valid_actions"] != [0, 1]
                        or decision["selected_action"] not in (0, 1)
                    ):
                        violations.append(
                            {
                                "cell_id": cell_id,
                                "method": method,
                                "kind": "illegal_prefix_decision",
                            }
                        )
                    prefix = prefix + (decision["selected_action"],)
                terminal = simulation["terminal_path_id"]
                if (
                    len(prefix) != depth
                    or terminal != "".join(str(action) for action in prefix)
                    or simulation["terminal_actions"] != list(prefix)
                    or not valid_path(terminal, depth)
                ):
                    violations.append(
                        {
                            "cell_id": cell_id,
                            "method": method,
                            "kind": "invalid_uct_terminal",
                        }
                    )
                if not 0.0 <= float(simulation["reward"]) <= 1.0:
                    violations.append(
                        {
                            "cell_id": cell_id,
                            "method": method,
                            "kind": "uct_reward_out_of_range",
                        }
                    )

    expected_cells = (
        len(results["config"]["depths"])
        * len(results["config"]["budget_ratios"])
        * len(results["config"]["landscapes"])
    )
    if len(results["cells"]) != expected_cells:
        violations.append({"kind": "wrong_cell_count"})
    output = {
        "schema_version": "dirs.tree_search_scaling_validation.v1",
        "expected_cell_count": expected_cells,
        "actual_cell_count": len(results["cells"]),
        "audited_uniform_samples": audited_uniform_samples,
        "audited_uct_simulations": audited_uct_simulations,
        "audited_uct_decisions": audited_uct_decisions,
        "violation_count": len(violations),
        "passed": not violations,
        "violations": violations,
    }
    write_json(args.output, output)
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
