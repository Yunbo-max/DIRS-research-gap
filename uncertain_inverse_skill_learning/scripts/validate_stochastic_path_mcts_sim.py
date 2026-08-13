#!/usr/bin/env python3
"""Validate saved stochastic connected-path MCTS simulation artifacts."""

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
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    graph = read_json(args.graph)
    results = read_json(args.results)
    path_nodes = {
        item["path_id"]: tuple(item["nodes"]) for item in graph["observed_paths"]
    }
    compatible: dict[tuple[str, ...], list[str]] = {}
    for path_id, nodes in path_nodes.items():
        for length in range(1, len(nodes) + 1):
            compatible.setdefault(nodes[:length], []).append(path_id)

    def expected_frontier(prefix: tuple[str, ...]) -> set[str]:
        return {
            path_nodes[path_id][len(prefix)]
            for path_id in compatible[prefix]
            if len(path_nodes[path_id]) > len(prefix)
        }

    violations = []
    trace_count = 0
    decision_count = 0
    backpropagation_count = 0
    for scenario in results["scenarios"]:
        audit = scenario["audit_example"]
        traces = audit["mcts_trace_at_max_budget"]
        trace_count += len(traces)
        for simulation in traces:
            decisions = simulation["decisions"]
            decision_count += len(decisions)
            backpropagation_count += len(simulation["backpropagation"])
            current_prefix = (simulation["terminal_nodes"][0],)
            for decision_index, decision in enumerate(decisions):
                logged_prefix = tuple(decision["state_prefix"])
                if logged_prefix != current_prefix:
                    violations.append(
                        {
                            "scenario_id": scenario["scenario_id"],
                            "simulation": simulation["simulation"],
                            "decision": decision_index,
                            "kind": "prefix_discontinuity",
                        }
                    )
                logged_frontier = {
                    item["action"] for item in decision["valid_frontier"]
                }
                actual_frontier = expected_frontier(logged_prefix)
                if logged_frontier != actual_frontier:
                    violations.append(
                        {
                            "scenario_id": scenario["scenario_id"],
                            "simulation": simulation["simulation"],
                            "decision": decision_index,
                            "kind": "incorrect_frontier",
                            "logged": sorted(logged_frontier),
                            "expected": sorted(actual_frontier),
                        }
                    )
                selected = decision["selected_next_node"]
                if selected not in actual_frontier:
                    violations.append(
                        {
                            "scenario_id": scenario["scenario_id"],
                            "simulation": simulation["simulation"],
                            "decision": decision_index,
                            "kind": "illegal_action",
                            "selected": selected,
                        }
                    )
                current_prefix = current_prefix + (selected,)

            path_id = simulation["terminal_path_id"]
            terminal_nodes = tuple(simulation["terminal_nodes"])
            if path_id not in path_nodes or terminal_nodes != path_nodes[path_id]:
                violations.append(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "simulation": simulation["simulation"],
                        "kind": "invalid_terminal",
                        "path_id": path_id,
                    }
                )
            if current_prefix != terminal_nodes:
                violations.append(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "simulation": simulation["simulation"],
                        "kind": "terminal_not_reached_by_logged_decisions",
                    }
                )
            reward = float(simulation["sampled_reward"])
            if not 0.0 <= reward <= 1.0:
                violations.append(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "simulation": simulation["simulation"],
                        "kind": "reward_out_of_range",
                        "reward": reward,
                    }
                )
            if len(simulation["backpropagation"]) != len(decisions):
                violations.append(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "simulation": simulation["simulation"],
                        "kind": "backpropagation_length_mismatch",
                    }
                )

    output = {
        "schema_version": "dirs.stochastic_prefix_path_mcts_validation.v1",
        "graph_id_matches": results["graph_id"] == graph["graph_hash_sha256"],
        "scenario_count": len(results["scenarios"]),
        "audited_simulation_count": trace_count,
        "audited_decision_count": decision_count,
        "audited_backpropagation_count": backpropagation_count,
        "violation_count": len(violations),
        "passed": (
            results["graph_id"] == graph["graph_hash_sha256"]
            and len(violations) == 0
        ),
        "violations": violations,
    }
    write_json(args.output, output)
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
