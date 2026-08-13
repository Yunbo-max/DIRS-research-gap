#!/usr/bin/env python3
"""Validate connected-path policy-comparison traces and allocations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
    paths = {
        item["path_id"]: tuple(item["nodes"]) for item in graph["observed_paths"]
    }
    path_ids = set(paths)
    compatible: dict[tuple[str, ...], list[str]] = {}
    graph_path_violations = []
    for record in graph["observed_paths"]:
        path_id = record["path_id"]
        nodes = tuple(record["nodes"])
        edge_set = {
            (edge["source"], edge["target"]) for edge in record["edges"]
        }
        for source, target in zip(nodes, nodes[1:]):
            if (source, target) not in edge_set:
                graph_path_violations.append(
                    {
                        "path_id": path_id,
                        "missing_consecutive_edge": [source, target],
                    }
                )
        for length in range(1, len(nodes) + 1):
            compatible.setdefault(nodes[:length], []).append(path_id)

    def expected_frontier(prefix: tuple[str, ...]) -> set[str]:
        return {
            paths[path_id][len(prefix)]
            for path_id in compatible[prefix]
            if len(paths[path_id]) > len(prefix)
        }

    violations = list(graph_path_violations)
    sample_count = 0
    mcts_simulation_count = 0
    mcts_decision_count = 0
    max_budget = max(results["config"]["budgets"])
    flat_methods = (
        "uniform_allocation_q",
        "sequential_halving",
        "successive_rejects",
        "ucb1_q",
        "top_two_thompson",
    )

    for scenario in results["scenarios"]:
        audit = scenario["audit_example"]
        if audit["oracle_path_id"] not in path_ids:
            violations.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "kind": "invalid_oracle_path",
                }
            )
        for method in flat_methods:
            method_audit = audit["methods_at_max_budget"][method]
            if method_audit["selected"] not in path_ids:
                violations.append(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "method": method,
                        "kind": "invalid_selected_path",
                    }
                )
            logged_counts = Counter(method_audit["counts"])
            if set(logged_counts) != path_ids or sum(logged_counts.values()) != max_budget:
                violations.append(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "method": method,
                        "kind": "invalid_allocation_counts",
                        "total": sum(logged_counts.values()),
                    }
                )
            raw_trace = method_audit["trace"]
            samples = (
                raw_trace[0]["samples"]
                if method in ("sequential_halving", "successive_rejects")
                else raw_trace
            )
            trace_counts = Counter()
            for sample in samples:
                sample_count += 1
                path_id = sample["path_id"]
                trace_counts[path_id] += 1
                if path_id not in path_ids:
                    violations.append(
                        {
                            "scenario_id": scenario["scenario_id"],
                            "method": method,
                            "kind": "sampled_invalid_path",
                        }
                    )
                reward = float(sample["reward"])
                if not 0.0 <= reward <= 1.0:
                    violations.append(
                        {
                            "scenario_id": scenario["scenario_id"],
                            "method": method,
                            "kind": "reward_out_of_range",
                        }
                    )
            if trace_counts != logged_counts:
                violations.append(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "method": method,
                        "kind": "trace_count_mismatch",
                    }
                )

        for trace_name in ("mcts_empirical_trace", "mcts_path_uniform_trace"):
            simulations = audit[trace_name]
            if len(simulations) != max_budget:
                violations.append(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "trace": trace_name,
                        "kind": "wrong_mcts_simulation_count",
                    }
                )
            for simulation in simulations:
                mcts_simulation_count += 1
                decisions = simulation["decisions"]
                current_prefix = (simulation["terminal_nodes"][0],)
                for decision in decisions:
                    mcts_decision_count += 1
                    prefix = tuple(decision["state_prefix"])
                    if prefix != current_prefix:
                        violations.append(
                            {
                                "scenario_id": scenario["scenario_id"],
                                "trace": trace_name,
                                "kind": "prefix_discontinuity",
                            }
                        )
                    logged_frontier = {
                        item["action"] for item in decision["valid_frontier"]
                    }
                    actual_frontier = expected_frontier(prefix)
                    selected = decision["selected_next_node"]
                    if logged_frontier != actual_frontier:
                        violations.append(
                            {
                                "scenario_id": scenario["scenario_id"],
                                "trace": trace_name,
                                "kind": "frontier_mismatch",
                            }
                        )
                    if selected not in actual_frontier:
                        violations.append(
                            {
                                "scenario_id": scenario["scenario_id"],
                                "trace": trace_name,
                                "kind": "illegal_mcts_action",
                            }
                        )
                    current_prefix = current_prefix + (selected,)
                terminal_path = simulation["terminal_path_id"]
                if (
                    terminal_path not in paths
                    or tuple(simulation["terminal_nodes"]) != paths[terminal_path]
                    or current_prefix != paths[terminal_path]
                ):
                    violations.append(
                        {
                            "scenario_id": scenario["scenario_id"],
                            "trace": trace_name,
                            "kind": "invalid_mcts_terminal",
                        }
                    )

    output = {
        "schema_version": "dirs.path_policy_comparison_validation.v1",
        "graph_id_matches": results["graph_id"] == graph["graph_hash_sha256"],
        "scenario_count": len(results["scenarios"]),
        "audited_flat_policy_samples": sample_count,
        "audited_mcts_simulations": mcts_simulation_count,
        "audited_mcts_decisions": mcts_decision_count,
        "violation_count": len(violations),
        "passed": (
            results["graph_id"] == graph["graph_hash_sha256"]
            and not violations
        ),
        "violations": violations,
    }
    write_json(args.output, output)
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
