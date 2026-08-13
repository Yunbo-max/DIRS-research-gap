#!/usr/bin/env python3
"""Learn a connected directed path-DAG baseline from extracted DIRS traces.

This script does not invent nodes, reorder nodes, or sample arbitrary subsets.
Every training example must provide a connected directed path through
``selected_nodes`` and ``selected_edges``. The learner aggregates those paths
into one union DAG, estimates node/edge/path support, and enumerates legal
root-to-terminal paths.

The input traces are proxy extractions, not historical expert cognition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def edge_tuple(edge: str) -> tuple[str, str]:
    parts = edge.split("->")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Invalid directed edge: {edge!r}")
    return parts[0], parts[1]


def validate_trace(row: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
    nodes = row.get("selected_nodes")
    edge_strings = row.get("selected_edges")
    case_id = row.get("chip_id", "<unknown>")
    if not isinstance(nodes, list) or len(nodes) < 2:
        raise ValueError(f"{case_id}: selected_nodes must contain a path")
    if len(nodes) != len(set(nodes)):
        raise ValueError(f"{case_id}: repeated node in directed path")
    if not isinstance(edge_strings, list):
        raise ValueError(f"{case_id}: selected_edges must be a list")
    edges = [edge_tuple(edge) for edge in edge_strings]
    expected = list(zip(nodes, nodes[1:]))
    if edges != expected:
        raise ValueError(
            f"{case_id}: edges are not the consecutive directed flow of selected_nodes"
        )
    return nodes, edges


def topological_order(
    nodes: set[str], adjacency: dict[str, set[str]]
) -> list[str]:
    indegree = {node: 0 for node in nodes}
    for source in nodes:
        for target in adjacency.get(source, set()):
            indegree[target] += 1
    frontier = sorted(node for node, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while frontier:
        node = frontier.pop(0)
        order.append(node)
        for target in sorted(adjacency.get(node, set())):
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(target)
                frontier.sort()
    if len(order) != len(nodes):
        raise ValueError("Union of training paths contains a directed cycle")
    return order


def enumerate_paths(
    sources: list[str],
    sinks: set[str],
    adjacency: dict[str, set[str]],
) -> list[list[str]]:
    paths: list[list[str]] = []

    def visit(node: str, prefix: list[str]) -> None:
        next_prefix = prefix + [node]
        if node in sinks:
            paths.append(next_prefix)
            return
        for target in sorted(adjacency.get(node, set())):
            visit(target, next_prefix)

    for source in sources:
        visit(source, [])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    rows = read_json(args.trace)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Trace input must be a non-empty JSON list")

    node_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    path_counts: Counter[tuple[str, ...]] = Counter()
    path_cases: dict[tuple[str, ...], list[str]] = defaultdict(list)
    all_nodes: set[str] = set()
    adjacency: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        nodes, edges = validate_trace(row)
        node_counts.update(nodes)
        edge_counts.update(edges)
        signature = tuple(nodes)
        path_counts[signature] += 1
        path_cases[signature].append(row["chip_id"])
        all_nodes.update(nodes)
        for source, target in edges:
            adjacency[source].add(target)

    order = topological_order(all_nodes, adjacency)
    indegree = {node: 0 for node in all_nodes}
    for source, targets in adjacency.items():
        for target in targets:
            indegree[target] += 1
    sources = sorted(node for node in all_nodes if indegree[node] == 0)
    sinks = sorted(node for node in all_nodes if not adjacency.get(node))
    all_paths = enumerate_paths(sources, set(sinks), adjacency)

    trace_count = len(rows)
    outgoing_total: Counter[str] = Counter()
    outgoing_options: dict[str, set[str]] = defaultdict(set)
    for (source, target), count in edge_counts.items():
        outgoing_total[source] += count
        outgoing_options[source].add(target)

    edge_records = []
    for (source, target), count in sorted(edge_counts.items()):
        option_count = len(outgoing_options[source])
        posterior_mean = (count + args.alpha) / (
            outgoing_total[source] + args.alpha * option_count
        )
        edge_records.append(
            {
                "source": source,
                "target": target,
                "support_count": count,
                "support_rate": round(count / trace_count, 6),
                "outgoing_posterior_mean": round(posterior_mean, 6),
            }
        )

    observed_paths = []
    for signature, count in sorted(
        path_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        observed_paths.append(
            {
                "path_id": f"path_{len(observed_paths) + 1:02d}",
                "nodes": list(signature),
                "edges": [
                    {"source": source, "target": target}
                    for source, target in zip(signature, signature[1:])
                ],
                "support_count": count,
                "empirical_probability": round(count / trace_count, 6),
                "training_cases": sorted(path_cases[signature]),
            }
        )

    observed_signatures = set(path_counts)
    legal_paths = [
        {
            "nodes": path,
            "observed_in_training": tuple(path) in observed_signatures,
            "support_count": path_counts.get(tuple(path), 0),
        }
        for path in all_paths
    ]

    graph_signature = {
        "nodes": sorted(all_nodes),
        "edges": [
            f"{source}->{target}" for source, target in sorted(edge_counts)
        ],
    }
    graph_hash = hashlib.sha256(
        json.dumps(graph_signature, sort_keys=True).encode("utf-8")
    ).hexdigest()

    output = {
        "schema_version": "dirs.learned_connected_path_dag.v1",
        "status": "learned_count_baseline_from_proxy_training_paths",
        "source_trace": str(args.trace),
        "training_trace_count": trace_count,
        "learning_rule": {
            "node_support": "empirical inclusion count",
            "edge_support": "empirical directed transition count",
            "outgoing_edge_posterior": f"Dirichlet-smoothed mean with alpha={args.alpha}",
            "path_support": "empirical count of complete connected directed paths",
            "arbitrary_node_subset_sampling": False,
        },
        "graph_hash_sha256": graph_hash,
        "sources": sources,
        "sinks": sinks,
        "topological_order": order,
        "nodes": [
            {
                "id": node,
                "support_count": node_counts[node],
                "support_rate": round(node_counts[node] / trace_count, 6),
            }
            for node in order
        ],
        "edges": edge_records,
        "observed_paths": observed_paths,
        "all_legal_source_to_sink_paths": legal_paths,
        "checks": {
            "all_input_traces_are_connected_directed_paths": True,
            "union_graph_is_acyclic": True,
            "union_graph_source_count": len(sources),
            "union_graph_sink_count": len(sinks),
            "observed_unique_path_count": len(observed_paths),
            "legal_union_path_count": len(legal_paths),
        },
        "claim_boundary": [
            "The learned graph reflects extracted proxy traces, not verified historical author processes.",
            "Unobserved union paths are structurally legal hypotheses, not learned successful paths.",
            "No MCTS was run by this learner.",
            "Context-conditioned path selection requires a separately trained or validated policy.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
