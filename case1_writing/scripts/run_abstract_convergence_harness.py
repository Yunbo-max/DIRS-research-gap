#!/usr/bin/env python3
"""Run a DIRS Case 1 abstract-training convergence harness.

This is a lightweight, reproducible harness for the training substrate:
- Loop 1 keeps the extended domain DAG learned from training examples.
- Loop 2 simulates MCTS-style connected sub-DAG selection for each example.
- The evaluator compares simulated selected paths with the training paths.

It does not call an external LLM and does not read the held-out original.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


NODE_ORDER = [
    "R1_abstract_as_argument",
    "G1_problem_gap",
    "C1_domain_context",
    "O1_named_method_or_object",
    "M1_architecture_or_mechanism",
    "M2_efficiency_or_theory_detail",
    "E1_evaluation_setup",
    "E2_result_outcome",
    "E3_quantitative_anchor",
    "I1_interpretation_or_tradeoff",
    "S1_bounded_takeaway",
    "P1_length_and_placement_prior",
]


REQUIRED_NODES = {
    "R1_abstract_as_argument",
    "G1_problem_gap",
    "O1_named_method_or_object",
    "M1_architecture_or_mechanism",
    "E1_evaluation_setup",
    "E2_result_outcome",
    "I1_interpretation_or_tradeoff",
    "S1_bounded_takeaway",
    "P1_length_and_placement_prior",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def path_edges(nodes: list[str]) -> list[str]:
    return [f"{a}->{b}" for a, b in zip(nodes, nodes[1:])]


def ordered(nodes: set[str]) -> list[str]:
    return [node for node in NODE_ORDER if node in nodes]


def score_path(
    candidate: list[str],
    target_nodes: set[str],
    target_edges: set[str],
    node_support: dict[str, float],
    edge_support: dict[str, float],
) -> float:
    candidate_set = set(candidate)
    candidate_edges = set(path_edges(candidate))
    node_recall = len(candidate_set & target_nodes) / max(1, len(target_nodes))
    node_precision = len(candidate_set & target_nodes) / max(1, len(candidate_set))
    edge_recall = len(candidate_edges & target_edges) / max(1, len(target_edges))
    support_bonus = statistics.mean([node_support.get(n, 0.0) for n in candidate]) if candidate else 0.0
    edge_bonus = statistics.mean([edge_support.get(e, 0.0) for e in candidate_edges]) if candidate_edges else 0.0
    missing_required = len(REQUIRED_NODES - candidate_set)
    extra_penalty = max(0, len(candidate_set - target_nodes) - 1) * 0.025
    return (
        0.36 * node_recall
        + 0.24 * node_precision
        + 0.18 * edge_recall
        + 0.13 * support_bonus
        + 0.09 * edge_bonus
        - 0.05 * missing_required
        - extra_penalty
    )


def rollout_path(
    target_nodes: set[str],
    node_support: dict[str, float],
    temperature: float,
    rng: random.Random,
) -> list[str]:
    selected = set(REQUIRED_NODES)
    for node in NODE_ORDER:
        if node in REQUIRED_NODES:
            continue
        support = node_support.get(node, 0.0)
        target_hint = 0.45 if node in target_nodes else 0.0
        logit = (support + target_hint - 0.45) / max(0.05, temperature)
        prob = 1.0 / (1.0 + math.exp(-logit))
        if rng.random() < prob:
            selected.add(node)
    return ordered(selected)


def choose_path(
    target_nodes: set[str],
    target_edges: set[str],
    node_support: dict[str, float],
    edge_support: dict[str, float],
    rollouts: int,
    loop_idx: int,
    rng: random.Random,
) -> tuple[list[str], float]:
    best_path: list[str] = []
    best_score = -1e9
    temperature = max(0.12, 0.85 * (0.96**loop_idx))
    for _ in range(rollouts):
        path = rollout_path(target_nodes, node_support, temperature, rng)
        score = score_path(path, target_nodes, target_edges, node_support, edge_support)
        if score > best_score:
            best_score = score
            best_path = path
    return best_path, round(best_score, 6)


def graph_signature(paths: list[list[str]]) -> str:
    unique_edges = sorted({edge for path in paths for edge in path_edges(path)})
    unique_nodes = sorted({node for path in paths for node in path})
    return "|".join(unique_nodes) + "::" + "|".join(unique_edges)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-loops", type=int, default=100)
    parser.add_argument("--min-loops", type=int, default=1)
    parser.add_argument("--mcts-rollouts", type=int, default=500)
    parser.add_argument("--stable-window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    trace = read_json(run_dir / "training_trace.json")
    node_support_rows = read_json(run_dir / "node_support_scores.json")
    edge_support_rows = read_json(run_dir / "edge_support_scores.json")
    style_profile = read_json(run_dir / "style_profile.json")
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    domain_name = manifest.get("domain") or style_profile.get("domain") or "unknown"
    holdout_chip_id = manifest.get("holdout_chip_id") or "unknown"
    min_loops = max(1, min(args.min_loops, args.max_loops))

    node_support = {row["id"]: row["support_rate"] for row in node_support_rows}
    edge_support = {row["id"]: row["support_rate"] for row in edge_support_rows}
    rng = random.Random(args.seed)

    records: list[dict[str, Any]] = []
    stable_count = 0
    previous_signature = ""
    converged_at: int | None = None

    for loop_idx in range(1, args.max_loops + 1):
        chosen_paths: list[list[str]] = []
        scores: list[float] = []
        for paper in trace:
            target_nodes = set(paper["selected_nodes"])
            target_edges = set(paper["selected_edges"])
            path, score = choose_path(
                target_nodes,
                target_edges,
                node_support,
                edge_support,
                args.mcts_rollouts,
                loop_idx,
                rng,
            )
            chosen_paths.append(path)
            scores.append(score)

        signature = graph_signature(chosen_paths)
        stable_count = stable_count + 1 if signature == previous_signature else 0
        previous_signature = signature

        record = {
            "loop": loop_idx,
            "mean_replay_score": round(statistics.mean(scores), 6),
            "min_replay_score": round(min(scores), 6),
            "stable_signature_count": stable_count,
            "unique_selected_nodes": sorted({node for path in chosen_paths for node in path}),
            "unique_selected_edges": sorted({edge for path in chosen_paths for edge in path_edges(path)}),
        }
        records.append(record)

        if loop_idx >= min_loops and stable_count >= args.stable_window:
            converged_at = loop_idx
            break

    final = records[-1]
    report = {
        "run_dir": str(run_dir),
        "domain": domain_name,
        "training_examples": len(trace),
        "heldout": holdout_chip_id,
        "max_loops": args.max_loops,
        "min_loops": min_loops,
        "completed_loops": final["loop"],
        "mcts_rollouts_per_example": args.mcts_rollouts,
        "stable_window": args.stable_window,
        "converged": converged_at is not None,
        "converged_at_loop": converged_at,
        "final_mean_replay_score": final["mean_replay_score"],
        "final_min_replay_score": final["min_replay_score"],
        "final_unique_selected_nodes": final["unique_selected_nodes"],
        "final_unique_selected_edges": final["unique_selected_edges"],
        "target_words_from_training": style_profile["recommended_target_words"],
        "target_band_from_training": style_profile["recommended_band"],
        "blind_rule": "held-out original remains in holdout_private_after_generation.json and is not read by this harness",
    }

    write_json(run_dir / "convergence_report.json", report)
    trace_path = run_dir / "convergence_trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    md = [
        "# DIRS Abstract Convergence Report",
        "",
        "Date: `2026-07-20`",
        "",
        f"Training examples: `{report['training_examples']}`",
        f"Domain: `{report['domain']}`",
        f"Held-out: `{report['heldout']}`",
        f"Completed loops: `{report['completed_loops']}`",
        f"Minimum loops before early stop: `{report['min_loops']}`",
        f"MCTS rollouts per example: `{report['mcts_rollouts_per_example']}`",
        f"Converged: `{report['converged']}`",
        f"Converged at loop: `{report['converged_at_loop']}`",
        f"Final mean replay score: `{report['final_mean_replay_score']}`",
        f"Final min replay score: `{report['final_min_replay_score']}`",
        "",
        "## Final Selected Full DAG Nodes",
        "",
        "```text",
        *report["final_unique_selected_nodes"],
        "```",
        "",
        "## Final Selected Full DAG Edges",
        "",
        "```text",
        *report["final_unique_selected_edges"],
        "```",
        "",
        "## Blind Rule",
        "",
        report["blind_rule"],
        "",
    ]
    (run_dir / "CONVERGENCE_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
