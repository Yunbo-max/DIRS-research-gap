#!/usr/bin/env python3
"""Controlled smoke test for uncertain inverse trace learning.

This test uses previously extracted DIRS training traces as proxy labels. It
creates multiple noisy trace hypotheses, tunes posterior hyperparameters on a
validation split, and evaluates on a held-out split.

It is deliberately labelled smoke_only:
- proposal noise and artifact-fit noise are synthetic;
- extracted traces are proxies, not historical expert behavior;
- no generated paper artifact or human evaluation is involved;
- results test the posterior/aggregation machinery, not method convergence.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def stable_case_seed(base_seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def ordered(nodes: set[str], node_order: list[str]) -> list[str]:
    return [node for node in node_order if node in nodes]


def path_edges(nodes: set[str], node_order: list[str]) -> set[str]:
    seq = ordered(nodes, node_order)
    return {f"{a}->{b}" for a, b in zip(seq, seq[1:])}


def f1(predicted: set[str], target: set[str]) -> float:
    if not predicted and not target:
        return 1.0
    tp = len(predicted & target)
    precision = tp / max(1, len(predicted))
    recall = tp / max(1, len(target))
    return 2 * precision * recall / max(1e-12, precision + recall)


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def softmax(values: list[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in values]
    pivot = max(scaled)
    exps = [math.exp(value - pivot) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def generate_proposals(
    row: dict[str, Any],
    node_order: list[str],
    proposal_count: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(stable_case_seed(base_seed, row["chip_id"]))
    target_nodes = set(row["selected_nodes"])
    proposals: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    # The first proposal represents a conventional single LLM extraction.
    # Later proposals span lower and higher corruption regimes.
    flip_rates = [0.34] + [
        0.10 + 0.35 * index / max(1, proposal_count - 2)
        for index in range(proposal_count - 1)
    ]

    for proposal_index, flip_rate in enumerate(flip_rates):
        candidate: set[str] = set()
        for node in node_order:
            present = node in target_nodes
            if rng.random() < flip_rate:
                present = not present
            if present:
                candidate.add(node)

        signature = tuple(ordered(candidate, node_order))
        if signature in seen:
            # Make a deterministic diversity attempt. Duplicate hypotheses are
            # still possible in a tiny graph.
            pivot = node_order[proposal_index % len(node_order)]
            candidate.symmetric_difference_update({pivot})
            signature = tuple(ordered(candidate, node_order))
        seen.add(signature)

        candidate_edges = path_edges(candidate, node_order)
        target_edges = set(row["selected_edges"])
        node_fit = f1(candidate, target_nodes)
        edge_fit = f1(candidate_edges, target_edges)
        # Synthetic verifier noise models an imperfect artifact-to-trace judge.
        verifier_noise = rng.gauss(0.0, 0.075)
        artifact_fit = max(
            0.0,
            min(1.0, 0.62 * node_fit + 0.38 * edge_fit + verifier_noise),
        )
        proposals.append(
            {
                "proposal_id": f"{row['chip_id']}:p{proposal_index:02d}",
                "nodes": ordered(candidate, node_order),
                "edges": sorted(candidate_edges),
                "artifact_fit": round(artifact_fit, 8),
                "synthetic_flip_rate": round(flip_rate, 8),
            }
        )
    return proposals


def graph_support(
    cases: list[dict[str, Any]],
    proposal_map: dict[str, list[dict[str, Any]]],
    node_order: list[str],
    temperature: float,
    complexity_penalty: float,
    mode: str,
) -> dict[str, float]:
    counts = {node: 0.0 for node in node_order}
    for row in cases:
        proposals = proposal_map[row["chip_id"]]
        if mode == "single":
            weights = [1.0] + [0.0] * (len(proposals) - 1)
        else:
            scores = [
                proposal["artifact_fit"]
                - complexity_penalty * len(proposal["nodes"]) / len(node_order)
                for proposal in proposals
            ]
            weights = softmax(scores, temperature)
        for proposal, weight in zip(proposals, weights):
            selected = set(proposal["nodes"])
            for node in node_order:
                counts[node] += weight * (node in selected)
    # Beta(1,1) smoothing prevents zero-probability graph priors.
    return {
        node: (1.0 + counts[node]) / (2.0 + len(cases))
        for node in node_order
    }


def infer_case(
    row: dict[str, Any],
    proposals: list[dict[str, Any]],
    support: dict[str, float],
    node_order: list[str],
    temperature: float,
    complexity_penalty: float,
    graph_weight: float,
    mode: str,
) -> dict[str, Any]:
    if mode == "single":
        weights = [1.0] + [0.0] * (len(proposals) - 1)
    else:
        scores = []
        for proposal in proposals:
            nodes = proposal["nodes"]
            mean_log_support = statistics.mean(
                math.log(max(1e-8, support[node])) for node in nodes
            )
            scores.append(
                proposal["artifact_fit"]
                + graph_weight * mean_log_support
                - complexity_penalty * len(nodes) / len(node_order)
            )
        weights = softmax(scores, temperature)

    map_index = max(range(len(weights)), key=weights.__getitem__)
    map_nodes = set(proposals[map_index]["nodes"])
    target_nodes = set(row["selected_nodes"])
    inclusion = {
        node: sum(
            weight * (node in set(proposal["nodes"]))
            for proposal, weight in zip(proposals, weights)
        )
        for node in node_order
    }
    brier = statistics.mean(
        (inclusion[node] - float(node in target_nodes)) ** 2
        for node in node_order
    )
    expected_f1 = sum(
        weight * f1(set(proposal["nodes"]), target_nodes)
        for proposal, weight in zip(proposals, weights)
    )
    sorted_indices = sorted(
        range(len(weights)), key=weights.__getitem__, reverse=True
    )
    cumulative = 0.0
    credible_indices: list[int] = []
    for index in sorted_indices:
        credible_indices.append(index)
        cumulative += weights[index]
        if cumulative >= 0.90:
            break
    best_available = max(
        f1(set(proposal["nodes"]), target_nodes) for proposal in proposals
    )
    credible_best = max(
        f1(set(proposals[index]["nodes"]), target_nodes)
        for index in credible_indices
    )
    return {
        "chip_id": row["chip_id"],
        "map_proposal_id": proposals[map_index]["proposal_id"],
        "map_nodes": ordered(map_nodes, node_order),
        "map_f1": f1(map_nodes, target_nodes),
        "expected_f1": expected_f1,
        "node_brier": brier,
        "posterior_entropy": -sum(
            weight * math.log(max(weight, 1e-12)) for weight in weights
        ),
        "credible_90_size": len(credible_indices),
        "credible_90_best_f1": credible_best,
        "best_available_f1": best_available,
        "posterior_weights": [
            {
                "proposal_id": proposal["proposal_id"],
                "weight": weight,
                "artifact_fit": proposal["artifact_fit"],
            }
            for proposal, weight in zip(proposals, weights)
        ],
    }


def evaluate(
    cases: list[dict[str, Any]],
    proposal_map: dict[str, list[dict[str, Any]]],
    support: dict[str, float],
    node_order: list[str],
    temperature: float,
    complexity_penalty: float,
    graph_weight: float,
    mode: str,
) -> dict[str, Any]:
    rows = [
        infer_case(
            row,
            proposal_map[row["chip_id"]],
            support,
            node_order,
            temperature,
            complexity_penalty,
            graph_weight,
            mode,
        )
        for row in cases
    ]
    return {
        "cases": len(rows),
        "mean_map_f1": statistics.mean(row["map_f1"] for row in rows),
        "mean_expected_f1": statistics.mean(
            row["expected_f1"] for row in rows
        ),
        "mean_node_brier": statistics.mean(row["node_brier"] for row in rows),
        "mean_posterior_entropy": statistics.mean(
            row["posterior_entropy"] for row in rows
        ),
        "credible_90_mean_size": statistics.mean(
            row["credible_90_size"] for row in rows
        ),
        "credible_90_mean_best_f1": statistics.mean(
            row["credible_90_best_f1"] for row in rows
        ),
        "mean_best_available_f1": statistics.mean(
            row["best_available_f1"] for row in rows
        ),
        "per_case": rows,
    }


def rounded_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in result.items()
        if key != "per_case"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--proposal-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    trace_path = Path(args.trace)
    output_dir = Path(args.output_dir)
    rows = read_json(trace_path)
    if len(rows) < 10:
        raise ValueError("The smoke test requires at least 10 proxy traces.")

    node_order = sorted(
        {node for row in rows for node in row["selected_nodes"]},
        key=lambda node: statistics.mean(
            [
                row["selected_nodes"].index(node)
                for row in rows
                if node in row["selected_nodes"]
            ]
        ),
    )
    split_rng = random.Random(args.seed)
    shuffled = list(rows)
    split_rng.shuffle(shuffled)
    train_end = max(6, int(0.60 * len(shuffled)))
    validation_end = max(train_end + 2, int(0.80 * len(shuffled)))
    train = shuffled[:train_end]
    validation = shuffled[train_end:validation_end]
    test = shuffled[validation_end:]
    prevalence = {
        node: sum(node in row["selected_nodes"] for row in train) / len(train)
        for node in node_order
    }
    proposal_map = {
        row["chip_id"]: generate_proposals(
            row,
            node_order,
            args.proposal_count,
            args.seed,
        )
        for row in rows
    }

    grid = list(
        itertools.product(
            [0.03, 0.06, 0.10, 0.16, 0.24, 0.36],
            [0.00, 0.04, 0.08],
            [0.00, 0.08, 0.16, 0.30, 0.50],
        )
    )
    trials: list[dict[str, Any]] = []
    for temperature, complexity_penalty, graph_weight in grid:
        support = graph_support(
            train,
            proposal_map,
            node_order,
            temperature,
            complexity_penalty,
            "posterior",
        )
        result = evaluate(
            validation,
            proposal_map,
            support,
            node_order,
            temperature,
            complexity_penalty,
            graph_weight,
            "posterior",
        )
        # Prefer high MAP trace recovery and calibrated node inclusion.
        objective = result["mean_map_f1"] - 0.35 * result["mean_node_brier"]
        trials.append(
            {
                "temperature": temperature,
                "complexity_penalty": complexity_penalty,
                "graph_weight": graph_weight,
                "objective": objective,
                **rounded_metrics(result),
            }
        )
    best = max(trials, key=lambda row: row["objective"])
    search_bounds = {
        "temperature": [0.03, 0.36],
        "complexity_penalty": [0.00, 0.08],
        "graph_weight": [0.00, 0.50],
    }
    boundary_hits = [
        name
        for name, bounds in search_bounds.items()
        if best[name] in bounds
    ]

    optimized_support = graph_support(
        train,
        proposal_map,
        node_order,
        best["temperature"],
        best["complexity_penalty"],
        "posterior",
    )
    single_support = graph_support(
        train,
        proposal_map,
        node_order,
        1.0,
        0.0,
        "single",
    )
    optimized_test = evaluate(
        test,
        proposal_map,
        optimized_support,
        node_order,
        best["temperature"],
        best["complexity_penalty"],
        best["graph_weight"],
        "posterior",
    )
    single_test = evaluate(
        test,
        proposal_map,
        single_support,
        node_order,
        1.0,
        0.0,
        0.0,
        "single",
    )

    report = {
        "evidence_status": "smoke_only_synthetic_proposal_noise",
        "convergence_claim": False,
        "source_trace": str(trace_path),
        "seed": args.seed,
        "proposal_count": args.proposal_count,
        "split": {
            "train": [row["chip_id"] for row in train],
            "validation": [row["chip_id"] for row in validation],
            "test": [row["chip_id"] for row in test],
        },
        "node_order": node_order,
        "training_only_proxy_node_prevalence": prevalence,
        "optimized_hyperparameters": {
            "temperature": best["temperature"],
            "complexity_penalty": best["complexity_penalty"],
            "graph_weight": best["graph_weight"],
            "validation_objective": round(best["objective"], 6),
            "search_boundary_hits": boundary_hits,
        },
        "test_metrics": {
            "single_hypothesis": rounded_metrics(single_test),
            "optimized_multi_hypothesis": rounded_metrics(optimized_test),
            "delta": {
                "mean_map_f1": round(
                    optimized_test["mean_map_f1"]
                    - single_test["mean_map_f1"],
                    6,
                ),
                "mean_expected_f1": round(
                    optimized_test["mean_expected_f1"]
                    - single_test["mean_expected_f1"],
                    6,
                ),
                "mean_node_brier": round(
                    optimized_test["mean_node_brier"]
                    - single_test["mean_node_brier"],
                    6,
                ),
            },
        },
        "limitations": [
            "Training traces are previous DIRS extractions, not historical expert traces.",
            "Proposal and verifier noise are synthetic.",
            "Synthetic corruption rates, verifier mixture, and tuning ranges are test-fixture constants, not DIRS policy.",
            "The graph has few node types and no generated-artifact execution.",
            "This run validates machinery only and is not convergence or publication evidence.",
        ],
    }

    write_json(output_dir / "report.json", report)
    write_json(output_dir / "hyperparameter_trials.json", trials)
    write_json(output_dir / "proposals.json", proposal_map)
    write_json(output_dir / "optimized_node_support.json", optimized_support)
    write_json(output_dir / "single_node_support.json", single_support)
    write_json(output_dir / "optimized_test_details.json", optimized_test)
    write_json(output_dir / "single_test_details.json", single_test)

    delta = report["test_metrics"]["delta"]
    markdown = f"""# Uncertain Inverse Trace Optimization Smoke Test

Evidence status: `smoke_only_synthetic_proposal_noise`

This run does **not** establish method convergence or recovery of historical
expert behavior. It checks whether multi-hypothesis posterior aggregation and
validation-tuned hyperparameters work on controlled noisy versions of existing
DIRS proxy traces.

## Data

```text
source traces: {len(rows)}
train: {len(train)}
validation: {len(validation)}
test: {len(test)}
hypotheses per case: {args.proposal_count}
seed: {args.seed}
```

## Optimized Hyperparameters

```json
{json.dumps(report["optimized_hyperparameters"], indent=2)}
```

## Held-Out Test

| Metric | Single hypothesis | Optimized multi-hypothesis | Delta |
|---|---:|---:|---:|
| MAP node F1 | {single_test["mean_map_f1"]:.6f} | {optimized_test["mean_map_f1"]:.6f} | {delta["mean_map_f1"]:+.6f} |
| Expected node F1 | {single_test["mean_expected_f1"]:.6f} | {optimized_test["mean_expected_f1"]:.6f} | {delta["mean_expected_f1"]:+.6f} |
| Node Brier (lower is better) | {single_test["mean_node_brier"]:.6f} | {optimized_test["mean_node_brier"]:.6f} | {delta["mean_node_brier"]:+.6f} |

## Interpretation Rule

Promote this only as a code-path smoke test. A real test must replace synthetic
proposals with independent LLM trace hypotheses, use artifact/tool evidence,
execute selected sub-DAGs blindly, and evaluate on partial-gold process traces
or expert outcomes.
"""
    (output_dir / "README.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
