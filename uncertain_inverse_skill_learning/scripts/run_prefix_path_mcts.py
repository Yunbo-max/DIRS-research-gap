#!/usr/bin/env python3
"""Run MCTS over prefixes of observed complete directed paths.

The action space at a state is exactly the set of next nodes that continue at
least one observed training path. No arbitrary node addition is possible.
Terminal rewards come from cached blind LLM rollout evaluations.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
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


def prefix_key(prefix: tuple[str, ...]) -> str:
    return "->".join(prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--rewards", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--simulations", type=int, default=48)
    parser.add_argument("--c-puct", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--checkpoints", default="2,4,6,12,24,48")
    args = parser.parse_args()

    graph = read_json(args.graph)
    reward_file = read_json(args.rewards)
    reward_by_path = {
        item["path_id"]: float(item["overall_preference_score"])
        for item in reward_file["path_rewards"]
    }
    path_records = graph["observed_paths"]
    paths = {item["path_id"]: tuple(item["nodes"]) for item in path_records}
    support = {item["path_id"]: int(item["support_count"]) for item in path_records}
    if set(paths) != set(reward_by_path):
        raise ValueError("Reward paths must exactly match observed graph paths")
    if args.simulations < 1:
        raise ValueError("simulations must be positive")

    terminal_lookup = {nodes: path_id for path_id, nodes in paths.items()}
    root_nodes = {nodes[0] for nodes in paths.values()}
    if len(root_nodes) != 1:
        raise ValueError("Observed paths must have exactly one root")
    root = (next(iter(root_nodes)),)

    compatible_paths: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for path_id, nodes in paths.items():
        for length in range(1, len(nodes) + 1):
            compatible_paths[nodes[:length]].append(path_id)

    def frontier(prefix: tuple[str, ...]) -> list[str]:
        options = set()
        for path_id in compatible_paths[prefix]:
            nodes = paths[path_id]
            if len(nodes) > len(prefix):
                options.add(nodes[len(prefix)])
        return sorted(options)

    def action_prior(prefix: tuple[str, ...], action: str) -> float:
        candidates = compatible_paths[prefix]
        denominator = sum(support[path_id] for path_id in candidates)
        numerator = sum(
            support[path_id]
            for path_id in candidates
            if len(paths[path_id]) > len(prefix)
            and paths[path_id][len(prefix)] == action
        )
        return numerator / denominator

    state_visits: Counter[tuple[str, ...]] = Counter()
    edge_visits: Counter[tuple[tuple[str, ...], str]] = Counter()
    edge_value_sum: defaultdict[tuple[tuple[str, ...], str], float] = defaultdict(float)
    terminal_visits: Counter[str] = Counter()
    simulations: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    checkpoint_set = {
        value
        for value in (int(part) for part in args.checkpoints.split(",") if part)
        if value <= args.simulations
    }
    checkpoint_records = []

    for simulation_index in range(1, args.simulations + 1):
        prefix = root
        decisions = []
        traversed: list[tuple[tuple[str, ...], str]] = []

        while prefix not in terminal_lookup:
            valid_actions = frontier(prefix)
            if not valid_actions:
                raise ValueError(f"No valid continuation from {prefix}")
            scored_actions = []
            for action in valid_actions:
                edge = (prefix, action)
                visits = edge_visits[edge]
                q_value = edge_value_sum[edge] / visits if visits else 0.0
                prior = action_prior(prefix, action)
                exploration = (
                    args.c_puct
                    * prior
                    * math.sqrt(max(1, state_visits[prefix]))
                    / (1 + visits)
                )
                scored_actions.append(
                    {
                        "action": action,
                        "prior": prior,
                        "visits_before": visits,
                        "q_before": q_value,
                        "puct_score": q_value + exploration,
                    }
                )
            best_score = max(item["puct_score"] for item in scored_actions)
            tied = [
                item
                for item in scored_actions
                if math.isclose(item["puct_score"], best_score, abs_tol=1e-12)
            ]
            selected = rng.choice(tied)
            decisions.append(
                {
                    "state_prefix": list(prefix),
                    "valid_outgoing_frontier": scored_actions,
                    "selected_next_node": selected["action"],
                }
            )
            traversed.append((prefix, selected["action"]))
            prefix = prefix + (selected["action"],)

        path_id = terminal_lookup[prefix]
        reward = reward_by_path[path_id]
        terminal_visits[path_id] += 1
        backpropagation = []
        for state, action in reversed(traversed):
            edge = (state, action)
            q_before = (
                edge_value_sum[edge] / edge_visits[edge]
                if edge_visits[edge]
                else 0.0
            )
            state_visits[state] += 1
            edge_visits[edge] += 1
            edge_value_sum[edge] += reward
            q_after = edge_value_sum[edge] / edge_visits[edge]
            backpropagation.append(
                {
                    "state_prefix": list(state),
                    "action": action,
                    "reward": reward,
                    "visits_after": edge_visits[edge],
                    "q_before": q_before,
                    "q_after": q_after,
                }
            )
        simulations.append(
            {
                "simulation": simulation_index,
                "decisions": decisions,
                "terminal_path_id": path_id,
                "terminal_nodes": list(prefix),
                "reward": reward,
                "backpropagation": backpropagation,
            }
        )

        if simulation_index in checkpoint_set:
            recommended = sorted(
                paths,
                key=lambda item: (
                    -terminal_visits[item],
                    -reward_by_path[item],
                    item,
                ),
            )[0]
            checkpoint_records.append(
                {
                    "simulation_budget": simulation_index,
                    "recommended_path_id": recommended,
                    "recommended_terminal_visits": terminal_visits[recommended],
                    "recommended_reward": reward_by_path[recommended],
                    "terminal_visits": dict(sorted(terminal_visits.items())),
                }
            )

    exhaustive_best = max(
        paths, key=lambda item: (reward_by_path[item], support[item], item)
    )
    frequency_greedy = max(
        paths, key=lambda item: (support[item], reward_by_path[item], item)
    )
    recommended = sorted(
        paths,
        key=lambda item: (
            -terminal_visits[item],
            -reward_by_path[item],
            item,
        ),
    )[0]

    final_edges = []
    for (state, action), visits in sorted(
        edge_visits.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        final_edges.append(
            {
                "state_prefix": list(state),
                "action": action,
                "prior": action_prior(state, action),
                "visits": visits,
                "q": edge_value_sum[(state, action)] / visits,
            }
        )

    output = {
        "schema_version": "dirs.prefix_path_mcts.v1",
        "graph_id": graph["graph_hash_sha256"],
        "search_space": {
            "observed_complete_paths_only": True,
            "path_count": len(paths),
            "arbitrary_node_selection": False,
            "state_is_complete_prefix": True,
            "action_is_legal_next_node": True,
        },
        "config": {
            "simulations": args.simulations,
            "c_puct": args.c_puct,
            "seed": args.seed,
            "prior": "empirical conditional prefix continuation frequency",
            "terminal_reward": "blind evaluator overall_preference_score",
        },
        "path_table": [
            {
                "path_id": path_id,
                "nodes": list(paths[path_id]),
                "training_support_count": support[path_id],
                "reward": reward_by_path[path_id],
            }
            for path_id in sorted(paths)
        ],
        "simulations": simulations,
        "checkpoints": checkpoint_records,
        "final_edge_statistics": final_edges,
        "terminal_visits": dict(sorted(terminal_visits.items())),
        "recommendation": {
            "path_id": recommended,
            "reward": reward_by_path[recommended],
            "terminal_visits": terminal_visits[recommended],
        },
        "baselines": {
            "exhaustive_best_path_id": exhaustive_best,
            "exhaustive_best_reward": reward_by_path[exhaustive_best],
            "frequency_greedy_path_id": frequency_greedy,
            "frequency_greedy_reward": reward_by_path[frequency_greedy],
            "uniform_random_valid_expected_reward": sum(reward_by_path.values())
            / len(reward_by_path),
            "uniform_random_one_draw_best_hit_probability": 1 / len(paths),
        },
        "checks": {
            "recommendation_matches_exhaustive_best": recommended == exhaustive_best,
            "every_selected_action_was_in_logged_frontier": True,
            "every_terminal_was_an_observed_complete_path": True,
        },
        "claim_boundary": [
            "This is cached-rollout MCTS, not an online new LLM call at every visit.",
            "Rewards come from one blind LLM evaluator and are not calibrated human utility.",
            "The six paths come from proxy traces with a fixed node vocabulary.",
        ],
    }
    write_json(args.output, output)


if __name__ == "__main__":
    main()
