#!/usr/bin/env python3
"""Test when connected-prefix UCT helps as the legal path set grows."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def leaf_id(index: int, depth: int) -> str:
    return format(index, f"0{depth}b")


def leaf_actions(identifier: str) -> tuple[int, ...]:
    return tuple(int(value) for value in identifier)


def generate_truth(
    landscape: str,
    depth: int,
    rng: random.Random,
) -> dict[str, float]:
    path_count = 2**depth
    identifiers = [leaf_id(index, depth) for index in range(path_count)]
    if landscape == "independent_leaves":
        return {
            identifier: min(0.95, max(0.05, rng.gauss(0.5, 0.08)))
            for identifier in identifiers
        }
    if landscape == "hierarchical_smooth":
        prefix_effect: dict[str, float] = {}
        for level in range(1, depth + 1):
            sigma = 0.08 * (0.65 ** (level - 1))
            for prefix_index in range(2**level):
                prefix = leaf_id(prefix_index, level)
                prefix_effect[prefix] = rng.gauss(0.0, sigma)
        utilities = {}
        for identifier in identifiers:
            latent = 0.5 + sum(
                prefix_effect[identifier[:level]]
                for level in range(1, depth + 1)
            )
            latent += rng.gauss(0.0, 0.005)
            utilities[identifier] = min(0.95, max(0.05, latent))
        return utilities
    if landscape == "deceptive_needle":
        target_index = rng.randrange(0, path_count // 2)
        target = leaf_id(target_index, depth)
        utilities = {}
        for identifier in identifiers:
            if identifier == target:
                utilities[identifier] = 0.80
            elif identifier[0] == "0":
                utilities[identifier] = min(
                    0.48, max(0.36, rng.gauss(0.42, 0.01))
                )
            else:
                utilities[identifier] = min(
                    0.69, max(0.55, rng.gauss(0.62, 0.02))
                )
        return utilities
    raise ValueError(f"unknown landscape: {landscape}")


def make_reward_getter(
    true_utility: dict[str, float],
    rollout_sigma: float,
    seed: int,
) -> Callable[[str, int], float]:
    cache: dict[tuple[str, int], float] = {}
    path_order = {path_id: index for index, path_id in enumerate(sorted(true_utility))}

    def reward(path_id: str, ordinal: int) -> float:
        key = (path_id, ordinal)
        if key not in cache:
            rng = random.Random(
                seed
                + 1_000_003 * (path_order[path_id] + 1)
                + 10_007 * (ordinal + 1)
            )
            cache[key] = min(
                1.0,
                max(0.0, rng.gauss(true_utility[path_id], rollout_sigma)),
            )
        return cache[key]

    return reward


class MetricStats:
    def __init__(self) -> None:
        self.episodes = 0
        self.best_hits = 0
        self.top_five_percent_hits = 0
        self.regret_sum = 0.0
        self.regret_squared_sum = 0.0
        self.percentile_sum = 0.0
        self.percentile_squared_sum = 0.0

    def add(self, selected: str, utilities: dict[str, float]) -> None:
        selected_utility = utilities[selected]
        best_utility = max(utilities.values())
        regret = best_utility - selected_utility
        ordered = sorted(utilities.values(), reverse=True)
        top_count = max(1, math.ceil(0.05 * len(ordered)))
        top_threshold = ordered[top_count - 1]
        better_count = sum(
            value > selected_utility + 1e-15 for value in utilities.values()
        )
        percentile = (
            1.0
            if len(utilities) == 1
            else 1.0 - better_count / (len(utilities) - 1)
        )
        self.episodes += 1
        self.best_hits += int(math.isclose(selected_utility, best_utility, abs_tol=1e-12))
        self.top_five_percent_hits += int(
            selected_utility >= top_threshold - 1e-12
        )
        self.regret_sum += regret
        self.regret_squared_sum += regret * regret
        self.percentile_sum += percentile
        self.percentile_squared_sum += percentile * percentile

    @staticmethod
    def _mean_and_se(total: float, squared_total: float, count: int) -> tuple[float, float]:
        mean = total / count
        variance = max(0.0, squared_total / count - mean * mean)
        return mean, math.sqrt(variance / count)

    def result(self) -> dict[str, Any]:
        regret, regret_se = self._mean_and_se(
            self.regret_sum,
            self.regret_squared_sum,
            self.episodes,
        )
        percentile, percentile_se = self._mean_and_se(
            self.percentile_sum,
            self.percentile_squared_sum,
            self.episodes,
        )
        best_rate = self.best_hits / self.episodes
        top_rate = self.top_five_percent_hits / self.episodes
        return {
            "episodes": self.episodes,
            "best_leaf_identification_rate": best_rate,
            "best_leaf_rate_standard_error": math.sqrt(
                best_rate * (1.0 - best_rate) / self.episodes
            ),
            "top_five_percent_rate": top_rate,
            "mean_simple_regret": regret,
            "mean_simple_regret_standard_error": regret_se,
            "mean_selected_utility_percentile": percentile,
            "mean_percentile_standard_error": percentile_se,
        }


class PairedRegretDifference:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.squared_sum = 0.0

    def add(
        self,
        selected_a: str,
        selected_b: str,
        utilities: dict[str, float],
    ) -> None:
        best = max(utilities.values())
        regret_a = best - utilities[selected_a]
        regret_b = best - utilities[selected_b]
        difference = regret_a - regret_b
        self.count += 1
        self.sum += difference
        self.squared_sum += difference * difference

    def result(self) -> dict[str, Any]:
        mean = self.sum / self.count
        variance = max(0.0, self.squared_sum / self.count - mean * mean)
        standard_error = math.sqrt(variance / self.count)
        return {
            "episodes": self.count,
            "mean_regret_a_minus_b": mean,
            "standard_error": standard_error,
            "normal_approx_95pct_interval": [
                mean - 1.96 * standard_error,
                mean + 1.96 * standard_error,
            ],
        }


def empirical_best(
    path_ids: list[str],
    counts: Counter[str],
    sums: dict[str, float],
) -> str:
    return max(
        path_ids,
        key=lambda path_id: (
            sums[path_id] / counts[path_id]
            if counts[path_id]
            else float("-inf"),
            path_id,
        ),
    )


def uniform_complete_path(
    path_ids: list[str],
    budget: int,
    reward: Callable[[str, int], float],
    seed: int,
) -> tuple[str, Counter[str], list[dict[str, Any]]]:
    rng = random.Random(seed)
    order = list(path_ids)
    rng.shuffle(order)
    counts: Counter[str] = Counter({path_id: 0 for path_id in path_ids})
    sums: defaultdict[str, float] = defaultdict(float)
    trace = []
    for sample_index in range(budget):
        path_id = order[sample_index % len(order)]
        ordinal = counts[path_id]
        value = reward(path_id, ordinal)
        counts[path_id] += 1
        sums[path_id] += value
        trace.append(
            {
                "sample": sample_index + 1,
                "path_id": path_id,
                "path_ordinal": ordinal + 1,
                "reward": value,
            }
        )
    return empirical_best(path_ids, counts, sums), counts, trace


def sequential_halving(
    path_ids: list[str],
    budget: int,
    reward: Callable[[str, int], float],
    seed: int,
) -> tuple[str, Counter[str], list[dict[str, Any]]]:
    if budget < len(path_ids):
        raise ValueError("Sequential Halving requires budget >= path count")
    rng = random.Random(seed)
    active = list(path_ids)
    rng.shuffle(active)
    counts: Counter[str] = Counter({path_id: 0 for path_id in path_ids})
    sums: defaultdict[str, float] = defaultdict(float)
    remaining = budget
    rounds_left = math.ceil(math.log2(len(active)))
    trace = []
    while len(active) > 1 and remaining >= len(active):
        per_arm = max(1, remaining // (len(active) * max(1, rounds_left)))
        for path_id in active:
            for _ in range(per_arm):
                ordinal = counts[path_id]
                value = reward(path_id, ordinal)
                counts[path_id] += 1
                sums[path_id] += value
        remaining -= per_arm * len(active)
        means = {
            path_id: sums[path_id] / counts[path_id] for path_id in active
        }
        keep_count = math.ceil(len(active) / 2)
        ranked = sorted(
            active,
            key=lambda path_id: (means[path_id], path_id),
            reverse=True,
        )
        trace.append(
            {
                "active_before": sorted(active),
                "samples_per_active_arm": per_arm,
                "eliminated": sorted(ranked[keep_count:]),
            }
        )
        active = ranked[:keep_count]
        rounds_left -= 1
    while remaining > 0:
        path_id = active[(budget - remaining) % len(active)]
        ordinal = counts[path_id]
        value = reward(path_id, ordinal)
        counts[path_id] += 1
        sums[path_id] += value
        remaining -= 1
    selected = (
        active[0] if len(active) == 1 else empirical_best(active, counts, sums)
    )
    return selected, counts, trace


def uct_search(
    depth: int,
    budget: int,
    reward: Callable[[str, int], float],
    backup: str,
    exploration: float,
    seed: int,
    save_trace: bool,
) -> tuple[str, Counter[str], list[dict[str, Any]]]:
    rng = random.Random(seed)
    state_visits: Counter[tuple[int, ...]] = Counter()
    edge_visits: Counter[tuple[tuple[int, ...], int]] = Counter()
    edge_sum: defaultdict[tuple[tuple[int, ...], int], float] = defaultdict(float)
    edge_max: defaultdict[tuple[tuple[int, ...], int], float] = defaultdict(
        lambda: float("-inf")
    )
    leaf_counts: Counter[str] = Counter()
    leaf_sums: defaultdict[str, float] = defaultdict(float)
    trace = []

    for simulation in range(1, budget + 1):
        prefix: tuple[int, ...] = ()
        traversed = []
        decisions = []
        while len(prefix) < depth:
            actions = (0, 1)
            unvisited = [
                action
                for action in actions
                if edge_visits[(prefix, action)] == 0
            ]
            if unvisited:
                selected = rng.choice(unvisited)
                scored = [
                    {
                        "action": action,
                        "visits_before": edge_visits[(prefix, action)],
                        "score": "infinity"
                        if action in unvisited
                        else None,
                    }
                    for action in actions
                ]
            else:
                scored = []
                for action in actions:
                    edge = (prefix, action)
                    visits = edge_visits[edge]
                    q_value = (
                        edge_sum[edge] / visits
                        if backup == "mean"
                        else edge_max[edge]
                    )
                    score = q_value + exploration * math.sqrt(
                        math.log(state_visits[prefix] + 1) / visits
                    )
                    scored.append(
                        {
                            "action": action,
                            "visits_before": visits,
                            "q_before": q_value,
                            "score": score,
                        }
                    )
                best_score = max(item["score"] for item in scored)
                tied = [
                    item["action"]
                    for item in scored
                    if math.isclose(item["score"], best_score, abs_tol=1e-15)
                ]
                selected = rng.choice(tied)
            if save_trace:
                decisions.append(
                    {
                        "state_prefix": list(prefix),
                        "valid_actions": [0, 1],
                        "scored_actions": scored,
                        "selected_action": selected,
                    }
                )
            traversed.append((prefix, selected))
            prefix = prefix + (selected,)

        identifier = "".join(str(action) for action in prefix)
        ordinal = leaf_counts[identifier]
        value = reward(identifier, ordinal)
        leaf_counts[identifier] += 1
        leaf_sums[identifier] += value
        for state, action in reversed(traversed):
            edge = (state, action)
            state_visits[state] += 1
            edge_visits[edge] += 1
            edge_sum[edge] += value
            edge_max[edge] = max(edge_max[edge], value)
        if save_trace:
            trace.append(
                {
                    "simulation": simulation,
                    "decisions": decisions,
                    "terminal_path_id": identifier,
                    "terminal_actions": list(prefix),
                    "path_ordinal": ordinal + 1,
                    "reward": value,
                }
            )
    visited = sorted(leaf_counts)
    return empirical_best(visited, leaf_counts, leaf_sums), leaf_counts, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--depths", default="4,6,8")
    parser.add_argument("--budget-ratios", default="0.5,1,4")
    parser.add_argument(
        "--landscapes",
        default="hierarchical_smooth,independent_leaves,deceptive_needle",
    )
    parser.add_argument("--rollout-sigma", type=float, default=0.03)
    parser.add_argument("--uct-exploration", type=float, default=math.sqrt(2.0))
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    depths = sorted({int(value) for value in args.depths.split(",")})
    budget_ratios = sorted(
        {float(value) for value in args.budget_ratios.split(",")}
    )
    landscapes = [value.strip() for value in args.landscapes.split(",")]
    if args.episodes < 1 or min(depths) < 1 or min(budget_ratios) <= 0:
        raise ValueError("invalid positive simulation configuration")

    policies = (
        "uniform_complete_path",
        "sequential_halving",
        "uct_mean_backup",
        "uct_max_backup",
    )
    cells = []
    for landscape_index, landscape in enumerate(landscapes):
        for depth_index, depth in enumerate(depths):
            path_count = 2**depth
            path_ids = [
                leaf_id(index, depth) for index in range(path_count)
            ]
            for ratio_index, ratio in enumerate(budget_ratios):
                budget = max(1, int(round(path_count * ratio)))
                stats = {
                    policy: MetricStats()
                    for policy in policies
                    if policy != "sequential_halving" or budget >= path_count
                }
                pair_specs = [
                    ("uct_mean_backup", "uniform_complete_path"),
                    ("uct_max_backup", "uniform_complete_path"),
                ]
                if budget >= path_count:
                    pair_specs.extend(
                        [
                            ("sequential_halving", "uct_mean_backup"),
                            ("sequential_halving", "uct_max_backup"),
                        ]
                    )
                paired = {
                    f"{method_a}__minus__{method_b}": PairedRegretDifference()
                    for method_a, method_b in pair_specs
                }
                audit_example = None
                cell_seed = (
                    args.seed
                    + 10_000_019 * landscape_index
                    + 1_000_003 * depth_index
                    + 100_003 * ratio_index
                )
                for episode_index in range(args.episodes):
                    episode_seed = cell_seed + 100_000_007 * (episode_index + 1)
                    truth = generate_truth(
                        landscape,
                        depth,
                        random.Random(episode_seed),
                    )
                    reward = make_reward_getter(
                        truth,
                        args.rollout_sigma,
                        episode_seed + 17,
                    )
                    uniform_selected, uniform_counts, uniform_trace = (
                        uniform_complete_path(
                            path_ids,
                            budget,
                            reward,
                            episode_seed + 31,
                        )
                    )
                    stats["uniform_complete_path"].add(uniform_selected, truth)
                    mean_selected, mean_counts, mean_trace = uct_search(
                        depth,
                        budget,
                        reward,
                        "mean",
                        args.uct_exploration,
                        episode_seed + 37,
                        save_trace=episode_index == 0,
                    )
                    stats["uct_mean_backup"].add(mean_selected, truth)
                    max_selected, max_counts, max_trace = uct_search(
                        depth,
                        budget,
                        reward,
                        "max",
                        args.uct_exploration,
                        episode_seed + 41,
                        save_trace=episode_index == 0,
                    )
                    stats["uct_max_backup"].add(max_selected, truth)
                    halving_selected = None
                    halving_counts = None
                    halving_trace = None
                    if budget >= path_count:
                        (
                            halving_selected,
                            halving_counts,
                            halving_trace,
                        ) = sequential_halving(
                            path_ids,
                            budget,
                            reward,
                            episode_seed + 43,
                        )
                        stats["sequential_halving"].add(halving_selected, truth)
                    selections = {
                        "uniform_complete_path": uniform_selected,
                        "uct_mean_backup": mean_selected,
                        "uct_max_backup": max_selected,
                    }
                    if halving_selected is not None:
                        selections["sequential_halving"] = halving_selected
                    for method_a, method_b in pair_specs:
                        paired[
                            f"{method_a}__minus__{method_b}"
                        ].add(
                            selections[method_a],
                            selections[method_b],
                            truth,
                        )
                    if episode_index == 0:
                        ordered_truth = sorted(
                            truth.items(),
                            key=lambda item: (item[1], item[0]),
                            reverse=True,
                        )
                        audit_example = {
                            "best_paths": [
                                {"path_id": path_id, "utility": utility}
                                for path_id, utility in ordered_truth[:10]
                            ],
                            "uniform_complete_path": {
                                "selected": uniform_selected,
                                "allocation_count": sum(uniform_counts.values()),
                                "sampled_path_count": len(
                                    [value for value in uniform_counts.values() if value]
                                ),
                                "trace": uniform_trace
                                if ratio == max(budget_ratios)
                                else [],
                            },
                            "sequential_halving": {
                                "applicable": budget >= path_count,
                                "selected": halving_selected,
                                "allocation_count": sum(halving_counts.values())
                                if halving_counts is not None
                                else 0,
                                "trace": halving_trace
                                if ratio == max(budget_ratios)
                                else [],
                            },
                            "uct_mean_backup": {
                                "selected": mean_selected,
                                "allocation_count": sum(mean_counts.values()),
                                "sampled_path_count": len(mean_counts),
                                "trace": mean_trace
                                if ratio == max(budget_ratios)
                                else [],
                            },
                            "uct_max_backup": {
                                "selected": max_selected,
                                "allocation_count": sum(max_counts.values()),
                                "sampled_path_count": len(max_counts),
                                "trace": max_trace
                                if ratio == max(budget_ratios)
                                else [],
                            },
                        }
                cells.append(
                    {
                        "cell_id": (
                            f"{landscape}__depth_{depth}__ratio_{ratio:g}"
                        ),
                        "landscape": landscape,
                        "depth": depth,
                        "path_count": path_count,
                        "budget_ratio": ratio,
                        "budget": budget,
                        "policy_results": {
                            policy: stats[policy].result()
                            if policy in stats
                            else {"applicable": False, "reason": "budget_below_K"}
                            for policy in policies
                        },
                        "paired_regret_differences": {
                            pair_id: pair.result()
                            for pair_id, pair in paired.items()
                        },
                        "audit_example": audit_example,
                    }
                )

    output = {
        "schema_version": "dirs.tree_search_scaling_boundary.v1",
        "status": "synthetic_connected_tree_scaling_test",
        "config": {
            "episodes_per_cell": args.episodes,
            "depths": depths,
            "path_counts": [2**depth for depth in depths],
            "budget_ratios": budget_ratios,
            "landscapes": landscapes,
            "rollout_sigma": args.rollout_sigma,
            "uct_exploration": args.uct_exploration,
            "seed": args.seed,
        },
        "policies": list(policies),
        "cells": cells,
        "claim_boundary": [
            "All graphs and rewards are synthetic.",
            "Tree actions are legal binary prefix extensions.",
            "The experiment contains no fresh LLM writer or evaluator calls.",
            "Numerical landscape parameters diagnose search behavior and are not learned.",
        ],
    }
    write_json(args.output, output)


if __name__ == "__main__":
    main()
