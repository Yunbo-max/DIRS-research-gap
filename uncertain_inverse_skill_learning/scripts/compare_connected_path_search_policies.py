#!/usr/bin/env python3
"""Compare MCTS with fixed-budget best-arm identification on legal DAG paths."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from run_stochastic_prefix_path_mcts_sim import (
    PathSpace,
    RunningMethodStats,
    clipped_gaussian,
    make_reward_streams,
    read_json,
    run_mcts_episode,
    write_json,
)


class PairedRegretDifference:
    """Track regret(method_a) - regret(method_b) on paired episodes."""

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.squared_sum = 0.0
        self.a_better = 0
        self.equal = 0
        self.b_better = 0

    def add(self, regret_a: float, regret_b: float) -> None:
        difference = regret_a - regret_b
        self.count += 1
        self.sum += difference
        self.squared_sum += difference * difference
        if difference < -1e-15:
            self.a_better += 1
        elif difference > 1e-15:
            self.b_better += 1
        else:
            self.equal += 1

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
            "a_lower_regret_count": self.a_better,
            "equal_regret_count": self.equal,
            "b_lower_regret_count": self.b_better,
        }


def empirical_best(
    path_ids: list[str],
    counts: Counter[str],
    sums: dict[str, float],
) -> str:
    means = {
        path_id: sums[path_id] / counts[path_id]
        if counts[path_id]
        else float("-inf")
        for path_id in path_ids
    }
    return max(path_ids, key=lambda path_id: (means[path_id], path_id))


def make_sampler(
    streams: dict[str, list[float]],
) -> tuple[
    Callable[[str], float],
    Counter[str],
    defaultdict[str, float],
    list[dict[str, Any]],
]:
    counts: Counter[str] = Counter({path_id: 0 for path_id in streams})
    sums: defaultdict[str, float] = defaultdict(float)
    trace = []

    def sample(path_id: str) -> float:
        ordinal = counts[path_id]
        reward = streams[path_id][ordinal]
        counts[path_id] += 1
        sums[path_id] += reward
        trace.append(
            {
                "sample": len(trace) + 1,
                "path_id": path_id,
                "path_ordinal": ordinal + 1,
                "reward": reward,
            }
        )
        return reward

    return sample, counts, sums, trace


def uniform_allocation(
    path_ids: list[str],
    streams: dict[str, list[float]],
    budget: int,
    seed: int,
) -> tuple[str, Counter[str], list[dict[str, Any]]]:
    sample, counts, sums, trace = make_sampler(streams)
    order = list(path_ids)
    random.Random(seed).shuffle(order)
    for index in range(budget):
        sample(order[index % len(order)])
    return empirical_best(path_ids, counts, sums), counts, trace


def sequential_halving(
    path_ids: list[str],
    streams: dict[str, list[float]],
    budget: int,
    seed: int,
) -> tuple[str, Counter[str], list[dict[str, Any]]]:
    sample, counts, sums, sample_trace = make_sampler(streams)
    rng = random.Random(seed)
    active = list(path_ids)
    rng.shuffle(active)
    remaining = budget
    rounds_left = math.ceil(math.log2(len(active)))
    elimination_trace = []

    while len(active) > 1 and remaining > 0:
        per_arm = max(1, remaining // (len(active) * max(1, rounds_left)))
        per_arm = min(per_arm, remaining // len(active))
        if per_arm < 1:
            break
        for path_id in active:
            for _ in range(per_arm):
                sample(path_id)
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
        eliminated = ranked[keep_count:]
        active = ranked[:keep_count]
        elimination_trace.append(
            {
                "active_before": sorted(ranked),
                "means": means,
                "eliminated": sorted(eliminated),
                "active_after": sorted(active),
            }
        )
        rounds_left -= 1

    while remaining > 0:
        target = active[(budget - remaining) % len(active)]
        sample(target)
        remaining -= 1
    selected = (
        active[0]
        if len(active) == 1
        else empirical_best(active, counts, sums)
    )
    return selected, counts, [
        {"samples": sample_trace},
        {"elimination_rounds": elimination_trace},
    ]


def successive_rejects(
    path_ids: list[str],
    streams: dict[str, list[float]],
    budget: int,
    seed: int,
) -> tuple[str, Counter[str], list[dict[str, Any]]]:
    sample, counts, sums, sample_trace = make_sampler(streams)
    rng = random.Random(seed)
    active = list(path_ids)
    rng.shuffle(active)
    arm_count = len(active)
    log_bar = 0.5 + sum(1.0 / index for index in range(2, arm_count + 1))
    previous_target = 0
    elimination_trace = []

    for round_index in range(1, arm_count):
        target = math.ceil(
            max(0, budget - arm_count)
            / (log_bar * (arm_count + 1 - round_index))
        )
        target = max(previous_target, target)
        for path_id in active:
            while counts[path_id] < target and sum(counts.values()) < budget:
                sample(path_id)
        means = {
            path_id: (
                sums[path_id] / counts[path_id]
                if counts[path_id]
                else float("-inf")
            )
            for path_id in active
        }
        worst_value = min(means.values())
        tied_worst = [
            path_id
            for path_id in active
            if math.isclose(means[path_id], worst_value, abs_tol=1e-15)
        ]
        eliminated = rng.choice(tied_worst)
        active.remove(eliminated)
        elimination_trace.append(
            {
                "target_samples_per_active_arm": target,
                "means": means,
                "eliminated": eliminated,
                "active_after": sorted(active),
            }
        )
        previous_target = target

    while sum(counts.values()) < budget:
        sample(active[0])
    return active[0], counts, [
        {"samples": sample_trace},
        {"elimination_rounds": elimination_trace},
    ]


def ucb1(
    path_ids: list[str],
    streams: dict[str, list[float]],
    budget: int,
    seed: int,
) -> tuple[str, Counter[str], list[dict[str, Any]]]:
    sample, counts, sums, trace = make_sampler(streams)
    order = list(path_ids)
    rng = random.Random(seed)
    rng.shuffle(order)
    for path_id in order:
        sample(path_id)
    while sum(counts.values()) < budget:
        total = sum(counts.values())
        scores = {
            path_id: (
                sums[path_id] / counts[path_id]
                + math.sqrt(2.0 * math.log(total + 1) / counts[path_id])
            )
            for path_id in path_ids
        }
        best_score = max(scores.values())
        tied = [
            path_id
            for path_id in path_ids
            if math.isclose(scores[path_id], best_score, abs_tol=1e-15)
        ]
        sample(rng.choice(tied))
    return empirical_best(path_ids, counts, sums), counts, trace


def top_two_thompson(
    path_ids: list[str],
    streams: dict[str, list[float]],
    budget: int,
    rollout_sigma: float,
    seed: int,
) -> tuple[str, Counter[str], list[dict[str, Any]]]:
    sample, counts, sums, trace = make_sampler(streams)
    rng = random.Random(seed)
    order = list(path_ids)
    rng.shuffle(order)
    for path_id in order:
        sample(path_id)

    def posterior_draw() -> dict[str, float]:
        return {
            path_id: rng.gauss(
                sums[path_id] / counts[path_id],
                max(rollout_sigma, 1e-6) / math.sqrt(counts[path_id]),
            )
            for path_id in path_ids
        }

    while sum(counts.values()) < budget:
        first_draw = posterior_draw()
        first = max(path_ids, key=lambda path_id: (first_draw[path_id], path_id))
        selected = first
        if rng.random() >= 0.5:
            for _ in range(100):
                second_draw = posterior_draw()
                second = max(
                    path_ids,
                    key=lambda path_id: (second_draw[path_id], path_id),
                )
                if second != first:
                    selected = second
                    break
        sample(selected)
    return empirical_best(path_ids, counts, sums), counts, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--reward-centres", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--budgets", default="12,24,48,96,192")
    parser.add_argument("--epistemic-sigmas", default="0.02,0.05")
    parser.add_argument("--rollout-sigma", type=float, default=0.03)
    parser.add_argument("--c-puct", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    budgets = sorted({int(value) for value in args.budgets.split(",")})
    epistemic_sigmas = [
        float(value) for value in args.epistemic_sigmas.split(",")
    ]
    if args.episodes < 1 or min(budgets) < 6:
        raise ValueError("positive episodes and budgets >= path count are required")

    graph = read_json(args.graph)
    reward_file = read_json(args.reward_centres)
    space = PathSpace.from_graph(graph)
    path_ids = sorted(space.paths)
    base_utility = {
        item["path_id"]: float(item["overall_preference_score"])
        for item in reward_file["path_rewards"]
    }
    if set(path_ids) != set(base_utility):
        raise ValueError("graph and reward-centre paths differ")
    max_budget = max(budgets)
    frequency_greedy = max(
        path_ids,
        key=lambda path_id: (
            space.support[path_id],
            base_utility[path_id],
            path_id,
        ),
    )
    method_names = (
        "frequency_greedy",
        "random_valid",
        "uniform_allocation_q",
        "sequential_halving",
        "successive_rejects",
        "ucb1_q",
        "top_two_thompson",
        "mcts_empirical_visit",
        "mcts_empirical_q",
        "mcts_path_uniform_visit",
        "mcts_path_uniform_q",
    )
    comparison_pairs = (
        ("top_two_thompson", "mcts_empirical_q"),
        ("sequential_halving", "mcts_empirical_q"),
        ("successive_rejects", "mcts_empirical_q"),
        ("uniform_allocation_q", "mcts_empirical_q"),
        ("mcts_path_uniform_q", "mcts_empirical_q"),
        ("top_two_thompson", "sequential_halving"),
    )
    scenarios = []

    for scenario_index, epistemic_sigma in enumerate(epistemic_sigmas):
        scenario_seed = args.seed + 1_000_003 * scenario_index
        stats = {
            budget: {
                method: RunningMethodStats(path_ids) for method in method_names
            }
            for budget in budgets
        }
        allocation_sums = {
            budget: {
                method: Counter({path_id: 0 for path_id in path_ids})
                for method in (
                    (
                        "uniform_allocation_q",
                        "sequential_halving",
                        "successive_rejects",
                        "ucb1_q",
                        "top_two_thompson",
                        "mcts_empirical_visit",
                        "mcts_empirical_q",
                        "mcts_path_uniform_visit",
                        "mcts_path_uniform_q",
                    )
                    if budget == max(budgets)
                    else (
                        "uniform_allocation_q",
                        "sequential_halving",
                        "successive_rejects",
                        "ucb1_q",
                        "top_two_thompson",
                    )
                )
            }
            for budget in budgets
        }
        oracle_counts = Counter({path_id: 0 for path_id in path_ids})
        paired_differences = {
            budget: {
                f"{method_a}__minus__{method_b}": PairedRegretDifference()
                for method_a, method_b in comparison_pairs
            }
            for budget in budgets
        }
        audit_example = None

        for episode_index in range(args.episodes):
            episode_seed = scenario_seed + 100_000_007 * (episode_index + 1)
            truth_rng = random.Random(episode_seed)
            true_utility = {
                path_id: clipped_gaussian(
                    truth_rng,
                    base_utility[path_id],
                    epistemic_sigma,
                )
                for path_id in path_ids
            }
            oracle_path = max(
                path_ids,
                key=lambda path_id: (true_utility[path_id], path_id),
            )
            oracle_counts[oracle_path] += 1
            streams = make_reward_streams(
                path_ids,
                true_utility,
                args.rollout_sigma,
                max_budget,
                episode_seed + 17,
            )
            empirical_mcts, empirical_visits, empirical_trace = run_mcts_episode(
                space,
                streams,
                budgets,
                args.c_puct,
                0.0,
                "terminal_path_mass",
                episode_seed + 31,
                save_trace=episode_index == 0,
            )
            uniform_mcts, uniform_visits, uniform_trace = run_mcts_episode(
                space,
                streams,
                budgets,
                args.c_puct,
                1.0,
                "terminal_path_mass",
                episode_seed + 37,
                save_trace=episode_index == 0,
            )
            random_path = random.Random(episode_seed + 41).choice(path_ids)
            audit_methods: dict[str, Any] = {}

            for budget in budgets:
                uniform_selected, uniform_counts, uniform_log = uniform_allocation(
                    path_ids, streams, budget, episode_seed + 43
                )
                halving_selected, halving_counts, halving_log = sequential_halving(
                    path_ids, streams, budget, episode_seed + 47
                )
                rejects_selected, rejects_counts, rejects_log = successive_rejects(
                    path_ids, streams, budget, episode_seed + 53
                )
                ucb_selected, ucb_counts, ucb_log = ucb1(
                    path_ids, streams, budget, episode_seed + 59
                )
                thompson_selected, thompson_counts, thompson_log = (
                    top_two_thompson(
                        path_ids,
                        streams,
                        budget,
                        args.rollout_sigma,
                        episode_seed + 61,
                    )
                )
                selections = {
                    "frequency_greedy": frequency_greedy,
                    "random_valid": random_path,
                    "uniform_allocation_q": uniform_selected,
                    "sequential_halving": halving_selected,
                    "successive_rejects": rejects_selected,
                    "ucb1_q": ucb_selected,
                    "top_two_thompson": thompson_selected,
                    "mcts_empirical_visit": empirical_mcts[budget]["mcts_visit"],
                    "mcts_empirical_q": empirical_mcts[budget]["mcts_terminal_q"],
                    "mcts_path_uniform_visit": uniform_mcts[budget]["mcts_visit"],
                    "mcts_path_uniform_q": uniform_mcts[budget]["mcts_terminal_q"],
                }
                counts_by_method = {
                    "uniform_allocation_q": uniform_counts,
                    "sequential_halving": halving_counts,
                    "successive_rejects": rejects_counts,
                    "ucb1_q": ucb_counts,
                    "top_two_thompson": thompson_counts,
                }
                for method, selected in selections.items():
                    stats[budget][method].add(selected, true_utility, oracle_path)
                for method_a, method_b in comparison_pairs:
                    regret_a = (
                        true_utility[oracle_path]
                        - true_utility[selections[method_a]]
                    )
                    regret_b = (
                        true_utility[oracle_path]
                        - true_utility[selections[method_b]]
                    )
                    paired_differences[budget][
                        f"{method_a}__minus__{method_b}"
                    ].add(regret_a, regret_b)
                for method, counts in counts_by_method.items():
                    allocation_sums[budget][method].update(counts)

                if episode_index == 0 and budget == max_budget:
                    audit_methods = {
                        "uniform_allocation_q": {
                            "selected": uniform_selected,
                            "counts": dict(uniform_counts),
                            "trace": uniform_log,
                        },
                        "sequential_halving": {
                            "selected": halving_selected,
                            "counts": dict(halving_counts),
                            "trace": halving_log,
                        },
                        "successive_rejects": {
                            "selected": rejects_selected,
                            "counts": dict(rejects_counts),
                            "trace": rejects_log,
                        },
                        "ucb1_q": {
                            "selected": ucb_selected,
                            "counts": dict(ucb_counts),
                            "trace": ucb_log,
                        },
                        "top_two_thompson": {
                            "selected": thompson_selected,
                            "counts": dict(thompson_counts),
                            "trace": thompson_log,
                        },
                    }

            for path_id, visits in empirical_visits.items():
                allocation_sums[max_budget]["mcts_empirical_visit"][path_id] += visits
                allocation_sums[max_budget]["mcts_empirical_q"][path_id] += visits
            for path_id, visits in uniform_visits.items():
                allocation_sums[max_budget]["mcts_path_uniform_visit"][path_id] += visits
                allocation_sums[max_budget]["mcts_path_uniform_q"][path_id] += visits

            if episode_index == 0:
                audit_example = {
                    "simulator_only_true_utility": true_utility,
                    "oracle_path_id": oracle_path,
                    "complete_path_nodes": {
                        path_id: list(space.paths[path_id]) for path_id in path_ids
                    },
                    "methods_at_max_budget": audit_methods,
                    "mcts_empirical_trace": empirical_trace,
                    "mcts_path_uniform_trace": uniform_trace,
                }

        scenarios.append(
            {
                "scenario_id": f"epistemic_{epistemic_sigma:.3f}",
                "epistemic_sigma": epistemic_sigma,
                "rollout_sigma": args.rollout_sigma,
                "oracle_path_counts": dict(sorted(oracle_counts.items())),
                "budgets": {
                    str(budget): {
                        method: stats[budget][method].result()
                        for method in method_names
                    }
                    for budget in budgets
                },
                "mean_allocation_counts": {
                    str(budget): {
                        method: {
                            path_id: count / args.episodes
                            for path_id, count in sorted(counts.items())
                        }
                        for method, counts in allocation_sums[budget].items()
                    }
                    for budget in budgets
                },
                "paired_regret_differences": {
                    str(budget): {
                        pair_id: paired.result()
                        for pair_id, paired in paired_differences[budget].items()
                    }
                    for budget in budgets
                },
                "audit_example": audit_example,
            }
        )

    output = {
        "schema_version": "dirs.connected_path_policy_comparison.v1",
        "status": "synthetic_paired_fixed_budget_policy_comparison",
        "graph_id": graph["graph_hash_sha256"],
        "config": {
            "episodes_per_scenario": args.episodes,
            "budgets": budgets,
            "epistemic_sigmas": epistemic_sigmas,
            "rollout_sigma": args.rollout_sigma,
            "c_puct": args.c_puct,
            "seed": args.seed,
            "paired_path_reward_streams": True,
        },
        "path_table": [
            {
                "path_id": path_id,
                "nodes": list(space.paths[path_id]),
                "training_support_count": space.support[path_id],
                "simulator_reward_centre": base_utility[path_id],
            }
            for path_id in path_ids
        ],
        "policies": list(method_names),
        "scenarios": scenarios,
        "claim_boundary": [
            "Synthetic scalar rewards are not fresh LLM artifacts.",
            "Top-two Thompson sampling is given the simulator rollout sigma.",
            "All six complete paths are enumerable; this favors flat bandit methods.",
            "Results do not establish behavior in a large implicit sub-DAG space.",
        ],
    }
    write_json(args.output, output)


if __name__ == "__main__":
    main()
