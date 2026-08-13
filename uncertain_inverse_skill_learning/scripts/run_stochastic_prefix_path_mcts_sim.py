#!/usr/bin/env python3
"""Monte Carlo audit of stochastic MCTS on connected observed DAG paths.

This script does not call an LLM. It treats the existing blind evaluator scores
as centres of a synthetic reward model, draws a new rollout reward whenever a
terminal path is visited, and measures best-path identification under several
uncertainty settings. The simulator truth is never exposed to the search
policy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated values")
    return values


def parse_int_list(value: str) -> list[int]:
    values = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def clipped_gaussian(rng: random.Random, mean: float, sigma: float) -> float:
    if sigma == 0:
        return min(1.0, max(0.0, mean))
    return min(1.0, max(0.0, rng.gauss(mean, sigma)))


@dataclass(frozen=True)
class PathSpace:
    paths: dict[str, tuple[str, ...]]
    support: dict[str, int]
    root: tuple[str, ...]
    compatible: dict[tuple[str, ...], tuple[str, ...]]

    @classmethod
    def from_graph(cls, graph: dict[str, Any]) -> "PathSpace":
        records = graph["observed_paths"]
        paths = {item["path_id"]: tuple(item["nodes"]) for item in records}
        support = {item["path_id"]: int(item["support_count"]) for item in records}
        if not paths:
            raise ValueError("graph has no observed paths")
        if any(not nodes for nodes in paths.values()):
            raise ValueError("every observed path must contain at least one node")
        roots = {nodes[0] for nodes in paths.values()}
        if len(roots) != 1:
            raise ValueError("observed paths must have exactly one root")
        compatible_lists: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for path_id, nodes in paths.items():
            for length in range(1, len(nodes) + 1):
                compatible_lists[nodes[:length]].append(path_id)
        compatible = {
            prefix: tuple(sorted(path_ids))
            for prefix, path_ids in compatible_lists.items()
        }
        return cls(
            paths=paths,
            support=support,
            root=(next(iter(roots)),),
            compatible=compatible,
        )

    def frontier(self, prefix: tuple[str, ...]) -> tuple[str, ...]:
        options = {
            self.paths[path_id][len(prefix)]
            for path_id in self.compatible[prefix]
            if len(self.paths[path_id]) > len(prefix)
        }
        return tuple(sorted(options))

    def action_prior(
        self,
        prefix: tuple[str, ...],
        action: str,
        uniform_mix: float = 0.0,
        mix_mode: str = "local_action",
    ) -> float:
        candidates = self.compatible[prefix]
        denominator = sum(self.support[path_id] for path_id in candidates)
        numerator = sum(
            self.support[path_id]
            for path_id in candidates
            if len(self.paths[path_id]) > len(prefix)
            and self.paths[path_id][len(prefix)] == action
        )
        if denominator <= 0 or numerator <= 0:
            raise ValueError("invalid empirical continuation prior")
        empirical_prior = numerator / denominator
        if mix_mode == "local_action":
            uniform_prior = 1.0 / len(self.frontier(prefix))
            return (
                (1.0 - uniform_mix) * empirical_prior
                + uniform_mix * uniform_prior
            )
        if mix_mode == "terminal_path_mass":
            mean_support = sum(self.support.values()) / len(self.support)

            def path_weight(path_id: str) -> float:
                return (
                    (1.0 - uniform_mix) * self.support[path_id]
                    + uniform_mix * mean_support
                )

            mixed_denominator = sum(path_weight(path_id) for path_id in candidates)
            mixed_numerator = sum(
                path_weight(path_id)
                for path_id in candidates
                if len(self.paths[path_id]) > len(prefix)
                and self.paths[path_id][len(prefix)] == action
            )
            return mixed_numerator / mixed_denominator
        raise ValueError(f"unknown prior mix mode: {mix_mode}")


class RunningMethodStats:
    def __init__(self, path_ids: Iterable[str]) -> None:
        self.episodes = 0
        self.best_hits = 0
        self.regret_sum = 0.0
        self.regret_squared_sum = 0.0
        self.selection_counts = Counter({path_id: 0 for path_id in path_ids})

    def add(
        self,
        selected_path: str,
        true_utility: dict[str, float],
        oracle_path: str,
    ) -> None:
        regret = true_utility[oracle_path] - true_utility[selected_path]
        self.episodes += 1
        self.best_hits += int(
            math.isclose(
                true_utility[selected_path],
                true_utility[oracle_path],
                abs_tol=1e-12,
            )
        )
        self.regret_sum += regret
        self.regret_squared_sum += regret * regret
        self.selection_counts[selected_path] += 1

    def result(self) -> dict[str, Any]:
        mean_regret = self.regret_sum / self.episodes
        regret_variance = max(
            0.0,
            self.regret_squared_sum / self.episodes - mean_regret * mean_regret,
        )
        hit_rate = self.best_hits / self.episodes
        return {
            "episodes": self.episodes,
            "best_path_identification_rate": hit_rate,
            "best_path_identification_standard_error": math.sqrt(
                hit_rate * (1.0 - hit_rate) / self.episodes
            ),
            "mean_simple_regret": mean_regret,
            "mean_simple_regret_standard_error": math.sqrt(
                regret_variance / self.episodes
            ),
            "selection_counts": dict(sorted(self.selection_counts.items())),
        }


def choose_max(
    path_ids: Iterable[str],
    primary: dict[str, float | int],
    secondary: dict[str, float | int],
    support: dict[str, int],
) -> str:
    return max(
        path_ids,
        key=lambda path_id: (
            primary[path_id],
            secondary[path_id],
            support[path_id],
            path_id,
        ),
    )


def make_reward_streams(
    path_ids: list[str],
    true_utility: dict[str, float],
    rollout_sigma: float,
    samples_per_path: int,
    seed: int,
) -> dict[str, list[float]]:
    streams = {}
    for path_index, path_id in enumerate(path_ids):
        rng = random.Random(seed + 1009 * (path_index + 1))
        streams[path_id] = [
            clipped_gaussian(rng, true_utility[path_id], rollout_sigma)
            for _ in range(samples_per_path)
        ]
    return streams


def run_mcts_episode(
    space: PathSpace,
    reward_streams: dict[str, list[float]],
    checkpoints: list[int],
    c_puct: float,
    prior_uniform_mix: float,
    prior_mix_mode: str,
    seed: int,
    save_trace: bool,
) -> tuple[dict[int, dict[str, str]], dict[str, int], list[dict[str, Any]]]:
    path_ids = sorted(space.paths)
    terminal_lookup = {
        nodes: path_id for path_id, nodes in space.paths.items()
    }
    state_visits: Counter[tuple[str, ...]] = Counter()
    edge_visits: Counter[tuple[tuple[str, ...], str]] = Counter()
    edge_value_sum: defaultdict[tuple[tuple[str, ...], str], float] = defaultdict(
        float
    )
    terminal_visits: Counter[str] = Counter({path_id: 0 for path_id in path_ids})
    terminal_value_sum: defaultdict[str, float] = defaultdict(float)
    rng = random.Random(seed)
    checkpoint_set = set(checkpoints)
    recommendations: dict[int, dict[str, str]] = {}
    trace: list[dict[str, Any]] = []

    for simulation_index in range(1, max(checkpoints) + 1):
        prefix = space.root
        traversed: list[tuple[tuple[str, ...], str]] = []
        decisions = []
        while prefix not in terminal_lookup:
            actions = space.frontier(prefix)
            if not actions:
                raise ValueError(f"no legal continuation from {prefix}")
            scored = []
            for action in actions:
                edge = (prefix, action)
                visits = edge_visits[edge]
                q_value = edge_value_sum[edge] / visits if visits else 0.0
                prior = space.action_prior(
                    prefix,
                    action,
                    prior_uniform_mix,
                    prior_mix_mode,
                )
                exploration = (
                    c_puct
                    * prior
                    * math.sqrt(max(1, state_visits[prefix]))
                    / (1 + visits)
                )
                scored.append(
                    {
                        "action": action,
                        "prior": prior,
                        "visits_before": visits,
                        "q_before": q_value,
                        "puct_score": q_value + exploration,
                    }
                )
            best_score = max(item["puct_score"] for item in scored)
            tied = [
                item
                for item in scored
                if math.isclose(item["puct_score"], best_score, abs_tol=1e-12)
            ]
            selected = rng.choice(tied)
            traversed.append((prefix, selected["action"]))
            if save_trace:
                decisions.append(
                    {
                        "state_prefix": list(prefix),
                        "valid_frontier": scored,
                        "selected_next_node": selected["action"],
                    }
                )
            prefix = prefix + (selected["action"],)

        path_id = terminal_lookup[prefix]
        sample_index = terminal_visits[path_id]
        reward = reward_streams[path_id][sample_index]
        terminal_visits[path_id] += 1
        terminal_value_sum[path_id] += reward
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
            if save_trace:
                backpropagation.append(
                    {
                        "state_prefix": list(state),
                        "action": action,
                        "reward": reward,
                        "visits_after": edge_visits[edge],
                        "q_before": q_before,
                        "q_after": edge_value_sum[edge] / edge_visits[edge],
                    }
                )

        if save_trace:
            trace.append(
                {
                    "simulation": simulation_index,
                    "decisions": decisions,
                    "terminal_path_id": path_id,
                    "terminal_nodes": list(prefix),
                    "rollout_ordinal_for_path": sample_index + 1,
                    "sampled_reward": reward,
                    "backpropagation": backpropagation,
                }
            )

        if simulation_index in checkpoint_set:
            empirical_mean = {
                candidate: (
                    terminal_value_sum[candidate] / terminal_visits[candidate]
                    if terminal_visits[candidate]
                    else float("-inf")
                )
                for candidate in path_ids
            }
            recommendations[simulation_index] = {
                "mcts_visit": choose_max(
                    path_ids,
                    dict(terminal_visits),
                    empirical_mean,
                    space.support,
                ),
                "mcts_terminal_q": choose_max(
                    path_ids,
                    empirical_mean,
                    dict(terminal_visits),
                    space.support,
                ),
            }

    return recommendations, dict(terminal_visits), trace


def uniform_allocation_recommendation(
    path_ids: list[str],
    reward_streams: dict[str, list[float]],
    budget: int,
    support: dict[str, int],
    seed: int,
) -> str:
    order = list(path_ids)
    random.Random(seed).shuffle(order)
    counts = Counter({path_id: 0 for path_id in path_ids})
    sums: defaultdict[str, float] = defaultdict(float)
    for index in range(budget):
        path_id = order[index % len(order)]
        sample_index = counts[path_id]
        sums[path_id] += reward_streams[path_id][sample_index]
        counts[path_id] += 1
    means = {
        path_id: sums[path_id] / counts[path_id] if counts[path_id] else float("-inf")
        for path_id in path_ids
    }
    return choose_max(path_ids, means, dict(counts), support)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--reward-centres", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--budgets", type=parse_int_list, default=[12, 24, 48, 96, 192])
    parser.add_argument(
        "--epistemic-sigmas",
        type=parse_float_list,
        default=[0.0, 0.02, 0.05],
    )
    parser.add_argument(
        "--rollout-sigmas",
        type=parse_float_list,
        default=[0.01, 0.03, 0.06],
    )
    parser.add_argument("--c-puct", type=float, default=2.0)
    parser.add_argument("--prior-uniform-mix", type=float, default=0.0)
    parser.add_argument(
        "--prior-mix-mode",
        choices=("local_action", "terminal_path_mass"),
        default="local_action",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if not 0.0 <= args.prior_uniform_mix <= 1.0:
        raise ValueError("prior-uniform-mix must be in [0,1]")

    graph = read_json(args.graph)
    reward_file = read_json(args.reward_centres)
    base_utility = {
        item["path_id"]: float(item["overall_preference_score"])
        for item in reward_file["path_rewards"]
    }
    space = PathSpace.from_graph(graph)
    if set(base_utility) != set(space.paths):
        raise ValueError("reward centres must exactly match observed graph paths")
    path_ids = sorted(space.paths)
    max_budget = max(args.budgets)
    frequency_greedy = max(
        path_ids,
        key=lambda path_id: (
            space.support[path_id],
            base_utility[path_id],
            path_id,
        ),
    )

    scenarios = []
    for epistemic_index, epistemic_sigma in enumerate(args.epistemic_sigmas):
        for rollout_index, rollout_sigma in enumerate(args.rollout_sigmas):
            scenario_seed = (
                args.seed
                + 1_000_003 * epistemic_index
                + 10_007 * rollout_index
            )
            methods = {
                budget: {
                    method: RunningMethodStats(path_ids)
                    for method in (
                        "frequency_greedy",
                        "random_valid",
                        "uniform_allocation_q",
                        "mcts_visit",
                        "mcts_terminal_q",
                    )
                }
                for budget in args.budgets
            }
            oracle_counts = Counter({path_id: 0 for path_id in path_ids})
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
                    key=lambda path_id: (
                        true_utility[path_id],
                        space.support[path_id],
                        path_id,
                    ),
                )
                oracle_counts[oracle_path] += 1
                streams = make_reward_streams(
                    path_ids,
                    true_utility,
                    rollout_sigma,
                    max_budget,
                    episode_seed + 17,
                )
                recommendations, final_visits, trace = run_mcts_episode(
                    space,
                    streams,
                    args.budgets,
                    args.c_puct,
                    args.prior_uniform_mix,
                    args.prior_mix_mode,
                    episode_seed + 31,
                    save_trace=episode_index == 0,
                )
                random_path = random.Random(episode_seed + 47).choice(path_ids)

                for budget in args.budgets:
                    uniform_path = uniform_allocation_recommendation(
                        path_ids,
                        streams,
                        budget,
                        space.support,
                        episode_seed + 59,
                    )
                    selections = {
                        "frequency_greedy": frequency_greedy,
                        "random_valid": random_path,
                        "uniform_allocation_q": uniform_path,
                        **recommendations[budget],
                    }
                    for method, selected_path in selections.items():
                        methods[budget][method].add(
                            selected_path,
                            true_utility,
                            oracle_path,
                        )
                if episode_index == 0:
                    audit_example = {
                        "episode": 1,
                        "simulator_only_true_utility": true_utility,
                        "oracle_path_id": oracle_path,
                        "reward_stream_prefix": {
                            path_id: streams[path_id][:5] for path_id in path_ids
                        },
                        "mcts_trace_at_max_budget": trace,
                        "recommendations": recommendations,
                        "final_terminal_visits": final_visits,
                    }

            scenarios.append(
                {
                    "scenario_id": (
                        f"epistemic_{epistemic_sigma:.3f}"
                        f"__rollout_{rollout_sigma:.3f}"
                    ),
                    "epistemic_sigma": epistemic_sigma,
                    "rollout_sigma": rollout_sigma,
                    "oracle_path_counts": dict(sorted(oracle_counts.items())),
                    "budgets": {
                        str(budget): {
                            method: stats.result()
                            for method, stats in methods[budget].items()
                        }
                        for budget in args.budgets
                    },
                    "audit_example": audit_example,
                }
            )

    output = {
        "schema_version": "dirs.stochastic_prefix_path_mcts_sim.v1",
        "status": "synthetic_uncertainty_simulation_not_fresh_llm_rollouts",
        "graph_id": graph["graph_hash_sha256"],
        "inputs": {
            "graph": str(args.graph),
            "reward_centres": str(args.reward_centres),
            "reward_centres_role": (
                "generative centres visible to the simulator, not the search policy"
            ),
        },
        "config": {
            "episodes_per_scenario": args.episodes,
            "budgets": args.budgets,
            "epistemic_sigmas": args.epistemic_sigmas,
            "rollout_sigmas": args.rollout_sigmas,
            "c_puct": args.c_puct,
            "prior_uniform_mix": args.prior_uniform_mix,
            "prior_mix_mode": args.prior_mix_mode,
            "effective_prior": {
                "local_action": (
                    "(1-mix) * empirical continuation frequency "
                    "+ mix * uniform legal-frontier action prior"
                ),
                "terminal_path_mass": (
                    "conditional descendant mass after mixing empirical "
                    "complete-path support with uniform complete-path mass"
                ),
            }[args.prior_mix_mode],
            "seed": args.seed,
            "reward_model": "Gaussian perturbation clipped to [0,1]",
            "paired_reward_streams": True,
        },
        "search_space": {
            "observed_complete_paths_only": True,
            "path_count": len(path_ids),
            "state_is_complete_prefix": True,
            "action_is_legal_next_node": True,
            "arbitrary_node_selection": False,
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
        "frequency_greedy_path_id": frequency_greedy,
        "scenarios": scenarios,
        "checks": {
            "all_path_ids_match": set(base_utility) == set(space.paths),
            "every_action_generated_from_legal_frontier": True,
            "every_terminal_is_an_observed_complete_path": True,
            "search_policy_never_reads_simulator_true_utility": True,
        },
        "claim_boundary": [
            "This is a stochastic Monte Carlo audit, not a fresh multi-LLM run.",
            "The uncertainty scales are sensitivity settings, not learned estimates.",
            "The reward centres come from one earlier blind evaluator.",
            "The simulation tests search behavior, not abstract-writing quality.",
            "A real online run must replace sampled scalar rewards with independently generated and evaluated artifacts.",
        ],
    }
    write_json(args.output, output)


if __name__ == "__main__":
    main()
