#!/usr/bin/env python3
"""Audit whether saved real writing traces identify prefix value."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROW_PATTERN = re.compile(
    r"^\|\s*(?P<index>\d+)\s*"
    r"\|\s*`(?P<chip>[^`]+)`\s*"
    r"\|\s*(?P<initial>\d+(?:\.\d+)?)\s*"
    r"\|\s*(?P<final>\d+(?:\.\d+)?)\s*"
    r"\|\s*(?P<unsupported>\d+)\s*"
    r"\|"
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    centre = mean(values)
    return math.sqrt(
        sum((value - centre) ** 2 for value in values) / (len(values) - 1)
    )


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("paired nonempty vectors required")
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss == 0 or right_ss == 0:
        return 0.0
    return numerator / math.sqrt(left_ss * right_ss)


def shared_prefix_length(left: list[str], right: list[str]) -> int:
    length = 0
    for left_node, right_node in zip(left, right):
        if left_node != right_node:
            break
        length += 1
    return length


def two_sided_permutation_pvalue(
    observed: float,
    statistic,
    values: list[float],
    permutations: int,
    rng: random.Random,
) -> float:
    extreme = 0
    shuffled = list(values)
    for _ in range(permutations):
        rng.shuffle(shuffled)
        candidate = statistic(shuffled)
        if abs(candidate) >= abs(observed) - 1e-15:
            extreme += 1
    return (extreme + 1) / (permutations + 1)


def extract_scores(status_files: list[Path]) -> dict[str, dict[str, Any]]:
    scores = {}
    for batch_index, path in enumerate(status_files, start=1):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = ROW_PATTERN.match(line)
            if not match:
                continue
            record = match.groupdict()
            chip_id = record["chip"]
            if chip_id in scores:
                raise ValueError(f"duplicate score row: {chip_id}")
            scores[chip_id] = {
                "batch": batch_index,
                "sample_index": int(record["index"]),
                "initial_score": float(record["initial"]),
                "final_score": float(record["final"]),
                "unsupported_claims_after_repair": int(record["unsupported"]),
                "score_source": str(path),
            }
    return scores


def prediction_errors(
    records: list[dict[str, Any]],
    score_key: str,
    exclude_same_batch: bool,
) -> dict[str, Any]:
    predictions = []
    global_absolute_errors = []
    path_absolute_errors = []
    global_squared_errors = []
    path_squared_errors = []
    fallback_count = 0
    for index, record in enumerate(records):
        other = [
            candidate
            for offset, candidate in enumerate(records)
            if offset != index
            and (
                not exclude_same_batch
                or candidate["batch"] != record["batch"]
            )
        ]
        if not other:
            raise ValueError("prediction fold has no training cases")
        global_prediction = mean([candidate[score_key] for candidate in other])
        same_path = [
            candidate[score_key]
            for candidate in other
            if candidate["path_id"] == record["path_id"]
        ]
        if same_path:
            path_prediction = mean(same_path)
            used_fallback = False
        else:
            path_prediction = global_prediction
            used_fallback = True
            fallback_count += 1
        global_error = global_prediction - record[score_key]
        path_error = path_prediction - record[score_key]
        global_absolute_errors.append(abs(global_error))
        path_absolute_errors.append(abs(path_error))
        global_squared_errors.append(global_error * global_error)
        path_squared_errors.append(path_error * path_error)
        predictions.append(
            {
                "chip_id": record["chip_id"],
                "batch": record["batch"],
                "path_id": record["path_id"],
                "observed_score": record[score_key],
                "global_prediction": global_prediction,
                "path_prediction": path_prediction,
                "path_used_global_fallback": used_fallback,
            }
        )
    return {
        "score_key": score_key,
        "exclusion": (
            "same_batch_and_same_case"
            if exclude_same_batch
            else "same_case_only"
        ),
        "global_mean_mae": mean(global_absolute_errors),
        "path_mean_mae": mean(path_absolute_errors),
        "global_mean_mse": mean(global_squared_errors),
        "path_mean_mse": mean(path_squared_errors),
        "path_model_mae_improvement": (
            mean(global_absolute_errors) - mean(path_absolute_errors)
        ),
        "path_model_mse_improvement": (
            mean(global_squared_errors) - mean(path_squared_errors)
        ),
        "paired_absolute_error_improvements": [
            global_error - path_error
            for global_error, path_error in zip(
                global_absolute_errors,
                path_absolute_errors,
            )
        ],
        "global_fallback_count_for_missing_path": fallback_count,
        "predictions": predictions,
    }


def bootstrap_mean_interval(
    values: list[float],
    draws: int,
    rng: random.Random,
) -> list[float]:
    bootstrap_means = []
    for _ in range(draws):
        bootstrap_means.append(
            mean([rng.choice(values) for _ in range(len(values))])
        )
    bootstrap_means.sort()
    lower_index = int(0.025 * (draws - 1))
    upper_index = int(0.975 * (draws - 1))
    return [bootstrap_means[lower_index], bootstrap_means[upper_index]]


def sign_flip_pvalue_positive(
    values: list[float],
    draws: int,
    rng: random.Random,
) -> float:
    observed = mean(values)
    count = 0
    for _ in range(draws):
        candidate = mean(
            [value if rng.random() < 0.5 else -value for value in values]
        )
        if candidate >= observed - 1e-15:
            count += 1
    return (count + 1) / (draws + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-trace", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--status-dir", required=True, type=Path)
    parser.add_argument("--dataset-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    if args.permutations < 1:
        raise ValueError("permutations must be positive")

    traces = read_json(args.training_trace)
    graph = read_json(args.graph)
    status_files = sorted(args.status_dir.glob("batch*_status_*.md"))
    if len(status_files) != 5:
        raise ValueError(f"expected five batch status files, found {len(status_files)}")
    scores = extract_scores(status_files)
    path_lookup = {
        tuple(record["nodes"]): record["path_id"]
        for record in graph["observed_paths"]
    }
    records = []
    for trace in traces:
        chip_id = trace["chip_id"]
        nodes = tuple(trace["selected_nodes"])
        if chip_id not in scores:
            raise ValueError(f"missing score for {chip_id}")
        if nodes not in path_lookup:
            raise ValueError(f"trace path not found in learned graph: {chip_id}")
        score = scores[chip_id]
        records.append(
            {
                "chip_id": chip_id,
                "title": trace["title"],
                "venue": trace["venue"],
                "path_id": path_lookup[nodes],
                "selected_nodes": list(nodes),
                "abstract_word_count": trace["abstract_word_count"],
                **score,
                "repair_gain": score["final_score"] - score["initial_score"],
            }
        )
    records.sort(key=lambda item: item["sample_index"])
    if len(records) != 19 or set(scores) != {record["chip_id"] for record in records}:
        raise ValueError("expected an exact 19-case score/trace join")
    write_json(
        args.dataset_output,
        {
            "schema_version": "dirs.real_trace_score_dataset.v1",
            "source_type": "saved_training_subagent_status_tables",
            "record_count": len(records),
            "records": records,
        },
    )

    final_scores = [record["final_score"] for record in records]
    initial_scores = [record["initial_score"] for record in records]
    repair_gains = [record["repair_gain"] for record in records]
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_path[record["path_id"]].append(record)
    path_summary = []
    for path_id, path_records in sorted(by_path.items()):
        values = [record["final_score"] for record in path_records]
        gains = [record["repair_gain"] for record in path_records]
        path_summary.append(
            {
                "path_id": path_id,
                "support": len(path_records),
                "mean_final_score": mean(values),
                "final_score_std": sample_std(values),
                "mean_repair_gain": mean(gains),
                "chip_ids": [record["chip_id"] for record in path_records],
            }
        )

    final_loo = prediction_errors(records, "final_score", False)
    final_batch_out = prediction_errors(records, "final_score", True)
    initial_loo = prediction_errors(records, "initial_score", False)
    robustness_rng = random.Random(args.seed + 313)
    for analysis in (final_loo, final_batch_out, initial_loo):
        improvements = analysis.pop("paired_absolute_error_improvements")
        analysis["paired_mae_improvement_bootstrap_95pct_interval"] = (
            bootstrap_mean_interval(
                improvements,
                args.permutations,
                robustness_rng,
            )
        )
        analysis["paired_mae_improvement_sign_flip_pvalue_one_sided"] = (
            sign_flip_pvalue_positive(
                improvements,
                args.permutations,
                robustness_rng,
            )
        )

    pair_prefix_lengths = []
    pair_score_similarities = []
    pair_indices = []
    for left_index in range(len(records)):
        for right_index in range(left_index + 1, len(records)):
            left = records[left_index]
            right = records[right_index]
            prefix_length = shared_prefix_length(
                left["selected_nodes"],
                right["selected_nodes"],
            )
            pair_prefix_lengths.append(float(prefix_length))
            pair_score_similarities.append(
                -abs(left["final_score"] - right["final_score"])
            )
            pair_indices.append((left_index, right_index))
    observed_prefix_correlation = pearson(
        pair_prefix_lengths,
        pair_score_similarities,
    )
    permutation_rng = random.Random(args.seed)

    def prefix_statistic(permuted_scores: list[float]) -> float:
        similarities = [
            -abs(permuted_scores[left] - permuted_scores[right])
            for left, right in pair_indices
        ]
        return pearson(pair_prefix_lengths, similarities)

    prefix_pvalue = two_sided_permutation_pvalue(
        observed_prefix_correlation,
        prefix_statistic,
        final_scores,
        args.permutations,
        permutation_rng,
    )

    node_effects = []
    for node in ("C1_domain_context", "M2_efficiency_or_theory_detail", "E3_quantitative_anchor"):
        present_mask = [node in record["selected_nodes"] for record in records]
        present = [
            record["final_score"]
            for record, is_present in zip(records, present_mask)
            if is_present
        ]
        absent = [
            record["final_score"]
            for record, is_present in zip(records, present_mask)
            if not is_present
        ]
        observed_difference = mean(present) - mean(absent)

        def node_statistic(permuted_scores: list[float]) -> float:
            permuted_present = [
                score
                for score, is_present in zip(permuted_scores, present_mask)
                if is_present
            ]
            permuted_absent = [
                score
                for score, is_present in zip(permuted_scores, present_mask)
                if not is_present
            ]
            return mean(permuted_present) - mean(permuted_absent)

        node_effects.append(
            {
                "node": node,
                "present_count": len(present),
                "absent_count": len(absent),
                "mean_present_final_score": mean(present),
                "mean_absent_final_score": mean(absent),
                "observed_mean_difference": observed_difference,
                "permutation_pvalue_two_sided": two_sided_permutation_pvalue(
                    observed_difference,
                    node_statistic,
                    final_scores,
                    args.permutations,
                    permutation_rng,
                ),
                "causal_interpretation_allowed": False,
            }
        )

    report = {
        "schema_version": "dirs.real_trace_prefix_predictiveness_audit.v1",
        "status": "observational_in_sample_audit",
        "config": {
            "permutations": args.permutations,
            "seed": args.seed,
            "holdout_used": False,
        },
        "data_summary": {
            "paper_count": len(records),
            "unique_path_count": len(by_path),
            "path_support_min": min(len(value) for value in by_path.values()),
            "path_support_max": max(len(value) for value in by_path.values()),
            "initial_score_mean": mean(initial_scores),
            "initial_score_std": sample_std(initial_scores),
            "final_score_mean": mean(final_scores),
            "final_score_std": sample_std(final_scores),
            "final_score_min": min(final_scores),
            "final_score_max": max(final_scores),
            "final_scores_at_least_0_94": sum(value >= 0.94 for value in final_scores),
            "repair_gain_mean": mean(repair_gains),
            "repair_gain_std": sample_std(repair_gains),
            "unsupported_claims_after_repair_total": sum(
                record["unsupported_claims_after_repair"] for record in records
            ),
        },
        "path_groups": path_summary,
        "prediction_robustness": {
            "final_leave_one_case_out": final_loo,
            "final_leave_one_batch_out": final_batch_out,
            "initial_leave_one_case_out": initial_loo,
            "status": "post_hoc_observational_robustness_check",
        },
        "shared_prefix_analysis": {
            "paper_pair_count": len(pair_indices),
            "pearson_shared_prefix_vs_negative_absolute_score_difference": (
                observed_prefix_correlation
            ),
            "permutation_pvalue_two_sided": prefix_pvalue,
            "interpretation": (
                "positive means longer shared prefixes coincide with more similar scores"
            ),
        },
        "optional_node_associations": node_effects,
        "identifiability": {
            "same_paper_counterfactual_paths_per_case": 0,
            "repeated_rollouts_per_paper_path": 0,
            "independent_blind_evaluator_replicates": 0,
            "training_target_visible_during_repair": True,
            "within_paper_deception_rate_estimable": False,
            "causal_node_or_edge_value_estimable": False,
        },
        "adequacy_decision": {
            "enable_real_task_mcts_from_this_dataset": False,
            "reason": (
                "one realized path per paper, no repeated conditional rollouts, "
                "in-sample repaired scores, sparse path groups, and no "
                "within-paper counterfactual utility"
            ),
        },
    }
    write_json(args.report_output, report)


if __name__ == "__main__":
    main()
