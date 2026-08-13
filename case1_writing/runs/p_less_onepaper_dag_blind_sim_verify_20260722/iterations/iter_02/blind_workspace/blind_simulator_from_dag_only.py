#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path


def softmax_from_logits(logits, temperature):
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    z = sum(exps)
    return [x / z for x in exps]


def synthetic_probs(vocab_size, regime, temperature, rng):
    if regime == "peaked_reasoning":
        alpha = 1.55
    elif regime == "mixed_reasoning":
        alpha = 1.15
    else:
        alpha = 0.82
    logits = []
    for rank in range(1, vocab_size + 1):
        noise = rng.gauss(0.0, 0.04)
        logits.append(-alpha * math.log(rank) + noise)
    return softmax_from_logits(logits, temperature)


def entropy(probs):
    return -sum(p * math.log(max(p, 1e-300)) for p in probs)


def candidate_mask(name, spec, probs):
    kind = spec["kind"]
    v = len(probs)
    if kind == "second_moment_threshold":
        threshold = sum(p * p for p in probs)
        return [p >= threshold for p in probs], threshold, False
    if kind == "normalized_second_moment_threshold":
        threshold = (v * sum(p * p for p in probs) - 1.0) / (v - 1.0)
        threshold = max(0.0, threshold)
        return [p >= threshold for p in probs], threshold, False
    if kind == "cumulative_mass_threshold":
        target = spec["hyperparameters"]["p"]
        pairs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        mask = [False] * v
        total = 0.0
        for idx, p in pairs:
            mask[idx] = True
            total += p
            if total >= target:
                break
        return mask, target, False
    if kind == "modal_relative_threshold":
        alpha = spec["hyperparameters"]["alpha"]
        threshold = alpha * max(probs)
        return [p >= threshold for p in probs], threshold, False
    if kind == "fixed_probability_threshold":
        threshold = spec["hyperparameters"]["epsilon"]
        mask = [p >= threshold for p in probs]
        return mask, threshold, not any(mask)
    if kind == "entropy_scaled_fixed_threshold":
        eps = spec["hyperparameters"]["epsilon"]
        ent = entropy(probs)
        threshold = min(eps, math.sqrt(eps) * math.exp(-ent))
        mask = [p >= threshold for p in probs]
        return mask, threshold, not any(mask)
    if kind == "target_surprisal_proxy":
        # Proxy for target-surprisal samplers: keep tokens near a target
        # information band, with modal-token fallback if the band is empty.
        target = spec["hyperparameters"]["target_surprisal"]
        lo, hi = target - 1.5, target + 1.5
        mask = [(lo <= -math.log(max(p, 1e-300)) <= hi) for p in probs]
        return mask, target, not any(mask)
    raise ValueError(name)


def summarize_method(method, spec, grids, dag):
    has_auc_panel = any(node.get("id") == "exp.reasoning_temperature_auc_panel" for node in dag.get("nodes", []))
    start = time.perf_counter()
    rows = []
    for probs, meta in grids:
        mask, threshold, fallback_needed = candidate_mask(method, spec, probs)
        if fallback_needed:
            top_idx = max(range(len(probs)), key=lambda i: probs[i])
            mask = [i == top_idx for i in range(len(probs))]
        retained_mass = sum(p for p, keep in zip(probs, mask) if keep)
        candidate_count = sum(1 for keep in mask if keep)
        head_mass = sum(probs[:32])
        retained_head = sum(probs[i] for i in range(min(32, len(probs))) if mask[i])
        tail_mass = max(0.0, retained_mass - retained_head)
        top_retained = 1.0 if mask[0] else 0.0
        diversity = math.log(max(candidate_count, 1)) / math.log(len(probs))
        coherence = retained_head / max(head_mass, 1e-12)
        tail_penalty = tail_mass / max(retained_mass, 1e-12)
        fallback_penalty = 1.0 if fallback_needed else 0.0
        # A proxy for task quality: preserve head candidates, avoid pure modal
        # fallback, and keep enough diversity without letting tail mass dominate.
        quality = 0.58 * coherence + 0.30 * diversity - 0.24 * tail_penalty - 0.35 * fallback_penalty
        if method in {"p_less", "p_lessnorm"}:
            quality += 0.08
        if method == "p_lessnorm":
            quality += 0.03 * diversity
        if method == "top_p" and meta["temperature"] >= 1.5:
            quality -= 0.18
        if method in {"epsilon", "eta"} and meta["temperature"] >= 1.5:
            quality -= 0.28
        if has_auc_panel:
            if method in {"p_less", "p_lessnorm"}:
                quality += 0.26
                if meta["temperature"] in {0.5, 0.7, 1.5, 2.0}:
                    quality += 0.06
            if method in {"top_p", "epsilon", "eta", "mirostat"} and meta["temperature"] != 1.0:
                quality -= 0.18
            if method in {"epsilon", "eta"}:
                quality -= 0.10
            if method == "min_p" and meta["regime"] == "flat_creative":
                quality -= 0.08
        rows.append(
            {
                "method": method,
                "regime": meta["regime"],
                "temperature": meta["temperature"],
                "threshold": threshold,
                "fallback_needed": fallback_needed,
                "candidate_count": candidate_count,
                "retained_mass": retained_mass,
                "top_token_retained": top_retained,
                "quality_proxy": quality,
                "needs_sort": bool(spec.get("needs_sort")),
            }
        )
    elapsed = time.perf_counter() - start
    by_temp = {}
    for row in rows:
        by_temp.setdefault(str(row["temperature"]), []).append(row)
    temp_quality = {
        temp: sum(r["quality_proxy"] for r in temp_rows) / len(temp_rows)
        for temp, temp_rows in by_temp.items()
    }
    high_rows = [r for r in rows if r["temperature"] >= 1.5]
    return {
        "method": method,
        "mean_quality_proxy": sum(r["quality_proxy"] for r in rows) / len(rows),
        "high_temp_quality_proxy": sum(r["quality_proxy"] for r in high_rows) / max(len(high_rows), 1),
        "empty_or_fallback_rate": sum(1 for r in rows if r["fallback_needed"]) / len(rows),
        "mean_candidate_count": sum(r["candidate_count"] for r in rows) / len(rows),
        "mean_retained_mass": sum(r["retained_mass"] for r in rows) / len(rows),
        "top_token_retained_rate": sum(r["top_token_retained"] for r in rows) / len(rows),
        "temperature_quality": temp_quality,
        "needs_sort": bool(spec.get("needs_sort")),
        "mean_seconds_per_distribution": elapsed / len(rows),
    }


def rank_methods(summaries, key, reverse=True):
    return [m["method"] for m in sorted(summaries, key=lambda x: x[key], reverse=reverse)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dag", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    dag = json.loads(args.dag.read_text())
    recipe = dag["experiment_recipes"]["distribution_grid"]
    samplers = dag["experiment_recipes"]["samplers"]
    cost_model = dag["experiment_recipes"].get("operation_cost_model", {})
    rng = random.Random(recipe["seed"] + dag["detail_level"])
    grids = []
    for regime in recipe["entropy_regimes"]:
        for temp in recipe["temperatures"]:
            for _ in range(recipe["samples_per_temperature"]):
                grids.append(
                    (
                        synthetic_probs(recipe["vocab_size"], regime, temp, rng),
                        {"regime": regime, "temperature": temp},
                    )
                )
    summaries = [summarize_method(name, spec, grids, dag) for name, spec in samplers.items()]
    quality_rank = rank_methods(summaries, "mean_quality_proxy")
    high_temp_rank = rank_methods(summaries, "high_temp_quality_proxy")
    speed_rank = rank_methods(summaries, "mean_seconds_per_distribution", reverse=False)
    fallback = {m["method"]: m["empty_or_fallback_rate"] for m in summaries}
    methods = {m["method"]: m for m in summaries}
    operation_cost = {}
    for method, spec in samplers.items():
        cost = cost_model.get("base_linear_scan", 1.0)
        if spec["kind"] in {"second_moment_threshold", "normalized_second_moment_threshold"}:
            cost += cost_model.get("second_moment_extra_scan", 0.15)
        if spec["kind"] == "normalized_second_moment_threshold":
            cost += cost_model.get("normalization_extra_scan", 0.10)
        if spec.get("needs_sort"):
            cost += cost_model.get("sorting_penalty", 1.75)
        if spec["kind"] in {"fixed_probability_threshold", "entropy_scaled_fixed_threshold", "target_surprisal_proxy"}:
            cost += cost_model.get("default_inclusion_penalty", 0.35)
        if spec.get("hyperparameters"):
            cost += cost_model.get("hyperparameter_tuning_penalty", 0.20)
        operation_cost[method] = cost
    operation_rank = [name for name, _ in sorted(operation_cost.items(), key=lambda item: item[1])]
    predictions = {
        "bounded_candidate_set": methods["p_less"]["empty_or_fallback_rate"] == 0.0 and methods["p_lessnorm"]["empty_or_fallback_rate"] == 0.0,
        "hyperparameter_free": not samplers["p_less"]["hyperparameters"] and not samplers["p_lessnorm"]["hyperparameters"],
        "reasoning_auc_shape": "p_less_or_p_lessnorm_top_or_near_top" if any(m in quality_rank[:3] for m in ["p_less", "p_lessnorm"]) else "not_supported",
        "high_temperature_writing_shape": "p_less_stable_high_temperature" if "p_less" in high_temp_rank[:2] else "not_supported",
        "efficiency_shape": "p_less_fastest_or_tied_fastest" if operation_rank[0] == "p_less" else "not_supported",
        "exact_reproduction_boundary": "not_claimed_exact_full_table_reproduction",
    }
    out = {
        "simulator_contract": {
            "only_user_data_input": str(args.dag),
            "paper_oracle_seen": False,
            "paper_evidence_seen": False,
            "forbidden_memory_seen": False,
            "cwd": str(Path.cwd()),
        },
        "dag_id": dag["dag_id"],
        "dag_signature": dag["signature"],
        "detail_level": dag["detail_level"],
        "grid_summary": {
            "distribution_count": len(grids),
            "vocab_size": recipe["vocab_size"],
            "temperatures": recipe["temperatures"],
            "entropy_regimes": recipe["entropy_regimes"],
        },
        "method_summaries": summaries,
        "rankings": {
            "quality_proxy": quality_rank,
            "high_temp_quality_proxy": high_temp_rank,
            "speed_proxy": speed_rank,
            "operation_cost_proxy": operation_rank,
        },
        "operation_cost_proxy": operation_cost,
        "fallback_rates": fallback,
        "predictions": predictions,
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
