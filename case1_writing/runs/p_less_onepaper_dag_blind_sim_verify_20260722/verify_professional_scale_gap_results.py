#!/usr/bin/env python3
"""Verifier for professional-scale p-less author simulation artifacts.

This verifier is separate from exact paper reproduction. It asks whether the
running/finished professional-scale author simulation has enough real evidence
to support the paper's research-gap shape:

- p-less/p-lessnorm top or near-top on reasoning AUC-style summaries
- p-less/p-lessnorm stable at high temperature
- p-less near fastest in sampler/full-generation timing channels

Reduced/smoke/preflight outputs never pass this verifier.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = RUN_DIR / "professional_scale_author_simulation" / "longrun_20260722_gpu3_interleaved"
OUTPUT_JSON = RUN_DIR / "professional_scale_gap_result_verification.json"
OUTPUT_MD = RUN_DIR / "PROFESSIONAL_SCALE_GAP_RESULT_VERIFICATION.md"


REASONING_DATASETS = {"gsm8k", "csqa", "qasc"}
PREFERRED_METHODS = {"p_less", "p_lessnorm"}
FRAGILE_BASELINES = {"top_p", "epsilon", "eta"}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def aggregate_scores(scores: list[dict]) -> dict:
    cell = defaultdict(lambda: {"n": 0, "correct": 0, "usable": 0})
    prompt_sets = defaultdict(set)
    for row in scores:
        key = (row["dataset"], str(row["temperature"]), row["sampler"])
        cell[key]["n"] += 1
        cell[key]["correct"] += int(bool(row.get("correct")))
        cell[key]["usable"] += int(bool(row.get("usable_story_proxy")))
        prompt_sets[row["dataset"]].add(row["prompt_idx"])

    cell_rows = []
    for (dataset, temperature, sampler), row in cell.items():
        if dataset == "writingprompts":
            metric = "usable_story_proxy_rate"
            value = row["usable"] / max(1, row["n"])
        else:
            metric = "accuracy"
            value = row["correct"] / max(1, row["n"])
        cell_rows.append({"dataset": dataset, "temperature": float(temperature), "sampler": sampler, "n": row["n"], "metric": metric, "value": value})

    auc_rows = []
    grouped = defaultdict(list)
    for row in cell_rows:
        if row["dataset"] in REASONING_DATASETS:
            grouped[(row["dataset"], row["sampler"])].append(row["value"])
    for (dataset, sampler), values in grouped.items():
        auc_rows.append({"dataset": dataset, "sampler": sampler, "mean_accuracy_over_temperatures": sum(values) / len(values), "temperature_count": len(values)})

    return {
        "cell_rows": sorted(cell_rows, key=lambda r: (r["dataset"], r["temperature"], r["sampler"])),
        "auc_rows": sorted(auc_rows, key=lambda r: (r["dataset"], r["sampler"])),
        "prompt_counts_by_dataset": {dataset: len(values) for dataset, values in sorted(prompt_sets.items())},
    }


def aggregate_timing(timing: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in timing:
        grouped[(row.get("dataset"), row.get("sampler"))].append(float(row.get("sample_seconds", 0.0)))
    out = []
    for (dataset, sampler), values in grouped.items():
        if not values:
            continue
        out.append({"dataset": dataset, "sampler": sampler, "mean_sample_seconds": statistics.mean(values), "n": len(values)})
    return sorted(out, key=lambda r: (str(r["dataset"]), r["mean_sample_seconds"]))


def rank_methods(rows: list[dict], value_key: str, higher_is_better: bool = True) -> dict[str, list[str]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    rankings = {}
    for dataset, values in grouped.items():
        ordered = sorted(values, key=lambda r: r[value_key], reverse=higher_is_better)
        rankings[dataset] = [row["sampler"] for row in ordered]
    return rankings


def near_top(rankings: dict[str, list[str]], max_rank: int = 3) -> tuple[float, dict]:
    details = {}
    ok = 0
    total = 0
    for dataset, ranking in rankings.items():
        if dataset not in REASONING_DATASETS:
            continue
        total += 1
        best_rank = min((ranking.index(method) + 1 for method in PREFERRED_METHODS if method in ranking), default=999)
        passed = best_rank <= max_rank
        ok += int(passed)
        details[dataset] = {"ranking": ranking, "best_p_less_rank": best_rank, "pass": passed}
    return ok / max(1, total), details


def high_temp_writing_check(cell_rows: list[dict], min_n: int) -> dict:
    rows = [row for row in cell_rows if row["dataset"] == "writingprompts" and float(row["temperature"]) >= 1.5 and row["n"] >= min_n]
    if not rows:
        return {"status": "pending", "reason": "no sufficient high-temperature writing cells yet"}
    by_temp = defaultdict(dict)
    for row in rows:
        by_temp[row["temperature"]][row["sampler"]] = row["value"]
    passed = 0
    total = 0
    details = {}
    for temperature, values in by_temp.items():
        if not all(method in values for method in PREFERRED_METHODS):
            continue
        total += 1
        p_best = max(values[method] for method in PREFERRED_METHODS)
        fragile_best = max((values.get(method, 0.0) for method in FRAGILE_BASELINES), default=0.0)
        ok = p_best >= fragile_best
        passed += int(ok)
        details[str(temperature)] = {"p_less_family_best": p_best, "fragile_baseline_best": fragile_best, "pass": ok, "values": values}
    if total == 0:
        return {"status": "pending", "reason": "high-temperature writing cells incomplete"}
    return {"status": "pass" if passed / total >= 0.5 else "fail", "pass_rate": passed / total, "details": details}


def timing_check(timing_rows: list[dict], min_token_rows: int) -> dict:
    rows = aggregate_timing(timing_rows)
    enough = [row for row in rows if row["n"] >= min_token_rows]
    if not enough:
        return {"status": "pending", "reason": "not enough timing rows per sampler yet", "rows": rows[:20]}
    grouped = defaultdict(list)
    for row in enough:
        grouped[row["dataset"]].append(row)
    passed = 0
    total = 0
    details = {}
    for dataset, values in grouped.items():
        ranking = [row["sampler"] for row in sorted(values, key=lambda r: r["mean_sample_seconds"])]
        best_rank = min((ranking.index(method) + 1 for method in PREFERRED_METHODS if method in ranking), default=999)
        ok = best_rank <= 3
        passed += int(ok)
        total += 1
        details[str(dataset)] = {"ranking": ranking, "best_p_less_rank": best_rank, "pass": ok}
    return {"status": "pass" if total and passed / total >= 0.5 else "fail", "pass_rate": passed / max(1, total), "details": details}


def verify(run_dir: Path, min_prompts_per_reasoning_dataset: int, min_writing_prompts: int, min_timing_rows: int) -> dict:
    status_path = run_dir / "professional_scale_status.json"
    manifest_path = run_dir / "professional_scale_manifest.json"
    scores = load_jsonl(run_dir / "scores.jsonl")
    timing = load_jsonl(run_dir / "sampling_time_by_token.jsonl")
    status = load_json(status_path) if status_path.exists() else {}
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    agg = aggregate_scores(scores)
    auc_rankings = rank_methods(agg["auc_rows"], "mean_accuracy_over_temperatures", higher_is_better=True)
    auc_pass_rate, auc_details = near_top(auc_rankings)
    prompt_counts = agg["prompt_counts_by_dataset"]
    reasoning_ready = all(prompt_counts.get(dataset, 0) >= min_prompts_per_reasoning_dataset for dataset in REASONING_DATASETS)
    writing_ready = prompt_counts.get("writingprompts", 0) >= min_writing_prompts
    high_temp = high_temp_writing_check(agg["cell_rows"], min_writing_prompts)
    timing = timing_check(timing, min_timing_rows)
    professional_plan = bool(status.get("professional_scale_for_gap_convergence") or manifest.get("professional_scale_for_gap_convergence"))

    checks = [
        {
            "name": "not_reduced_or_smoke",
            "status": "pass" if professional_plan else "fail",
            "detail": {"scale": status.get("scale") or manifest.get("scale"), "manifest_scale_gate": manifest.get("scale_gate")},
        },
        {
            "name": "reasoning_coverage_ready",
            "status": "pass" if reasoning_ready else "pending",
            "detail": {"prompt_counts_by_dataset": prompt_counts, "minimum": min_prompts_per_reasoning_dataset},
        },
        {
            "name": "reasoning_auc_close_shape",
            "status": "pass" if reasoning_ready and auc_pass_rate >= 2 / 3 else ("pending" if not reasoning_ready else "fail"),
            "detail": {"pass_rate": auc_pass_rate, "details": auc_details},
        },
        {
            "name": "writing_coverage_ready",
            "status": "pass" if writing_ready else "pending",
            "detail": {"prompt_counts_by_dataset": prompt_counts, "minimum": min_writing_prompts},
        },
        {
            "name": "high_temperature_writing_close_shape",
            "status": high_temp["status"] if writing_ready else "pending",
            "detail": high_temp,
        },
        {
            "name": "timing_close_shape",
            "status": timing["status"],
            "detail": timing,
        },
    ]
    accepted = all(check["status"] == "pass" for check in checks)
    return {
        "created_at_utc": now_utc(),
        "run_dir": str(run_dir),
        "accepted_professional_gap_close_match": accepted,
        "status": "accepted" if accepted else "running_or_waiting_for_more_evidence",
        "runner_status": status,
        "checks": checks,
        "score_summary": agg,
    }


def write_report(result: dict) -> None:
    lines = [
        "# Professional-Scale Gap Result Verification",
        "",
        f"Date: {result['created_at_utc']}",
        "",
        f"- Accepted professional gap close match: `{str(result['accepted_professional_gap_close_match']).lower()}`",
        f"- Status: `{result['status']}`",
        f"- Run dir: `{result['run_dir']}`",
        "",
        "## Checks",
        "",
    ]
    for check in result["checks"]:
        lines.append(f"- `{check['name']}`: `{check['status']}`")
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--min-prompts-per-reasoning-dataset", type=int, default=25)
    parser.add_argument("--min-writing-prompts", type=int, default=25)
    parser.add_argument("--min-timing-rows", type=int, default=25 * 64)
    args = parser.parse_args()
    result = verify(Path(args.run_dir), args.min_prompts_per_reasoning_dataset, args.min_writing_prompts, args.min_timing_rows)
    write_json(OUTPUT_JSON, result)
    write_report(result)
    print(json.dumps({k: result[k] for k in ["status", "accepted_professional_gap_close_match", "created_at_utc"]}, indent=2))


if __name__ == "__main__":
    main()
