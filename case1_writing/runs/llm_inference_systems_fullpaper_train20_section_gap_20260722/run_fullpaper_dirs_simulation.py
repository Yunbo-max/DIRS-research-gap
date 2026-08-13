#!/usr/bin/env python3
"""Local DIRS-style full-paper section gap simulation.

This harness is intentionally local/deterministic: it uses the already-built
20-paper evidence table, simulates a content system and action system, runs
connected-subgraph search, and writes convergence artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
DEFAULT_EVIDENCE = RUN_DIR / "paper_section_evidence_table.json"
DEFAULT_EXECUTION_PROBE = RUN_DIR / "real_execution_probe.json"
DEFAULT_GPU_MICROBENCH = RUN_DIR / "gpu_microbenchmark.json"
DEFAULT_REPRO_CAMPAIGN = RUN_DIR / "author_style_gpu_reproduction_campaign.json"


NODES = [
    ("root.fullpaper_gap_argument", "root", "Full paper gap argument unit."),
    ("intro.deployment_pressure", "introduction", "Why the systems constraint matters now."),
    ("intro.prior_family_map", "introduction", "Closest prior family and what it already solves."),
    ("intro.failure_mode", "introduction", "Exact failure under the target constraint."),
    ("intro.gap_claim", "introduction", "Bounded research gap claim."),
    ("intro.method_need", "introduction", "Why a new mechanism is needed."),
    ("related.closest_baselines", "related_work", "Closest strong baselines."),
    ("related.axis_of_difference", "related_work", "Narrow difference versus prior work."),
    ("related.near_miss_handling", "related_work", "Partial prior solutions and novelty boundary."),
    ("method.object_definition", "method", "The introduced method object."),
    ("method.mechanism_delta", "method", "Mechanism change from baseline."),
    ("method.constraint_binding", "method", "Design choices tied to gap constraint."),
    ("method.operational_steps", "method", "Executable pipeline or algorithm."),
    ("method.theory_or_invariant", "method", "Theorem, invariant, or objective when present."),
    ("experiments.axis_match", "experiments", "Experiment axes match the gap."),
    ("experiments.baseline_strength", "experiments", "Strong baseline family included."),
    ("experiments.metric_pairing", "experiments", "Efficiency paired with correctness or quality."),
    ("experiments.execution_surface", "experiments", "Declare local GPU, external API, repo run, or paper-only backend provenance."),
    ("experiments.gpu_hardware_profile", "experiments", "Record GPU model, CUDA/driver, memory, precision, and backend assumptions."),
    ("experiments.api_or_local_backend", "experiments", "Record whether results use external API calls, local serving, or no live backend."),
    ("experiments.real_benchmark_command", "experiments", "Persist exact runnable command or script, config, seed, batch/context, and artifact path."),
    ("experiments.exact_rerun_feasibility_audit", "experiments", "Audit every paper for exact-rerun readiness, code availability, missing checkpoints, data, and APIs."),
    ("experiments.author_style_reproduction_campaign", "experiments", "Run GPU stress tests with baselines, controls, repeated seeds, and domain-relevant motifs."),
    ("experiments.scale_or_stress", "experiments", "Stress setting or scale dimension."),
    ("experiments.ablation_or_control", "experiments", "Ablation or control tests the mechanism."),
    ("results.runtime_measurement", "results", "Report latency, throughput, memory, GPU occupancy, or API cost from the actual run."),
    ("results.measured_gap_derivation", "results", "Derive research gaps from measured tradeoff failures, not only from paper prose."),
    ("results.primary_table_read", "results", "Main result table answers the gap."),
    ("results.tradeoff_interpretation", "results", "Tradeoff interpretation."),
    ("results.reproduction_status", "results", "Classify evidence as rerun, microbench-only, code-inspected-only, API-only, or paper-only."),
    ("results.proxy_vs_exact_boundary", "results", "Separate exact reproduction, proxy experiment, code audit, and blocked-paper evidence."),
    ("results.exception_scan", "results", "Exceptions, saturation, or weak settings checked."),
    ("results.mechanism_attribution", "results", "Ablation/theory attributes the gain."),
    ("results.scope_boundary", "limitations", "Limitations bound the claim."),
    ("appendix.raw_measurement_table", "appendix", "Attach machine-readable raw rows and grouped statistics for every GPU experiment family."),
    ("appendix.artifact_log", "appendix", "Store probe JSON, benchmark JSON, logs, commit IDs, and commands used by the run."),
    ("appendix.reproducibility_or_extra_evidence", "appendix", "Appendix/code artifacts support trust."),
]

EDGES = [
    ("root.fullpaper_gap_argument", "intro.deployment_pressure"),
    ("intro.deployment_pressure", "intro.prior_family_map"),
    ("intro.prior_family_map", "intro.failure_mode"),
    ("intro.failure_mode", "intro.gap_claim"),
    ("intro.gap_claim", "intro.method_need"),
    ("intro.gap_claim", "related.closest_baselines"),
    ("related.closest_baselines", "related.axis_of_difference"),
    ("related.axis_of_difference", "related.near_miss_handling"),
    ("related.near_miss_handling", "intro.method_need"),
    ("intro.method_need", "method.object_definition"),
    ("method.object_definition", "method.mechanism_delta"),
    ("method.mechanism_delta", "method.constraint_binding"),
    ("method.constraint_binding", "method.operational_steps"),
    ("method.operational_steps", "method.theory_or_invariant"),
    ("method.constraint_binding", "experiments.axis_match"),
    ("experiments.axis_match", "experiments.baseline_strength"),
    ("experiments.baseline_strength", "experiments.metric_pairing"),
    ("experiments.metric_pairing", "experiments.execution_surface"),
    ("experiments.execution_surface", "experiments.gpu_hardware_profile"),
    ("experiments.execution_surface", "experiments.api_or_local_backend"),
    ("experiments.gpu_hardware_profile", "experiments.real_benchmark_command"),
    ("experiments.api_or_local_backend", "experiments.real_benchmark_command"),
    ("experiments.execution_surface", "experiments.exact_rerun_feasibility_audit"),
    ("experiments.exact_rerun_feasibility_audit", "experiments.author_style_reproduction_campaign"),
    ("experiments.real_benchmark_command", "experiments.author_style_reproduction_campaign"),
    ("experiments.author_style_reproduction_campaign", "results.runtime_measurement"),
    ("experiments.author_style_reproduction_campaign", "results.measured_gap_derivation"),
    ("experiments.real_benchmark_command", "experiments.scale_or_stress"),
    ("experiments.real_benchmark_command", "results.runtime_measurement"),
    ("experiments.metric_pairing", "experiments.scale_or_stress"),
    ("experiments.scale_or_stress", "experiments.ablation_or_control"),
    ("experiments.ablation_or_control", "results.primary_table_read"),
    ("results.runtime_measurement", "results.primary_table_read"),
    ("results.runtime_measurement", "results.reproduction_status"),
    ("results.measured_gap_derivation", "results.tradeoff_interpretation"),
    ("results.primary_table_read", "results.tradeoff_interpretation"),
    ("results.tradeoff_interpretation", "results.exception_scan"),
    ("results.reproduction_status", "results.proxy_vs_exact_boundary"),
    ("results.proxy_vs_exact_boundary", "results.exception_scan"),
    ("results.reproduction_status", "results.exception_scan"),
    ("results.exception_scan", "results.mechanism_attribution"),
    ("results.mechanism_attribution", "results.scope_boundary"),
    ("results.scope_boundary", "appendix.raw_measurement_table"),
    ("appendix.raw_measurement_table", "appendix.artifact_log"),
    ("results.scope_boundary", "appendix.artifact_log"),
    ("appendix.artifact_log", "appendix.reproducibility_or_extra_evidence"),
    ("results.scope_boundary", "appendix.reproducibility_or_extra_evidence"),
]

MANDATORY_NODES = {
    "root.fullpaper_gap_argument",
    "intro.deployment_pressure",
    "intro.prior_family_map",
    "intro.failure_mode",
    "intro.gap_claim",
    "intro.method_need",
    "method.object_definition",
    "method.mechanism_delta",
    "method.constraint_binding",
    "method.operational_steps",
    "experiments.axis_match",
    "experiments.baseline_strength",
    "experiments.metric_pairing",
    "experiments.execution_surface",
    "experiments.gpu_hardware_profile",
    "experiments.api_or_local_backend",
    "experiments.real_benchmark_command",
    "experiments.exact_rerun_feasibility_audit",
    "experiments.author_style_reproduction_campaign",
    "results.runtime_measurement",
    "results.measured_gap_derivation",
    "results.primary_table_read",
    "results.tradeoff_interpretation",
    "results.reproduction_status",
    "results.proxy_vs_exact_boundary",
    "results.exception_scan",
    "results.scope_boundary",
    "appendix.raw_measurement_table",
}

CONSTRAINT_TERMS = {
    "latency",
    "speed",
    "throughput",
    "memory",
    "gpu",
    "hardware",
    "token",
    "tokens",
    "cache",
    "kv",
    "decoding",
    "compression",
    "rate",
    "quality",
    "fidelity",
    "exact",
    "lossless",
    "correctness",
    "long-context",
    "long context",
    "sparse",
    "adaptive",
    "dynamic",
    "training-free",
    "distortion",
}

QUALITY_TERMS = {
    "accuracy",
    "quality",
    "fid",
    "lpips",
    "psnr",
    "ssim",
    "f1",
    "exact",
    "lossless",
    "correctness",
    "reward",
    "win",
    "clip",
    "perplexity",
    "fidelity",
}

EFFICIENCY_TERMS = {
    "speed",
    "latency",
    "throughput",
    "memory",
    "cache",
    "token",
    "tokens/sec",
    "gpu",
    "rate",
    "bpp",
    "flops",
    "cost",
    "compression",
}

GPU_TERMS = {
    "gpu",
    "cuda",
    "h100",
    "a100",
    "4090",
    "rtx",
    "v100",
    "l40",
    "driver",
    "hardware",
    "device",
    "memory",
    "vram",
}

API_TERMS = {
    "api",
    "openai",
    "gemini",
    "claude",
    "gpt-",
    "endpoint",
    "server",
    "serving",
    "hosted",
    "external",
    "local",
    "offline",
    "model_access",
}

EXECUTION_TERMS = {
    "benchmark",
    "microbenchmark",
    "runtime",
    "latency",
    "throughput",
    "tokens/sec",
    "memory",
    "code",
    "repo",
    "script",
    "command",
    "seed",
    "compileall",
    "rerun",
    "not_rerun",
    "checkpoint",
    "profile",
}


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stringify(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True).lower()


def has_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def listish(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def has_eval_baseline(paper: dict) -> bool:
    text = stringify(
        [
            paper.get("evaluation_evidence"),
            paper.get("experimental_setting_evidence"),
            paper.get("footprint"),
        ]
    )
    return "baseline" in text or "baselines" in text or "strong_baselines" in text


def has_metrics(paper: dict) -> bool:
    text = stringify([paper.get("evaluation_evidence"), paper.get("footprint")])
    return "metric" in text or has_any(text, QUALITY_TERMS | EFFICIENCY_TERMS)


def has_ablation(paper: dict) -> bool:
    text = stringify(
        [
            paper.get("evaluation_evidence"),
            paper.get("result_and_limitation_evidence"),
            paper.get("footprint"),
        ]
    )
    return "ablation" in text or "control" in text or "sensitivity" in text


def has_theory_or_invariant(paper: dict) -> bool:
    text = stringify([paper.get("method_evidence"), paper.get("result_and_limitation_evidence")])
    terms = {
        "lossless",
        "theorem",
        "proof",
        "objective",
        "equation",
        "optimal",
        "transport",
        "entropy",
        "rate-distortion",
        "renyi",
        "radix",
        "sensitivity",
        "invariant",
        "distribution",
        "exact",
    }
    return has_any(text, terms)


def has_structured_section(value: object) -> bool:
    if not value:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        if any(value.get(key) for key in ("summary", "central_gap", "core_method", "pin_summary", "claim")):
            return True
        if any(value.get(key) for key in ("nodes", "events", "mechanisms", "key_steps", "results", "main_results")):
            return True
        return any(has_structured_section(item) for item in value.values())
    return True


def load_json_if_present(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"error": f"invalid_json:{path}"}


def load_execution_evidence(probe_path: Path, microbench_path: Path, campaign_path: Path) -> dict:
    probe = load_json_if_present(probe_path)
    microbench = load_json_if_present(microbench_path)
    campaign = load_json_if_present(campaign_path)
    campaign_rows = campaign.get("experiment_rows", []) if isinstance(campaign.get("experiment_rows"), list) else []
    campaign_gaps = campaign.get("inferred_research_gaps", []) if isinstance(campaign.get("inferred_research_gaps"), list) else []
    repo_audit = campaign.get("repo_audit", []) if isinstance(campaign.get("repo_audit"), list) else []
    return {
        "probe_path": str(probe_path),
        "microbench_path": str(microbench_path),
        "campaign_path": str(campaign_path),
        "probe": probe,
        "microbench": microbench,
        "campaign": campaign,
        "cuda_available": bool(probe.get("cuda_available")),
        "gpu_count": probe.get("cuda_device_count", 0),
        "gpu_name": ((probe.get("selected_gpu_for_microbenchmark") or {}).get("name")),
        "selected_gpu_index": ((probe.get("selected_gpu_for_microbenchmark") or {}).get("index")),
        "openai_api_key_present": bool(probe.get("openai_api_key_present")),
        "paid_api_invoked": bool(probe.get("paid_api_invoked")),
        "microbenchmark_ran": bool(microbench.get("ran")),
        "microbenchmark_type": microbench.get("benchmark_type"),
        "runtime_rows": microbench.get("results", []),
        "max_allocated_mib": (microbench.get("memory") or {}).get("max_allocated_mib"),
        "campaign_ran": len(campaign_rows) > 0,
        "campaign_rows": len(campaign_rows),
        "campaign_repeats": campaign.get("repeats"),
        "campaign_runtime_seconds": campaign.get("runtime_seconds"),
        "campaign_gap_count": len(campaign_gaps),
        "campaign_families": sorted({row.get("family") for row in campaign_rows if row.get("family")}),
        "repo_audit_count": len(repo_audit),
    }


def feature_support(paper: dict, execution: dict | None = None) -> dict[str, bool]:
    execution = execution or {}
    cov = paper.get("source_coverage", {})
    gap_text = stringify(paper.get("gap_evidence"))
    method_text = stringify(paper.get("method_evidence"))
    eval_text = stringify(paper.get("evaluation_evidence"))
    exp_text = stringify(paper.get("experimental_setting_evidence"))
    result_text = stringify(paper.get("result_and_limitation_evidence"))
    footprint_text = stringify(paper.get("footprint"))
    all_text = " ".join([gap_text, method_text, eval_text, exp_text, result_text, footprint_text])
    has_gap = has_structured_section(paper.get("gap_evidence"))
    has_method = has_structured_section(paper.get("method_evidence"))
    has_eval = has_structured_section(paper.get("evaluation_evidence")) or has_structured_section(
        paper.get("experimental_setting_evidence")
    )
    has_result = has_structured_section(paper.get("result_and_limitation_evidence"))
    has_constraints = has_any(all_text, CONSTRAINT_TERMS)
    has_quality = has_any(all_text, QUALITY_TERMS)
    has_eff = has_any(all_text, EFFICIENCY_TERMS)
    local = paper.get("local_artifacts", {})
    has_gpu_or_hardware = has_any(all_text, GPU_TERMS) or bool(execution.get("cuda_available"))
    has_backend_provenance = has_any(all_text, API_TERMS) or bool(local.get("text")) or bool(
        execution.get("probe")
    )
    has_real_benchmark_artifact = has_any(all_text, EXECUTION_TERMS) or bool(
        execution.get("microbenchmark_ran")
    )
    has_runtime_measurement = (
        has_any(all_text, EFFICIENCY_TERMS | {"runtime", "latency", "throughput", "tokens/sec", "memory", "cost"})
        or bool(execution.get("microbenchmark_ran"))
        or bool(execution.get("campaign_ran"))
    )
    has_reproduction_status = (
        "not_rerun" in all_text
        or "rerun" in all_text
        or "code_inspected" in all_text
        or "compileall" in all_text
        or "official_repo" in all_text
        or "checkpoint" in all_text
        or bool(local.get("text"))
        or bool(execution.get("microbenchmark_ran"))
        or bool(execution.get("campaign_ran"))
    )
    has_campaign = bool(execution.get("campaign_ran"))
    has_campaign_gaps = bool(execution.get("campaign_gap_count"))
    has_repo_audit = bool(execution.get("repo_audit_count"))

    support = {
        "root.fullpaper_gap_argument": True,
        "intro.deployment_pressure": cov.get("introduction") is True and has_gap,
        "intro.prior_family_map": has_gap and ("prior" in gap_text or "existing" in gap_text or has_eval_baseline(paper)),
        "intro.failure_mode": has_gap
        and (
            "fail" in gap_text
            or "lack" in gap_text
            or "limitation" in gap_text
            or "need" in gap_text
            or "but" in gap_text
            or "gap" in gap_text
            or "requires" in gap_text
        ),
        "intro.gap_claim": has_gap,
        "intro.method_need": has_gap and has_method,
        "related.closest_baselines": cov.get("related_work") is True or has_eval_baseline(paper),
        "related.axis_of_difference": has_eval_baseline(paper)
        and ("versus" in all_text or "baseline" in all_text or "prior" in all_text or "delta" in all_text),
        "related.near_miss_handling": "prior_limitations" in gap_text
        or "failure_modes" in gap_text
        or "limitations" in gap_text
        or "near" in all_text,
        "method.object_definition": cov.get("method") is True and has_method,
        "method.mechanism_delta": has_method and has_gap,
        "method.constraint_binding": has_method and has_gap and has_constraints,
        "method.operational_steps": has_method,
        "method.theory_or_invariant": has_theory_or_invariant(paper),
        "experiments.axis_match": cov.get("experiments") is True and has_eval and has_gap,
        "experiments.baseline_strength": has_eval_baseline(paper),
        "experiments.metric_pairing": has_metrics(paper) and has_eff and has_quality,
        "experiments.execution_surface": has_eval and has_backend_provenance,
        "experiments.gpu_hardware_profile": has_eval and has_gpu_or_hardware,
        "experiments.api_or_local_backend": has_eval and has_backend_provenance,
        "experiments.real_benchmark_command": has_eval and has_real_benchmark_artifact,
        "experiments.exact_rerun_feasibility_audit": has_eval and has_repo_audit,
        "experiments.author_style_reproduction_campaign": has_eval and has_campaign,
        "experiments.scale_or_stress": "ablation" in all_text
        or "hardware" in all_text
        or "larger" in all_text
        or "long" in all_text
        or "temperature" in all_text
        or "model" in all_text
        or "stress" in all_text,
        "experiments.ablation_or_control": has_ablation(paper),
        "results.runtime_measurement": has_result and has_runtime_measurement,
        "results.measured_gap_derivation": has_result and has_campaign_gaps,
        "results.primary_table_read": cov.get("results") is True and has_result,
        "results.tradeoff_interpretation": has_result and (has_eff and has_quality or "tradeoff" in all_text or "trade-off" in all_text),
        "results.reproduction_status": has_result and has_reproduction_status,
        "results.proxy_vs_exact_boundary": has_result and has_reproduction_status and (has_campaign or "not_rerun" in all_text),
        "results.exception_scan": cov.get("limitations") is True
        or "negative_or_tradeoff" in result_text
        or "limitation" in result_text
        or "weak" in result_text,
        "results.mechanism_attribution": has_ablation(paper) or has_theory_or_invariant(paper),
        "results.scope_boundary": cov.get("limitations") is True or "limitations" in result_text or "not_rerun" in all_text,
        "appendix.raw_measurement_table": has_campaign,
        "appendix.artifact_log": bool(local.get("text"))
        or bool(local.get("pdf"))
        or bool(cov.get("code_repo_inspected"))
        or bool(execution.get("probe"))
        or bool(execution.get("microbenchmark_ran"))
        or bool(execution.get("campaign_ran")),
        "appendix.reproducibility_or_extra_evidence": bool(cov.get("appendix_or_supplement"))
        or bool(local.get("text"))
        or bool(local.get("pdf"))
        or bool(cov.get("code_repo_inspected")),
    }
    return support


def compute_support(
    papers: list[dict],
    execution: dict | None = None,
) -> tuple[dict[str, float], dict[str, int], dict[str, float], dict[str, int], dict[str, dict[str, bool]]]:
    matrix = {paper["chip_id"]: feature_support(paper, execution) for paper in papers}
    n = len(papers)
    node_counts = {
        node_id: sum(1 for paper in papers if matrix[paper["chip_id"]].get(node_id))
        for node_id, _, _ in NODES
    }
    node_rates = {node_id: node_counts[node_id] / n for node_id, _, _ in NODES}
    edge_counts = {}
    for src, dst in EDGES:
        edge_id = f"{src}->{dst}"
        edge_counts[edge_id] = sum(
            1
            for paper in papers
            if matrix[paper["chip_id"]].get(src) and matrix[paper["chip_id"]].get(dst)
        )
    edge_rates = {edge_id: edge_counts[edge_id] / n for edge_id in edge_counts}
    return node_rates, node_counts, edge_rates, edge_counts, matrix


def closure_score(nodes: set[str], edges: set[str]) -> float:
    if not nodes:
        return 0.0
    reachable = {"root.fullpaper_gap_argument"}
    changed = True
    while changed:
        changed = False
        for edge_id in edges:
            src, dst = edge_id.split("->")
            if src in reachable and dst in nodes and dst not in reachable:
                reachable.add(dst)
                changed = True
    return len(reachable & nodes) / len(nodes)


def score_graph(nodes: set[str], edges: set[str], node_rates: dict[str, float], edge_rates: dict[str, float]) -> float:
    mandatory_hit = sum(1 for n in MANDATORY_NODES if n in nodes) / len(MANDATORY_NODES)
    section_names = {"introduction", "related_work", "method", "experiments", "results", "limitations", "appendix"}
    node_section = {node_id: section for node_id, section, _ in NODES}
    section_hit = sum(1 for s in section_names if any(node_section[n] == s for n in nodes)) / len(section_names)
    avg_node = sum(node_rates[n] for n in nodes) / max(len(nodes), 1)
    avg_edge = sum(edge_rates[e] for e in edges) / max(len(edges), 1)
    connected = closure_score(nodes, edges)
    unsupported_penalty = sum(max(0.0, 0.55 - node_rates[n]) for n in nodes) / max(len(nodes), 1)
    bloat_penalty = max(0, len(nodes) - 40) * 0.005
    return (
        0.30 * mandatory_hit
        + 0.18 * section_hit
        + 0.24 * avg_node
        + 0.16 * avg_edge
        + 0.14 * connected
        - unsupported_penalty
        - bloat_penalty
    )


def select_edges(nodes: set[str], edge_rates: dict[str, float], threshold: float) -> set[str]:
    edges = set()
    for src, dst in EDGES:
        edge_id = f"{src}->{dst}"
        if src in nodes and dst in nodes and edge_rates[edge_id] >= threshold:
            edges.add(edge_id)
    return edges


def rollout_select(
    rng: random.Random,
    posterior: dict[str, float],
    node_rates: dict[str, float],
    edge_rates: dict[str, float],
    rollouts: int,
) -> tuple[set[str], set[str], float]:
    best_nodes: set[str] = set()
    best_edges: set[str] = set()
    best_score = -1e9
    all_nodes = [node_id for node_id, _, _ in NODES]

    for _ in range(rollouts):
        node_threshold = rng.uniform(0.50, 0.78)
        edge_threshold = rng.uniform(0.45, 0.72)
        nodes = set(MANDATORY_NODES)
        nodes.add("root.fullpaper_gap_argument")
        for node_id in all_nodes:
            if node_id in nodes:
                continue
            noise = rng.gauss(0.0, 0.035)
            if posterior[node_id] + noise >= node_threshold:
                nodes.add(node_id)

        # Keep prerequisites for selected downstream nodes.
        changed = True
        while changed:
            changed = False
            for src, dst in EDGES:
                if dst in nodes and src not in nodes and src in MANDATORY_NODES:
                    nodes.add(src)
                    changed = True

        edges = select_edges(nodes, edge_rates, edge_threshold)
        score = score_graph(nodes, edges, node_rates, edge_rates)
        score += rng.gauss(0.0, 0.002)
        if score > best_score:
            best_score = score
            best_nodes = nodes
            best_edges = edges

    return best_nodes, best_edges, best_score


def run_simulation(args: argparse.Namespace) -> dict:
    evidence = json.loads(args.evidence.read_text())
    papers = evidence["papers"]
    execution = load_execution_evidence(args.execution_probe, args.gpu_microbenchmark, args.repro_campaign)
    node_rates, node_counts, edge_rates, edge_counts, matrix = compute_support(papers, execution)

    posterior = {node_id: 0.50 for node_id, _, _ in NODES}
    prev_signature = None
    stable = 0
    trace_rows = []
    best = None

    trace_path = args.output_dir / "dirs_simulation_trace.jsonl"
    with trace_path.open("w") as fh:
        for loop in range(1, args.max_loops + 1):
            rng = random.Random(args.seed + loop)
            learning_rate = min(0.35, 0.10 + loop / (args.max_loops * 5.0))
            for node_id in posterior:
                posterior[node_id] = (1.0 - learning_rate) * posterior[node_id] + learning_rate * node_rates[node_id]

            nodes, edges, score = rollout_select(rng, posterior, node_rates, edge_rates, args.rollouts)
            signature = stable_hash({"nodes": sorted(nodes), "edges": sorted(edges)})
            if signature == prev_signature:
                stable += 1
            else:
                stable = 0
            prev_signature = signature

            row = {
                "loop": loop,
                "score": round(score, 6),
                "stable_signature_count": stable,
                "signature": signature,
                "selected_nodes": sorted(nodes),
                "selected_edges": sorted(edges),
                "mean_selected_node_support": round(sum(node_rates[n] for n in nodes) / len(nodes), 6),
                "mean_selected_edge_support": round(sum(edge_rates[e] for e in edges) / max(len(edges), 1), 6),
                "node_count": len(nodes),
                "edge_count": len(edges),
            }
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            trace_rows.append(row)
            if best is None or row["score"] > best["score"]:
                best = row
            if loop >= args.min_loops and stable >= args.stable_window:
                break

    final = trace_rows[-1]
    converged = final["stable_signature_count"] >= args.stable_window and final["loop"] >= args.min_loops
    selected_nodes = final["selected_nodes"]
    selected_edges = final["selected_edges"]

    node_library = [
        {
            "id": node_id,
            "section": section,
            "role": role,
            "support_count": node_counts[node_id],
            "support_rate": round(node_rates[node_id], 4),
            "selected": node_id in selected_nodes,
            "mandatory": node_id in MANDATORY_NODES,
        }
        for node_id, section, role in NODES
    ]
    edge_library = [
        {
            "id": f"{src}->{dst}",
            "source": src,
            "target": dst,
            "support_count": edge_counts[f"{src}->{dst}"],
            "support_rate": round(edge_rates[f"{src}->{dst}"], 4),
            "selected": f"{src}->{dst}" in selected_edges,
        }
        for src, dst in EDGES
    ]

    by_paper = []
    for paper in papers:
        sup = matrix[paper["chip_id"]]
        by_paper.append(
            {
                "chip_id": paper["chip_id"],
                "title": paper["title"],
                "supported_selected_nodes": [n for n in selected_nodes if sup.get(n)],
                "missing_selected_nodes": [n for n in selected_nodes if not sup.get(n)],
            }
        )

    report = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "simulation_type": "local_deterministic_dirs_fullpaper_section_gap",
        "domain": evidence["domain"],
        "paper_count": len(papers),
        "source_evidence_table": str(args.evidence),
        "execution_probe": {
            "probe_path": execution["probe_path"],
            "gpu_microbenchmark_path": execution["microbench_path"],
            "author_style_campaign_path": execution["campaign_path"],
            "cuda_available": execution["cuda_available"],
            "gpu_count": execution["gpu_count"],
            "selected_gpu_index": execution["selected_gpu_index"],
            "gpu_name": execution["gpu_name"],
            "openai_api_key_present": execution["openai_api_key_present"],
            "paid_api_invoked": execution["paid_api_invoked"],
            "microbenchmark_ran": execution["microbenchmark_ran"],
            "microbenchmark_type": execution["microbenchmark_type"],
            "max_allocated_mib": execution["max_allocated_mib"],
            "runtime_rows": execution["runtime_rows"],
            "campaign_ran": execution["campaign_ran"],
            "campaign_rows": execution["campaign_rows"],
            "campaign_repeats": execution["campaign_repeats"],
            "campaign_runtime_seconds": execution["campaign_runtime_seconds"],
            "campaign_gap_count": execution["campaign_gap_count"],
            "campaign_families": execution["campaign_families"],
            "repo_audit_count": execution["repo_audit_count"],
        },
        "private_holdout_read": False,
        "max_loops": args.max_loops,
        "min_loops": args.min_loops,
        "stable_window": args.stable_window,
        "rollouts_per_loop": args.rollouts,
        "seed": args.seed,
        "completed_loops": final["loop"],
        "converged": converged,
        "converged_at_loop": final["loop"] if converged else None,
        "final_score": final["score"],
        "final_signature": final["signature"],
        "final_selected_nodes": selected_nodes,
        "final_selected_edges": selected_edges,
        "mean_selected_node_support": final["mean_selected_node_support"],
        "mean_selected_edge_support": final["mean_selected_edge_support"],
        "node_library": node_library,
        "edge_library": edge_library,
        "paper_replay": by_paper,
        "gap_taxonomy": [
            "fragmentation_or_unification_gap",
            "constraint_mismatch_gap",
            "static_policy_gap",
            "exactness_versus_speed_gap",
            "objective_misalignment_gap",
            "mechanism_specificity_gap",
        ],
        "verifier_reward": {
            "positive": [
                "introduction gap pressure",
                "closest prior baseline fairness",
                "method mechanism bound to gap",
                "experiment axis matches gap",
                "metric pairing of efficiency and correctness/quality",
                "execution surface declared as GPU, API, local, or paper-only",
                "GPU hardware/API backend provenance logged",
                "exact benchmark command or script persisted",
                "author-style GPU reproduction campaign with repeated seeds",
                "exact-rerun feasibility audit across all papers",
                "measured gap derivation from GPU tradeoff failures",
                "runtime/memory/API-cost measurement attached to results",
                "reproduction status stated separately from paper claims",
                "scale or stress testing",
                "ablation/control support",
                "limitation boundary",
            ],
            "negative": [
                "unsupported section node",
                "unconnected graph",
                "single-metric tradeoff claim",
                "GPU speed claim without hardware profile",
                "API result without external-call or cost provenance",
                "paper-only result phrased as a local reproduction",
                "runtime number without command/config/log artifact",
                "proxy experiment reported as exact paper reproduction",
                "gap claim derived only from reading, without measurement pressure",
                "mechanism attribution without ablation/theory",
                "broad novelty claim without near-miss handling",
            ],
        },
    }

    (args.output_dir / "dirs_final_dag.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (args.output_dir / "dirs_node_support_scores.json").write_text(json.dumps(node_library, indent=2, sort_keys=True))
    (args.output_dir / "dirs_edge_support_scores.json").write_text(json.dumps(edge_library, indent=2, sort_keys=True))
    write_markdown_report(args.output_dir / "DIRS_FULLPAPER_SIMULATION_REPORT.md", report, trace_rows)
    return report


def write_markdown_report(path: Path, report: dict, trace_rows: list[dict]) -> None:
    selected_nodes = {n["id"]: n for n in report["node_library"] if n["selected"]}
    selected_edges = {e["id"]: e for e in report["edge_library"] if e["selected"]}
    lines = [
        "# DIRS Full-Paper Section Gap Simulation Report",
        "",
        f"Date: `{report['created_at_utc']}`",
        f"Domain: `{report['domain']}`",
        f"Papers: `{report['paper_count']}`",
        f"Simulation: `{report['simulation_type']}`",
        f"Private holdout read: `{str(report['private_holdout_read']).lower()}`",
        "",
        "## Convergence",
        "",
        f"- Completed loops: `{report['completed_loops']}`",
        f"- Minimum loops: `{report['min_loops']}`",
        f"- Stable window: `{report['stable_window']}`",
        f"- Rollouts per loop: `{report['rollouts_per_loop']}`",
        f"- Converged: `{str(report['converged']).lower()}`",
        f"- Final score: `{report['final_score']}`",
        f"- Mean selected node support: `{report['mean_selected_node_support']}`",
        f"- Mean selected edge support: `{report['mean_selected_edge_support']}`",
        "",
        "## Real Execution Layer",
        "",
        f"- CUDA available: `{str(report['execution_probe']['cuda_available']).lower()}`",
        f"- Visible GPU count: `{report['execution_probe']['gpu_count']}`",
        f"- Selected GPU: `{report['execution_probe']['selected_gpu_index']}` `{report['execution_probe']['gpu_name']}`",
        f"- OpenAI API key present: `{str(report['execution_probe']['openai_api_key_present']).lower()}`",
        f"- Paid/external API invoked: `{str(report['execution_probe']['paid_api_invoked']).lower()}`",
        f"- GPU microbenchmark ran: `{str(report['execution_probe']['microbenchmark_ran']).lower()}`",
        f"- Microbenchmark type: `{report['execution_probe']['microbenchmark_type']}`",
        f"- Max allocated memory: `{report['execution_probe']['max_allocated_mib']} MiB`",
        f"- Probe artifact: `{report['execution_probe']['probe_path']}`",
        f"- Benchmark artifact: `{report['execution_probe']['gpu_microbenchmark_path']}`",
        f"- Author-style campaign ran: `{str(report['execution_probe']['campaign_ran']).lower()}`",
        f"- Campaign rows: `{report['execution_probe']['campaign_rows']}`",
        f"- Campaign repeats: `{report['execution_probe']['campaign_repeats']}`",
        f"- Campaign runtime: `{report['execution_probe']['campaign_runtime_seconds']}s`",
        f"- Campaign families: `{', '.join(report['execution_probe']['campaign_families'])}`",
        f"- Inferred measured gaps: `{report['execution_probe']['campaign_gap_count']}`",
        f"- Campaign artifact: `{report['execution_probe']['author_style_campaign_path']}`",
        "",
        "Runtime rows:",
        "",
    ]
    for row in report["execution_probe"].get("runtime_rows", []):
        lines.append(
            "- context `{context_tokens}` tokens, `{dtype}`, `{ms_per_decode_token}` ms/decode-token, "
            "`{decode_tokens_per_second}` decode-tokens/s".format(**row)
        )
    lines += [
        "",
        "## Final Selected DAG Nodes",
        "",
    ]
    for node_id in report["final_selected_nodes"]:
        node = selected_nodes[node_id]
        lines.append(
            f"- `{node_id}` ({node['section']}): support `{node['support_count']}/20`, {node['role']}"
        )
    lines += ["", "## Final Selected DAG Edges", ""]
    for edge_id in report["final_selected_edges"]:
        edge = selected_edges[edge_id]
        lines.append(f"- `{edge_id}`: support `{edge['support_count']}/20`")
    lines += [
        "",
        "## Learned Full-Paper Gap Policy",
        "",
        "```text",
        "introduction pressure",
        "  -> prior family and near-miss boundary",
        "  -> exact failure mode",
        "  -> bounded gap claim",
        "  -> method mechanism bound to the gap",
        "  -> experiments whose axes match the gap",
        "  -> metric pairs that protect tradeoff claims",
        "  -> execution surface: local GPU / API / paper-only provenance",
        "  -> GPU hardware and API/backend nodes before benchmark claims",
        "  -> exact benchmark command and artifact log",
        "  -> exact-rerun feasibility audit across the full paper set",
        "  -> author-style GPU reproduction campaign with baselines/controls/repeats",
        "  -> measured gap derivation from observed tradeoff failures",
        "  -> proxy-vs-exact reproduction boundary",
        "  -> runtime measurement and reproduction status",
        "  -> scale/stress and ablation evidence",
        "  -> scoped result and limitation boundary",
        "```",
        "",
        "## Verifier Reward",
        "",
        "Positive signals:",
    ]
    for item in report["verifier_reward"]["positive"]:
        lines.append(f"- {item}")
    lines += ["", "Negative signals:"]
    for item in report["verifier_reward"]["negative"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Replay Gaps",
        "",
        "Selected nodes with less than full support are conditional moves, not universal requirements.",
    ]
    for node_id, node in selected_nodes.items():
        if node["support_count"] < report["paper_count"]:
            lines.append(f"- `{node_id}`: `{node['support_count']}/20` support")
    lines += [
        "",
        "## Trace Tail",
        "",
        "```jsonl",
    ]
    for row in trace_rows[-5:]:
        lines.append(json.dumps(row, sort_keys=True))
    lines += ["```", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--execution-probe", type=Path, default=DEFAULT_EXECUTION_PROBE)
    parser.add_argument("--gpu-microbenchmark", type=Path, default=DEFAULT_GPU_MICROBENCH)
    parser.add_argument("--repro-campaign", type=Path, default=DEFAULT_REPRO_CAMPAIGN)
    parser.add_argument("--output-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max-loops", type=int, default=80)
    parser.add_argument("--min-loops", type=int, default=24)
    parser.add_argument("--stable-window", type=int, default=10)
    parser.add_argument("--rollouts", type=int, default=5000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_simulation(args)
    print(json.dumps({
        "converged": report["converged"],
        "completed_loops": report["completed_loops"],
        "final_score": report["final_score"],
        "nodes": len(report["final_selected_nodes"]),
        "edges": len(report["final_selected_edges"]),
        "execution_probe": report["execution_probe"],
        "report": str(args.output_dir / "DIRS_FULLPAPER_SIMULATION_REPORT.md"),
        "dag": str(args.output_dir / "dirs_final_dag.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
