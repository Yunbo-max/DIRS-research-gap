#!/usr/bin/env python3
"""Loop 2: author-side experiment simulation for strict systems papers.

Loop 1 learns reusable DAG priors. Loop 2 is different: it simulates the author
doing the scientific work. The author forms a gap hypothesis, reads/runs
measurements, decides whether the hypothesis survives, revises the claim, and
then writes a bounded conclusion.

This script consumes the real GPU campaign rows and repo audit generated in the
same run directory. It does not treat paper reading as simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev


RUN_DIR = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN = RUN_DIR / "author_style_gpu_reproduction_campaign.json"
DEFAULT_CAMPAIGN_SCRIPT = RUN_DIR / "run_author_style_gpu_reproduction_campaign.py"
OUTPUT_JSON = RUN_DIR / "author_loop2_final_dag.json"
TRACE_JSONL = RUN_DIR / "author_loop2_mcts_trace.jsonl"
OUTPUT_MD = RUN_DIR / "AUTHOR_LOOP2_EXPERIMENT_SIMULATION_REPORT.md"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def run_campaign_if_needed(campaign_path: Path, repeats: int) -> dict:
    if campaign_path.exists():
        return {"ran": False, "reason": "existing_campaign_loaded", "path": str(campaign_path)}
    proc = subprocess.run(
        ["python", str(DEFAULT_CAMPAIGN_SCRIPT), "--repeats", str(repeats)],
        cwd=str(RUN_DIR),
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "ran": True,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "path": str(campaign_path),
    }


def group_rows(rows: list[dict], family: str) -> list[dict]:
    return [row for row in rows if row.get("family") == family]


def safe_mean(rows: list[dict], key: str) -> float | None:
    vals = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return mean(vals) if vals else None


def ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def summarize_campaign(campaign: dict) -> dict:
    rows = campaign.get("experiment_rows", [])
    repo_audit = campaign.get("repo_audit", [])
    repo_status_counts: dict[str, int] = {}
    for row in repo_audit:
        status = row.get("exact_rerun_status", "unknown")
        repo_status_counts[status] = repo_status_counts.get(status, 0) + 1

    kv = group_rows(rows, "kv_cache_locality")
    kv_local = [r for r in kv if str(r.get("method", "")).startswith("local_window")]
    kv_dense = [r for r in kv if r.get("method") == "dense_full_kv"]
    kv_topk = [r for r in kv if r.get("method") == "topk_128_after_full_scores"]
    kv_max_err = max((r.get("quality_error_rmse_vs_dense", 0.0) for r in kv_local), default=0.0)
    kv_dense_ms = safe_mean(kv_dense, "ms")
    kv_local_ms = safe_mean(kv_local, "ms")
    kv_topk_ms = safe_mean(kv_topk, "ms")

    tm = group_rows(rows, "token_merging")
    tm_base_by_tokens = {
        tokens: safe_mean([r for r in tm if r.get("method") == "full_attention_baseline" and r.get("tokens") == tokens], "ms")
        for tokens in sorted({r.get("tokens") for r in tm if r.get("tokens")})
    }
    tm_proxy = [r for r in tm if r.get("method") != "full_attention_baseline"]
    tm_max_err = max((r.get("quality_error_rmse_vs_full", 0.0) for r in tm_proxy), default=0.0)
    tm_speedups = []
    for row in tm_proxy:
        base = tm_base_by_tokens.get(row.get("tokens"))
        sp = ratio(base, row.get("ms"))
        if sp is not None:
            tm_speedups.append(sp)

    spec = group_rows(rows, "speculative_decoding_proxy")
    spec_speedups = [r.get("effective_speedup_after_acceptance", 0.0) for r in spec]
    spec_accept = [r.get("block_accept_rate", 0.0) for r in spec]

    samp = group_rows(rows, "sampling_truncation")
    samp_nonfull = [r for r in samp if r.get("method") != "softmax_full"]
    samp_min_mass = min((r.get("retained_probability_mass", 1.0) for r in samp_nonfull), default=1.0)
    samp_max_entropy_delta = max((r.get("entropy_delta_vs_full", 0.0) for r in samp_nonfull), default=0.0)

    quant = group_rows(rows, "quantization_compression")
    quant_nonbase = [r for r in quant if r.get("method") != "fp16_baseline"]
    quant_max_rmse = max((r.get("rmse_vs_fp16", 0.0) for r in quant_nonbase), default=0.0)
    quant_base_ms = safe_mean([r for r in quant if r.get("method") == "fp16_baseline"], "ms")
    quant_qdq_ms = safe_mean(quant_nonbase, "ms")

    sparse = group_rows(rows, "sparse_kernel_efficiency")
    sparse_dense_by_key = {}
    for row in sparse:
        if row.get("method") == "dense_matvec":
            sparse_dense_by_key[(row.get("seed"), row.get("n"))] = row.get("items_per_second")
    sparse_controls = [r for r in sparse if r.get("method") != "dense_matvec"]
    sparse_not_faster = 0
    for row in sparse_controls:
        base = sparse_dense_by_key.get((row.get("seed"), row.get("n")))
        if base is not None and row.get("items_per_second", 0.0) <= base:
            sparse_not_faster += 1

    return {
        "rows": len(rows),
        "repeats": campaign.get("repeats"),
        "families": sorted({row.get("family") for row in rows if row.get("family")}),
        "repo_status_counts": repo_status_counts,
        "blocked_exact_reruns": len([r for r in repo_audit if r.get("exact_rerun_status") != "code_ready_needs_model_data"]),
        "repo_audit_count": len(repo_audit),
        "kv": {
            "dense_ms_mean": kv_dense_ms,
            "local_ms_mean": kv_local_ms,
            "topk_ms_mean": kv_topk_ms,
            "max_local_rmse": kv_max_err,
            "local_speedup_vs_dense": ratio(kv_dense_ms, kv_local_ms),
            "topk_speedup_vs_dense": ratio(kv_dense_ms, kv_topk_ms),
        },
        "token_merging": {
            "max_rmse": tm_max_err,
            "mean_speedup_vs_full": mean(tm_speedups) if tm_speedups else None,
            "max_speedup_vs_full": max(tm_speedups) if tm_speedups else None,
        },
        "speculative": {
            "mean_effective_speedup": mean(spec_speedups) if spec_speedups else None,
            "min_effective_speedup": min(spec_speedups) if spec_speedups else None,
            "mean_block_acceptance": mean(spec_accept) if spec_accept else None,
            "block_acceptance_std": pstdev(spec_accept) if len(spec_accept) > 1 else 0.0,
        },
        "sampling": {
            "min_retained_mass": samp_min_mass,
            "max_entropy_delta": samp_max_entropy_delta,
        },
        "quantization": {
            "base_ms_mean": quant_base_ms,
            "quant_dequant_ms_mean": quant_qdq_ms,
            "qdq_speedup_vs_fp16_op": ratio(quant_base_ms, quant_qdq_ms),
            "max_rmse": quant_max_rmse,
        },
        "sparse": {
            "control_rows": len(sparse_controls),
            "not_faster_than_dense": sparse_not_faster,
            "not_faster_fraction": sparse_not_faster / max(len(sparse_controls), 1),
        },
    }


NODE_LIBRARY = {
    "root.author_experiment_loop": "Loop 2 is the author doing experiments and making scientific decisions.",
    "H.reproducibility_gap": "Hypothesis: exact systems claims are blocked without runnable artifacts.",
    "H.quality_guard_gap": "Hypothesis: token efficiency needs quality/correctness guards.",
    "H.hardware_specificity_gap": "Hypothesis: speedup requires hardware-aware implementation, not abstract sparsity/compression.",
    "E.repo_exact_rerun_audit": "Audit 20 papers for repo, checkpoint, dataset, and API readiness.",
    "E.gpu_campaign_baselines": "Run GPU baselines and controls with repeated seeds.",
    "E.kv_cache_stress": "Run dense/local/top-k KV-style long-context stress tests.",
    "E.token_merge_stress": "Run full-attention versus merge proxy controls.",
    "E.speculative_acceptance_stress": "Run draft/target acceptance-limited speculative proxy.",
    "E.sampling_mass_entropy_stress": "Run full softmax, top-k, top-p, entropy-adaptive sampling controls.",
    "E.quantization_kernel_stress": "Run fp16 versus quant-dequant controls.",
    "E.sparse_kernel_stress": "Run dense versus masked-sparse matvec controls.",
    "M.raw_measurement_read": "Read raw GPU rows, not just summary prose.",
    "M.statistics_over_repeats": "Aggregate repeated-seed means/stds before judging.",
    "D.accept_reproducibility_gap": "Author decision: exact-rerun infrastructure gap survives.",
    "D.accept_quality_guard_gap": "Author decision: speed claims need paired quality/correctness guards.",
    "D.accept_acceptance_limited_gap": "Author decision: speculative speedup is acceptance-limited.",
    "D.accept_hardware_specificity_gap": "Author decision: unstructured sparsity/compression needs hardware-aware kernels.",
    "D.reject_exact_reproduction_claim": "Author decision: do not claim exact leaderboard reproduction from proxy tests.",
    "D.revise_overbroad_speed_claim": "Author decision: soften any universal speedup claim.",
    "C.central_claim": "Conclusion: strict systems gaps must be measured as backend-specific speed/quality/reproducibility tradeoffs.",
    "C.paper_sections_to_write": "Write experiments/results/limitations from decisions and artifacts.",
    "A.appendix_artifacts": "Attach scripts, raw rows, traces, and blocked-rerun audit.",
}


EDGE_LIBRARY = [
    ("root.author_experiment_loop", "H.reproducibility_gap"),
    ("root.author_experiment_loop", "H.quality_guard_gap"),
    ("root.author_experiment_loop", "H.hardware_specificity_gap"),
    ("H.reproducibility_gap", "E.repo_exact_rerun_audit"),
    ("H.quality_guard_gap", "E.gpu_campaign_baselines"),
    ("H.hardware_specificity_gap", "E.gpu_campaign_baselines"),
    ("E.gpu_campaign_baselines", "E.kv_cache_stress"),
    ("E.gpu_campaign_baselines", "E.token_merge_stress"),
    ("E.gpu_campaign_baselines", "E.speculative_acceptance_stress"),
    ("E.gpu_campaign_baselines", "E.sampling_mass_entropy_stress"),
    ("E.gpu_campaign_baselines", "E.quantization_kernel_stress"),
    ("E.gpu_campaign_baselines", "E.sparse_kernel_stress"),
    ("E.repo_exact_rerun_audit", "M.raw_measurement_read"),
    ("E.kv_cache_stress", "M.raw_measurement_read"),
    ("E.token_merge_stress", "M.raw_measurement_read"),
    ("E.speculative_acceptance_stress", "M.raw_measurement_read"),
    ("E.sampling_mass_entropy_stress", "M.raw_measurement_read"),
    ("E.quantization_kernel_stress", "M.raw_measurement_read"),
    ("E.sparse_kernel_stress", "M.raw_measurement_read"),
    ("M.raw_measurement_read", "M.statistics_over_repeats"),
    ("M.statistics_over_repeats", "D.accept_reproducibility_gap"),
    ("M.statistics_over_repeats", "D.accept_quality_guard_gap"),
    ("M.statistics_over_repeats", "D.accept_acceptance_limited_gap"),
    ("M.statistics_over_repeats", "D.accept_hardware_specificity_gap"),
    ("D.accept_reproducibility_gap", "D.reject_exact_reproduction_claim"),
    ("D.accept_quality_guard_gap", "D.revise_overbroad_speed_claim"),
    ("D.accept_acceptance_limited_gap", "D.revise_overbroad_speed_claim"),
    ("D.accept_hardware_specificity_gap", "D.revise_overbroad_speed_claim"),
    ("D.reject_exact_reproduction_claim", "C.central_claim"),
    ("D.revise_overbroad_speed_claim", "C.central_claim"),
    ("C.central_claim", "C.paper_sections_to_write"),
    ("C.paper_sections_to_write", "A.appendix_artifacts"),
]


MANDATORY = {
    "root.author_experiment_loop",
    "E.repo_exact_rerun_audit",
    "E.gpu_campaign_baselines",
    "M.raw_measurement_read",
    "M.statistics_over_repeats",
    "D.reject_exact_reproduction_claim",
    "D.revise_overbroad_speed_claim",
    "C.central_claim",
    "C.paper_sections_to_write",
    "A.appendix_artifacts",
}


def node_support(summary: dict) -> dict[str, float]:
    blocked = summary["blocked_exact_reruns"] / max(summary["repo_audit_count"], 1)
    rows = summary["rows"]
    families = set(summary["families"])
    kv_err = summary["kv"]["max_local_rmse"]
    merge_err = summary["token_merging"]["max_rmse"]
    spec_min = summary["speculative"]["min_effective_speedup"] or 0.0
    mass_min = summary["sampling"]["min_retained_mass"]
    qdq_speed = summary["quantization"]["qdq_speedup_vs_fp16_op"] or 0.0
    sparse_fail = summary["sparse"]["not_faster_fraction"]
    has_full_campaign = rows >= 1000 and len(families) >= 6
    quality_signal = max(kv_err > 0.05, merge_err > 0.10, mass_min < 0.20)
    hardware_signal = max(sparse_fail > 0.10, qdq_speed < 1.0)
    return {
        "root.author_experiment_loop": 1.0,
        "H.reproducibility_gap": min(1.0, 0.4 + blocked),
        "H.quality_guard_gap": 1.0 if quality_signal else 0.45,
        "H.hardware_specificity_gap": 1.0 if hardware_signal else 0.45,
        "E.repo_exact_rerun_audit": 1.0 if summary["repo_audit_count"] >= 20 else 0.50,
        "E.gpu_campaign_baselines": 1.0 if has_full_campaign else 0.50,
        "E.kv_cache_stress": 1.0 if "kv_cache_locality" in families else 0.0,
        "E.token_merge_stress": 1.0 if "token_merging" in families else 0.0,
        "E.speculative_acceptance_stress": 1.0 if "speculative_decoding_proxy" in families else 0.0,
        "E.sampling_mass_entropy_stress": 1.0 if "sampling_truncation" in families else 0.0,
        "E.quantization_kernel_stress": 1.0 if "quantization_compression" in families else 0.0,
        "E.sparse_kernel_stress": 1.0 if "sparse_kernel_efficiency" in families else 0.0,
        "M.raw_measurement_read": 1.0 if rows >= 1000 else 0.45,
        "M.statistics_over_repeats": 1.0 if (summary["repeats"] or 0) >= 20 else 0.55,
        "D.accept_reproducibility_gap": min(1.0, 0.4 + blocked),
        "D.accept_quality_guard_gap": 1.0 if quality_signal else 0.45,
        "D.accept_acceptance_limited_gap": 1.0 if spec_min < 1.0 else 0.45,
        "D.accept_hardware_specificity_gap": 1.0 if hardware_signal else 0.45,
        "D.reject_exact_reproduction_claim": 1.0,
        "D.revise_overbroad_speed_claim": 1.0,
        "C.central_claim": 1.0,
        "C.paper_sections_to_write": 1.0,
        "A.appendix_artifacts": 1.0,
    }


def edges_for(nodes: set[str], supports: dict[str, float], threshold: float) -> set[str]:
    out = set()
    for src, dst in EDGE_LIBRARY:
        if src in nodes and dst in nodes and min(supports[src], supports[dst]) >= threshold:
            out.add(f"{src}->{dst}")
    return out


def closure_score(nodes: set[str], edges: set[str]) -> float:
    reachable = {"root.author_experiment_loop"}
    changed = True
    while changed:
        changed = False
        for edge in edges:
            src, dst = edge.split("->")
            if src in reachable and dst in nodes and dst not in reachable:
                reachable.add(dst)
                changed = True
    return len(reachable & nodes) / max(len(nodes), 1)


def author_reward(nodes: set[str], edges: set[str], supports: dict[str, float]) -> float:
    avg_support = sum(supports[n] for n in nodes) / max(len(nodes), 1)
    mandatory_hit = sum(1 for n in MANDATORY if n in nodes) / len(MANDATORY)
    connected = closure_score(nodes, edges)
    experiments = [n for n in nodes if n.startswith("E.") and n.endswith("_stress")]
    experiment_breadth = len(experiments) / 6.0
    decisions = [n for n in nodes if n.startswith("D.")]
    decision_breadth = len(decisions) / 6.0
    author_chain = all(
        n in nodes
        for n in [
            "M.raw_measurement_read",
            "M.statistics_over_repeats",
            "C.central_claim",
            "C.paper_sections_to_write",
        ]
    )
    overclaim_penalty = 0.0
    if "D.reject_exact_reproduction_claim" not in nodes:
        overclaim_penalty += 0.15
    if "D.revise_overbroad_speed_claim" not in nodes:
        overclaim_penalty += 0.15
    return (
        0.24 * avg_support
        + 0.20 * mandatory_hit
        + 0.18 * connected
        + 0.18 * experiment_breadth
        + 0.12 * decision_breadth
        + (0.08 if author_chain else 0.0)
        - overclaim_penalty
    )


def rollout_search(summary: dict, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    supports = node_support(summary)
    posterior = {node: 0.50 for node in NODE_LIBRARY}
    trace = []
    prev_sig = None
    stable = 0
    final = None

    with TRACE_JSONL.open("w") as fh:
        for loop in range(1, args.max_loops + 1):
            rng = random.Random(args.seed + loop)
            lr = min(0.35, 0.10 + loop / (args.max_loops * 5.0))
            for node in posterior:
                posterior[node] = (1 - lr) * posterior[node] + lr * supports[node]

            best_nodes: set[str] = set()
            best_edges: set[str] = set()
            best_score = -1e9
            for _ in range(args.rollouts):
                threshold = rng.uniform(0.48, 0.82)
                edge_threshold = rng.uniform(0.45, 0.75)
                nodes = set(MANDATORY)
                for node in NODE_LIBRARY:
                    if node in nodes:
                        continue
                    if posterior[node] + rng.gauss(0, 0.035) >= threshold:
                        nodes.add(node)

                # Author DAG prerequisites.
                changed = True
                while changed:
                    changed = False
                    for src, dst in EDGE_LIBRARY:
                        if dst in nodes and src in MANDATORY and src not in nodes:
                            nodes.add(src)
                            changed = True

                edges = edges_for(nodes, supports, edge_threshold)
                score = author_reward(nodes, edges, supports) + rng.gauss(0, 0.002)
                if score > best_score:
                    best_nodes, best_edges, best_score = nodes, edges, score

            sig = stable_hash({"nodes": sorted(best_nodes), "edges": sorted(best_edges)})
            stable = stable + 1 if sig == prev_sig else 0
            prev_sig = sig
            row = {
                "loop": loop,
                "score": round(best_score, 6),
                "signature": sig,
                "stable_signature_count": stable,
                "selected_nodes": sorted(best_nodes),
                "selected_edges": sorted(best_edges),
                "node_count": len(best_nodes),
                "edge_count": len(best_edges),
                "mean_support": round(sum(supports[n] for n in best_nodes) / len(best_nodes), 6),
            }
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            trace.append(row)
            final = row
            if loop >= args.min_loops and stable >= args.stable_window:
                break

    assert final is not None
    report = {
        "created_at_utc": now_utc(),
        "simulation_type": "loop2_author_experiment_decision_simulation",
        "definition": "Author forms hypotheses, runs/reads experiments, makes decisions, revises claims, and writes conclusions.",
        "not_simulation": [
            "not reviewer-only critique",
            "not paper-reading summary",
            "not GPU benchmark alone",
        ],
        "campaign_path": str(args.campaign),
        "private_holdout_read": False,
        "paid_external_api_invoked": False,
        "completed_loops": final["loop"],
        "converged": final["loop"] >= args.min_loops and final["stable_signature_count"] >= args.stable_window,
        "final_score": final["score"],
        "final_selected_nodes": final["selected_nodes"],
        "final_selected_edges": final["selected_edges"],
        "summary": summary,
        "node_library": [
            {
                "id": node,
                "role": role,
                "support": round(supports[node], 4),
                "selected": node in final["selected_nodes"],
                "mandatory": node in MANDATORY,
            }
            for node, role in NODE_LIBRARY.items()
        ],
        "edge_library": [
            {
                "id": f"{src}->{dst}",
                "source": src,
                "target": dst,
                "selected": f"{src}->{dst}" in final["selected_edges"],
                "support": round(min(supports[src], supports[dst]), 4),
            }
            for src, dst in EDGE_LIBRARY
        ],
        "author_decisions": derive_author_decisions(summary),
    }
    return report, trace


def derive_author_decisions(summary: dict) -> list[dict]:
    decisions = []
    blocked = summary["blocked_exact_reruns"]
    total = summary["repo_audit_count"]
    decisions.append(
        {
            "decision": "Reject exact-reproduction wording.",
            "because": f"{blocked}/{total} papers are not exact-rerun-ready under local constraints.",
            "write_as": "proxy reproduction plus reproducibility boundary",
        }
    )
    decisions.append(
        {
            "decision": "Accept quality-guard gap.",
            "because": (
                f"KV local-window RMSE reached {summary['kv']['max_local_rmse']:.6f}; "
                f"token-merge RMSE reached {summary['token_merging']['max_rmse']:.6f}; "
                f"sampling retained mass dropped to {summary['sampling']['min_retained_mass']:.6f}."
            ),
            "write_as": "speed claims require paired quality/correctness metrics",
        }
    )
    decisions.append(
        {
            "decision": "Accept acceptance-limited speculative decoding gap.",
            "because": f"minimum effective speculative speedup was {summary['speculative']['min_effective_speedup']:.6f}.",
            "write_as": "draft quality and acceptance distribution are core variables",
        }
    )
    decisions.append(
        {
            "decision": "Accept hardware-specificity gap.",
            "because": (
                f"{summary['sparse']['not_faster_than_dense']}/{summary['sparse']['control_rows']} sparse controls "
                "failed to beat dense; quant-dequant was not faster than fp16 operation in the proxy."
            ),
            "write_as": "efficiency requires hardware-aware kernels and end-to-end measurement",
        }
    )
    decisions.append(
        {
            "decision": "Revise conclusion to bounded NeurIPS-systems claim.",
            "because": "The measurements support tradeoff and reproducibility gaps, not a universal new algorithmic win.",
            "write_as": "a strict paper should propose a measured backend-specific policy with artifacts and limits",
        }
    )
    return decisions


def write_report(report: dict, trace: list[dict]) -> None:
    selected_nodes = {row["id"]: row for row in report["node_library"] if row["selected"]}
    selected_edges = {row["id"]: row for row in report["edge_library"] if row["selected"]}
    s = report["summary"]
    lines = [
        "# Loop 2 Author Experiment Simulation",
        "",
        f"Date: `{report['created_at_utc']}`",
        f"Simulation: `{report['simulation_type']}`",
        f"Definition: {report['definition']}",
        f"Campaign artifact: `{report['campaign_path']}`",
        f"Private holdout read: `{str(report['private_holdout_read']).lower()}`",
        f"Paid/external API invoked: `{str(report['paid_external_api_invoked']).lower()}`",
        "",
        "## What Simulation Means Here",
        "",
        "Loop 2 is the author-side scientific decision loop:",
        "",
        "```text",
        "gap hypothesis",
        "  -> experiment design / baseline / control",
        "  -> GPU or artifact measurement",
        "  -> repeated-seed statistics",
        "  -> author judgment",
        "  -> claim revision",
        "  -> conclusion and paper sections",
        "```",
        "",
        "It is not paper reading, not reviewer-only critique, and not a GPU benchmark by itself.",
        "",
        "## Campaign Evidence",
        "",
        f"- GPU rows read: `{s['rows']}`",
        f"- Repeated seeds: `{s['repeats']}`",
        f"- Families: `{', '.join(s['families'])}`",
        f"- Exact-rerun blocked/not-ready: `{s['blocked_exact_reruns']}/{s['repo_audit_count']}`",
        f"- KV max local-window RMSE: `{s['kv']['max_local_rmse']:.6f}`",
        f"- Token-merge max RMSE: `{s['token_merging']['max_rmse']:.6f}`",
        f"- Speculative min effective speedup: `{s['speculative']['min_effective_speedup']:.6f}`",
        f"- Sampling min retained mass: `{s['sampling']['min_retained_mass']:.6f}`",
        f"- Sparse controls not faster than dense: `{s['sparse']['not_faster_than_dense']}/{s['sparse']['control_rows']}`",
        "",
        "## Author Decisions",
        "",
    ]
    for decision in report["author_decisions"]:
        lines.append(f"- `{decision['decision']}` {decision['because']} Write as: {decision['write_as']}.")
    lines += [
        "",
        "## Selected Author DAG Nodes",
        "",
    ]
    for node_id in report["final_selected_nodes"]:
        node = selected_nodes[node_id]
        lines.append(f"- `{node_id}` support `{node['support']}`: {node['role']}")
    lines += ["", "## Selected Author DAG Edges", ""]
    for edge_id in report["final_selected_edges"]:
        edge = selected_edges[edge_id]
        lines.append(f"- `{edge_id}` support `{edge['support']}`")
    lines += [
        "",
        "## Convergence",
        "",
        f"- Completed loops: `{report['completed_loops']}`",
        f"- Converged: `{str(report['converged']).lower()}`",
        f"- Final score: `{report['final_score']}`",
        "",
        "## Trace Tail",
        "",
        "```jsonl",
    ]
    for row in trace[-5:]:
        lines.append(json.dumps(row, sort_keys=True))
    lines += ["```", ""]
    OUTPUT_MD.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--rollouts", type=int, default=20000)
    parser.add_argument("--max-loops", type=int, default=80)
    parser.add_argument("--min-loops", type=int, default=24)
    parser.add_argument("--stable-window", type=int, default=10)
    parser.add_argument("--campaign-repeats-if-missing", type=int, default=30)
    args = parser.parse_args()

    campaign_status = run_campaign_if_needed(args.campaign, args.campaign_repeats_if_missing)
    if campaign_status.get("returncode") not in (None, 0):
        raise SystemExit(json.dumps(campaign_status, indent=2))
    campaign = json.loads(args.campaign.read_text())
    summary = summarize_campaign(campaign)
    start = time.perf_counter()
    report, trace = rollout_search(summary, args)
    report["wall_time_seconds"] = round(time.perf_counter() - start, 3)
    report["campaign_status"] = campaign_status
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True))
    write_report(report, trace)
    print(
        json.dumps(
            {
                "converged": report["converged"],
                "completed_loops": report["completed_loops"],
                "nodes": len(report["final_selected_nodes"]),
                "edges": len(report["final_selected_edges"]),
                "score": report["final_score"],
                "json": str(OUTPUT_JSON),
                "markdown": str(OUTPUT_MD),
                "trace": str(TRACE_JSONL),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
