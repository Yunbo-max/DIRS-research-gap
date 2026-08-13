#!/usr/bin/env python3
"""Diagnostic fallback for the DIRS abstract-writing simulation with LLM roles.

Roles:
- editor: selects/repairs a connected abstract sub-DAG from chip facts and feedback
- simulator: writes the abstract from chip facts and the selected sub-DAG
- evaluator: compares generated text to the training abstract and returns feedback

Normal heavy DIRS runs should use Codex subagents instead of this API runner.
Non-dry-run API execution requires DIRS_ALLOW_OPENAI_API=1 so accidental runs do
not consume hosted API quota.

The held-out original stays private. This script uses training examples only
unless --include-holdout-blind is set, and even then it does not read the
held-out private comparison file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any


NODE_ORDER = [
    "R1_abstract_as_argument",
    "G1_problem_gap",
    "C1_domain_context",
    "O1_named_method_or_object",
    "M1_architecture_or_mechanism",
    "M2_efficiency_or_theory_detail",
    "E1_evaluation_setup",
    "E2_result_outcome",
    "E3_quantitative_anchor",
    "I1_interpretation_or_tradeoff",
    "S1_bounded_takeaway",
    "P1_length_and_placement_prior",
]

REQUIRED_NODES = [
    "R1_abstract_as_argument",
    "G1_problem_gap",
    "O1_named_method_or_object",
    "M1_architecture_or_mechanism",
    "E1_evaluation_setup",
    "E2_result_outcome",
    "I1_interpretation_or_tradeoff",
    "S1_bounded_takeaway",
    "P1_length_and_placement_prior",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def truncate_text(value: Any, limit: int = 1800) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20].rstrip() + " ...[truncated]"


def chip_summary(chip: dict[str, Any]) -> dict[str, Any]:
    meta = chip.get("chip_metadata", {})
    return {
        "chip_id": chip.get("chip_id"),
        "title": meta.get("title") or chip.get("title"),
        "venue": meta.get("venue"),
        "year": meta.get("year"),
        "domain_tags": meta.get("domain_tags", []),
        "problem_gap": truncate_text(chip.get("problem_gap")),
        "method_mechanism": truncate_text(chip.get("method_mechanism")),
        "evaluation_validation": truncate_text(chip.get("evaluation_validation")),
        "experimental_setting": truncate_text(chip.get("experimental_setting")),
        "result_outcome": truncate_text(chip.get("result_outcome")),
        "implementation": truncate_text(chip.get("implementation"), 900),
        "limitations": truncate_text(chip.get("limitations"), 700),
    }


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w.-]+\b", text))


def path_edges(nodes: list[str]) -> list[str]:
    return [f"{a}->{b}" for a, b in zip(nodes, nodes[1:])]


def ordered(nodes: list[str]) -> list[str]:
    node_set = set(nodes)
    return [node for node in NODE_ORDER if node in node_set]


def supported_default_path(chip: dict[str, Any], style_profile: dict[str, Any]) -> list[str]:
    text = json.dumps(chip_summary(chip), ensure_ascii=False).lower()
    nodes = set(REQUIRED_NODES)
    if chip.get("chip_metadata", {}).get("domain_tags"):
        nodes.add("C1_domain_context")
    if any(term in text for term in ["efficient", "efficiency", "linear", "quadratic", "theorem", "bound", "complexity", "state space", "attention"]):
        nodes.add("M2_efficiency_or_theory_detail")
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:%|x|b|m|k|points?|tokens?|samples?|tasks?)?\b", text):
        nodes.add("E3_quantitative_anchor")
    return ordered(list(nodes))


def rows_to_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {str(row["id"]): float(row.get("support_rate", 0.0)) for row in rows}


def edge_pair(edge: str) -> tuple[str, str] | None:
    if "->" not in edge:
        return None
    left, right = edge.split("->", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return None
    return left, right


def chip_text_for_compatibility(chip: dict[str, Any]) -> str:
    return json.dumps(chip_summary(chip), ensure_ascii=False).lower()


def node_compatible(node: str, chip: dict[str, Any]) -> bool:
    if node in REQUIRED_NODES:
        return True
    text = chip_text_for_compatibility(chip)
    if node == "C1_domain_context":
        return bool(chip.get("chip_metadata", {}).get("domain_tags"))
    if node == "M2_efficiency_or_theory_detail":
        return any(term in text for term in ["efficient", "efficiency", "linear", "quadratic", "theorem", "bound", "complexity", "state space", "attention"])
    if node == "E3_quantitative_anchor":
        return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:%|x|b|m|k|points?|tokens?|samples?|tasks?)?\b", text))
    return True


def make_dag_state(run_dir: Path, node_support: list[dict[str, Any]], edge_support: list[dict[str, Any]]) -> dict[str, Any]:
    node_rates = rows_to_rates(node_support)
    edge_rates = rows_to_rates(edge_support)
    convergence_path = run_dir / "convergence_report.json"
    if convergence_path.exists():
        report = read_json(convergence_path)
        nodes = report.get("final_unique_selected_nodes", [])
        edges = report.get("final_unique_selected_edges", [])
    else:
        nodes = [row["id"] for row in node_support]
        edges = [row["id"] for row in edge_support]
    return {
        "nodes": sorted(set(nodes), key=lambda n: NODE_ORDER.index(n) if n in NODE_ORDER else 999),
        "edges": sorted(set(edges)),
        "node_weights": {node: node_rates.get(node, 0.1) for node in nodes},
        "edge_weights": {edge: edge_rates.get(edge, 0.1) for edge in edges},
        "feedback_log": [],
        "loop_updates": [],
    }


def dag_adjacency(dag_state: dict[str, Any], chip: dict[str, Any]) -> dict[str, list[str]]:
    edge_weights = dag_state.get("edge_weights", {})
    adjacency: dict[str, list[str]] = {}
    for edge in dag_state.get("edges", []):
        pair = edge_pair(edge)
        if not pair:
            continue
        src, dst = pair
        if src not in NODE_ORDER or dst not in NODE_ORDER:
            continue
        if NODE_ORDER.index(src) >= NODE_ORDER.index(dst):
            continue
        if not node_compatible(dst, chip):
            continue
        adjacency.setdefault(src, []).append(dst)
    for src in list(adjacency):
        adjacency[src] = sorted(adjacency[src], key=lambda dst: (-edge_weights.get(f"{src}->{dst}", 0.0), NODE_ORDER.index(dst)))
    return adjacency


def path_score(nodes: list[str], chip: dict[str, Any], dag_state: dict[str, Any]) -> float:
    node_weights = dag_state.get("node_weights", {})
    edge_weights = dag_state.get("edge_weights", {})
    edges = path_edges(nodes)
    node_score = statistics.mean([float(node_weights.get(node, 0.0)) for node in nodes]) if nodes else 0.0
    edge_score = statistics.mean([float(edge_weights.get(edge, 0.0)) for edge in edges]) if edges else 0.0
    compat_score = sum(1 for node in nodes if node_compatible(node, chip)) / max(1, len(nodes))
    missing_required = len(set(REQUIRED_NODES) - set(nodes))
    length_prior = 1.0 - min(0.5, abs(len(nodes) - 10) * 0.05)
    return 0.34 * node_score + 0.31 * edge_score + 0.25 * compat_score + 0.10 * length_prior - 0.12 * missing_required


def random_rollout_path(chip: dict[str, Any], dag_state: dict[str, Any], rng: random.Random) -> list[str]:
    adjacency = dag_adjacency(dag_state, chip)
    current = "R1_abstract_as_argument"
    path = [current]
    guard = 0
    while current != "P1_length_and_placement_prior" and guard < len(NODE_ORDER) + 3:
        guard += 1
        choices = adjacency.get(current, [])
        if not choices:
            break
        weights = []
        for dst in choices:
            edge = f"{current}->{dst}"
            weights.append(max(0.01, float(dag_state.get("edge_weights", {}).get(edge, 0.1))))
        current = rng.choices(choices, weights=weights, k=1)[0]
        if current in path:
            break
        path.append(current)
    if path[-1] != "P1_length_and_placement_prior":
        fallback = supported_default_path(chip, {})
        return fallback
    return path


def mcts_candidate_paths(
    chip: dict[str, Any],
    dag_state: dict[str, Any],
    rollouts: int,
    top_k: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    best: dict[tuple[str, ...], float] = {}
    for _ in range(max(1, rollouts)):
        path = random_rollout_path(chip, dag_state, rng)
        key = tuple(path)
        score = path_score(path, chip, dag_state)
        if score > best.get(key, -1e9):
            best[key] = score
    default = tuple(supported_default_path(chip, {}))
    best[default] = max(best.get(default, -1e9), path_score(list(default), chip, dag_state))
    rows = [
        {
            "selected_nodes": list(path),
            "selected_edges": path_edges(list(path)),
            "mcts_score": round(score, 6),
        }
        for path, score in sorted(best.items(), key=lambda item: -item[1])[:top_k]
    ]
    return rows


def json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    obj = re.search(r"(\{.*\})", text, flags=re.S)
    if obj:
        try:
            return json.loads(obj.group(1))
        except Exception:
            pass
    return {"raw_text": text}


class LLMClient:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.client = None
        if not dry_run:
            if os.environ.get("DIRS_ALLOW_OPENAI_API") != "1":
                raise RuntimeError(
                    "Hosted API execution is disabled for DIRS by default. "
                    "Use Codex subagents for heavy runs, pass --dry-run for a "
                    "local smoke test, or set DIRS_ALLOW_OPENAI_API=1 only for "
                    "an explicitly requested API diagnostic."
                )
            try:
                from openai import OpenAI
            except Exception as exc:
                raise RuntimeError("openai package is required for non-dry-run mode") from exc
            self.client = OpenAI()

    def call(self, model: str, system: str, user: str, max_output_tokens: int = 1800) -> str:
        if self.dry_run:
            return json.dumps(
                {
                    "dry_run": True,
                    "model": model,
                    "system_preview": system[:180],
                    "user_preview": user[:500],
                },
                ensure_ascii=False,
            )
        assert self.client is not None
        response = self.client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_output_tokens=max_output_tokens,
        )
        return getattr(response, "output_text", "") or str(response)


def editor_prompt(
    chip: dict[str, Any],
    style_profile: dict[str, Any],
    node_support: list[dict[str, Any]],
    edge_support: list[dict[str, Any]],
    dag_state: dict[str, Any],
    candidate_paths: list[dict[str, Any]],
    feedback: dict[str, Any] | None,
) -> str:
    payload = {
        "task": "Select a connected DIRS abstract sub-DAG for this paper chip.",
        "chip_summary": chip_summary(chip),
        "style_profile": style_profile,
        "node_support": node_support,
        "edge_support": edge_support,
        "current_shared_dag": {
            "nodes": dag_state.get("nodes", []),
            "edges": dag_state.get("edges", []),
            "recent_loop_updates": dag_state.get("loop_updates", [])[-5:],
        },
        "mcts_candidate_paths": candidate_paths,
        "previous_feedback": feedback or {},
        "allowed_node_order": NODE_ORDER,
        "rules": [
            "Use only chip-supported content.",
            "Keep nodes connected and in DAG order.",
            "Prefer one of the MCTS candidate paths unless feedback clearly justifies a repair.",
            "Include enough compatible nodes to reach target length.",
            "Do not read or infer from any held-out original abstract.",
        ],
        "return_json_schema": {
            "selected_nodes": ["node_id"],
            "selected_edges": ["node_a->node_b"],
            "budget_plan": {"context_gap": "words", "method": "words", "evidence": "words", "takeaway": "words"},
            "writing_constraints": ["constraint"],
            "repair_note": "short note",
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def simulator_prompt(chip: dict[str, Any], style_profile: dict[str, Any], editor_plan: dict[str, Any]) -> str:
    payload = {
        "task": "Write a paper abstract from chip facts and the selected DIRS sub-DAG.",
        "chip_summary": chip_summary(chip),
        "target_words": style_profile["recommended_target_words"],
        "target_band": style_profile["recommended_band"],
        "editor_plan": editor_plan,
        "rules": [
            "Write one abstract paragraph only.",
            "No markdown.",
            "Follow selected_nodes in order.",
            "No unsupported claims.",
            "Put result/evidence after method/setup.",
            "End with a bounded takeaway.",
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def evaluator_prompt(
    chip: dict[str, Any],
    generated: str,
    target_abstract: str,
    editor_plan: dict[str, Any],
    style_profile: dict[str, Any],
) -> str:
    payload = {
        "task": "Evaluate generated abstract against the training target and chip facts.",
        "chip_summary": chip_summary(chip),
        "generated_abstract": generated,
        "target_training_abstract": target_abstract,
        "editor_plan": editor_plan,
        "target_band": style_profile["recommended_band"],
        "return_json_schema": {
            "overall_score": "0-1",
            "coverage_score": "0-1",
            "order_score": "0-1",
            "style_score": "0-1",
            "length_score": "0-1",
            "unsupported_claims": ["claim"],
            "missing_supported_claims": ["claim"],
            "no_jump_violations": ["violation"],
            "feedback_for_editor": "specific repair feedback",
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def normalize_editor_plan(raw: dict[str, Any], candidate_paths: list[dict[str, Any]], dag_state: dict[str, Any]) -> dict[str, Any]:
    fallback = candidate_paths[0] if candidate_paths else {
        "selected_nodes": REQUIRED_NODES,
        "selected_edges": path_edges(REQUIRED_NODES),
        "mcts_score": None,
    }
    dag_edges = set(dag_state.get("edges", []))
    nodes = raw.get("selected_nodes")
    if not isinstance(nodes, list):
        nodes = fallback["selected_nodes"]
    nodes = [str(n) for n in nodes if str(n) in NODE_ORDER]
    if not nodes:
        nodes = fallback["selected_nodes"]
    nodes = ordered(nodes)
    inferred_edges = path_edges(nodes)
    raw_edges = raw.get("selected_edges")
    if isinstance(raw_edges, list):
        normalized_edges: list[str] = []
        for edge in raw_edges:
            if isinstance(edge, list) and len(edge) == 2:
                normalized_edges.append(f"{edge[0]}->{edge[1]}")
            else:
                normalized_edges.append(str(edge))
        selected_edges = [edge for edge in normalized_edges if edge in dag_edges]
    else:
        selected_edges = [edge for edge in inferred_edges if edge in dag_edges]
    valid_path_edges = len(selected_edges) == max(0, len(nodes) - 1)
    if not valid_path_edges:
        nodes = list(fallback["selected_nodes"])
        selected_edges = list(fallback["selected_edges"])
    return {
        "selected_nodes": nodes,
        "selected_edges": selected_edges,
        "mcts_candidates": candidate_paths,
        "used_fallback": not valid_path_edges,
        "budget_plan": raw.get("budget_plan", {}),
        "writing_constraints": raw.get("writing_constraints", []),
        "repair_note": raw.get("repair_note", ""),
        "raw_editor": raw,
    }


def evaluator_score(raw: dict[str, Any]) -> float | None:
    value = raw.get("overall_score")
    try:
        score = float(value)
        if 1.0 < score <= 10.0:
            score = score / 10.0
        return score
    except Exception:
        return None


def list_field(raw: dict[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    return value if isinstance(value, list) else []


def update_dag_state(
    dag_state: dict[str, Any],
    chip_id: str,
    loop_idx: int,
    editor_plan: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    score = evaluator_score(evaluation)
    unsupported = list_field(evaluation, "unsupported_claims")
    no_jump = list_field(evaluation, "no_jump_violations")
    missing = list_field(evaluation, "missing_supported_claims")
    multiplier = 1.0
    if score is not None:
        multiplier += (score - 0.75) * 0.08
    if unsupported or no_jump:
        multiplier -= 0.05
    if missing:
        multiplier -= 0.02

    for node in editor_plan.get("selected_nodes", []):
        if node not in dag_state["nodes"]:
            dag_state["nodes"].append(node)
        old = float(dag_state["node_weights"].get(node, 0.1))
        dag_state["node_weights"][node] = round(max(0.01, min(1.25, old * multiplier)), 6)

    for edge in editor_plan.get("selected_edges", []):
        if edge not in dag_state["edges"]:
            dag_state["edges"].append(edge)
        old = float(dag_state["edge_weights"].get(edge, 0.1))
        dag_state["edge_weights"][edge] = round(max(0.01, min(1.25, old * multiplier)), 6)

    update = {
        "loop": loop_idx,
        "chip_id": chip_id,
        "score": score,
        "selected_nodes": editor_plan.get("selected_nodes", []),
        "selected_edges": editor_plan.get("selected_edges", []),
        "unsupported_claim_count": len(unsupported),
        "missing_claim_count": len(missing),
        "no_jump_count": len(no_jump),
        "feedback_for_editor": evaluation.get("feedback_for_editor", ""),
    }
    dag_state["loop_updates"].append(update)
    dag_state["feedback_log"].append(update)


def loop_signature(events: list[dict[str, Any]]) -> str:
    edges = sorted({edge for event in events for edge in event.get("selected_edges", [])})
    nodes = sorted({node for event in events for node in event.get("selected_nodes", [])})
    return "|".join(nodes) + "::" + "|".join(edges)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-name", default="")
    parser.add_argument("--max-loops", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all training samples")
    parser.add_argument("--mcts-rollouts", type=int, default=500)
    parser.add_argument("--candidate-paths", type=int, default=4)
    parser.add_argument("--stable-window", type=int, default=3)
    parser.add_argument("--min-mean-score", type=float, default=0.88)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--editor-model", default=os.environ.get("DIRS_EDITOR_MODEL", "gpt-5-mini"))
    parser.add_argument("--simulator-model", default=os.environ.get("DIRS_SIMULATOR_MODEL", "gpt-5-mini"))
    parser.add_argument("--evaluator-model", default=os.environ.get("DIRS_EVALUATOR_MODEL", "gpt-5-mini"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-holdout-blind", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest = read_json(run_dir / "manifest.json")
    style_profile = read_json(run_dir / "style_profile.json")
    node_support = read_json(run_dir / "node_support_scores.json")
    edge_support = read_json(run_dir / "edge_support_scores.json")
    dag_state = make_dag_state(run_dir, node_support, edge_support)
    rng = random.Random(args.seed)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output_name or f"heavy_llm_{timestamp}"
    out_dir = run_dir / "heavy_llm_runs" / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = [p for p in manifest["papers"] if p["split"] == "train"]
    if args.max_samples > 0:
        papers = papers[: args.max_samples]

    client = LLMClient(dry_run=args.dry_run)
    events_path = out_dir / "events.jsonl"
    feedback_by_chip: dict[str, dict[str, Any]] = {}
    scores: list[float] = []

    config = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "max_loops": args.max_loops,
        "max_samples": args.max_samples or "all",
        "mcts_rollouts": args.mcts_rollouts,
        "candidate_paths": args.candidate_paths,
        "stable_window": args.stable_window,
        "min_mean_score": args.min_mean_score,
        "seed": args.seed,
        "editor_model": args.editor_model,
        "simulator_model": args.simulator_model,
        "evaluator_model": args.evaluator_model,
        "dry_run": args.dry_run,
        "include_holdout_blind": args.include_holdout_blind,
        "heldout_private_rule": "not read by this script",
    }
    write_json(out_dir / "heavy_llm_config.json", config)

    completed_loops = 0
    stable_signature_count = 0
    previous_signature = ""
    converged = False

    for loop_idx in range(1, args.max_loops + 1):
        completed_loops = loop_idx
        loop_events: list[dict[str, Any]] = []
        loop_scores: list[float] = []
        for paper_idx, paper in enumerate(papers, start=1):
            chip = read_json(Path(paper["chip_path"]))
            chip_id = paper["chip_id"]
            candidate_paths = mcts_candidate_paths(chip, dag_state, args.mcts_rollouts, args.candidate_paths, rng)
            feedback = feedback_by_chip.get(chip_id)

            editor_raw_text = client.call(
                args.editor_model,
                "You are the DIRS Loop 1 editor. Return strict JSON only.",
                editor_prompt(chip, style_profile, node_support, edge_support, dag_state, candidate_paths, feedback),
                max_output_tokens=1200,
            )
            editor_plan = normalize_editor_plan(json_from_text(editor_raw_text), candidate_paths, dag_state)

            generated = client.call(
                args.simulator_model,
                "You are the DIRS Loop 2 simulator. Write the abstract only.",
                simulator_prompt(chip, style_profile, editor_plan),
                max_output_tokens=1100,
            ).strip()
            if args.dry_run:
                generated = f"DRY RUN abstract placeholder for {chip_id}."

            eval_raw_text = client.call(
                args.evaluator_model,
                "You are the DIRS evaluator. Return strict JSON only.",
                evaluator_prompt(chip, generated, paper["abstract_text"], editor_plan, style_profile),
                max_output_tokens=1500,
            )
            evaluation = json_from_text(eval_raw_text)
            score = evaluator_score(evaluation)
            if score is not None:
                scores.append(score)
                loop_scores.append(score)
            feedback_by_chip[chip_id] = evaluation
            update_dag_state(dag_state, chip_id, loop_idx, editor_plan, evaluation)

            event = {
                "split": "train",
                "loop": loop_idx,
                "sample_index": paper_idx,
                "chip_id": chip_id,
                "title": paper["title"],
                "editor_model": args.editor_model,
                "simulator_model": args.simulator_model,
                "evaluator_model": args.evaluator_model,
                "mcts_candidate_paths": candidate_paths,
                "selected_nodes": editor_plan["selected_nodes"],
                "selected_edges": editor_plan["selected_edges"],
                "used_fallback": editor_plan["used_fallback"],
                "generated_word_count": word_count(generated),
                "generated_abstract": generated,
                "evaluation": evaluation,
            }
            append_jsonl(events_path, event)
            loop_events.append(event)

        signature = loop_signature(loop_events)
        if signature == previous_signature:
            stable_signature_count += 1
        else:
            stable_signature_count = 1
        previous_signature = signature
        mean_loop_score = statistics.mean(loop_scores) if loop_scores else None
        if (
            stable_signature_count >= args.stable_window
            and mean_loop_score is not None
            and mean_loop_score >= args.min_mean_score
        ):
            converged = True
            break

    holdout_generated = False
    if args.include_holdout_blind:
        holdouts = [p for p in manifest["papers"] if p["split"] == "holdout"]
        if holdouts:
            paper = holdouts[0]
            chip = read_json(Path(paper["chip_path"]))
            candidate_paths = mcts_candidate_paths(chip, dag_state, args.mcts_rollouts, args.candidate_paths, rng)
            editor_raw_text = client.call(
                args.editor_model,
                "You are the DIRS Loop 1 editor for a blind held-out paper. Return strict JSON only.",
                editor_prompt(chip, style_profile, node_support, edge_support, dag_state, candidate_paths, None),
                max_output_tokens=1200,
            )
            editor_plan = normalize_editor_plan(json_from_text(editor_raw_text), candidate_paths, dag_state)
            generated = client.call(
                args.simulator_model,
                "You are the DIRS Loop 2 simulator. Write the held-out abstract only. Do not ask for or use the original abstract.",
                simulator_prompt(chip, style_profile, editor_plan),
                max_output_tokens=1100,
            ).strip()
            if args.dry_run:
                generated = f"DRY RUN held-out abstract placeholder for {paper['chip_id']}."
            holdout_event = {
                "split": "holdout_blind",
                "loop": completed_loops,
                "chip_id": paper["chip_id"],
                "title": paper["title"],
                "editor_model": args.editor_model,
                "simulator_model": args.simulator_model,
                "mcts_candidate_paths": candidate_paths,
                "selected_nodes": editor_plan["selected_nodes"],
                "selected_edges": editor_plan["selected_edges"],
                "used_fallback": editor_plan["used_fallback"],
                "generated_word_count": word_count(generated),
                "generated_abstract": generated,
                "evaluation": {
                    "blind_holdout": True,
                    "target_original_read": False,
                    "feedback_for_editor": "No evaluator comparison before reveal.",
                },
            }
            append_jsonl(events_path, holdout_event)
            holdout_generated = True

    write_json(out_dir / "dag_state.json", dag_state)

    summary = {
        "output_dir": str(out_dir),
        "completed_loops": completed_loops,
        "completed_training_samples": len(papers),
        "total_training_evaluations": completed_loops * len(papers),
        "stable_signature_count": stable_signature_count,
        "converged": converged,
        "dry_run": args.dry_run,
        "models": {
            "editor": args.editor_model,
            "simulator": args.simulator_model,
            "evaluator": args.evaluator_model,
        },
        "score_count": len(scores),
        "mean_overall_score": round(statistics.mean(scores), 4) if scores else None,
        "min_overall_score": round(min(scores), 4) if scores else None,
        "max_overall_score": round(max(scores), 4) if scores else None,
        "holdout_blind_generated": holdout_generated,
        "heldout_private_read": False,
    }
    write_json(out_dir / "summary.json", summary)

    readme = [
        "# Heavy LLM DIRS Abstract Simulation",
        "",
        f"Date: `{dt.datetime.now().strftime('%Y-%m-%d')}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Files",
        "",
        "```text",
        "heavy_llm_config.json",
        "dag_state.json",
        "events.jsonl",
        "summary.json",
        "```",
        "",
        "Held-out private original was not read by this script.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
