#!/usr/bin/env python3
"""Verifier for the DIRS research-gap long-goal supervisor.

This is a control harness, not a score generator. It prevents false convergence
from a run that stops too early, skips the initial DAG proposal phase, or never
executes simulation/verification loops after the first proposal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_SNAPSHOT_FILES = [
    "domain_skill_library.json",
    "skill_graph.yaml",
    "node_library.json",
    "edge_library.json",
    "ranked_research_questions.json",
    "gpu_probe.json",
    "gpu_execution_plan.json",
    "verifier_result.json",
    "training_trace.jsonl",
]

REQUIRED_EXECUTION_NODES = [
    "X0_gpu_probe",
    "X1_repo_readiness",
    "X2_smoke_benchmark_plan",
    "X3_budget_metric_schema",
    "X4_execution_verifier",
]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--min-loops", type=int, required=True)
    parser.add_argument("--stable-window", type=int, required=True)
    parser.add_argument("--write-report", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    events = read_jsonl(run_dir / "longgoal_gap_supervisor.events.jsonl")
    finished = [e for e in events if e.get("event") == "loop_finished"]
    errors = []
    warnings = []

    if len(finished) < args.min_loops:
        errors.append(f"only {len(finished)} finished loops, required {args.min_loops}")

    if not finished or finished[0].get("stage") != "initial_dag_proposal":
        errors.append("loop 1 did not finish as initial_dag_proposal")

    later = finished[1:]
    if not later:
        errors.append("no post-DAG simulation/verification loops finished")
    elif any(e.get("stage") != "simulation_verify_repair" for e in later):
        errors.append("one or more post-loop stages were not simulation_verify_repair")

    if len(finished) >= args.stable_window:
        tail = finished[-args.stable_window :]
        stable_keys = {
            (e.get("top_question"), e.get("verdict"), e.get("graph_signature"))
            for e in tail
        }
        if len(stable_keys) != 1:
            errors.append("tail window is not stable on top_question/verdict/graph_signature")
    else:
        errors.append("not enough loops for requested stability window")

    if any(e.get("returncode") != 0 for e in finished):
        errors.append("at least one loop finished with nonzero return code")

    node_path = run_dir / "node_library.json"
    if not node_path.exists():
        errors.append("missing node_library.json")
        nodes = []
    else:
        nodes = json.loads(node_path.read_text(encoding="utf-8"))
    node_ids = {node.get("id") for node in nodes}
    missing_execution_nodes = [node_id for node_id in REQUIRED_EXECUTION_NODES if node_id not in node_ids]
    if missing_execution_nodes:
        errors.append(f"missing execution DAG nodes: {missing_execution_nodes}")
    for node in nodes:
        if node.get("id") in REQUIRED_EXECUTION_NODES:
            props = node.get("properties", {})
            if not props.get("tools"):
                errors.append(f"execution node {node.get('id')} missing tools property")
            if "gpu_required" not in props:
                errors.append(f"execution node {node.get('id')} missing gpu_required property")
            if not props.get("success_criteria"):
                errors.append(f"execution node {node.get('id')} missing success_criteria")

    gpu_probe_path = run_dir / "gpu_probe.json"
    if not gpu_probe_path.exists():
        errors.append("missing gpu_probe.json")
        gpu_probe = {}
    else:
        gpu_probe = json.loads(gpu_probe_path.read_text(encoding="utf-8"))
    if gpu_probe.get("gpu_count", 0) < 1:
        errors.append("GPU probe did not detect any GPU")

    plan_path = run_dir / "gpu_execution_plan.json"
    if not plan_path.exists():
        errors.append("missing gpu_execution_plan.json")
        gpu_plan = {}
    else:
        gpu_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if gpu_plan.get("status") != "execution_ready":
        errors.append("GPU execution plan is not execution_ready")
    if len(gpu_plan.get("metrics_to_log", [])) < 5:
        errors.append("GPU execution plan has too few budget metrics")

    iter_dir = run_dir / "longgoal_iterations"
    snapshot_dirs = sorted(p for p in iter_dir.glob("loop_*") if p.is_dir())
    if len(snapshot_dirs) < len(finished):
        errors.append("missing snapshot directories for some finished loops")

    for snap in snapshot_dirs[-max(1, min(len(snapshot_dirs), args.stable_window)) :]:
        missing = [name for name in REQUIRED_SNAPSHOT_FILES if not (snap / name).exists()]
        if missing:
            errors.append(f"snapshot {snap.name} missing files: {missing}")

    verifier_path = run_dir / "verifier_result.json"
    if not verifier_path.exists():
        errors.append("missing verifier_result.json")
        verifier = {}
    else:
        verifier = json.loads(verifier_path.read_text(encoding="utf-8"))

    do_not_claim = verifier.get("do_not_claim", [])
    if len(do_not_claim) < 3:
        warnings.append("verifier_result.json has weak do_not_claim guard")

    top = verifier.get("top_question", {})
    required_dimensions = top.get("required_dimensions")
    supported_dimensions = top.get("supported_dimensions")
    if required_dimensions is None or supported_dimensions is None:
        errors.append("top question missing supported/required dimension counts")
    elif supported_dimensions < required_dimensions:
        errors.append("top question is not fully supported by selected-paper dimensions")

    report = {
        "control_verdict": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "finished_loops": len(finished),
        "min_loops": args.min_loops,
        "stable_window": args.stable_window,
        "first_stage": finished[0].get("stage") if finished else None,
        "post_loop_stage_count": len(later),
        "latest_top_question": finished[-1].get("top_question") if finished else None,
        "latest_verdict": finished[-1].get("verdict") if finished else None,
    }

    if args.write_report:
        Path(args.write_report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
