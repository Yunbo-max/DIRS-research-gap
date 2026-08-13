#!/usr/bin/env python3
"""Run GPU blind simulation and force verifier failures back into the DAG.

This script implements the stricter policy:

- A verifier `partial`, `fail`, or `blocked` status is not convergence.
- The loop writes a DAG update request and injects missing nodes into the next
  DAG version.
- If the same exact-table blocker repeats, the run stops as blocked, not as
  success, with concrete success criteria for the next real reproduction pass.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
BASE_DAG = RUN_DIR / "paper_author_dag.json"
ORACLE = RUN_DIR / "paper_oracle_results.json"
SOURCE_SCRIPT = RUN_DIR / "run_onepaper_dag_blind_gpu_table_verify.py"
LOOP_DIR = RUN_DIR / "gpu_table_dag_update_loop"
OUTPUT_JSON = RUN_DIR / "onepaper_dag_blind_gpu_update_loop_summary.json"
OUTPUT_MD = RUN_DIR / "ONEPAPER_DAG_BLIND_GPU_UPDATE_LOOP_REPORT.md"


def load_gpu_module():
    spec = importlib.util.spec_from_file_location("gpu_table_verify", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SOURCE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:16]


def append_node_once(dag: dict, node: dict) -> None:
    if not any(existing.get("id") == node["id"] for existing in dag["nodes"]):
        dag["nodes"].append(node)


def append_edge_once(dag: dict, edge: list[str]) -> None:
    if edge not in dag["edges"]:
        dag["edges"].append(edge)


def update_dag_from_verifier(dag: dict, verification: dict, iteration: int) -> tuple[dict, list[dict]]:
    next_dag = copy.deepcopy(dag)
    updates = []
    statuses = {check["name"]: check["status"] for check in verification["checks"]}
    details = {check["name"]: check["detail"] for check in verification["checks"]}

    if statuses.get("paper_table3_gpu_timing_shape") == "partial":
        update = {
            "id": "update.require_full_generation_timing_pipeline",
            "reason": "GPU sampler-only proxy did not exactly match the paper Table 3 timing shape discussed in the efficiency paragraph.",
            "success_criteria": [
                "run Mistral-7B generations on GSM8K/GPQA or paper-equivalent prompts",
                "measure average sampling time per token for epsilon, eta, min-p, mirostat, top-p, and p-less",
                "compare p-less seconds/token against paper Table 3 anchors",
            ],
            "verifier_detail": details["paper_table3_gpu_timing_shape"],
        }
        updates.append(update)
        append_node_once(
            next_dag,
            {
                "id": "exp.full_generation_timing_table3",
                "type": "required_exact_reproduction_experiment",
                "purpose": "reproduce paper Table 3 using model generation, not sampler-only tensor probes",
                "requires": ["Mistral-7B or equivalent checkpoint", "GSM8K/GPQA prompts", "full generation harness", "per-token sampler instrumentation"],
                "success_criteria": update["success_criteria"],
            },
        )
        append_edge_once(next_dag, ["exp.sampling_efficiency_profile", "exp.full_generation_timing_table3"])
        append_edge_once(next_dag, ["exp.full_generation_timing_table3", "decision.claim_revision"])

    if statuses.get("paper_figures16_17_cpu_ram") == "blocked" or statuses.get("paper_appendix_table15_cpu_ram_values") == "blocked":
        update = {
            "id": "update.require_cpu_ram_profile_figures16_17_table15",
            "reason": "Verifier cannot compare the simulation to the paper's CPU/RAM figures and appendix Table 15 without CPU-time and memory instrumentation.",
            "success_criteria": [
                "instrument top-p, min-p, and p-less CPU processing time during generation",
                "record RAM usage with the same binning/aggregation used for Figures 16 and 17 where possible",
                "emit a Table 15-style CPU time and RAM summary",
                "compare p-less CPU time and RAM against top-p and min-p paper anchors",
            ],
            "verifier_detail": details.get("paper_figures16_17_cpu_ram") or details.get("paper_appendix_table15_cpu_ram_values"),
        }
        updates.append(update)
        append_node_once(
            next_dag,
            {
                "id": "exp.cpu_ram_profile_figures16_17_table15",
                "type": "required_figure_appendix_reproduction_experiment",
                "purpose": "reproduce or approximate paper Figures 16/17 and Table 15 CPU/RAM profiling",
                "requires": ["generation loop instrumentation", "CPU timing", "RAM sampler/process memory tracking", "top-p/min-p/p-less implementations"],
                "success_criteria": update["success_criteria"],
            },
        )
        append_edge_once(next_dag, ["exp.sampling_efficiency_profile", "exp.cpu_ram_profile_figures16_17_table15"])
        append_edge_once(next_dag, ["exp.cpu_ram_profile_figures16_17_table15", "decision.claim_revision"])

    if statuses.get("exact_numeric_reproduction") == "blocked":
        update = {
            "id": "update.require_exact_table_reproduction_artifacts",
            "reason": "Verifier cannot accept exact numeric paper-table, paragraph, or figure reproduction from proxy measurements alone.",
            "success_criteria": [
                "obtain or build the full benchmark harness",
                "run Llama-2-7B, Mistral-7B, and Llama3-70B settings where feasible",
                "recompute Table 1 AUC, Table 2 Writing Prompts win-rate, Table 3 sampling-time tables, Figure 2 curves, and Figures 16/17 CPU/RAM profiles",
                "store raw generations, scoring scripts, seeds, and hardware logs",
            ],
            "verifier_detail": details["exact_numeric_reproduction"],
        }
        updates.append(update)
        append_node_once(
            next_dag,
            {
                "id": "artifact.exact_reproduction_package",
                "type": "required_artifact_package",
                "purpose": "make exact paper-table comparison possible rather than proxy-only",
                "requires": ["datasets", "prompts", "model checkpoints", "scoring/evaluation scripts", "raw outputs", "hardware/runtime logs"],
                "success_criteria": update["success_criteria"],
            },
        )
        append_edge_once(next_dag, ["artifact.full_eval_pipeline_gate", "artifact.exact_reproduction_package"])
        append_edge_once(next_dag, ["artifact.exact_reproduction_package", "decision.main_result_shape"])

    failed = [name for name, status in statuses.items() if status == "fail"]
    for name in failed:
        update = {
            "id": f"update.fix_{name}",
            "reason": f"Verifier check failed: {name}.",
            "success_criteria": ["add missing DAG experiment/control details until blind simulator can recover the paper-shaped result"],
            "verifier_detail": details[name],
        }
        updates.append(update)

    if updates:
        next_dag.setdefault("verifier_feedback_history", []).append(
            {
                "iteration": iteration,
                "created_at_utc": now_utc(),
                "updates": updates,
            }
        )
        next_dag["detail_level"] = max(int(next_dag.get("detail_level", 0)), 6)
        next_dag["dag_id"] = f"{next_dag.get('dag_id', 'p_less_dag')}_updated_iter_{iteration}"
        next_dag["signature"] = stable_hash(
            {
                "nodes": next_dag["nodes"],
                "edges": next_dag["edges"],
                "recipes": next_dag.get("experiment_recipes", {}),
                "feedback": next_dag.get("verifier_feedback_history", []),
            }
        )
    return next_dag, updates


def run_blind_iteration(mod, dag: dict, iter_dir: Path, args) -> tuple[dict, float]:
    blind = iter_dir / "blind_workspace"
    blind.mkdir(parents=True, exist_ok=True)
    dag_path = blind / "paper_author_dag.json"
    dag_path.write_text(json.dumps(dag, indent=2, sort_keys=True))
    shutil.copy2(mod.SIMULATOR, blind / "blind_gpu_table_simulator_from_dag_only.py")
    cmd = [
        sys.executable,
        "blind_gpu_table_simulator_from_dag_only.py",
        "--dag",
        "paper_author_dag.json",
        "--output",
        "blind_gpu_simulation_result.json",
        "--batch",
        str(args.batch),
        "--vocab",
        str(args.vocab),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=blind, text=True, capture_output=True, check=False)
    runtime = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"blind GPU sim failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    sim = json.loads((blind / "blind_gpu_simulation_result.json").read_text())
    sim["subprocess"] = {"runtime_seconds": round(runtime, 3), "returncode": proc.returncode}
    (blind / "blind_gpu_simulation_result.json").write_text(json.dumps(sim, indent=2, sort_keys=True))
    return sim, runtime


def write_report(summary: dict) -> None:
    lines = [
        "# One-Paper Blind GPU DAG Update Loop",
        "",
        f"Date: `{summary['created_at_utc']}`",
        f"Target: `{summary['target']}`",
        f"Final status: `{summary['status']}`",
        f"Iterations: `{len(summary['iterations'])}`",
        f"Total GPU simulator runtime seconds: `{summary['total_gpu_runtime_seconds']}`",
        "",
        "## Policy",
        "",
        "`partial`, `fail`, and `blocked` verifier outputs do not count as convergence. They must become DAG update requests.",
        "",
        "## Iterations",
        "",
    ]
    for item in summary["iterations"]:
        status_line = ", ".join(f"{c['name']}={c['status']}" for c in item["verification"]["checks"])
        lines.append(f"- Iteration `{item['iteration']}` score `{item['verification']['score']}` updates `{len(item['dag_updates'])}`: {status_line}")
    lines += [
        "",
        "## Final Required DAG Updates",
        "",
    ]
    for update in summary["final_required_updates"]:
        lines.append(f"- `{update['id']}`: {update['reason']}")
        lines.append(f"  Success criteria: {'; '.join(update['success_criteria'])}")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Summary JSON: `{OUTPUT_JSON}`",
        f"- Loop directory: `{LOOP_DIR}`",
    ]
    OUTPUT_MD.write_text("\n".join(lines))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--stable-blocker-window", type=int, default=2)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    mod = load_gpu_module()
    mod.ensure_simulator()
    oracle = json.loads(ORACLE.read_text())
    dag = json.loads(BASE_DAG.read_text())
    if LOOP_DIR.exists():
        shutil.rmtree(LOOP_DIR)
    LOOP_DIR.mkdir(parents=True, exist_ok=True)

    iterations = []
    blocker_signature = None
    stable_blocker_count = 0
    total_runtime = 0.0
    final_required_updates = []
    status = "max_iterations_reached"

    for iteration in range(1, args.max_iterations + 1):
        iter_dir = LOOP_DIR / f"iter_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "paper_author_dag_input.json").write_text(json.dumps(dag, indent=2, sort_keys=True))
        sim, runtime = run_blind_iteration(mod, dag, iter_dir, args)
        total_runtime += runtime
        verification = mod.verify_against_paper_tables(sim, oracle)
        (iter_dir / "verification_result.json").write_text(json.dumps(verification, indent=2, sort_keys=True))
        next_dag, updates = update_dag_from_verifier(dag, verification, iteration)
        (iter_dir / "dag_update_request.json").write_text(json.dumps(updates, indent=2, sort_keys=True))
        (iter_dir / "paper_author_dag_updated.json").write_text(json.dumps(next_dag, indent=2, sort_keys=True))
        statuses = tuple(sorted((check["name"], check["status"]) for check in verification["checks"] if check["status"] != "pass"))
        sig = stable_hash(statuses)
        stable_blocker_count = stable_blocker_count + 1 if sig == blocker_signature else 1
        blocker_signature = sig
        iterations.append(
            {
                "iteration": iteration,
                "gpu_runtime_seconds": round(runtime, 3),
                "verification": verification,
                "dag_updates": updates,
                "blocking_status_signature": sig,
                "stable_blocker_count": stable_blocker_count,
                "paths": {
                    "iteration_dir": str(iter_dir),
                    "blind_workspace": str(iter_dir / "blind_workspace"),
                    "dag_update_request": str(iter_dir / "dag_update_request.json"),
                },
            }
        )
        final_required_updates = updates
        if not updates:
            status = "converged_no_table_mismatch"
            dag = next_dag
            break
        dag = next_dag
        if stable_blocker_count >= args.stable_blocker_window:
            status = "blocked_waiting_for_exact_artifacts_after_dag_update"
            break

    summary = {
        "created_at_utc": now_utc(),
        "target": oracle["chip_id"],
        "status": status,
        "total_gpu_runtime_seconds": round(total_runtime, 3),
        "blind_simulator_only_input": "paper_author_dag.json",
        "iterations": iterations,
        "final_required_updates": final_required_updates,
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_report(summary)
    print(
        json.dumps(
            {
                "status": status,
                "iterations": len(iterations),
                "total_gpu_runtime_seconds": round(total_runtime, 3),
                "final_required_update_count": len(final_required_updates),
                "report": str(OUTPUT_MD),
                "summary_json": str(OUTPUT_JSON),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
