#!/usr/bin/env python3
"""Long-goal supervisor for DIRS research-gap convergence.

The intended loop is:

1. propose an initial gap-finding DAG from the 20 selected code-fit papers;
2. simulate candidate research questions through that DAG;
3. verify support, novelty boundary, feasibility, and 4090 fit;
4. repair/re-rank until the top gap and graph structure stabilize.

This supervisor is local and deterministic by default. It does not call hosted
model APIs. It repeatedly executes `run_gap_convergence.py`, snapshots each
iteration, and checks stability of the top question plus verifier verdict.
"""

from __future__ import annotations

import json
import os
import hashlib
import shutil
import subprocess
import sys
import time
import re
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
HARNESS = RUN_DIR / "run_gap_convergence.py"
CONTROL_VERIFIER = RUN_DIR / "verify_longgoal_gap_control.py"
ITER_DIR = RUN_DIR / "longgoal_iterations"


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def read_result() -> dict:
    verifier_path = RUN_DIR / "verifier_result.json"
    ranked_path = RUN_DIR / "ranked_research_questions.json"
    graph_path = RUN_DIR / "skill_graph.yaml"
    gpu_probe_path = RUN_DIR / "gpu_probe.json"
    verifier = json.loads(verifier_path.read_text(encoding="utf-8")) if verifier_path.exists() else {}
    ranked = json.loads(ranked_path.read_text(encoding="utf-8")) if ranked_path.exists() else []
    top = ranked[0] if ranked else {}
    graph = graph_path.read_text(encoding="utf-8") if graph_path.exists() else ""
    gpu_probe = json.loads(gpu_probe_path.read_text(encoding="utf-8")) if gpu_probe_path.exists() else {}
    stable_gpus = [
        {
            "index": gpu.get("index"),
            "name": gpu.get("name"),
            "memory_total": gpu.get("memory_total"),
            "driver_version": gpu.get("driver_version"),
        }
        for gpu in gpu_probe.get("gpus", [])
    ]
    # `memory_free` is intentionally excluded: it changes as other processes use
    # the cards and should not make a stable DAG look unstable.
    signature_payload = {
        "graph": re.sub(r'observed_gpus: .*', 'observed_gpus: <stable_gpu_probe>', graph),
        "gpu_available": gpu_probe.get("available"),
        "gpu_count": gpu_probe.get("gpu_count"),
        "stable_gpus": stable_gpus,
    }
    return {
        "top_question": top.get("id") or verifier.get("top_question", {}).get("id"),
        "top_score": top.get("scores", {}).get("total"),
        "verdict": verifier.get("verdict"),
        "confidence": verifier.get("confidence"),
        "graph_signature": hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def stable(history: list[dict], window: int) -> bool:
    if len(history) < window:
        return False
    tail = history[-window:]
    keys = [(x.get("top_question"), x.get("verdict"), x.get("graph_signature")) for x in tail]
    return len(set(keys)) == 1


def run_control_verifier(config: dict) -> dict:
    report_path = RUN_DIR / "longgoal_gap_control_verifier.json"
    result = subprocess.run(
        [
            sys.executable,
            str(CONTROL_VERIFIER),
            "--run-dir",
            str(RUN_DIR),
            "--min-loops",
            str(config["min_loops"]),
            "--stable-window",
            str(config["stable_window"]),
            "--write-report",
            str(report_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {
            "control_verdict": "fail",
            "errors": ["control verifier did not write report"],
            "stdout": result.stdout,
        }
    report["returncode"] = result.returncode
    report["stdout"] = result.stdout
    return report


def write_report(config: dict, history: list[dict], converged: bool, control: dict | None = None) -> None:
    latest = history[-1] if history else {}
    control = control or {"control_verdict": "not_run"}
    report = {
        **config,
        "updated_at_utc": now_utc(),
        "completed_loops": len(history),
        "converged": converged,
        "control_verdict": control.get("control_verdict"),
        "control_errors": control.get("errors", []),
        "control_warnings": control.get("warnings", []),
        "convergence_window": config["stable_window"],
        "stable_top_question": latest.get("top_question"),
        "stable_verdict": latest.get("verdict"),
        "stable_confidence": latest.get("confidence"),
        "loop_history": history,
    }
    write_json(RUN_DIR / "longgoal_convergence_report.json", report)

    lines = [
        "# Long-Goal DIRS Research-Gap Convergence",
        "",
        f"Updated: `{report['updated_at_utc']}`",
        "",
        "## Status",
        "",
        f"- Completed loops: `{report['completed_loops']}`",
        f"- Converged: `{str(converged).lower()}`",
        f"- Control verifier: `{control.get('control_verdict')}`",
        f"- Stable top question: `{latest.get('top_question')}`",
        f"- Stable verdict: `{latest.get('verdict')}`",
        f"- Confidence: `{latest.get('confidence')}`",
        "",
        "## Loop Shape",
        "",
        "```text",
        "loop 1: propose initial research-gap DAG",
        "loop n: simulate candidate questions -> verify -> repair/rerank",
        "stop: minimum loops complete + control verifier passes + top question,",
        "      verdict, and graph signature stable over window",
        "```",
        "",
        "## Current Top Gap",
        "",
        "Can a 7B/8B tool agent improve dynamic multi-turn task success by planning",
        "under an explicit joint budget for tokens, tool latency, GPU time, and",
        "simulator-verified state reliability?",
        "",
    ]
    (RUN_DIR / "LONGGOAL_CONVERGENCE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    duration_seconds = env_int("DIRS_GAP_LONGGOAL_SECONDS", 86400)
    min_loops = env_int("DIRS_GAP_MIN_LOOPS", 12)
    max_loops = env_int("DIRS_GAP_MAX_LOOPS", 1000)
    stable_window = env_int("DIRS_GAP_STABLE_WINDOW", 5)
    loop_sleep_seconds = env_int("DIRS_GAP_LOOP_SLEEP_SECONDS", 0)
    stop_after_converged = os.environ.get("DIRS_GAP_STOP_AFTER_CONVERGED", "0") == "1"
    reset_events = os.environ.get("DIRS_GAP_RESET_EVENTS", "0") == "1"

    if reset_events:
        for path in [
            RUN_DIR / "longgoal_gap_supervisor.events.jsonl",
            RUN_DIR / "longgoal_gap_supervisor.latest.json",
            RUN_DIR / "longgoal_gap_supervisor.ended.json",
            RUN_DIR / "longgoal_convergence_report.json",
            RUN_DIR / "LONGGOAL_CONVERGENCE_REPORT.md",
            RUN_DIR / "longgoal_gap_control_verifier.json",
        ]:
            if path.exists():
                path.unlink()
        if ITER_DIR.exists():
            shutil.rmtree(ITER_DIR)
    ITER_DIR.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    config = {
        "pid": pid,
        "started_at_utc": now_utc(),
        "run_dir": str(RUN_DIR),
        "harness": str(HARNESS),
        "duration_seconds": duration_seconds,
        "min_loops": min_loops,
        "max_loops": max_loops,
        "stable_window": stable_window,
        "loop_sleep_seconds": loop_sleep_seconds,
        "stop_after_converged": stop_after_converged,
        "reset_events": reset_events,
        "runtime": "local_deterministic_dirs_case3_case4_no_api",
        "training_set": "20 selected code-fit papers",
        "focus": "learn how to find a good research gap",
        "control_verifier": str(CONTROL_VERIFIER),
    }
    (RUN_DIR / "longgoal_gap_supervisor.pid").write_text(f"{pid}\n", encoding="utf-8")
    write_json(RUN_DIR / "longgoal_gap_supervisor.status.json", config)
    append_jsonl(RUN_DIR / "longgoal_gap_supervisor.events.jsonl", {"event": "started", **config})

    started = time.time()
    history: list[dict] = []
    loop = 1
    converged = False

    while loop <= max_loops and time.time() - started < duration_seconds:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        snap_dir = ITER_DIR / f"loop_{loop:04d}_{stamp}"
        log_path = snap_dir / "harness.log"
        snap_dir.mkdir(parents=True, exist_ok=True)

        stage = "initial_dag_proposal" if loop == 1 else "simulation_verify_repair"
        event = {"event": "loop_started", "loop": loop, "stage": stage, "started_at_utc": now_utc()}
        append_jsonl(RUN_DIR / "longgoal_gap_supervisor.events.jsonl", event)
        write_json(snap_dir / "loop_started.json", event)

        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run([sys.executable, str(HARNESS)], stdout=log_file, stderr=subprocess.STDOUT, text=True)

        summary = {
            "event": "loop_finished",
            "loop": loop,
            "stage": stage,
            "finished_at_utc": now_utc(),
            "returncode": result.returncode,
            **read_result(),
        }
        history.append(summary)
        append_jsonl(RUN_DIR / "longgoal_gap_supervisor.events.jsonl", summary)
        write_json(snap_dir / "loop_finished.json", summary)
        write_json(RUN_DIR / "longgoal_gap_supervisor.latest.json", summary)

        for name in [
            "README.md",
            "domain_skill_library.json",
            "skill_graph.yaml",
            "node_library.json",
            "edge_library.json",
            "ranked_research_questions.json",
            "gpu_probe.json",
            "gpu_execution_plan.json",
            "verifier_result.json",
            "training_trace.jsonl",
        ]:
            copy_if_exists(RUN_DIR / name, snap_dir / name)

        phase_stable = loop >= min_loops and stable(history, stable_window)
        control = run_control_verifier(config) if phase_stable else {"control_verdict": "not_ready"}
        converged = phase_stable and control.get("control_verdict") == "pass"
        write_report(config, history, converged, control)
        if converged and stop_after_converged:
            break

        if result.returncode != 0:
            time.sleep(10)
        elif loop_sleep_seconds > 0:
            time.sleep(loop_sleep_seconds)
        loop += 1

    ended = {
        "event": "ended",
        "ended_at_utc": now_utc(),
        "completed_loops": len(history),
        "converged": converged,
    }
    append_jsonl(RUN_DIR / "longgoal_gap_supervisor.events.jsonl", ended)
    write_json(RUN_DIR / "longgoal_gap_supervisor.ended.json", ended)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
