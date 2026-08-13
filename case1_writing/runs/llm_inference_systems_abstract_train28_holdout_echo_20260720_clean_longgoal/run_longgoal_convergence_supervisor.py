#!/usr/bin/env python3
"""Run repeated local DIRS convergence passes for a long-goal session.

This supervisor intentionally uses the local deterministic harness only. It
does not call external APIs and does not read holdout_private_after_generation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path("/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_abstract_train28_holdout_echo_20260720_clean_longgoal")
HARNESS = Path("/tf/notebooks/yunbo/DIRS/case1_writing/scripts/run_abstract_convergence_harness.py")
ITER_DIR = RUN_DIR / "longgoal_iterations"


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


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


def main() -> int:
    duration_seconds = env_int("DIRS_LONGGOAL_SECONDS", 86400)
    min_loops = env_int("DIRS_MIN_LOOPS", 24)
    max_loops = env_int("DIRS_MAX_LOOPS", 1000)
    mcts_rollouts = env_int("DIRS_MCTS_ROLLOUTS", 5000)
    stable_window = env_int("DIRS_STABLE_WINDOW", 10)
    base_seed = env_int("DIRS_BASE_SEED", 20260720)
    stop_after_converged = os.environ.get("DIRS_STOP_AFTER_CONVERGED", "0") == "1"

    ITER_DIR.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    (RUN_DIR / "longgoal_supervisor.pid").write_text(f"{pid}\n", encoding="utf-8")

    started = time.time()
    config = {
        "pid": pid,
        "started_at_utc": now_utc(),
        "run_dir": str(RUN_DIR),
        "duration_seconds": duration_seconds,
        "min_loops": min_loops,
        "max_loops": max_loops,
        "mcts_rollouts": mcts_rollouts,
        "stable_window": stable_window,
        "base_seed": base_seed,
        "stop_after_converged": stop_after_converged,
        "runtime": "local_deterministic_no_api",
        "blind_rule": "do not open holdout_private_after_generation.json",
    }
    write_json(RUN_DIR / "longgoal_supervisor.status.json", config)
    append_jsonl(RUN_DIR / "longgoal_supervisor.events.jsonl", {"event": "started", **config})

    iteration = 1
    while time.time() - started < duration_seconds:
        seed = base_seed + iteration
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = f"iter_{iteration:04d}_seed_{seed}_{stamp}"
        snap_dir = ITER_DIR / prefix
        log_path = ITER_DIR / f"{prefix}.log"
        snap_dir.mkdir(parents=True, exist_ok=True)

        event = {
            "event": "iteration_started",
            "iteration": iteration,
            "seed": seed,
            "started_at_utc": now_utc(),
            "snapshot_dir": str(snap_dir),
            "log_path": str(log_path),
        }
        write_json(snap_dir / "iteration_started.json", event)
        append_jsonl(RUN_DIR / "longgoal_supervisor.events.jsonl", event)

        cmd = [
            sys.executable,
            str(HARNESS),
            "--run-dir",
            str(RUN_DIR),
            "--max-loops",
            str(max_loops),
            "--min-loops",
            str(min_loops),
            "--mcts-rollouts",
            str(mcts_rollouts),
            "--stable-window",
            str(stable_window),
            "--seed",
            str(seed),
        ]
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

        report_path = RUN_DIR / "convergence_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        copy_if_exists(report_path, snap_dir / "convergence_report.json")
        copy_if_exists(RUN_DIR / "convergence_trace.jsonl", snap_dir / "convergence_trace.jsonl")
        copy_if_exists(RUN_DIR / "CONVERGENCE_REPORT.md", snap_dir / "CONVERGENCE_REPORT.md")

        finished_event = {
            "event": "iteration_finished",
            "iteration": iteration,
            "seed": seed,
            "finished_at_utc": now_utc(),
            "returncode": result.returncode,
            "converged": report.get("converged"),
            "converged_at_loop": report.get("converged_at_loop"),
            "completed_loops": report.get("completed_loops"),
            "final_mean_replay_score": report.get("final_mean_replay_score"),
            "final_min_replay_score": report.get("final_min_replay_score"),
            "snapshot_dir": str(snap_dir),
            "log_path": str(log_path),
        }
        write_json(snap_dir / "iteration_finished.json", finished_event)
        write_json(RUN_DIR / "longgoal_supervisor.latest.json", finished_event)
        append_jsonl(RUN_DIR / "longgoal_supervisor.events.jsonl", finished_event)

        if result.returncode != 0:
            time.sleep(30)
        if stop_after_converged and report.get("converged"):
            break
        iteration += 1

    ended_event = {"event": "ended", "ended_at_utc": now_utc(), "iterations_completed": iteration - 1}
    write_json(RUN_DIR / "longgoal_supervisor.ended.json", ended_event)
    append_jsonl(RUN_DIR / "longgoal_supervisor.events.jsonl", ended_event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
