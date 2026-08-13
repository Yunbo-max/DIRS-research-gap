#!/usr/bin/env python
"""Refresh long-goal status files from supervisor events and process state."""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import subprocess
from typing import Any


RUN_DIR = pathlib.Path(__file__).resolve().parent
STATUS_JSON = RUN_DIR / "longgoal_supervisor.status.json"
EVENTS_JSONL = RUN_DIR / "longgoal_supervisor.events.jsonl"
LATEST_JSON = RUN_DIR / "longgoal_supervisor.latest.json"
ENDED_JSON = RUN_DIR / "longgoal_supervisor.ended.json"
CONFIG_JSON = RUN_DIR / "longrun_config.json"
STATUS_MD = RUN_DIR / "LONGGOAL_STATUS.md"


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def iso_z(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def read_json(path: pathlib.Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def read_events() -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not EVENTS_JSONL.exists():
        return events
    for line in EVENTS_JSONL.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"event": "malformed_event_line", "raw": line})
    return events


def ps_rows(args: list[str]) -> list[str]:
    proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    lines = [line.rstrip() for line in proc.stdout.splitlines()]
    return lines


def process_status(pid: int | None) -> dict[str, Any]:
    if not pid:
        return {"alive": False, "rows": [], "children": []}
    rows = ps_rows(["ps", "-p", str(pid), "-o", "pid,ppid,stat,etime,pcpu,pmem,cmd"])
    child_rows = ps_rows(["ps", "--ppid", str(pid), "-o", "pid,ppid,stat,etime,pcpu,pmem,cmd"])
    return {
        "alive": len(rows) > 1,
        "rows": rows,
        "children": child_rows[1:] if len(child_rows) > 1 else [],
    }


def latest_finished(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event") == "iteration_finished":
            return event
    return None


def active_started(events: list[dict[str, Any]], latest_done: dict[str, Any] | None) -> dict[str, Any] | None:
    done_iter = latest_done.get("iteration") if latest_done else None
    for event in reversed(events):
        if event.get("event") == "iteration_started" and event.get("iteration") != done_iter:
            return event
        if event.get("event") == "iteration_finished":
            return None
    return None


def iteration_summary(event: dict[str, Any] | None) -> dict[str, Any] | None:
    if not event:
        return None
    keys = [
        "iteration",
        "seed",
        "completed_loops",
        "converged",
        "converged_at_loop",
        "final_mean_replay_score",
        "final_min_replay_score",
        "returncode",
    ]
    return {key: event[key] for key in keys if key in event}


def first_finished(events: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any] | None:
    existing = config.get("first_iteration")
    if existing:
        return existing
    for event in events:
        if event.get("event") == "iteration_finished":
            return iteration_summary(event)
    return None


def all_completed_converged(events: list[dict[str, Any]]) -> bool:
    finished = [event for event in events if event.get("event") == "iteration_finished"]
    return bool(finished) and all(event.get("returncode") == 0 and event.get("converged") is True for event in finished)


def finished_count(events: list[dict[str, Any]]) -> int:
    return sum(1 for event in events if event.get("event") == "iteration_finished")


def snapshot_count() -> int:
    path = RUN_DIR / "longgoal_iterations"
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_dir() and item.name.startswith("iter_"))


def current_state_line(active: dict[str, Any] | None, count: int, ended: dict[str, Any] | None) -> str:
    if ended:
        return f"ended after `{ended.get('iterations_completed', count)}` completed iterations"
    if active:
        if count == 0:
            return f"iteration {active.get('iteration')} running"
        if all_completed_converged(read_events()):
            return f"iteration {active.get('iteration')} started after successful iterations 1 through {count}"
        return f"iteration {active.get('iteration')} running after {count} completed iterations"
    return f"no active child observed after {count} completed iterations"


def code(value: Any) -> str:
    if isinstance(value, bool):
        value = str(value).lower()
    return f"`{value}`"


def build_markdown(
    config: dict[str, Any],
    status: dict[str, Any],
    events: list[dict[str, Any]],
    now: dt.datetime,
    proc_status: dict[str, Any],
    latest_done: dict[str, Any] | None,
    active: dict[str, Any] | None,
    elapsed: int,
    remaining: int,
    ended: dict[str, Any] | None,
) -> str:
    supervisor = config.get("supervisor", {})
    duration = status.get("duration_seconds", supervisor.get("duration_seconds", 86400))
    min_loops = supervisor.get("min_loops", status.get("min_loops", "unknown"))
    mcts = supervisor.get("mcts_rollouts", "unknown")
    stable = supervisor.get("stable_window", "unknown")
    pid = status.get("pid", supervisor.get("pid", "unknown"))
    first = first_finished(events, config) or {}
    latest = iteration_summary(latest_done) or {}
    active_child = proc_status.get("children", [])
    active_cmd = ""
    if active_child:
        active_cmd = active_child[0].split(None, 6)[-1] if len(active_child[0].split(None, 6)) >= 7 else active_child[0]
    elif active:
        active_cmd = "no child process observed at refresh time"

    lines: list[str] = [
        "# Long-Goal Status",
        "",
        f"Status as of {code(iso_z(now))}:",
        "",
        f"- Supervisor PID: {code(pid)}",
        f"- Supervisor alive: {code(str(proc_status.get('alive')).lower())}",
        f"- Runtime target: {code(duration)} seconds",
        f"- Runtime mode: {code(config.get('runtime_mode', 'local_deterministic_no_api'))}",
        f"- Current state: {current_state_line(active, finished_count(events), ended)}",
        f"- Elapsed runtime: about {code(elapsed)} seconds",
        f"- Remaining target runtime: about {code(remaining)} seconds",
        f"- Snapshot directories: {code(snapshot_count())}",
        f"- Domain: {code(config.get('domain', 'LLM Inference / Systems / Token Efficiency'))}",
        f"- Held-out paper: {code(config.get('holdout_chip_id', 'ICML2026_71057_echo_elastic_speculative_decoding'))}",
        f"- Training examples: {code(config.get('training_examples', 28))}",
        f"- Blind rule: do not open {code('holdout_private_after_generation.json')}",
        "",
        "## First Iteration Result",
        "",
    ]
    for key, label in [
        ("iteration", "Iteration"),
        ("seed", "Seed"),
        ("completed_loops", "Completed loops"),
        ("converged", "Converged"),
        ("converged_at_loop", "Converged at loop"),
        ("final_mean_replay_score", "Final mean replay score"),
        ("final_min_replay_score", "Final minimum replay score"),
        ("target_words_from_training", "Target abstract length"),
        ("target_band_from_training", "Target band"),
    ]:
        if key in first:
            value = first[key]
            if key == "target_words_from_training":
                lines.append(f"- {label}: {code(value)} words")
            elif key == "target_band_from_training" and isinstance(value, list) and len(value) == 2:
                lines.append(f"- {label}: {code(value[0])} to {code(value[1])} words")
            else:
                lines.append(f"- {label}: {code(value)}")
    lines.extend(
        [
            f"- Minimum loops required: {code(min_loops)}",
            f"- MCTS rollouts per example: {code(mcts)}",
            f"- Stable window: {code(stable)}",
            "",
            "## Latest Completed Iteration",
            "",
        ]
    )
    if latest:
        for key, label in [
            ("iteration", "Iteration"),
            ("seed", "Seed"),
            ("completed_loops", "Completed loops"),
            ("converged", "Converged"),
            ("converged_at_loop", "Converged at loop"),
            ("final_mean_replay_score", "Final mean replay score"),
            ("final_min_replay_score", "Final minimum replay score"),
        ]:
            if key in latest:
                lines.append(f"- {label}: {code(latest[key])}")
        lines.extend(
            [
                f"- Minimum loops required: {code(min_loops)}",
                f"- MCTS rollouts per example: {code(mcts)}",
                f"- Stable window: {code(stable)}",
            ]
        )
    else:
        lines.append("- No completed iteration has been recorded yet.")
    lines.extend(["", "## Active Iteration", ""])
    if active:
        lines.extend(
            [
                f"- Iteration: {code(active.get('iteration'))}",
                f"- Seed: {code(active.get('seed'))}",
                f"- Started at: {code(active.get('started_at_utc'))}",
                f"- Observed running at: {code(iso_z(now))}",
                f"- Child harness observed: {code(str(bool(active_child)).lower())}",
                f"- Child harness command: {code(active_cmd)}",
            ]
        )
    else:
        lines.append("- No active iteration was observed at refresh time.")
    lines.extend(
        [
            "",
            "## Two-Loop Process",
            "",
            "Loop 1 updates and audits the learned abstract DAG prior from the 28 training abstracts.",
            "",
            "Loop 2 runs connected-subgraph selection and replay evaluation through the MCTS-style harness, then records whether the graph is stable across the required window.",
            "",
            "The long supervisor repeats these two loops with fresh seeds and snapshots every pass under `longgoal_iterations/`.",
            "",
            "## Public Audit Notes",
            "",
            "The cleaned split keeps the ECHO holdout out of `training_trace.json`, with no public `abstract_text` for the holdout. The remaining extraction issues are minor scars in a few training abstracts rather than severe figure, introduction, or arXiv spillover.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    status = read_json(STATUS_JSON, {})
    config = read_json(CONFIG_JSON, {})
    events = read_events()
    latest_done = latest_finished(events)
    active = active_started(events, latest_done)
    ended = read_json(ENDED_JSON, None) if ENDED_JSON.exists() else None
    now = utc_now()

    supervisor = config.setdefault("supervisor", {})
    pid = status.get("pid", supervisor.get("pid"))
    proc = process_status(pid if isinstance(pid, int) else None)
    started_raw = status.get("started_at_utc", supervisor.get("started_at_utc"))
    started = parse_utc(started_raw) if started_raw else now
    duration = int(status.get("duration_seconds", supervisor.get("duration_seconds", 86400)))
    elapsed = max(0, round((now - started).total_seconds()))
    remaining = max(0, duration - elapsed)

    supervisor.update(
        {
            "pid": pid,
            "started_at_utc": started_raw,
            "last_observed_at_utc": iso_z(now),
            "last_observed_elapsed_seconds": elapsed,
            "last_observed_remaining_seconds": remaining,
            "duration_seconds": duration,
            "supervisor_alive": bool(proc.get("alive")),
            "child_processes": proc.get("children", []),
            "snapshot_count": snapshot_count(),
        }
    )
    if latest_done:
        config["latest_completed_iteration"] = iteration_summary(latest_done)
    if active:
        config["active_iteration"] = {
            "iteration": active.get("iteration"),
            "seed": active.get("seed"),
            "started_at_utc": active.get("started_at_utc"),
            "observed_at_utc": iso_z(now),
        }
    else:
        config["active_iteration"] = None
    if ended:
        config["ended"] = ended

    CONFIG_JSON.write_text(json.dumps(config, indent=2) + "\n")
    STATUS_MD.write_text(
        build_markdown(config, status, events, now, proc, latest_done, active, elapsed, remaining, ended)
    )
    print(
        json.dumps(
            {
                "observed_at_utc": iso_z(now),
                "supervisor_alive": proc.get("alive"),
                "latest_completed_iteration": config.get("latest_completed_iteration"),
                "active_iteration": config.get("active_iteration"),
                "elapsed_seconds": elapsed,
                "remaining_seconds": remaining,
                "snapshot_count": supervisor.get("snapshot_count"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
