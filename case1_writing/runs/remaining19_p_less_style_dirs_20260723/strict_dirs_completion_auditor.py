#!/usr/bin/env python3
"""Audit strict DIRS long-goal completion from current artifacts.

This is intentionally conservative. A paper is only counted as accepted when
its verifier says it converged under professional gates. A non-converged paper
is only counted as explicitly blocked when the current DAG has the Loop-1
verifier-feedback gates and the latest verifier/queue names concrete
professional blockers. Running jobs remain running, not blocked or complete.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent
PAPER_RUNS = RUN_ROOT / "paper_runs"
SUMMARY_PATH = RUN_ROOT / "remaining19_strict_dirs_summary.json"
QUEUE_PATH = RUN_ROOT / "specialized_runner_queue.json"
AUDIT_JSON = RUN_ROOT / "strict_dirs_completion_audit_20260723.json"
AUDIT_MD = RUN_ROOT / "STRICT_DIRS_COMPLETION_AUDIT_20260723.md"

REQUIRED_DAG_NODES = {
    "loop1.latest_verifier_feedback",
    "ops.non_reduced_artifact_completion_gate",
    "reviewer.compare_simulation_to_paper_evidence_channels",
    "decision.continue_or_block_after_professional_verifier",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def latest_verifier_path(paper_dir: Path) -> Path | None:
    candidates = sorted(paper_dir.glob("verifier_result_iter_*.json"))
    return candidates[-1] if candidates else None


def process_alive(pid: Any) -> bool | None:
    if pid in (None, "", 0):
        return None
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid="],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    return result.returncode == 0 and bool(result.stdout.strip())


def queue_items_by_paper() -> dict[str, dict[str, Any]]:
    if not QUEUE_PATH.exists():
        return {}
    queue_obj = read_json(QUEUE_PATH)
    queue = queue_obj.get("queue", queue_obj if isinstance(queue_obj, list) else [])
    return {item.get("paper_id"): item for item in queue if item.get("paper_id")}


def summary_papers_by_id() -> dict[str, dict[str, Any]]:
    if not SUMMARY_PATH.exists():
        return {}
    summary = read_json(SUMMARY_PATH)
    return {item.get("paper_id"): item for item in summary.get("papers", []) if item.get("paper_id")}


def status_text(verifier: dict[str, Any], queue_item: dict[str, Any] | None, summary_item: dict[str, Any] | None) -> str:
    for source in [verifier, queue_item or {}, summary_item or {}]:
        for key in ["status", "specialized_runner_status", "professional_blocker", "final_status"]:
            value = source.get(key)
            if value:
                return str(value)
    return "unknown"


def verifier_blocked_checks(verifier: dict[str, Any]) -> list[str]:
    names = []
    for check in verifier.get("checks", []):
        if check.get("status") in {"blocked", "fail", "missing", "support_only"}:
            names.append(str(check.get("name")))
    nested = verifier.get("verifier", {})
    if isinstance(nested, dict):
        for key in ["unresolved_professional_debt", "loop1_required_dag_update", "support_only_reasons"]:
            if nested.get(key):
                names.append(key)
    for key in ["support_only_until", "exact_artifact_debt", "blocking_reasons", "blockers", "required_updates"]:
        if verifier.get(key):
            names.append(key)
    return sorted(set(name for name in names if name and name != "None"))


def explicit_blocker(status: str, blocked_checks: list[str], verifier: dict[str, Any]) -> bool:
    if verifier.get("converged"):
        return False
    status_is_blocked = (
        status.startswith("blocked_by_")
        or status.startswith("not_converged_operational_blocker")
        or "blocked" in status
        or "support_only" in status
    )
    return bool(status_is_blocked and blocked_checks)


def fatal_blocked_status(status: str) -> bool:
    return status.startswith("blocked_by_") or status.startswith("not_converged_operational_blocker")


def running_status(status: str, summary_item: dict[str, Any] | None, verifier: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    active = "running" in status
    evidence: dict[str, Any] = {}
    for source_key in ["live_full_gsm8k_runner", "active_run", "runner"]:
        runner = verifier.get(source_key)
        if isinstance(runner, dict):
            evidence.update(
                {
                    f"{source_key}_{k}": v
                    for k, v in runner.items()
                    if k
                    in {
                        "pid",
                        "pid_alive",
                        "paired_completed_samples",
                        "total_samples",
                        "jsonl_rows",
                        "cuda_visible_devices",
                        "full_gsm8k_complete",
                    }
                }
            )
            alive = process_alive(runner.get("pid"))
            if alive is not None:
                evidence[f"{source_key}_pid_alive"] = alive
                active = active or (alive and not runner.get("full_gsm8k_complete"))
    if summary_item:
        runs = summary_item.get("active_specialized_runs", [])
        if runs:
            active = True
    campaign_keys = [
        "gsm8k_protocol_repair_campaign",
        "ablation_grid_campaign",
        "multibenchmark_grid_campaign",
        "table2_acceleration_campaign",
        "dream7b_axis_campaign",
    ]
    running_campaign_status_markers = {
        "running",
        "ready",
        "waiting_for_gpu_capacity",
    }
    terminal_blocked_markers = {
        "explicit",
        "blocked_by_missing",
        "no_runnable_configs",
    }
    for campaign_key in campaign_keys:
        campaign = verifier.get(campaign_key)
        if not isinstance(campaign, dict):
            continue
        campaign_status = str(campaign.get("status") or "")
        running_count = int(campaign.get("running_config_count") or 0)
        pending_count = int(campaign.get("pending_config_count") or 0)
        runnable_count = int(campaign.get("runnable_config_count") or 0)
        completed_count = int(campaign.get("completed_config_count") or 0)
        evidence[f"{campaign_key}_status"] = campaign_status
        evidence[f"{campaign_key}_running_config_count"] = running_count
        evidence[f"{campaign_key}_pending_config_count"] = pending_count
        evidence[f"{campaign_key}_runnable_config_count"] = runnable_count
        evidence[f"{campaign_key}_completed_config_count"] = completed_count
        status_lower = campaign_status.lower()
        status_has_running_marker = any(marker in status_lower for marker in running_campaign_status_markers)
        status_is_terminal_blocked = any(marker in status_lower for marker in terminal_blocked_markers)
        if running_count > 0 or (
            (pending_count > 0 or runnable_count > 0)
            and status_has_running_marker
            and not status_is_terminal_blocked
        ):
            active = True
    for item in verifier.get("unresolved_professional_debt", []) or []:
        if not isinstance(item, dict):
            continue
        debt_status = str(item.get("status") or "")
        if any(marker in debt_status.lower() for marker in running_campaign_status_markers):
            active = True
            evidence.setdefault("running_professional_debt", []).append(
                {
                    "id": item.get("id"),
                    "status": debt_status,
                }
            )
    return active, evidence


def audit_paper(paper_dir: Path, queue_by_id: dict[str, dict[str, Any]], summary_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dag_path = paper_dir / "paper_author_gap_dag.json"
    dag_exists = dag_path.exists()
    dag = read_json(dag_path) if dag_exists else {}
    verifier_path = latest_verifier_path(paper_dir)
    verifier = read_json(verifier_path) if verifier_path else {}
    paper_id = dag.get("target_paper_id") or verifier.get("paper_id") or paper_dir.name
    queue_item = queue_by_id.get(paper_id)
    summary_item = summary_by_id.get(paper_id)
    node_ids = {node.get("id") for node in dag.get("nodes", [])}
    missing_nodes = sorted(REQUIRED_DAG_NODES - node_ids)
    blind_contract = dag.get("blind_contract", {})
    strict_policy = dag.get("strict_policy", {})
    status = status_text(verifier, queue_item, summary_item)
    blocked_checks = verifier_blocked_checks(verifier)
    is_running, running_evidence = running_status(status, summary_item, verifier)
    accepted = bool(verifier.get("converged") and (verifier.get("professional_ready") or verifier.get("professional_package_ready")))
    blocked = explicit_blocker(status, blocked_checks, verifier)
    fatal_blocked = fatal_blocked_status(status)

    if accepted:
        classification = "accepted"
    elif is_running:
        classification = "running"
    elif fatal_blocked and blocked and not missing_nodes:
        classification = "explicitly_blocked"
    elif fatal_blocked and blocked:
        classification = "blocked_but_dag_repair_missing"
    elif blocked and not missing_nodes:
        classification = "explicitly_blocked"
    elif blocked:
        classification = "blocked_but_dag_repair_missing"
    else:
        classification = "unresolved"

    return {
        "paper_id": paper_id,
        "paper_dir": str(paper_dir),
        "classification": classification,
        "status": status,
        "accepted": accepted,
        "explicitly_blocked": classification == "explicitly_blocked",
        "running": classification == "running",
        "dag_path": str(dag_path) if dag_exists else None,
        "verifier_path": str(verifier_path) if verifier_path else None,
        "missing_required_dag_nodes": missing_nodes,
        "blind_contract_dag_only": {
            "paper_text_visible_to_loop2": blind_contract.get("paper_text_visible_to_loop2"),
            "oracle_results_visible_to_loop2": blind_contract.get("oracle_results_visible_to_loop2"),
            "previous_memory_visible_to_loop2": blind_contract.get("previous_memory_visible_to_loop2"),
            "only_input_file": blind_contract.get("only_input_file"),
        },
        "non_reduced_policy": {
            "reduced_or_small_runs_are_convergence_evidence": strict_policy.get("reduced_or_small_runs_are_convergence_evidence"),
            "repo_syntax_or_readme_audit_is_convergence_evidence": strict_policy.get("repo_syntax_or_readme_audit_is_convergence_evidence"),
            "minimum_for_gap_convergence": strict_policy.get("minimum_for_gap_convergence"),
        },
        "blocked_checks": blocked_checks,
        "fatal_blocked_status": fatal_blocked,
        "running_evidence": running_evidence,
    }


def update_summary(report: dict[str, Any]) -> None:
    if not SUMMARY_PATH.exists():
        return
    summary = read_json(SUMMARY_PATH)
    counts = report["counts"]
    summary["completion_audit"] = {
        "path": str(AUDIT_JSON),
        "created_at_utc": report["created_at_utc"],
        "counts": counts,
        "goal_complete": report["goal_complete"],
    }
    summary["accepted_count"] = counts["accepted"]
    summary["blocked_count"] = counts["explicitly_blocked"]
    summary["running_count"] = counts["running"]
    summary["unresolved_count"] = counts["unresolved"]
    summary["updated_at_utc"] = report["created_at_utc"]
    summary["final_status"] = report["final_status"]
    write_json(SUMMARY_PATH, summary)


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Strict DIRS Completion Audit",
        "",
        f"- Updated: `{report['created_at_utc']}`",
        f"- Final status: `{report['final_status']}`",
        f"- Goal complete: `{report['goal_complete']}`",
        f"- Counts: accepted=`{report['counts']['accepted']}`, explicitly_blocked=`{report['counts']['explicitly_blocked']}`, running=`{report['counts']['running']}`, unresolved=`{report['counts']['unresolved']}`",
        "- Completion rule: all 19 papers must be accepted or explicitly blocked under non-reduced professional gates; running papers keep the goal active.",
        "",
        "## Paper Classifications",
        "",
    ]
    for paper in report["papers"]:
        lines.append(
            f"- `{paper['paper_id']}`: `{paper['classification']}` status=`{paper['status']}` "
            f"blocked_checks=`{len(paper['blocked_checks'])}` missing_dag_nodes=`{len(paper['missing_required_dag_nodes'])}`"
        )
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    queue_by_id = queue_items_by_paper()
    summary_by_id = summary_papers_by_id()
    papers = [
        audit_paper(path, queue_by_id, summary_by_id)
        for path in sorted(PAPER_RUNS.iterdir())
        if path.is_dir()
    ]
    counts = {
        "accepted": sum(1 for item in papers if item["classification"] == "accepted"),
        "explicitly_blocked": sum(1 for item in papers if item["classification"] == "explicitly_blocked"),
        "running": sum(1 for item in papers if item["classification"] == "running"),
        "unresolved": sum(1 for item in papers if item["classification"] not in {"accepted", "explicitly_blocked", "running"}),
        "total": len(papers),
    }
    goal_complete = counts["total"] == 19 and counts["running"] == 0 and counts["unresolved"] == 0
    final_status = "complete_all_papers_accepted_or_explicitly_blocked" if goal_complete else "running_professional_two_loop_not_converged"
    report = {
        "artifact_kind": "strict_dirs_completion_audit",
        "created_at_utc": utc_now(),
        "goal_complete": goal_complete,
        "final_status": final_status,
        "counts": counts,
        "requirements": {
            "paper_count_required": 19,
            "current_dag_required_nodes": sorted(REQUIRED_DAG_NODES),
            "loop2_input_contract": "DAG-only; no paper text, oracle values, or previous memory visible to Loop 2",
            "non_reduced_gate": "No reduced/small/proxy/syntax-only/support-only evidence can converge",
        },
        "papers": papers,
    }
    write_json(AUDIT_JSON, report)
    write_markdown(report)
    update_summary(report)
    print(json.dumps({"audit": str(AUDIT_JSON), "markdown": str(AUDIT_MD), "counts": counts, "goal_complete": goal_complete}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
