#!/usr/bin/env python3
"""Feed strict verifier failures back into the paper-specific gap DAGs.

Loop 2 is allowed to see the DAG, encoded repo paths, and artifacts produced
from those DAG instructions. It must not browse prior memory or paper text.
This pass keeps that contract honest: verifier failures are converted into
operational DAG nodes that name missing artifact classes, evidence channels,
and continuation gates without exposing oracle result values.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent
PAPER_RUNS = RUN_ROOT / "paper_runs"
SUMMARY_PATH = RUN_ROOT / "remaining19_strict_dirs_summary.json"
QUEUE_PATH = RUN_ROOT / "specialized_runner_queue.json"
REPORT_JSON = RUN_ROOT / "loop1_dag_repair_audit_20260723.json"
REPORT_MD = RUN_ROOT / "LOOP1_DAG_REPAIR_AUDIT_20260723.md"

PAPER_EVIDENCE_CHANNELS = [
    "paper tables",
    "paper figures",
    "result paragraphs",
    "appendix tables/figures",
    "released code/config/checkpoint/data artifacts",
    "runtime/hardware traces",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def latest_dag_path(paper_dir: Path) -> Path:
    iter_paths = sorted(paper_dir.glob("paper_author_gap_dag_iter_*.json"))
    return iter_paths[-1] if iter_paths else paper_dir / "paper_author_gap_dag.json"


def next_iter_path(paper_dir: Path) -> Path:
    max_iter = 0
    for path in paper_dir.glob("paper_author_gap_dag_iter_*.json"):
        match = re.search(r"_iter_(\d+)\.json$", path.name)
        if match:
            max_iter = max(max_iter, int(match.group(1)))
    return paper_dir / f"paper_author_gap_dag_iter_{max_iter + 1:02d}.json"


def latest_verifier_path(paper_dir: Path) -> Path | None:
    candidates = sorted(paper_dir.glob("verifier_result_iter_*.json"))
    return candidates[-1] if candidates else None


def node_by_id(dag: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    for node in dag.setdefault("nodes", []):
        if node.get("id") == node_id:
            return node
    return None


def ensure_node(dag: dict[str, Any], node: dict[str, Any]) -> bool:
    existing = node_by_id(dag, node["id"])
    if existing is None:
        dag.setdefault("nodes", []).append(node)
        return True
    changed = existing != {**existing, **node}
    existing.update(node)
    return changed


def ensure_edge(dag: dict[str, Any], source: str, target: str) -> bool:
    edge = [source, target]
    if edge in dag.setdefault("edges", []):
        return False
    dag["edges"].append(edge)
    return True


def signature_for(dag: dict[str, Any]) -> str:
    payload = {
        "nodes": dag.get("nodes", []),
        "edges": dag.get("edges", []),
        "strict_policy": dag.get("strict_policy", {}),
        "blind_contract": dag.get("blind_contract", {}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def compact_detail(detail: Any) -> Any:
    if isinstance(detail, str):
        return detail[:600]
    if isinstance(detail, list):
        out = []
        for item in detail[:12]:
            if isinstance(item, dict):
                out.append(
                    {
                        key: compact_detail(value)
                        for key, value in item.items()
                        if key
                        in {
                            "blocked_config_ids",
                            "detail",
                            "id",
                            "manifest_path",
                            "name",
                            "needed",
                            "linked_existing_artifact_count",
                            "completed_config_count",
                            "pending_config_count",
                            "reason",
                            "required",
                            "runnable_config_count",
                            "status",
                        }
                    }
                )
            else:
                out.append(str(item)[:300])
        return out
    if isinstance(detail, dict):
        out = {}
        for key, value in detail.items():
            if key in {
                "blocked_config_ids",
                "blockers",
                "explicit_debt_comparison",
                "manifest_path",
                "needed",
                "linked_existing_artifact_count",
                "completed_config_count",
                "paper_shaped_outputs_required",
                "pending_config_count",
                "reason",
                "required",
                "runnable_config_count",
                "status",
                "support_only_until",
            } or (
                isinstance(value, dict)
                and any(nested_key in value for nested_key in {"status", "manifest_path", "needed"})
            ):
                out[key] = compact_detail(value)
        return out or str(detail)[:600]
    return detail


def collect_verifier_feedback(verifier: dict[str, Any]) -> dict[str, Any]:
    blocked_checks = []
    for check in verifier.get("checks", []):
        status = check.get("status")
        detail = check.get("detail")
        # Running checks with explicit blocker/debt lists still need to be
        # routed into Loop 1, but only as artifact requirements. The compact
        # detail filter below strips observed/oracle values before the DAG is
        # written for Loop 2.
        has_running_debt = (
            status == "running"
            and isinstance(detail, dict)
            and bool(
                detail.get("blockers")
                or detail.get("required")
                or detail.get("needed")
                or detail.get("pending_config_count")
                or detail.get("blocked_config_ids")
                or detail.get("manifest_path")
            )
        )
        if status in {"blocked", "fail", "missing", "support_only"} or has_running_debt:
            blocked_checks.append(
                {
                    "name": check.get("name"),
                    "status": status,
                    "detail": compact_detail(detail),
                }
            )

    nested = verifier.get("verifier", {})
    if isinstance(nested, dict):
        for key in ["unresolved_professional_debt", "loop1_required_dag_update", "support_only_reasons"]:
            value = nested.get(key)
            if value:
                blocked_checks.append({"name": key, "status": "blocked", "detail": compact_detail(value)})

    for key in [
        "paper_result_comparison",
        "unresolved_professional_debt",
        "ablation_grid_campaign",
        "multibenchmark_grid_campaign",
        "table2_acceleration_campaign",
        "dream7b_axis_campaign",
        "trajectory_metric_semantics_audit",
        "support_only_until",
        "exact_artifact_debt",
        "blocking_reasons",
        "blockers",
    ]:
        value = verifier.get(key)
        if value:
            blocked_checks.append({"name": key, "status": "blocked", "detail": compact_detail(value)})

    required_updates = []
    for update in verifier.get("required_updates", []):
        if isinstance(update, dict):
            item = {
                "id": update.get("id"),
                "reason": update.get("reason"),
                "success_criteria": compact_detail(update.get("success_criteria", [])),
            }
            for extra_key in [
                "artifact_family",
                "blocked_check",
                "failed_metric_ids",
                "failed_simulation_setting_ids",
                "manifest_path",
                "oracle_values_exposed_to_loop2",
                "protocol_finding_statuses",
                "repair_axis_ids",
                "report_path",
                "risk_audit_status",
            ]:
                if extra_key in update and update.get(extra_key) not in (None, "", []):
                    value = update.get(extra_key)
                    if extra_key in {"failed_metric_ids", "failed_simulation_setting_ids", "repair_axis_ids"} and isinstance(value, list):
                        item[extra_key] = [str(entry)[:200] for entry in value]
                    elif extra_key == "protocol_finding_statuses" and isinstance(value, dict):
                        item[extra_key] = {
                            str(status_key)[:200]: compact_detail(status_value)
                            for status_key, status_value in value.items()
                        }
                    else:
                        item[extra_key] = compact_detail(value)
            required_updates.append(item)
        else:
            text = str(update)
            required_updates.append(
                {
                    "id": f"update.{slug(text)[:80]}",
                    "reason": text,
                    "success_criteria": [text],
                }
            )

    return {
        "status": verifier.get("status") or verifier.get("final_status") or verifier.get("professional_blocker"),
        "converged": bool(verifier.get("converged")),
        "professional_ready": bool(verifier.get("professional_ready") or verifier.get("professional_package_ready")),
        "blocked_checks": blocked_checks,
        "required_updates": required_updates,
    }


def update_dag_with_feedback(paper_dir: Path, verifier_path: Path, queue_item: dict[str, Any] | None) -> dict[str, Any]:
    dag_path = latest_dag_path(paper_dir)
    dag = read_json(dag_path)
    verifier = read_json(verifier_path)
    feedback = collect_verifier_feedback(verifier)
    paper_id = dag.get("target_paper_id") or verifier.get("paper_id") or paper_dir.name
    short = slug(paper_id)
    now = utc_now()
    existing_feedback = node_by_id(dag, "loop1.latest_verifier_feedback")
    already_synced = bool(existing_feedback and verifier_path.name in str(existing_feedback.get("content", "")))
    planned_iter_path = dag_path if already_synced and dag_path.name.startswith("paper_author_gap_dag_iter_") else next_iter_path(paper_dir)
    match = re.search(r"_iter_(\d+)\.json$", planned_iter_path.name)
    iteration_number = int(match.group(1)) if match else 0

    dag["updated_at_utc"] = now
    dag["graph_id"] = f"{paper_id}_gap_dag_iter_{iteration_number:02d}"
    dag.setdefault("blind_contract", {}).update(
        {
            "only_input_file": "paper_author_gap_dag.json",
            "paper_text_visible_to_loop2": False,
            "oracle_results_visible_to_loop2": False,
            "previous_memory_visible_to_loop2": False,
            "verifier_feedback_visible_only_as_encoded_dag_nodes": True,
        }
    )
    dag.setdefault("strict_policy", {}).update(
        {
            "loop2_may_use_only_encoded_dag_dependencies": True,
            "reduced_or_small_runs_are_convergence_evidence": False,
            "repo_syntax_or_readme_audit_is_convergence_evidence": False,
            "minimum_for_gap_convergence": (
                "DAG-only Loop 2 must produce paper-shaped operational artifacts; "
                "reviewer then compares gap and result shape against paper tables, figures, paragraphs, appendix, and released artifacts."
            ),
        }
    )

    queue_status = (queue_item or {}).get("specialized_runner_status") or (queue_item or {}).get("professional_blocker")
    repo_paths = (queue_item or {}).get("repo_paths", [])
    blocker_names = [
        item.get("name")
        for item in feedback["blocked_checks"]
        if item.get("name")
    ]
    update_ids = [item.get("id") for item in feedback["required_updates"] if item.get("id")]

    changed = False
    changed |= ensure_node(
        dag,
        {
            "id": "loop1.latest_verifier_feedback",
            "type": "verifier_feedback_packet",
            "skill_role": "feed Loop 2 failures back into the DAG without exposing oracle values",
            "content": (
                f"latest_verifier={verifier_path.name}; status={feedback['status'] or queue_status}; "
                f"converged={feedback['converged']}; professional_ready={feedback['professional_ready']}; "
                f"blocked_checks={blocker_names}; required_updates={update_ids}"
            ),
            "verifier_feedback": feedback,
        },
    )
    changed |= ensure_node(
        dag,
        {
            "id": "ops.non_reduced_artifact_completion_gate",
            "type": "operational_execution_gate",
            "skill_role": "force full author-style execution before any gap claim can converge",
            "content": (
                "Loop 2 must execute the repo/model/data/API/GPU steps encoded in this DAG, emit raw outputs, "
                "metrics, tables/figure summaries, and hardware traces, and keep all reduced, proxy, syntax-only, "
                "README-only, or support-only artifacts non-convergent."
            ),
            "repo_paths_visible_to_loop2": repo_paths,
            "blocked_checks_to_satisfy": blocker_names,
        },
    )
    changed |= ensure_node(
        dag,
        {
            "id": "reviewer.compare_simulation_to_paper_evidence_channels",
            "type": "reviewer_result_shape_gate",
            "skill_role": "verify whether DAG-only simulation recovered the paper's real research gap",
            "content": (
                "After Loop 2 produces operational artifacts, compare the simulated gap, decisions, result shape, "
                "and exceptions against paper tables, figures, result paragraphs, appendix evidence, and released artifacts. "
                "Close result shape can pass; absent outputs or generic claims cannot pass."
            ),
            "paper_evidence_channels": PAPER_EVIDENCE_CHANNELS,
        },
    )
    changed |= ensure_node(
        dag,
        {
            "id": "decision.continue_or_block_after_professional_verifier",
            "type": "author_reviewer_decision",
            "skill_role": "continue loops until accepted or explicitly blocked by exact professional requirements",
            "content": (
                "If verifier accepts close paper-shaped results, promote the learned research-gap DAG. "
                "If exact model/data/API/GPU/release artifacts are unavailable, record explicit blockers and route them back to Loop 1. "
                "Do not stop on missing results without updating the DAG."
            ),
        },
    )

    for update in feedback["required_updates"]:
        update_id = update.get("id")
        if not update_id:
            continue
        update_node_id = f"loop1.required_update.{slug(update_id)[:80]}"
        changed |= ensure_node(
            dag,
            {
                "id": update_node_id,
                "type": "verifier_required_dag_repair",
                "skill_role": "turn a verifier mismatch into a concrete Loop 2 repair action",
                "content": update.get("reason"),
                "required_update_id": update_id,
                "artifact_family": update.get("artifact_family"),
                "success_criteria": update.get("success_criteria", []),
                "failed_metric_ids": update.get("failed_metric_ids", []),
                "failed_simulation_setting_ids": update.get("failed_simulation_setting_ids", []),
                "repair_axis_ids": update.get("repair_axis_ids", []),
                "protocol_finding_statuses": update.get("protocol_finding_statuses", {}),
                "risk_audit_status": update.get("risk_audit_status"),
                "verifier_report_path": update.get("report_path"),
                "oracle_values_exposed_to_loop2": False,
            },
        )
        changed |= ensure_edge(dag, "loop1.latest_verifier_feedback", update_node_id)
        changed |= ensure_edge(dag, update_node_id, "ops.non_reduced_artifact_completion_gate")

    for source, target in [
        ("reviewer.keep_exact_artifact_debt", "loop1.latest_verifier_feedback"),
        ("loop1.latest_verifier_feedback", "ops.non_reduced_artifact_completion_gate"),
        ("ops.non_reduced_artifact_completion_gate", "loop2.execute_operational_dag"),
        ("loop2.execute_operational_dag", "reviewer.compare_simulation_to_paper_evidence_channels"),
        ("reviewer.compare_simulation_to_paper_evidence_channels", "decision.continue_or_block_after_professional_verifier"),
        ("decision.continue_or_block_after_professional_verifier", "decision.promote_research_gap"),
    ]:
        changed |= ensure_edge(dag, source, target)

    update_record = {
        "id": "update.sync_latest_verifier_blockers_to_operational_dag",
        "iteration": iteration_number,
        "created_at_utc": now,
        "source": "sync_verifier_blockers_to_dags.py",
        "status": feedback["status"] or queue_status or "not_converged",
        "blocker_check_names": blocker_names,
        "required_update_ids": update_ids,
        "converged": False,
        "success_criteria": [
            "Loop 2 input remains DAG-only",
            "missing operational artifacts are named as DAG gates",
            "reduced/proxy/support-only evidence remains non-convergent",
            "verifier compares against tables, figures, paragraphs, appendix, and released artifacts only after operational outputs exist",
        ],
    }
    previous_updates = dag.setdefault("previous_loop_updates", [])
    existing_update = None
    for previous in previous_updates:
        if previous.get("id") == update_record["id"] and previous.get("source") == update_record["source"]:
            existing_update = previous
            break
    if existing_update is None:
        previous_updates.append(update_record)
        changed = True
    else:
        preserved_created_at = existing_update.get("created_at_utc", now)
        existing_update.clear()
        existing_update.update(update_record)
        existing_update["created_at_utc"] = preserved_created_at
        existing_update["updated_at_utc"] = now
        changed = True

    dag["signature"] = signature_for(dag)
    current_path = paper_dir / "paper_author_gap_dag.json"
    iter_path = planned_iter_path
    write_json(current_path, dag)
    write_json(iter_path, dag)
    return {
        "paper_dir": str(paper_dir),
        "paper_id": paper_id,
        "dag_source": str(dag_path),
        "dag_current": str(current_path),
        "dag_iteration": str(iter_path),
        "verifier": str(verifier_path),
        "queue_status": queue_status,
        "feedback_status": feedback["status"],
        "blocked_check_count": len(feedback["blocked_checks"]),
        "required_update_count": len(feedback["required_updates"]),
        "added_or_updated": changed,
        "signature": dag["signature"],
    }


def load_queue_by_paper() -> dict[str, dict[str, Any]]:
    if not QUEUE_PATH.exists():
        return {}
    queue_obj = read_json(QUEUE_PATH)
    queue_items = queue_obj.get("queue", queue_obj if isinstance(queue_obj, list) else [])
    return {item.get("paper_id"): item for item in queue_items if item.get("paper_id")}


def sync_summary(report: dict[str, Any]) -> None:
    if not SUMMARY_PATH.exists():
        return
    summary = read_json(SUMMARY_PATH)
    by_id = {item["paper_id"]: item for item in report["papers"]}
    for paper in summary.get("papers", []):
        item = by_id.get(paper.get("paper_id"))
        if not item:
            continue
        paper["latest_dag_repair_iter"] = item["dag_iteration"]
        paper["latest_dag_repair_signature"] = item["signature"]
        statuses = paper.setdefault("implementation_statuses", [])
        status = "loop1_latest_verifier_feedback_encoded_in_dag"
        if status not in statuses:
            statuses.append(status)
    summary["updated_at_utc"] = report["created_at_utc"]
    summary["final_status"] = "running_professional_two_loop_not_converged"
    write_json(SUMMARY_PATH, summary)


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Loop 1 DAG Repair Audit",
        "",
        f"- Updated: `{report['created_at_utc']}`",
        f"- Papers repaired: `{len(report['papers'])}`",
        "- Scope: verifier failures were fed back into DAG-only operational nodes; no oracle table values were exposed to Loop 2.",
        "- Policy: reduced/small/proxy/syntax-only/support-only evidence remains non-convergent.",
        "",
        "## Per-Paper Repair",
        "",
    ]
    for item in report["papers"]:
        lines.append(
            f"- `{item['paper_id']}`: blocked_checks=`{item['blocked_check_count']}`, "
            f"required_updates=`{item['required_update_count']}`, iter=`{Path(item['dag_iteration']).name}`, "
            f"status=`{item['feedback_status'] or item['queue_status']}`"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    queue_by_paper = load_queue_by_paper()
    papers = []
    for paper_dir in sorted(path for path in PAPER_RUNS.iterdir() if path.is_dir()):
        verifier_path = latest_verifier_path(paper_dir)
        if verifier_path is None:
            continue
        dag = read_json(latest_dag_path(paper_dir))
        paper_id = dag.get("target_paper_id") or read_json(verifier_path).get("paper_id") or paper_dir.name
        papers.append(update_dag_with_feedback(paper_dir, verifier_path, queue_by_paper.get(paper_id)))

    report = {
        "artifact_kind": "loop1_dag_repair_audit",
        "created_at_utc": utc_now(),
        "policy": {
            "loop2_input": "paper_author_gap_dag.json only",
            "oracle_values_exposed_to_loop2": False,
            "reduced_or_proxy_converges": False,
            "verifier_evidence_channels": PAPER_EVIDENCE_CHANNELS,
        },
        "papers": papers,
    }
    write_json(REPORT_JSON, report)
    write_markdown(report)
    sync_summary(report)
    print(json.dumps({"report": str(REPORT_JSON), "markdown": str(REPORT_MD), "papers": len(papers)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
