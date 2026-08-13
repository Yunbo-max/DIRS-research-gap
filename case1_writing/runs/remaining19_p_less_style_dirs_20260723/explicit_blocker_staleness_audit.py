#!/usr/bin/env python3
"""Audit explicit blockers for stale or weak operational evidence.

This is a Loop 1 hygiene artifact. It does not accept papers and does not
weaken the non-reduced convergence policy. It checks whether blocked papers
still have DAG/verifier/artifact evidence attached, and which blockers should
be rechecked after the active Prophet GPU job releases GPU 3.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent
COMPLETION_AUDIT = RUN_ROOT / "strict_dirs_completion_audit_20260723.json"
QUEUE_PATH = RUN_ROOT / "specialized_runner_queue.json"
SUMMARY_PATH = RUN_ROOT / "remaining19_strict_dirs_summary.json"
GPU_RECHECK_STATE = RUN_ROOT / "gpu_recheck_dispatcher_state.json"
REPORT_PATH = RUN_ROOT / "explicit_blocker_staleness_audit_20260723.json"
STATUS_MD = RUN_ROOT / "EXPLICIT_BLOCKER_STALENESS_AUDIT_20260723.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def gpu_inventory() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        idx, name, used, total, util = parts
        used_i = int(float(used))
        total_i = int(float(total))
        gpus.append(
            {
                "index": idx,
                "name": name,
                "memory_used_mib": used_i,
                "memory_total_mib": total_i,
                "memory_free_mib": total_i - used_i,
                "utilization_gpu_pct": int(float(util)),
            }
        )
    return gpus


def process_table() -> list[str]:
    result = subprocess.run(
        ["ps", "aux"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    return [
        line
        for line in result.stdout.splitlines()
        if "prophet_custom_full_gsm8k_runner.py" in line
        or "strict_dirs_goal_supervisor.py" in line
    ]


def map_queue() -> dict[str, dict[str, Any]]:
    queue = read_json(QUEUE_PATH, {}).get("queue", [])
    return {item.get("paper_id"): item for item in queue if item.get("paper_id")}


def map_summary() -> dict[str, dict[str, Any]]:
    papers = read_json(SUMMARY_PATH, {}).get("papers", [])
    return {item.get("paper_id"): item for item in papers if item.get("paper_id")}


def repo_paths_for(paper: dict[str, Any], queue_item: dict[str, Any], summary_item: dict[str, Any], op: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for source in (queue_item, summary_item):
        for path in source.get("repo_paths") or []:
            if path not in paths:
                paths.append(path)
    for audit in op.get("repo_audits") or []:
        path = audit.get("repo")
        if path and path not in paths:
            paths.append(path)
    return paths


def latest_verifier_path(paper_dir: Path, audit_path: str | None) -> Path | None:
    if audit_path:
        candidate = Path(audit_path)
        if candidate.exists():
            return candidate
    candidates = sorted(paper_dir.glob("verifier_result_iter_*.json"))
    return candidates[-1] if candidates else None


def status_words(*items: Any) -> str:
    return " ".join(json.dumps(item, sort_keys=True) for item in items if item is not None).lower()


def blocker_text(*items: Any) -> str:
    return status_words(*items)


def blocker_tags(text: str) -> list[str]:
    tags = []
    if any(marker in text for marker in ["clean_gpu_slot", "free gpu", "memory_free", "free_mib", "insufficient_free_gpu_memory"]):
        tags.append("active_gpu_capacity")
    if any(marker in text for marker in ["h100", "a100", "a800", "h20", "l40", "mi300x", "rtx pro 6000"]):
        tags.append("exact_hardware_class")
    if any(marker in text for marker in ["checkpoint", "weights", "model_artifact", "pretrained"]):
        tags.append("checkpoint_or_model")
    if any(marker in text for marker in ["dataset", "imagenet", "coco", "nsd", "videomme", "suiteSparse".lower(), "longbench"]):
        tags.append("dataset")
    if any(marker in text for marker in ["api key", "openrouter", "openai api", "terms/form", "access-gated", "license"]):
        tags.append("api_or_access")
    if any(marker in text for marker in ["source code missing", "placeholder", "official_source_repo_not_found", "github_repository_not_found"]):
        tags.append("source_release")
    if any(marker in text for marker in ["python311", "transformers 4.57", "flash_attention", "flash-attn", "cuda nvcc"]):
        tags.append("software_runtime")
    return sorted(set(tags))


def blocker_fingerprint(
    *,
    tags: list[str],
    operational_blockers: list[Any],
    required_update_nodes: list[str],
    verifier_required_updates: list[Any],
    weak_evidence: list[str],
) -> str:
    payload = {
        "blocker_tags": tags,
        "operational_blocker_count": len(operational_blockers),
        "required_update_nodes": required_update_nodes,
        "verifier_required_update_count": len(verifier_required_updates),
        "weak_evidence": weak_evidence,
    }
    return __import__("hashlib").sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def audit_one(
    paper: dict[str, Any],
    queue_by_id: dict[str, dict[str, Any]],
    summary_by_id: dict[str, dict[str, Any]],
    gpu_recheck_fingerprints: dict[str, str],
) -> dict[str, Any]:
    paper_id = paper["paper_id"]
    paper_dir = Path(paper["paper_dir"])
    dag_path = Path(paper["dag_path"])
    verifier_path = latest_verifier_path(paper_dir, paper.get("verifier_path"))
    op_path = paper_dir / "operational_artifacts.json"
    dag = read_json(dag_path, {})
    verifier = read_json(verifier_path, {}) if verifier_path else {}
    op = read_json(op_path, {})
    queue_item = queue_by_id.get(paper_id, {})
    summary_item = summary_by_id.get(paper_id, {})
    repo_paths = repo_paths_for(paper, queue_item, summary_item, op)
    nodes = dag.get("nodes") or []
    required_update_nodes = [
        node.get("id")
        for node in nodes
        if isinstance(node, dict) and str(node.get("id", "")).startswith("loop1.required_update")
    ]
    operational_blockers = op.get("blockers") or []
    verifier_required_updates = verifier.get("required_updates") or []
    verifier_checks = verifier.get("checks") or []
    blocked_checks = [
        check
        for check in verifier_checks
        if check.get("status") in {"blocked", "fail"}
        or "blocked" in str(check.get("status", "")).lower()
    ]
    repo_status = []
    for path in repo_paths:
        p = Path(path)
        repo_status.append(
            {
                "path": path,
                "exists": p.exists(),
                "file_count_probe": sum(1 for _ in p.rglob("*")) if p.exists() and p.is_dir() else None,
            }
        )
    current_blocker_text = blocker_text(
        paper.get("status"),
        verifier_required_updates,
        blocked_checks,
        operational_blockers,
    )
    tags = blocker_tags(current_blocker_text)
    needs_gpu_recheck = "active_gpu_capacity" in tags
    current_blocker_count = len(operational_blockers) + len(verifier_required_updates) + len(blocked_checks)
    weak_evidence = []
    if not dag_path.exists():
        weak_evidence.append("missing_dag")
    if not verifier_path or not verifier_path.exists():
        weak_evidence.append("missing_verifier")
    if not required_update_nodes and paper.get("explicitly_blocked"):
        weak_evidence.append("missing_loop1_required_update_node")
    if not current_blocker_count and paper.get("explicitly_blocked"):
        weak_evidence.append("no_current_blocker_payload")
    if any(not item["exists"] for item in repo_status):
        weak_evidence.append("repo_path_missing_now")
    blind_contract = dag.get("blind_contract") or {}
    if blind_contract.get("paper_text_visible_to_loop2") is not False:
        weak_evidence.append("blind_contract_not_strict_on_paper_text")
    if blind_contract.get("oracle_results_visible_to_loop2") is not False:
        weak_evidence.append("blind_contract_not_strict_on_oracle_results")
    fingerprint = blocker_fingerprint(
        tags=tags,
        operational_blockers=operational_blockers,
        required_update_nodes=required_update_nodes,
        verifier_required_updates=verifier_required_updates,
        weak_evidence=weak_evidence,
    )
    gpu_rechecked_for_current_blockers = gpu_recheck_fingerprints.get(paper_id) == fingerprint
    if weak_evidence:
        status = "needs_loop1_repair"
    elif needs_gpu_recheck and not paper.get("running") and gpu_rechecked_for_current_blockers:
        status = "explicit_blocker_evidence_bound_after_gpu_recheck"
    elif needs_gpu_recheck and not paper.get("running"):
        status = "explicit_blocker_valid_recheck_after_gpu_release"
    elif paper.get("running"):
        status = "running_not_explicit_blocker"
    else:
        status = "explicit_blocker_evidence_bound"
    return {
        "paper_id": paper_id,
        "classification": paper.get("classification"),
        "status": status,
        "paper_status": paper.get("status"),
        "dag_path": str(dag_path),
        "verifier_path": str(verifier_path) if verifier_path else None,
        "operational_artifacts_path": str(op_path) if op_path.exists() else None,
        "repo_status": repo_status,
        "required_update_node_count": len(required_update_nodes),
        "required_update_nodes": required_update_nodes,
        "verifier_required_update_count": len(verifier_required_updates),
        "blocked_verifier_check_count": len(blocked_checks),
        "operational_blocker_count": len(operational_blockers),
        "needs_recheck_after_active_gpu_release": needs_gpu_recheck and not paper.get("running"),
        "gpu_rechecked_for_current_blockers": gpu_rechecked_for_current_blockers,
        "blocker_fingerprint": fingerprint,
        "blocker_tags": tags,
        "weak_evidence": weak_evidence,
    }


def render_md(report: dict[str, Any]) -> None:
    counts = report["counts"]
    lines = [
        "# Explicit Blocker Staleness Audit",
        "",
        f"- Updated: `{report['created_at_utc']}`",
        "- Purpose: verify explicit blockers remain evidence-bound while the active non-reduced GPU run continues.",
        "- Policy: this audit cannot accept a paper and cannot treat reduced/proxy evidence as convergence.",
        f"- Counts: evidence_bound=`{counts['explicit_blocker_evidence_bound']}`, after_gpu_recheck=`{counts['explicit_blocker_evidence_bound_after_gpu_recheck']}`, gpu_recheck=`{counts['explicit_blocker_valid_recheck_after_gpu_release']}`, needs_repair=`{counts['needs_loop1_repair']}`, running=`{counts['running_not_explicit_blocker']}`",
        "",
        "## Current GPU State",
        "",
    ]
    for gpu in report["gpu_inventory"]:
        lines.append(
            f"- GPU `{gpu['index']}` {gpu['name']} free=`{gpu['memory_free_mib']}` MiB "
            f"used=`{gpu['memory_used_mib']}` MiB util=`{gpu['utilization_gpu_pct']}`%"
        )
    lines += ["", "## Paper Statuses", ""]
    for paper in report["papers"]:
        lines.append(
            f"- `{paper['paper_id']}`: `{paper['status']}` "
            f"updates=`{paper['required_update_node_count']}` verifier_updates=`{paper['verifier_required_update_count']}` "
            f"op_blockers=`{paper['operational_blocker_count']}` tags=`{paper['blocker_tags']}` weak=`{paper['weak_evidence']}`"
        )
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = read_json(COMPLETION_AUDIT, {})
    queue_by_id = map_queue()
    summary_by_id = map_summary()
    gpu_recheck_fingerprints = read_json(GPU_RECHECK_STATE, {}).get("attempted_recheck_fingerprints", {})
    papers = [
        audit_one(paper, queue_by_id, summary_by_id, gpu_recheck_fingerprints)
        for paper in audit.get("papers", [])
    ]
    counts: dict[str, int] = {}
    for paper in papers:
        counts[paper["status"]] = counts.get(paper["status"], 0) + 1
    for key in [
        "explicit_blocker_evidence_bound",
        "explicit_blocker_evidence_bound_after_gpu_recheck",
        "explicit_blocker_valid_recheck_after_gpu_release",
        "needs_loop1_repair",
        "running_not_explicit_blocker",
    ]:
        counts.setdefault(key, 0)
    report = {
        "artifact_kind": "explicit_blocker_staleness_audit",
        "created_at_utc": utc_now(),
        "completion_audit_path": str(COMPLETION_AUDIT),
        "gpu_inventory": gpu_inventory(),
        "relevant_processes": process_table(),
        "policy": {
            "can_converge_papers": False,
            "reduced_or_proxy_evidence_allowed": False,
            "purpose": "Loop 1 blocker hygiene and recheck scheduling",
        },
        "counts": counts,
        "papers": papers,
    }
    write_json(REPORT_PATH, report)
    render_md(report)
    print(json.dumps({"report": str(REPORT_PATH), "markdown": str(STATUS_MD), "counts": counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
