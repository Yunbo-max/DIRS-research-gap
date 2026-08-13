#!/usr/bin/env python3
"""Strict p-less-style DIRS run for the remaining 19 systems papers.

This runner applies the lesson from the p-less one-paper convergence case:

- Loop 1 repairs the paper-specific gap DAG from verifier failures.
- Loop 2 is a blind author simulation from the DAG only.
- Repo audits and generic GPU motif rows are support evidence only.
- Reduced, smoke, toy, proxy, and syntax-only checks never count as
  convergence.
- A paper converges only if a professional paper-shaped artifact package is
  accepted; otherwise the verifier emits explicit operational blockers.

The script intentionally records blockers instead of silently upgrading weak
evidence into success.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


RUN_DIR = Path(__file__).resolve().parent
SOURCE_RUN = RUN_DIR.parent / "llm_inference_systems_fullpaper_train20_section_gap_20260722"
EVIDENCE_PATH = SOURCE_RUN / "paper_section_evidence_table.json"
CAMPAIGN_PATH = SOURCE_RUN / "author_style_gpu_reproduction_campaign.json"
PLESS_ID = "ICLR2026_ItFuNJQGH4_p_less_sampling"
OUTPUT_JSON = RUN_DIR / "remaining19_strict_dirs_summary.json"
OUTPUT_MD = RUN_DIR / "REMAINING19_STRICT_DIRS_REPORT.md"
STATUS_MD = RUN_DIR / "LONGGOAL_STATUS.md"
QUEUE_JSON = RUN_DIR / "specialized_runner_queue.json"
QUEUE_MD = RUN_DIR / "SPECIALIZED_RUNNER_QUEUE.md"
PAPER_DIR = RUN_DIR / "paper_runs"


STRICT_POLICY = {
    "reduced_or_small_runs_are_convergence_evidence": False,
    "repo_syntax_or_readme_audit_is_convergence_evidence": False,
    "generic_gpu_motif_rows_are_convergence_evidence": False,
    "minimum_for_gap_convergence": (
        "professional paper-shaped artifacts with paper-appropriate models, "
        "datasets, baselines, metrics, timing/compute traces, and verifier "
        "comparison to paper tables/figures/paragraph claims"
    ),
    "allowed_nonconverged_support": [
        "repo audit",
        "README/script inventory",
        "syntax/import checks",
        "generic GPU motif stress rows",
        "planning manifests",
    ],
}


FAMILY_KEYWORDS = {
    "visual_tokenizer": ["tokenizer", "vq", "autoencoder", "reconstruction", "vision tokenizer", "latents"],
    "compression_vq": ["compression", "rate-distortion", "bitrate", "vq", "quantization", "codec"],
    "diffusion_cache": ["cache", "diffusion", "denoising", "scheduler", "latency"],
    "speculative_or_early_decoding": ["speculative", "early", "commit", "decoding", "draft"],
    "video_token_merging": ["video", "token merging", "merge", "llava", "qwen", "vlm"],
    "long_context_or_rope": ["long context", "rope", "kv", "context", "locality"],
    "sparse_cuda_rl": ["sparse", "cuda", "kernel", "rl", "speedup"],
    "reasoning_agents": ["reasoning", "agents", "information seeking", "trajectory"],
    "structured_text_or_fmri": ["fmri", "structured text", "representation"],
    "3d_generation": ["3d", "voxel", "mesh", "gaussian", "reconstruction", "articulated"],
}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug[:140] or "paper"


def as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def stringify(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True).lower()


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 20) -> dict:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": proc.returncode,
            "elapsed_s": round(time.perf_counter() - start, 3),
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-3000:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "elapsed_s": round(time.perf_counter() - start, 3),
            "stdout_tail": (exc.stdout or "")[-3000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-3000:] if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }


def load_inputs() -> tuple[list[dict], dict]:
    evidence = read_json(EVIDENCE_PATH)
    campaign = read_json(CAMPAIGN_PATH)
    papers = evidence.get("papers", [])
    remaining = [paper for paper in papers if paper.get("chip_id") != PLESS_ID]
    return remaining, campaign


def repo_status_by_chip(campaign: dict) -> dict[str, dict]:
    return {row["chip_id"]: row for row in campaign.get("repo_audit", [])}


def family_tags(paper: dict) -> list[str]:
    fp = footprint_lists(paper)
    text = stringify(
        {
            "chip_id": paper.get("chip_id"),
            "title": paper.get("title"),
            "gaps": fp["gaps"],
            "failure_modes": fp["failure_modes"],
            "method_components": fp["method_components"],
            "benchmarks": fp["benchmarks"],
        }
    )
    tags = []
    chip_title = f"{paper.get('chip_id', '')} {paper.get('title', '')}".lower()
    if any(term in chip_title for term in ["seacache", "sencache"]):
        tags.append("diffusion_cache")
    if any(term in chip_title for term in ["flashvid", "token merging"]):
        tags.append("video_token_merging")
    if any(term in chip_title for term in ["infotok", "atoken", "tokenizer"]):
        tags.append("visual_tokenizer")
    if any(term in chip_title for term in ["rdvq", "rate-distortion", "compression", "vq"]):
        tags.append("compression_vq")
    if any(term in chip_title for term in ["prophet", "hsd", "speculative", "early commit", "decoding"]):
        tags.append("speculative_or_early_decoding")
    if any(term in chip_title for term in ["loongrl", "mrrope", "rope", "long_context", "long context"]):
        tags.append("long_context_or_rope")
    if any(term in chip_title for term in ["sparserl", "sparse cuda", "cuda generation", "nuwa", "pruning"]):
        tags.append("sparse_cuda_rl")
    if any(term in chip_title for term in ["trellis", "spark", "3d", "articulated"]):
        tags.append("3d_generation")
    if any(term in chip_title for term in ["rational", "agents", "reasoning"]):
        tags.append("reasoning_agents")
    if any(term in chip_title for term in ["fmri", "brain"]):
        tags.append("structured_text_or_fmri")
    for family, terms in FAMILY_KEYWORDS.items():
        if family not in tags and any(term in text for term in terms):
            tags.append(family)
    return tags or ["systems_token_efficiency"]


def footprint_lists(paper: dict) -> dict[str, list]:
    fp = paper.get("footprint") or {}
    return {
        "gaps": as_list(fp.get("gaps")),
        "failure_modes": as_list(fp.get("failure_modes")),
        "method_components": as_list(fp.get("method_components")),
        "baselines": as_list(fp.get("baselines")) + as_list(fp.get("external_strong_baselines")),
        "benchmarks": as_list(fp.get("benchmarks")),
        "metrics": as_list(fp.get("metrics")),
        "eval_protocols": as_list(fp.get("eval_protocols")),
        "datasets": as_list(fp.get("datasets")),
        "hardware_environments": as_list(fp.get("hardware_environments")),
        "code_artifacts": as_list(fp.get("code_artifacts")),
        "data_artifacts": as_list(fp.get("data_artifacts")),
        "implementation_statuses": as_list(fp.get("implementation_statuses")),
        "framework_runtimes": as_list(fp.get("framework_runtimes")),
    }


def compact(items: list, n: int = 10) -> str:
    text_items = [str(item) for item in items if item not in (None, "")]
    if not text_items:
        return "not specified in the available footprint"
    return "; ".join(text_items[:n])


def build_gap_dag(paper: dict, repo_status: dict, iteration: int, previous_updates: list[dict]) -> dict:
    fp = footprint_lists(paper)
    repos = repo_status.get("repos", [])
    strict_nodes = []
    if iteration >= 2:
        strict_nodes = [
            {
                "id": "reviewer.reject_reduced_convergence",
                "type": "scale_gate",
                "skill_role": "prevent false convergence",
                "content": "Reduced, smoke, toy, syntax-only, README-only, or generic proxy runs are support/debug only and cannot converge this paper.",
            },
            {
                "id": "reviewer.require_professional_artifact_package",
                "type": "professional_gate",
                "skill_role": "require paper-shaped evidence",
                "content": "Require paper-appropriate models/checkpoints, datasets, baselines, metrics, timing/compute traces, raw outputs, and verifier-comparable tables/figures.",
            },
        ]

    nodes = [
        {
            "id": "paper_context.title_domain",
            "type": "domain_context",
            "skill_role": "identify the paper target",
            "content": f"{paper.get('title')} | families={', '.join(family_tags(paper))}",
        },
        {
            "id": "loop1.extract_evidence_channels",
            "type": "reviewer_inventory",
            "skill_role": "turn the paper into evidence channels",
            "content": "Extract gap, failure modes, baselines, method mechanism, experiments, datasets, metrics, figures/tables, code, model/data dependencies, and hardware.",
        },
        {
            "id": "gap.paper_gap_claims",
            "type": "gap_hypothesis",
            "skill_role": "state the missing capability",
            "content": compact(fp["gaps"]),
        },
        {
            "id": "gap.failure_mode_stressors",
            "type": "gap_decomposition",
            "skill_role": "make the gap falsifiable",
            "content": compact(fp["failure_modes"]),
        },
        {
            "id": "related.baseline_axis",
            "type": "baseline_map",
            "skill_role": "name the near-miss methods",
            "content": compact(fp["baselines"]),
        },
        {
            "id": "method.bind_gap_to_mechanism",
            "type": "mechanism_binding",
            "skill_role": "make the solution answer the gap",
            "content": compact(fp["method_components"]),
        },
        {
            "id": "experiments.benchmark_metric_grid",
            "type": "experiment_design",
            "skill_role": "design paper-shaped comparisons",
            "content": f"benchmarks={compact(fp['benchmarks'])}; metrics={compact(fp['metrics'])}; protocols={compact(fp['eval_protocols'])}",
        },
        {
            "id": "experiments.system_surface",
            "type": "systems_measurement",
            "skill_role": "bind claims to hardware/runtime",
            "content": f"hardware={compact(fp['hardware_environments'])}; runtimes={compact(fp['framework_runtimes'])}",
        },
        {
            "id": "ops.resolve_repo_code",
            "type": "operational_dependency",
            "skill_role": "find code and runnable entrypoints",
            "content": f"repos={compact(repos)}; code_artifacts={compact(fp['code_artifacts'])}",
            "repo_paths": repos,
        },
        {
            "id": "ops.resolve_models_data",
            "type": "operational_dependency",
            "skill_role": "find models, datasets, and external artifacts",
            "content": f"datasets={compact(fp['datasets'])}; data_artifacts={compact(fp['data_artifacts'])}; implementation_statuses={compact(fp['implementation_statuses'])}",
        },
        {
            "id": "loop2.execute_operational_dag",
            "type": "operational_execution",
            "skill_role": "act like the author",
            "content": "Run repo audits, dependency probes, available scripts/unit checks, GPU-ready checks, and paper-shaped experiments only when exact models/data/scripts are present.",
        },
        *strict_nodes,
        {
            "id": "reviewer.compare_gap_semantics",
            "type": "verification",
            "skill_role": "compare simulated gap to paper evidence",
            "content": "Check whether the simulated gap preserves the paper's missing capability, failure mode, baseline axis, and mechanism fit.",
        },
        {
            "id": "reviewer.compare_result_shapes",
            "type": "verification",
            "skill_role": "compare results to paper tables/figures/paragraphs",
            "content": "Accept close result shape only from paper-shaped operational artifacts; otherwise emit blockers and DAG updates.",
        },
        {
            "id": "reviewer.keep_exact_artifact_debt",
            "type": "verification_boundary",
            "skill_role": "do not hide missing exact reruns",
            "content": "Track missing main tables, figures, appendix tables, raw outputs, scoring scripts, timing traces, memory traces, and seeds as exact artifact debt.",
        },
        {
            "id": "decision.promote_research_gap",
            "type": "author_reviewer_decision",
            "skill_role": "decide whether the paper-specific gap is learned",
            "content": "Promote only if semantic gap and professional result evidence pass; otherwise record explicit blocker and update the DAG.",
        },
    ]
    edges = [
        ["paper_context.title_domain", "loop1.extract_evidence_channels"],
        ["loop1.extract_evidence_channels", "gap.paper_gap_claims"],
        ["gap.paper_gap_claims", "gap.failure_mode_stressors"],
        ["gap.failure_mode_stressors", "related.baseline_axis"],
        ["related.baseline_axis", "method.bind_gap_to_mechanism"],
        ["method.bind_gap_to_mechanism", "experiments.benchmark_metric_grid"],
        ["experiments.benchmark_metric_grid", "experiments.system_surface"],
        ["experiments.system_surface", "ops.resolve_repo_code"],
        ["ops.resolve_repo_code", "ops.resolve_models_data"],
        ["ops.resolve_models_data", "loop2.execute_operational_dag"],
        ["loop2.execute_operational_dag", "reviewer.compare_gap_semantics"],
        ["loop2.execute_operational_dag", "reviewer.compare_result_shapes"],
        ["reviewer.compare_gap_semantics", "decision.promote_research_gap"],
        ["reviewer.compare_result_shapes", "reviewer.keep_exact_artifact_debt"],
        ["reviewer.keep_exact_artifact_debt", "decision.promote_research_gap"],
    ]
    if iteration >= 2:
        edges.extend(
            [
                ["loop2.execute_operational_dag", "reviewer.reject_reduced_convergence"],
                ["reviewer.reject_reduced_convergence", "reviewer.require_professional_artifact_package"],
                ["reviewer.require_professional_artifact_package", "reviewer.compare_result_shapes"],
                ["reviewer.require_professional_artifact_package", "decision.promote_research_gap"],
            ]
        )

    dag = {
        "graph_id": f"{paper['chip_id']}_gap_dag_iter_{iteration:02d}",
        "created_at_utc": now_utc(),
        "target_paper_id": paper["chip_id"],
        "target_title": paper.get("title"),
        "blind_contract": {
            "only_input_file": "paper_author_gap_dag.json",
            "paper_text_visible_to_loop2": False,
            "oracle_results_visible_to_loop2": False,
            "previous_memory_visible_to_loop2": False,
            "repo_paths_visible_only_if_encoded_in_dag": True,
        },
        "strict_policy": STRICT_POLICY,
        "nodes": nodes,
        "edges": edges,
        "previous_loop_updates": previous_updates,
    }
    dag["signature"] = stable_hash({"nodes": nodes, "edges": edges, "policy": STRICT_POLICY})
    return dag


def audit_repo(repo: Path, max_python: int = 250) -> dict:
    record = {
        "repo": str(repo),
        "exists": repo.exists(),
        "git_head": None,
        "readme_files": [],
        "python_file_count": 0,
        "cuda_cpp_file_count": 0,
        "shell_file_count": 0,
        "entrypoint_candidates": [],
        "syntax_checked_python_files": 0,
        "syntax_error_count": 0,
        "syntax_errors": [],
        "status": "missing",
    }
    if not repo.exists():
        return record
    readmes = [p for p in repo.rglob("*") if p.is_file() and p.name.lower() in {"readme.md", "readme.rst", "readme.txt", "readme"}]
    py_files = list(repo.rglob("*.py"))
    cu_files = list(repo.rglob("*.cu")) + list(repo.rglob("*.cuh")) + list(repo.rglob("*.cpp"))
    shell_files = list(repo.rglob("*.sh"))
    entrypoints = []
    for p in py_files + shell_files:
        name = p.name.lower()
        if any(term in name for term in ["train", "eval", "test", "infer", "demo", "benchmark", "generate", "app"]):
            entrypoints.append(str(p.relative_to(repo)))
    git = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo, timeout=10) if (repo / ".git").exists() else {}

    errors = []
    for py in py_files[:max_python]:
        try:
            ast.parse(py.read_text(encoding="utf-8", errors="ignore"), filename=str(py))
        except SyntaxError as exc:
            errors.append({"file": str(py.relative_to(repo)), "line": exc.lineno, "msg": exc.msg})
        except ValueError as exc:
            errors.append({"file": str(py.relative_to(repo)), "line": None, "msg": str(exc)})
        except OSError as exc:
            errors.append({"file": str(py.relative_to(repo)), "line": None, "msg": str(exc)})

    record.update(
        {
            "git_head": (git.get("stdout_tail") or "").strip() if git else None,
            "readme_files": [str(p.relative_to(repo)) for p in readmes[:10]],
            "python_file_count": len(py_files),
            "cuda_cpp_file_count": len(cu_files),
            "shell_file_count": len(shell_files),
            "entrypoint_candidates": entrypoints[:30],
            "syntax_checked_python_files": min(len(py_files), max_python),
            "syntax_error_count": len(errors),
            "syntax_errors": errors[:20],
            "status": "code_inventory_ready" if len(errors) == 0 else "code_inventory_has_syntax_errors",
        }
    )
    return record


def device_probe() -> dict:
    smi = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    return {
        "status": "pass" if smi["returncode"] == 0 else "blocked",
        "nvidia_smi": smi,
    }


def simulate_from_dag_only(dag: dict, paper_dir: Path) -> dict:
    """Blind author simulator.

    The simulator only reads the DAG object passed here. It can inspect repos
    only when repo paths appear in DAG nodes.
    """

    node_by_id = {node["id"]: node for node in dag["nodes"]}
    gap = node_by_id["gap.paper_gap_claims"]["content"]
    failures = node_by_id["gap.failure_mode_stressors"]["content"]
    method = node_by_id["method.bind_gap_to_mechanism"]["content"]
    experiment = node_by_id["experiments.benchmark_metric_grid"]["content"]
    repo_node = node_by_id["ops.resolve_repo_code"]
    repo_paths = [Path(p) for p in repo_node.get("repo_paths", [])]
    repo_audits = [audit_repo(path) for path in repo_paths]
    gpu_probe = device_probe()

    blockers = []
    if not repo_paths:
        blockers.append("no_local_repository_encoded_in_dag")
    if repo_paths and not any(audit["exists"] for audit in repo_audits):
        blockers.append("encoded_repository_paths_missing")
    data_node = node_by_id["ops.resolve_models_data"]["content"].lower()
    for term in ["not downloaded", "weights_not_downloaded", "datasets_not_downloaded", "missing", "not_rerun", "result_not_rerun"]:
        if term in data_node:
            blockers.append(term.replace(" ", "_"))

    has_strict_gate = "reviewer.reject_reduced_convergence" in node_by_id
    has_professional_gate = "reviewer.require_professional_artifact_package" in node_by_id
    professional_package_ready = False
    package_reason = "no_paper_shaped_runner_has_completed_for_this_paper"
    if blockers:
        package_reason = "blocked_by_" + "+".join(sorted(set(blockers))[:6])

    simulation = {
        "created_at_utc": now_utc(),
        "paper_id": dag["target_paper_id"],
        "paper_title": dag["target_title"],
        "paper_text_seen": False,
        "oracle_results_seen": False,
        "previous_memory_seen": False,
        "input_contract": dag["blind_contract"],
        "gap_prediction": f"{gap}; stressors: {failures}",
        "method_prediction": method,
        "experiment_prediction": experiment,
        "repo_audits": repo_audits,
        "gpu_probe": gpu_probe,
        "contains_reduced_rejection_gate": has_strict_gate,
        "contains_professional_artifact_gate": has_professional_gate,
        "professional_package_ready": professional_package_ready,
        "professional_package_reason": package_reason,
        "raw_artifact_level": "repo_and_dependency_audit_only",
        "reduced_or_proxy_used_for_convergence": False,
        "author_decision": (
            "explicit_operational_blocker"
            if not professional_package_ready
            else "candidate_professional_close_match"
        ),
    }
    write_json(paper_dir / "blind_author_simulation.json", simulation)
    write_json(paper_dir / "operational_artifacts.json", {"repo_audits": repo_audits, "gpu_probe": gpu_probe, "blockers": blockers})
    return simulation


def keyword_score(prediction: str, references: list) -> tuple[float, list[str]]:
    ref_terms = []
    for ref in references:
        words = re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(ref).lower())
        ref_terms.extend(words)
    counts = Counter(ref_terms)
    candidates = [word for word, count in counts.items() if count >= 1]
    # Keep distinctive terms; generic systems words are too easy.
    stop = {
        "the", "and", "with", "for", "from", "that", "this", "using", "use", "paper",
        "model", "models", "method", "methods", "results", "data", "dataset", "datasets",
        "benchmark", "benchmarks", "evaluation", "training", "inference", "performance",
    }
    candidates = [w for w in candidates if w not in stop]
    if not candidates:
        return 0.0, []
    pred = prediction.lower()
    hits = sorted({w for w in candidates if w in pred})
    return len(hits) / min(len(candidates), 12), hits[:30]


def verify_simulation(paper: dict, dag: dict, simulation: dict, iteration: int) -> dict:
    fp = footprint_lists(paper)
    gap_score, gap_hits = keyword_score(
        simulation["gap_prediction"],
        fp["gaps"] + fp["failure_modes"] + fp["baselines"],
    )
    method_score, method_hits = keyword_score(
        simulation["method_prediction"],
        fp["method_components"] + fp["code_artifacts"],
    )
    experiment_score, experiment_hits = keyword_score(
        simulation["experiment_prediction"],
        fp["benchmarks"] + fp["metrics"] + fp["eval_protocols"] + fp["datasets"],
    )

    professional_ready = bool(simulation["professional_package_ready"])
    has_scale_gate = bool(simulation["contains_reduced_rejection_gate"])
    has_prof_gate = bool(simulation["contains_professional_artifact_gate"])
    exact_debt = exact_artifact_debt(fp)

    checks = [
        {
            "name": "blind_contract",
            "status": "pass" if not simulation["paper_text_seen"] and not simulation["oracle_results_seen"] else "fail",
            "detail": simulation["input_contract"],
        },
        {
            "name": "gap_semantic_match",
            "status": "pass" if gap_score >= 0.35 else "fail",
            "score": round(gap_score, 3),
            "hits": gap_hits,
            "oracle_gap_terms": fp["gaps"][:12],
        },
        {
            "name": "method_gap_binding_match",
            "status": "pass" if method_score >= 0.25 or not fp["method_components"] else "fail",
            "score": round(method_score, 3),
            "hits": method_hits,
            "oracle_method_terms": fp["method_components"][:12],
        },
        {
            "name": "experiment_axis_match",
            "status": "pass" if experiment_score >= 0.35 else "fail",
            "score": round(experiment_score, 3),
            "hits": experiment_hits,
            "oracle_experiment_terms": (fp["benchmarks"] + fp["eval_protocols"])[:12],
        },
        {
            "name": "reduced_proxy_rejection_gate",
            "status": "pass" if has_scale_gate and not simulation["reduced_or_proxy_used_for_convergence"] else "fail",
            "detail": STRICT_POLICY,
        },
        {
            "name": "professional_artifact_package",
            "status": "pass" if professional_ready and has_prof_gate else "blocked",
            "detail": {
                "ready": professional_ready,
                "reason": simulation["professional_package_reason"],
                "raw_artifact_level": simulation["raw_artifact_level"],
                "repo_audit_count": len(simulation["repo_audits"]),
            },
        },
        {
            "name": "exact_artifact_debt_recorded",
            "status": "pass" if exact_debt else "fail",
            "detail": exact_debt,
        },
    ]
    pass_count = sum(1 for check in checks if check["status"] == "pass")
    semantic_ready = all(
        check["status"] == "pass"
        for check in checks
        if check["name"] in {"blind_contract", "gap_semantic_match", "method_gap_binding_match", "experiment_axis_match", "reduced_proxy_rejection_gate", "exact_artifact_debt_recorded"}
    )
    converged = semantic_ready and professional_ready and has_prof_gate

    updates = []
    if iteration == 1:
        updates.append(
            {
                "id": "update.add_reduced_proxy_rejection_gate",
                "reason": "The p-less case showed that proxy, syntax, and reduced runs must not count as convergence.",
                "success_criteria": ["add reviewer.reject_reduced_convergence", "mark repo/generic GPU checks as support only"],
            }
        )
        updates.append(
            {
                "id": "update.add_professional_artifact_package_gate",
                "reason": "The blind author simulation must know exactly which model/data/table/figure artifacts are required.",
                "success_criteria": ["add reviewer.require_professional_artifact_package", "record exact artifact debt"],
            }
        )
    if not professional_ready:
        updates.append(
            {
                "id": "update.build_paper_shaped_runner_or_record_external_blocker",
                "reason": simulation["professional_package_reason"],
                "success_criteria": [
                    "resolve checkpoints/models/datasets/APIs named by the DAG",
                    "run paper-appropriate baselines and ablations",
                    "emit raw outputs, scoring, timing, GPU/CPU/RAM traces",
                    "compare to paper tables, figures, paragraphs, and appendix",
                ],
            }
        )

    status = (
        "converged_professional_close_match"
        if converged
        else "blocked_waiting_for_professional_artifacts_after_dag_update"
        if has_scale_gate and has_prof_gate and not professional_ready
        else "not_converged_needs_dag_update"
    )
    return {
        "created_at_utc": now_utc(),
        "paper_id": paper["chip_id"],
        "paper_title": paper.get("title"),
        "iteration": iteration,
        "checks": checks,
        "score": round(pass_count / len(checks), 6),
        "semantic_ready": semantic_ready,
        "professional_ready": professional_ready,
        "converged": converged,
        "status": status,
        "required_updates": updates,
    }


def exact_artifact_debt(fp: dict[str, list]) -> list[dict]:
    debt = []
    if fp["benchmarks"] or fp["eval_protocols"]:
        debt.append(
            {
                "id": "main_benchmark_tables",
                "required": compact(fp["benchmarks"] + fp["eval_protocols"], 16),
            }
        )
    if fp["metrics"]:
        debt.append({"id": "metric_scoring_outputs", "required": compact(fp["metrics"], 16)})
    if fp["datasets"] or fp["data_artifacts"]:
        debt.append({"id": "datasets_and_model_artifacts", "required": compact(fp["datasets"] + fp["data_artifacts"], 16)})
    if fp["hardware_environments"] or fp["framework_runtimes"]:
        debt.append({"id": "hardware_runtime_traces", "required": compact(fp["hardware_environments"] + fp["framework_runtimes"], 16)})
    if fp["method_components"] or fp["code_artifacts"]:
        debt.append({"id": "method_specific_code_path", "required": compact(fp["method_components"] + fp["code_artifacts"], 16)})
    if not debt:
        debt.append({"id": "paper_specific_artifacts", "required": "tables, figures, metrics, raw outputs, and appendix artifacts"})
    return debt


def run_paper(paper: dict, repo_status: dict, max_loops: int) -> dict:
    paper_dir = PAPER_DIR / slugify(paper["chip_id"])
    paper_dir.mkdir(parents=True, exist_ok=True)
    previous_updates: list[dict] = []
    iterations = []
    for iteration in range(1, max_loops + 1):
        dag = build_gap_dag(paper, repo_status, iteration, previous_updates)
        write_json(paper_dir / f"paper_author_gap_dag_iter_{iteration:02d}.json", dag)
        if iteration == max_loops:
            write_json(paper_dir / "paper_author_gap_dag.json", dag)
        simulation = simulate_from_dag_only(dag, paper_dir)
        verification = verify_simulation(paper, dag, simulation, iteration)
        write_json(paper_dir / f"verifier_result_iter_{iteration:02d}.json", verification)
        iterations.append({"iteration": iteration, "dag_signature": dag["signature"], "simulation": simulation, "verification": verification})
        previous_updates = verification["required_updates"]
        if verification["converged"]:
            break
    final = iterations[-1]["verification"]
    status_lines = [
        f"# {paper.get('title')}",
        "",
        f"- Paper id: `{paper['chip_id']}`",
        f"- Final status: `{final['status']}`",
        f"- Converged: `{str(final['converged']).lower()}`",
        f"- Score: `{final['score']}`",
        f"- Semantic ready: `{str(final['semantic_ready']).lower()}`",
        f"- Professional ready: `{str(final['professional_ready']).lower()}`",
        "",
        "## Checks",
        "",
    ]
    for check in final["checks"]:
        status_lines.append(f"- `{check['name']}`: `{check['status']}`")
    status_lines.extend(["", "## Required Updates", ""])
    for update in final["required_updates"]:
        status_lines.append(f"- `{update['id']}`: {update['reason']}")
    (paper_dir / "STATUS.md").write_text("\n".join(status_lines) + "\n", encoding="utf-8")
    return {
        "paper_id": paper["chip_id"],
        "title": paper.get("title"),
        "family_tags": family_tags(paper),
        "final_status": final["status"],
        "converged": final["converged"],
        "score": final["score"],
        "semantic_ready": final["semantic_ready"],
        "professional_ready": final["professional_ready"],
        "repo_count": len(repo_status.get("repos", [])),
        "repo_exact_rerun_status": repo_status.get("exact_rerun_status", "unknown"),
        "implementation_statuses": footprint_lists(paper)["implementation_statuses"],
        "required_update_ids": [u["id"] for u in final["required_updates"]],
        "paper_dir": str(paper_dir),
        "iterations": iterations,
    }


def choose_runner_type(row: dict) -> str:
    tags = set(row["family_tags"])
    title = row["title"].lower()
    paper_id = row["paper_id"].lower()
    if "flashvid" in paper_id or "flashvid" in title:
        return "vlm_video_token_merging_runner"
    if "sparserl" in paper_id or "sparse cuda" in title:
        return "sparse_cuda_kernel_quality_runner"
    if "prophet" in paper_id or "hsd" in paper_id or "speculative" in title or "early commit" in title:
        return "llm_decoding_acceptance_runner"
    if "loongrl" in paper_id:
        return "long_context_reasoning_rl_runner"
    if "mrrope" in paper_id or "rope" in title:
        return "long_context_position_encoding_runner"
    if "seacache" in paper_id or "sencache" in paper_id or "cache" in title:
        return "diffusion_cache_latency_quality_runner"
    if "rdvq" in paper_id or "rate-distortion" in title or "codec" in title:
        return "compression_rate_distortion_runner"
    if "atoken" in paper_id or "infotok" in paper_id or "tokenizer" in title:
        return "visual_tokenizer_multitask_runner"
    if "nuwa" in paper_id or "pruning" in title:
        return "edge_vit_pruning_runner"
    if "clot" in paper_id or "lagrangian optimal transport" in title:
        return "hyperparameter_trajectory_ot_runner"
    if "prism" in paper_id or "fmri" in title or "brain" in title:
        return "fmri_reconstruction_qa_runner"
    if "rational" in paper_id or "agents" in title:
        return "agent_decision_benchmark_runner"
    if "spark" in paper_id or "trellis" in paper_id or "3d" in title:
        return "3d_generation_reconstruction_runner"
    if "dto_kd" in paper_id or "knowledge distillation" in title:
        return "distillation_tradeoff_runner"
    if "diffusion_cache" in tags:
        return "diffusion_cache_latency_quality_runner"
    if "compression_vq" in tags:
        return "compression_rate_distortion_runner"
    if "visual_tokenizer" in tags:
        return "visual_tokenizer_multitask_runner"
    if "long_context_or_rope" in tags:
        return "long_context_position_encoding_runner"
    if "sparse_cuda_rl" in tags:
        return "sparse_cuda_kernel_quality_runner"
    return "paper_specific_artifact_runner"


def build_specialized_runner_queue(summary: dict) -> list[dict]:
    queue = []
    for row in summary["papers"]:
        final = row["iterations"][-1]["verification"]
        sim = row["iterations"][-1]["simulation"]
        artifact_check = next(c for c in final["checks"] if c["name"] == "professional_artifact_package")
        exact_debt = next(c for c in final["checks"] if c["name"] == "exact_artifact_debt_recorded")["detail"]
        repo_audits = sim.get("repo_audits", [])
        repo_paths = [audit["repo"] for audit in repo_audits if audit.get("exists")]
        if row["converged"]:
            priority = "done"
        elif row["repo_exact_rerun_status"] == "code_ready_needs_model_data":
            priority = "high"
        elif row["repo_count"] > 0:
            priority = "medium"
        else:
            priority = "external_blocked"
        queue.append(
            {
                "paper_id": row["paper_id"],
                "title": row["title"],
                "priority": priority,
                "runner_type": choose_runner_type(row),
                "repo_exact_rerun_status": row["repo_exact_rerun_status"],
                "repo_paths": repo_paths,
                "implementation_statuses": row["implementation_statuses"],
                "professional_blocker": artifact_check["detail"]["reason"],
                "exact_artifact_debt": exact_debt,
                "next_actions": [
                    "resolve all DAG-named checkpoints/models/datasets/APIs",
                    "build a paper-shaped runner for the listed runner_type",
                    "run paper baselines plus method on the paper's stated benchmarks/metrics",
                    "emit raw outputs, scores, timing, GPU/CPU/RAM traces, and table/figure summaries",
                    "rerun the verifier; do not count reduced/proxy/syntax-only evidence",
                ],
            }
        )
    priority_order = {"high": 0, "medium": 1, "external_blocked": 2, "done": 3}
    return sorted(queue, key=lambda item: (priority_order.get(item["priority"], 9), item["paper_id"]))


def write_queue(summary: dict) -> None:
    queue = build_specialized_runner_queue(summary)
    write_json(QUEUE_JSON, {"created_at_utc": now_utc(), "queue": queue})
    lines = [
        "# Specialized Runner Queue",
        "",
        f"Date: `{now_utc()}`",
        "",
        "This queue is the Loop 1 repair output after applying the p-less strict gate to the remaining 19 papers. Items are not converged; they are ordered by operational readiness.",
        "",
    ]
    for item in queue:
        lines.extend(
            [
                f"## `{item['paper_id']}`",
                "",
                f"- Title: {item['title']}",
                f"- Priority: `{item['priority']}`",
                f"- Runner type: `{item['runner_type']}`",
                f"- Repo exact-rerun status: `{item['repo_exact_rerun_status']}`",
                f"- Repos: `{compact(item['repo_paths'], 4)}`",
                f"- Professional blocker: `{item['professional_blocker']}`",
                "- Required artifact debt:",
            ]
        )
        for debt in item["exact_artifact_debt"]:
            lines.append(f"  - `{debt['id']}`: {debt['required']}")
        lines.append("")
    QUEUE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(summary: dict) -> None:
    lines = [
        "# Remaining 19 Strict DIRS Report",
        "",
        f"Date: `{summary['created_at_utc']}`",
        f"Domain: `{summary['domain']}`",
        f"Papers: `{summary['paper_count']}`",
        f"Excluded completed template paper: `{PLESS_ID}`",
        "",
        "## Strict Rule",
        "",
        "The p-less case is treated as the template: paper reading, repo syntax checks, README/script inventories, and generic GPU motif rows are not convergence evidence. A paper converges only with verifier-accepted professional paper-shaped artifacts. Otherwise the run records explicit blockers and DAG updates.",
        "",
        "## Summary",
        "",
        f"- Accepted professional close match: `{summary['accepted_count']}` / `{summary['paper_count']}`",
        f"- Explicitly blocked after DAG update: `{summary['blocked_count']}` / `{summary['paper_count']}`",
        f"- Need specialized runner/artifact resolution: `{summary['needs_specialized_runner_count']}` / `{summary['paper_count']}`",
        f"- GPU available during run: `{str(summary['gpu_available']).lower()}`",
        "",
        "## Per Paper",
        "",
    ]
    for row in summary["papers"]:
        lines.append(
            f"- `{row['paper_id']}`: `{row['final_status']}`, score `{row['score']}`, "
            f"repos `{row['repo_count']}`, families `{', '.join(row['family_tags'])}`"
        )
    lines.extend(["", "## Artifacts", ""])
    lines.append(f"- JSON: `{OUTPUT_JSON}`")
    lines.append(f"- Paper run directory: `{PAPER_DIR}`")
    lines.append(f"- Specialized runner queue: `{QUEUE_MD}`")
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_status(summary: dict) -> None:
    lines = [
        "# Remaining 19 p-less-Style DIRS Long Goal Status",
        "",
        f"Date: `{summary['created_at_utc']}`",
        "",
        f"- Final status: `{summary['final_status']}`",
        f"- Accepted professional close match: `{summary['accepted_count']}` / `{summary['paper_count']}`",
        f"- Explicitly blocked after DAG update: `{summary['blocked_count']}` / `{summary['paper_count']}`",
        f"- Reduced/smoke/proxy convergence disallowed: `true`",
        f"- GPU available during run: `{str(summary['gpu_available']).lower()}`",
        "",
        "The run created a paper-specific DAG and DAG-only author simulation for each remaining paper. It did not promote repo audits or generic GPU motif rows into convergence. Papers without verifier-accepted professional artifacts remain explicitly blocked with required update nodes.",
        "",
    ]
    STATUS_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-loops", type=int, default=2)
    args = parser.parse_args()

    papers, campaign = load_inputs()
    repo_map = repo_status_by_chip(campaign)
    gpu = device_probe()
    results = []
    for paper in papers:
        results.append(run_paper(paper, repo_map.get(paper["chip_id"], {}), args.max_loops))

    accepted = [row for row in results if row["final_status"] == "converged_professional_close_match"]
    blocked = [row for row in results if row["final_status"] == "blocked_waiting_for_professional_artifacts_after_dag_update"]
    needs_specialized = [
        row
        for row in results
        if "update.build_paper_shaped_runner_or_record_external_blocker" in row["required_update_ids"]
    ]
    summary = {
        "created_at_utc": now_utc(),
        "domain": campaign.get("domain", "LLM Inference / Systems / Token Efficiency"),
        "source_evidence": str(EVIDENCE_PATH),
        "source_campaign": str(CAMPAIGN_PATH),
        "completed_template_paper": PLESS_ID,
        "paper_count": len(results),
        "strict_policy": STRICT_POLICY,
        "gpu_available": gpu["status"] == "pass",
        "accepted_count": len(accepted),
        "blocked_count": len(blocked),
        "needs_specialized_runner_count": len(needs_specialized),
        "final_status": (
            "converged_all_remaining19"
            if len(accepted) == len(results)
            else "completed_with_explicit_operational_blockers"
        ),
        "papers": results,
    }
    write_json(OUTPUT_JSON, summary)
    write_queue(summary)
    write_report(summary)
    write_status(summary)
    print(json.dumps({
        "final_status": summary["final_status"],
        "papers": summary["paper_count"],
        "accepted": summary["accepted_count"],
        "blocked": summary["blocked_count"],
        "needs_specialized_runner": summary["needs_specialized_runner_count"],
        "output": str(OUTPUT_JSON),
    }, indent=2))


if __name__ == "__main__":
    main()
