#!/usr/bin/env python3
"""Audit released-source parity blockers for the Prophet full-paper simulation.

This is a verifier-side artifact. It does not promote any reduced run to
convergence; it records when a paper-required experimental axis cannot be
executed from the released repo without inventing author-only code or models.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNNER_DIR = Path(__file__).resolve().parent
REPO = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/Prophet")
PAPER_TEXT = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/text/ICLR2026_g88nt4ieTG_openreview.txt")
REPORT_PATH = RUNNER_DIR / "source_parity_blocker_audit.json"
STATUS_PATH = RUNNER_DIR / "SOURCE_PARITY_BLOCKER_AUDIT.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_command(args: list[str], cwd: Path | None = None, timeout: int = 30) -> dict[str, Any]:
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        "args": args,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def line_anchor(path: Path, line_no: int, label: str) -> dict[str, Any]:
    lines = read_lines(path)
    text = lines[line_no - 1].strip() if 1 <= line_no <= len(lines) else ""
    return {
        "id": label,
        "path": str(path),
        "file": safe_rel(path),
        "line": line_no,
        "text": text,
    }


def regex_anchors(path: Path, patterns: dict[str, str], max_hits_per_pattern: int = 8) -> list[dict[str, Any]]:
    lines = read_lines(path)
    anchors: list[dict[str, Any]] = []
    for label, pattern in patterns.items():
        rx = re.compile(pattern, re.IGNORECASE)
        hits = 0
        for line_no, line in enumerate(lines, 1):
            if not rx.search(line):
                continue
            anchors.append(
                {
                    "id": f"{label}:{line_no}",
                    "path": str(path),
                    "file": safe_rel(path),
                    "line": line_no,
                    "text": line.strip(),
                }
            )
            hits += 1
            if hits >= max_hits_per_pattern:
                break
    return anchors


def repo_files() -> list[str]:
    if not REPO.exists():
        return []
    return sorted(str(path.relative_to(REPO)) for path in REPO.rglob("*") if path.is_file() and ".git/" not in str(path))


def repository_snapshot() -> dict[str, Any]:
    local_head = run_command(["git", "rev-parse", "HEAD"], cwd=REPO)
    remote_v = run_command(["git", "remote", "-v"], cwd=REPO)
    remote_head = run_command(["git", "ls-remote", "origin", "HEAD", "refs/heads/main"], cwd=REPO)
    status = run_command(["git", "status", "--short"], cwd=REPO)
    remote_hashes = [
        line.split()[0]
        for line in remote_head["stdout"].splitlines()
        if line.split()
    ]
    local_hash = local_head["stdout"].strip()
    return {
        "repo": str(REPO),
        "remote": remote_v["stdout"],
        "local_head": local_hash,
        "remote_heads": remote_head["stdout"].splitlines(),
        "remote_matches_local": bool(local_hash and local_hash in remote_hashes),
        "git_status_short": status["stdout"].splitlines(),
        "released_file_count": len(repo_files()),
        "released_files": repo_files(),
    }


def code_evidence() -> dict[str, list[dict[str, Any]]]:
    readme = REPO / "README.md"
    generate = REPO / "generate.py"
    earlyexit = REPO / "generate_earlyexit.py"
    eval_llada = REPO / "eval_llada.py"
    eval_sh = REPO / "eval.sh"
    return {
        "readme_claims_and_usage": [
            line_anchor(readme, 1, "readme_title"),
            line_anchor(readme, 3, "paper_title"),
            line_anchor(readme, 15, "dream_claim"),
            line_anchor(readme, 86, "eval_command"),
            line_anchor(readme, 87, "gsm8k_task"),
            line_anchor(readme, 88, "llada_model_registration"),
            line_anchor(readme, 89, "llada_model_args"),
            line_anchor(readme, 108, "remasking_declared_options"),
            line_anchor(readme, 112, "generate_earlyexit_component"),
            line_anchor(readme, 120, "eval_llada_component"),
        ],
        "baseline_generation": [
            line_anchor(generate, 31, "baseline_generate_signature"),
            line_anchor(generate, 32, "baseline_default_remasking"),
            line_anchor(generate, 76, "baseline_low_confidence_branch"),
            line_anchor(generate, 79, "baseline_random_branch"),
            line_anchor(generate, 82, "baseline_unknown_remasking_rejected"),
            line_anchor(generate, 113, "baseline_hardcoded_llada_model"),
            line_anchor(generate, 114, "baseline_hardcoded_llada_tokenizer"),
        ],
        "prophet_generation": [
            line_anchor(earlyexit, 52, "prophet_generate_signature"),
            line_anchor(earlyexit, 53, "prophet_default_remasking"),
            line_anchor(earlyexit, 108, "prophet_low_confidence_branch"),
            line_anchor(earlyexit, 111, "prophet_random_branch"),
            line_anchor(earlyexit, 114, "prophet_unknown_remasking_rejected"),
            line_anchor(earlyexit, 197, "prophet_hardcoded_llada_model"),
            line_anchor(earlyexit, 198, "prophet_hardcoded_llada_tokenizer"),
        ],
        "evaluation_harness": [
            line_anchor(eval_llada, 1, "llada_eval_docstring"),
            line_anchor(eval_llada, 49, "llada_only_model_registration"),
            line_anchor(eval_llada, 80, "automodel_loader_from_model_path"),
            line_anchor(eval_llada, 93, "tokenizer_loader_from_model_path"),
            line_anchor(eval_llada, 283, "prophet_generation_branch"),
            line_anchor(eval_llada, 305, "baseline_generation_branch"),
            line_anchor(eval_llada, 319, "generated_answer_decode"),
            line_anchor(eval_llada, 325, "generated_answer_retokenize"),
        ],
        "released_eval_script": [
            line_anchor(eval_sh, 6, "accelerate_eval_llada"),
            line_anchor(eval_sh, 7, "gsm8k_task_only"),
            line_anchor(eval_sh, 8, "llada_dist_only"),
            line_anchor(eval_sh, 9, "llada_model_path_and_prophet_args"),
        ],
        "repo_regex_hits": regex_anchors(
            REPO / "README.md",
            {
                "dream": r"dream",
                "remasking": r"remasking",
                "simple_eval": r"simple[-_ ]eval",
            },
        )
        + regex_anchors(
            REPO / "eval_llada.py",
            {
                "dream": r"dream",
                "top_k": r"top[-_ ]?k|margin",
                "simple_eval": r"simple[-_ ]eval",
            },
        )
        + regex_anchors(
            REPO / "generate.py",
            {"top_k": r"top[-_ ]?k|margin", "not_implemented": r"NotImplementedError"},
        )
        + regex_anchors(
            REPO / "generate_earlyexit.py",
            {"top_k": r"top[-_ ]?k|margin", "not_implemented": r"NotImplementedError"},
        ),
    }


def paper_evidence() -> dict[str, list[dict[str, Any]]]:
    p = PAPER_TEXT
    return {
        "table1_and_model_axis": [
            line_anchor(p, 626, "experiments_on_two_models"),
            line_anchor(p, 627, "llada_and_dream_compare_full_prophet"),
            line_anchor(p, 632, "table1_caption"),
            line_anchor(p, 633, "table1_accuracy_metric"),
            line_anchor(p, 634, "table1_speed_eval_notes"),
            line_anchor(p, 635, "appendix_config_pointer"),
            line_anchor(p, 779, "simple_evals_prompt"),
            line_anchor(p, 780, "generation_length_policy"),
            line_anchor(p, 782, "greedy_deterministic_policy"),
        ],
        "table2_acceleration_axis": [
            line_anchor(p, 825, "sdtt_method"),
            line_anchor(p, 826, "sdtt_distillation_setup"),
            line_anchor(p, 827, "sdtt_table2a_claim"),
            line_anchor(p, 829, "sdtt_prophet_combination"),
            line_anchor(p, 832, "fastdllm_method"),
            line_anchor(p, 834, "fastdllm_prophet_combination"),
            line_anchor(p, 835, "fastdllm_table2b_claim"),
            line_anchor(p, 838, "table2_caption"),
        ],
        "table3_table4_ablation_axis": [
            line_anchor(p, 902, "table3_caption"),
            line_anchor(p, 903, "table3_remasking_caption"),
            line_anchor(p, 948, "top_k_margin_row"),
            line_anchor(p, 1016, "remasking_strategy_discussion"),
            line_anchor(p, 1017, "three_remasking_heuristics"),
            line_anchor(p, 1020, "top_k_margin_gain"),
            line_anchor(p, 962, "table4_caption"),
            line_anchor(p, 1022, "block_length_discussion"),
            line_anchor(p, 1026, "large_block_gains"),
        ],
        "answer_extraction_axis": [
            line_anchor(p, 1516, "dynamic_answer_extraction_note"),
            line_anchor(p, 1525, "generate_and_extract_final_answer"),
        ],
        "paper_regex_hits": regex_anchors(
            p,
            {
                "dream": r"Dream-7B|Dream",
                "simple_evals": r"simple-evals",
                "sdtt": r"SDTT",
                "fastdllm": r"Fast-dLLM",
                "top_k_margin": r"top-k margin|Top-k margin",
                "table3": r"Table 3",
                "table4": r"Table 4",
                "extract_final": r"extract the final answer|final answer",
            },
            max_hits_per_pattern=10,
        ),
    }


def blocker_audits() -> list[dict[str, Any]]:
    return [
        {
            "id": "dream7b_axis",
            "paper_requirement": "Table 1 includes Dream-7B-Instruct full-step and Prophet rows across the benchmark suite.",
            "status": "evidence_bound_blocked_missing_exact_dream_operational_path",
            "why_not_runnable_now": [
                "The release advertises Dream compatibility but exposes eval_llada.py as the only benchmark harness.",
                "The command examples and generation script main paths instantiate GSAI-ML/LLaDA-8B-Instruct.",
                "No eval_dream.py, Dream model identifier, Dream mask-token check, or Dream prompt/scorer parity wrapper is present in the current release.",
            ],
            "professional_gate": "Do not substitute an arbitrary Dream checkpoint or LLaDA runner without an explicit parity node.",
            "next_required_artifact": "Exact Dream model id plus generation/evaluation wrapper matching the paper's simple-evals protocol.",
        },
        {
            "id": "top_k_margin_remasking",
            "paper_requirement": "Table 3b compares random, low-confidence, and top-k margin remasking.",
            "status": "evidence_bound_blocked_missing_released_top_k_margin_code_path",
            "why_not_runnable_now": [
                "generate.py implements low_confidence and random only.",
                "generate_earlyexit.py implements low_confidence and random only.",
                "Unknown remasking strategies raise NotImplementedError, so a top-k margin run would require new implementation beyond released-source parity.",
            ],
            "professional_gate": "Do not count a hand-rolled top-k margin as paper reproduction unless Loop 1 adds an author-parity implementation node and verifier accepts it.",
            "next_required_artifact": "Released or reconstructed top-k margin remasking definition with tests against static and Prophet branches.",
        },
        {
            "id": "simple_evals_table1_prompt_scorer_parity",
            "paper_requirement": "Table 1 follows simple-evals prompts for LLaDA and Dream with generated-answer extraction.",
            "status": "partially_runnable_for_gsm8k_custom_runner_but_table1_suite_parity_blocked",
            "why_not_runnable_now": [
                "The custom GSM8K node runs generated-answer extraction for the active full split.",
                "The released benchmark script is an lm-evaluation-harness LLaDA integration, not the full simple-evals setup named in the paper.",
                "Non-GSM8K Table 1 tasks require task-specific prompt/extractor parity before being accepted as paper-shaped results.",
            ],
            "professional_gate": "Do not let lm-eval defaults silently stand in for the paper's simple-evals protocol across the whole Table 1 grid.",
            "next_required_artifact": "Per-task simple-evals prompts, extraction rules, and Dream/LLaDA scorer parity encoded in DAG or released code.",
        },
        {
            "id": "table2_sdtt_fastdllm_combinations",
            "paper_requirement": "Table 2 reports SDTT, SDTT+Prophet, Fast-dLLM, and Fast-dLLM+Prophet on GSM8K.",
            "status": "evidence_bound_blocked_missing_external_sdtt_fastdllm_artifacts",
            "why_not_runnable_now": [
                "The Prophet release does not include an SDTT-distilled checkpoint or distillation recipe artifact.",
                "The Prophet release does not include Fast-dLLM integration code or a KV-cache/parallel-decoding runner.",
                "Combining Prophet with either baseline requires exact external artifacts, not a proxy speed calculation.",
            ],
            "professional_gate": "Do not infer multiplicative speedup from paper text; require runnable SDTT/Fast-dLLM artifacts or keep this debt explicit.",
            "next_required_artifact": "SDTT checkpoint/training recipe and Fast-dLLM integration runnable on the same GSM8K protocol.",
        },
    ]


def build_report() -> dict[str, Any]:
    code = code_evidence()
    paper = paper_evidence()
    blockers = blocker_audits()
    return {
        "artifact_kind": "prophet_source_parity_blocker_audit",
        "created_at_utc": utc_now(),
        "paper_id": "ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding",
        "paper_title": "Diffusion Language Models Know the Answer Before Decoding",
        "status": "evidence_bound_source_parity_blockers_ready",
        "policy": {
            "reduced_or_proxy_results_can_converge": False,
            "loop2_may_use_paper_oracle_values": False,
            "real_runnable_nodes_must_keep_running": True,
            "missing_released_axes_must_be_explicitly_blocked_or_repaired_in_loop1": True,
        },
        "repository_snapshot": repository_snapshot(),
        "paper_evidence": paper,
        "code_evidence": code,
        "blocker_audits": blockers,
        "runnable_new_nodes": [],
        "verifier_implication": {
            "can_converge_from_this_audit_alone": False,
            "accepted_use": "support explicit blocker classification for paper-required axes that cannot be run from released artifacts",
            "still_required": [
                "complete active full GSM8K paired run",
                "complete or explicitly block Table 1 suite parity",
                "complete running Table 3/4 ablation grid where code supports it",
                "run paper comparator after final artifacts arrive",
            ],
        },
        "recommended_loop1_update": {
            "needed": False,
            "reason": "The current DAG already treats these as operational artifact blockers; this audit strengthens the evidence channel.",
        },
        "counts": {
            "blocker_count": len(blockers),
            "runnable_new_node_count": 0,
            "paper_evidence_anchor_count": sum(len(items) for items in paper.values()),
            "code_evidence_anchor_count": sum(len(items) for items in code.values()),
        },
    }


def render_status(report: dict[str, Any]) -> None:
    lines = [
        "# Prophet Source-Parity Blocker Audit",
        "",
        f"- Updated: `{report['created_at_utc']}`",
        f"- Status: `{report['status']}`",
        "- Policy: no reduced/proxy result can converge a paper.",
        f"- Repo: `{report['repository_snapshot']['repo']}`",
        f"- Local HEAD: `{report['repository_snapshot']['local_head']}`",
        f"- Remote matches local: `{report['repository_snapshot']['remote_matches_local']}`",
        f"- Git status entries: `{len(report['repository_snapshot']['git_status_short'])}`",
        f"- Paper evidence anchors: `{report['counts']['paper_evidence_anchor_count']}`",
        f"- Code evidence anchors: `{report['counts']['code_evidence_anchor_count']}`",
        f"- Runnable new nodes: `{report['counts']['runnable_new_node_count']}`",
        "",
        "## Explicit Source-Parity Blockers",
        "",
    ]
    for item in report["blocker_audits"]:
        lines.append(f"- `{item['id']}`: `{item['status']}`")
    lines += [
        "",
        "## Verifier Implication",
        "",
        f"- Can converge from this audit alone: `{report['verifier_implication']['can_converge_from_this_audit_alone']}`",
        f"- Accepted use: `{report['verifier_implication']['accepted_use']}`",
        "",
        "## Still Required",
        "",
    ]
    for item in report["verifier_implication"]["still_required"]:
        lines.append(f"- {item}")
    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report = build_report()
    write_json(REPORT_PATH, report)
    render_status(report)
    print(json.dumps({"report": str(REPORT_PATH), "status": str(STATUS_PATH), "audit_status": report["status"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
