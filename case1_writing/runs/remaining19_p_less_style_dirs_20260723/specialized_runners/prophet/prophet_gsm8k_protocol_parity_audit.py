#!/usr/bin/env python3
"""Source-bound GSM8K protocol parity audit for Prophet.

This is a verifier/Loop-1 repair artifact, not a Loop-2 oracle input. It checks
whether the current custom full-split runner has enough prompt, suffix,
answer-region, harness, and scorer parity to interpret GSM8K result shape.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ROOT = Path("/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723")
RUNNER_DIR = RUN_ROOT / "specialized_runners/prophet"
PAPER_RUN = RUN_ROOT / "paper_runs/iclr2026_g88nt4ietg_prophet_dlm_early_commit_decoding"
REPO = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/Prophet")
PAPER_TEXT = Path("/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/text/ICLR2026_g88nt4ieTG_openreview.txt")

CUSTOM_RUNNER = RUNNER_DIR / "prophet_custom_full_gsm8k_runner.py"
README = REPO / "README.md"
EVAL_LLADA = REPO / "eval_llada.py"
GENERATE_EARLYEXIT = REPO / "generate_earlyexit.py"
DAG_PATH = PAPER_RUN / "paper_author_gap_dag.json"

AUDIT_PATH = RUNNER_DIR / "gsm8k_protocol_parity_audit.json"
AUDIT_STATUS_PATH = RUNNER_DIR / "GSM8K_PROTOCOL_PARITY_AUDIT.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def line_numbers(path: Path, patterns: list[str]) -> dict[str, list[int]]:
    text = read_text(path)
    out: dict[str, list[int]] = {}
    lines = text.splitlines()
    for pattern in patterns:
        regex = re.compile(pattern)
        out[pattern] = [idx for idx, line in enumerate(lines, start=1) if regex.search(line)]
    return out


def contains(path: Path, needle: str) -> bool:
    return needle in read_text(path)


def build_payload() -> dict[str, Any]:
    now = utc_now()
    custom_text = read_text(CUSTOM_RUNNER)
    readme_text = read_text(README)
    eval_text = read_text(EVAL_LLADA)
    early_text = read_text(GENERATE_EARLYEXIT)
    dag = read_json(DAG_PATH, {})
    dag_node_ids = [
        node.get("id")
        for node in dag.get("nodes", [])
        if isinstance(node, dict) and node.get("id")
    ]

    source_facts = {
        "custom_runner": {
            "path": str(CUSTOM_RUNNER),
            "uses_raw_zero_shot_prompt_constant": "OFFICIAL_ZERO_SHOT_PROMPT" in custom_text,
            "uses_tokenizer_apply_chat_template": "apply_chat_template" in custom_text,
            "default_constraints_text": "200:The|201:answer|202:is"
            if '"200:The|201:answer|202:is"' in custom_text
            else None,
            "answer_start_formula_matches_release_eval": "max(constraints.keys()) + 2" in custom_text,
            "uses_flexible_numeric_extractor": "extract_flexible_answer" in custom_text,
            "line_anchors": line_numbers(
                CUSTOM_RUNNER,
                [
                    r"OFFICIAL_ZERO_SHOT_PROMPT",
                    r"constraints-text",
                    r"answer_start = max",
                    r"extract_flexible_answer",
                    r"prompt_text = OFFICIAL_ZERO_SHOT_PROMPT",
                ],
            ),
        },
        "released_eval": {
            "eval_llada_path": str(EVAL_LLADA),
            "readme_path": str(README),
            "readme_example_uses_chat_template": "apply_chat_template" in readme_text,
            "readme_eval_command_uses_lm_eval_gsm8k_cot_zeroshot": "gsm8k_cot_zeroshot" in readme_text,
            "readme_eval_command_uses_same_constraints": "constraints_text=\"200:The|201:answer|202:is\"" in readme_text,
            "eval_llada_answer_start_formula_matches_custom": "answer_start = max(constraints.keys()) + 2" in eval_text,
            "eval_llada_decodes_generated_answer": "generated_answer = self.tokenizer.decode" in eval_text,
            "generate_earlyexit_fixed_answer_length": "answer_length = 5" in early_text,
            "line_anchors": {
                "README.md": line_numbers(
                    README,
                    [
                        r"apply_chat_template",
                        r"gsm8k_cot_zeroshot",
                        r"constraints_text",
                        r"answer_start_pos",
                    ],
                ),
                "eval_llada.py": line_numbers(
                    EVAL_LLADA,
                    [
                        r"answer_start = max",
                        r"generated_answer = self.tokenizer.decode",
                        r"constraints = _parse_constraints",
                    ],
                ),
                "generate_earlyexit.py": line_numbers(
                    GENERATE_EARLYEXIT,
                    [
                        r"answer_length = 5",
                        r"answer_positions",
                        r"should_early_exit",
                    ],
                ),
            },
        },
        "paper_protocol_anchors": {
            "path": str(PAPER_TEXT),
            "line_anchors": {
                "simple_evals_prompt_and_generated_answer_extraction": [779, 780, 781, 782, 783, 1505],
                "suffix_prompt_structure": [1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496],
                "answer_region_determination": [1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516],
            },
            "note": "Line anchors identify protocol evidence only; this audit intentionally omits table target metric values.",
        },
    }

    prompt_status = (
        "partial_unproven_equivalence"
        if source_facts["custom_runner"]["uses_raw_zero_shot_prompt_constant"]
        and source_facts["released_eval"]["readme_example_uses_chat_template"]
        else "unknown"
    )
    suffix_status = (
        "matches_released_eval_formula_but_not_full_paper_protocol"
        if source_facts["custom_runner"]["answer_start_formula_matches_release_eval"]
        and source_facts["released_eval"]["eval_llada_answer_start_formula_matches_custom"]
        else "unresolved"
    )
    harness_status = (
        "custom_fallback_support_only_until_equivalence_audit"
        if source_facts["released_eval"]["readme_eval_command_uses_lm_eval_gsm8k_cot_zeroshot"]
        else "missing_released_eval_path"
    )
    scorer_status = (
        "custom_extractor_not_exact_simple_evals_extractor"
        if source_facts["custom_runner"]["uses_flexible_numeric_extractor"]
        else "unresolved"
    )

    findings = [
        {
            "id": "prompt_template_parity",
            "status": prompt_status,
            "evidence": [
                "Custom runner builds a raw Q/A zero-shot prompt constant.",
                "Released examples show chat-template prompting; README evaluation uses lm-eval GSM8K CoT task.",
                "Paper says simple-evals prompt with step-by-step reasoning.",
            ],
            "repair_implication": "Do not treat a final speed/accuracy mismatch as a method failure until exact prompt/template parity is encoded or explicitly blocked.",
        },
        {
            "id": "suffix_answer_region_parity",
            "status": suffix_status,
            "evidence": [
                "Custom runner and eval_llada both set answer_start to max constraint position plus two.",
                "Released early-exit code monitors a fixed five-token answer region.",
                "Paper appendix describes a suffix prompt followed by a reserved final-result region.",
            ],
            "repair_implication": "If step savings remain low, verify whether fixed answer_start/answer_length and suffix placement match the paper evaluator for GSM8K.",
        },
        {
            "id": "harness_parity",
            "status": harness_status,
            "evidence": [
                "Release exposes eval_llada through lm-evaluation-harness.",
                "Current custom runner bypasses that harness to preserve a full non-reduced run after official harness failure.",
            ],
            "repair_implication": "Custom full-split rows are real operational evidence, but exact paper-table parity still needs lm-eval/simple-evals equivalence or a narrower blocker.",
        },
        {
            "id": "generated_answer_extractor_parity",
            "status": scorer_status,
            "evidence": [
                "Custom runner computes strict and flexible numeric extraction.",
                "Paper appendix says generated final answers are extracted, not multiple-choice log-prob compared.",
                "Exact simple-evals extractor is not recovered in the local released path.",
            ],
            "repair_implication": "Final accuracy-delta mismatch should trigger extractor parity audit or rescoring before any conclusion.",
        },
    ]

    repair_axis_ids = [
        "prompt_template_parity",
        "suffix_answer_region_parity",
        "harness_parity",
        "generated_answer_extractor_parity",
    ]
    dag_coverage = {
        "dag_path": str(DAG_PATH),
        "covered_repair_nodes": [
            node_id
            for node_id in dag_node_ids
            if node_id
            in {
                "protocol.prompt_template_parity_gate",
                "runner.suffix_answer_region_parity_gate",
                "protocol.simple_evals_vs_lmeval_harness_gate",
                "scoring.generated_answer_extractor_parity_gate",
            }
        ],
        "required_repair_axis_ids": repair_axis_ids,
    }
    status = "protocol_parity_partial_with_repair_nodes_encoded"
    if len(dag_coverage["covered_repair_nodes"]) < 4:
        status = "protocol_parity_partial_dag_repair_nodes_missing"

    return {
        "artifact_kind": "prophet_gsm8k_protocol_parity_audit",
        "created_at_utc": now,
        "status": status,
        "paper_id": "ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding",
        "paper_title": "Diffusion Language Models Know the Answer Before Decoding",
        "visibility_contract": {
            "loop2_author_can_read": False,
            "verifier_can_read": True,
            "paper_oracle_target_values_included": False,
            "oracle_values_exposed_to_loop2": False,
        },
        "source_facts": source_facts,
        "findings": findings,
        "dag_coverage": dag_coverage,
        "verifier_implication": {
            "can_converge_from_this_audit_alone": False,
            "partial_rows_are_convergence_evidence": False,
            "recommended_action_if_final_shape_fails": "activate encoded prompt/suffix/harness/scorer repair gates and rerun or rescore from real full artifacts",
            "recommended_action_if_final_shape_passes": "keep audit as protocol caveat and continue remaining professional debt gates",
        },
        "report_path": str(AUDIT_PATH),
        "status_path": str(AUDIT_STATUS_PATH),
    }


def write_status(payload: dict[str, Any]) -> None:
    lines = [
        "# GSM8K Protocol Parity Audit",
        "",
        f"- Updated: `{payload['created_at_utc']}`",
        f"- Status: `{payload['status']}`",
        f"- Loop 2 can read this: `{payload['visibility_contract']['loop2_author_can_read']}`",
        f"- Can converge from this audit alone: `{payload['verifier_implication']['can_converge_from_this_audit_alone']}`",
        f"- Report: `{payload['report_path']}`",
        "",
        "## Findings",
        "",
    ]
    for finding in payload["findings"]:
        lines.append(f"- `{finding['id']}`: `{finding['status']}`")
    lines.extend(
        [
            "",
            "## DAG Coverage",
            "",
            f"- Covered repair nodes: `{payload['dag_coverage']['covered_repair_nodes']}`",
        ]
    )
    AUDIT_STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = build_payload()
    write_json(AUDIT_PATH, payload)
    write_status(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
