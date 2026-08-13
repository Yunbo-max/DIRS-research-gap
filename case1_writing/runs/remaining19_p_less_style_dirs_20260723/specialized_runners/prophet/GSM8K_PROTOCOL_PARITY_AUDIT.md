# GSM8K Protocol Parity Audit

- Updated: `2026-07-24T14:02:56Z`
- Status: `protocol_parity_partial_with_repair_nodes_encoded`
- Loop 2 can read this: `False`
- Can converge from this audit alone: `False`
- Report: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/gsm8k_protocol_parity_audit.json`

## Findings

- `prompt_template_parity`: `unknown`
- `suffix_answer_region_parity`: `matches_released_eval_formula_but_not_full_paper_protocol`
- `harness_parity`: `custom_fallback_support_only_until_equivalence_audit`
- `generated_answer_extractor_parity`: `custom_extractor_not_exact_simple_evals_extractor`

## DAG Coverage

- Covered repair nodes: `['protocol.prompt_template_parity_gate', 'runner.suffix_answer_region_parity_gate', 'protocol.simple_evals_vs_lmeval_harness_gate', 'scoring.generated_answer_extractor_parity_gate']`
