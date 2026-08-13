# GSM8K Table 1 Protocol Selection Audit

- Updated: `2026-07-24T10:26:51Z`
- Status: `completed_table1_candidate_selection_repaired_primary_still_mismatch`
- Loop2 visibility: `false`
- Oracle target values included: `false`

## Findings
- Table 1 GSM8K protocol is the released lm-eval `gsm8k_cot_zeroshot` path with `200:The|201:answer|202:is`, not the trajectory prompt.
- The trajectory prompt / `220:Answer` repair remains useful support evidence for suffix analysis, but it must not be selected as the primary Table 1 candidate.
- The comparator now routes Table 1 primary comparison to the Table1-compatible custom full run.
- The primary result still fails on `accuracy_delta`, `prophet_avg_steps`, and `step_speedup`, so convergence remains blocked.

## Artifacts
- JSON: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/gsm8k_table1_protocol_selection_audit.json`
- Comparator: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/prophet_paper_result_comparator.py`
- Comparison report: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/prophet_paper_result_comparison.json`
