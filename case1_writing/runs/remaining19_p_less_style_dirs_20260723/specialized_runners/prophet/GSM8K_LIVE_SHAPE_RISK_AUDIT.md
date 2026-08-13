# GSM8K Live Shape Risk Audit

- Updated: `2026-07-24T14:02:56Z`
- Status: `postcompletion_shape_mismatch_requires_loop1_dag_repair`
- Samples: `1319/1319`
- Full split complete: `True`
- Comparison status: `blocked_paper_result_comparison`
- Primary GSM8K status: `blocked_gsm8k_result_shape_mismatch`
- Failing metrics: `['accuracy_delta', 'prophet_avg_steps', 'step_speedup']`
- Can converge from this audit alone: `False`
- Do not stop before full split: `False`
- Loop 2 can read this: `False`
- Report: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/gsm8k_live_shape_risk_audit.json`

## Possible DAG Repair Axes

- `prompt_template_parity`: The answer-emergence and exit point can shift if the simulation prompt is not the paper's exact evaluation prompt.
- `suffix_constraint_semantics`: Prophet monitors a final-answer region; a different suffix or constrained-token layout can delay or advance exits.
- `answer_region_start_and_length`: Step savings depend on the exact answer-region token span used by the early-commit rule.
- `simple_evals_vs_lm_eval_protocol`: The paper describes simple-evals-style scoring, while the release exposes an lm-eval integration and custom runner glue.
- `generated_answer_extractor_parity`: A scoring/extraction mismatch can change the direction of the accuracy delta without changing the underlying samples.
- `released_eval_harness_vs_custom_runner_semantics`: The custom full-split runner must be checked against the released eval path before a result-shape mismatch is attributed to the idea.
