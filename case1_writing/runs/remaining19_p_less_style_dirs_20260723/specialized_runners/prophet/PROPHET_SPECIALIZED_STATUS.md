# Prophet Specialized Runner Status

- Updated: `2026-07-24T14:02:56Z`
- Paper: `Diffusion Language Models Know the Answer Before Decoding`
- Status: `blocked_by_table1_table2_ablation_and_dream_axis_debt_after_full_gsm8k_trajectory`
- Full GSM8K paired samples: `1319` / `1319`
- JSONL rows: `2638`
- GPU: `3`
- Runner PID: `17791`
- JSONL integrity: `pass_complete_jsonl_integrity` gate=`pass` rows=`2638`
- Integrity report: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/prophet_live_integrity_report.json`
- Paper comparison: `blocked_paper_result_comparison` gate=`blocked`
- Paper comparison report: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/prophet_paper_result_comparison.json`
- Trajectory settings: `8` / `8` status=`completed`
- Trajectory rows: `17888`
- Ablation grid: `ready_waiting_for_gpu_capacity` runnable=`17` completed=`13`
- Ablation manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/ablation_grid_campaign.json`
- Ablation integrity: `blocked_ablation_grid_integrity_failure` running=`0` complete=`13` integrity_blocked=`4` manifest_blocked=`2`
- Ablation integrity report: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/ablation_grid_integrity_report.json`
- Table 1 threshold repair: `running_full_table1_threshold_repair_candidates` baseline_ready=`True` runnable=`4` completed=`0` running=`4`
- Table 1 threshold manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/table1_threshold_repair_full_gsm8k/table1_threshold_repair_campaign.json`
- Multi-benchmark grid: `ready_waiting_for_gpu_capacity_and_prompt_scorer_parity_resolution` runnable=`30` completed=`0`
- Multi-benchmark manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/multibenchmark_table1_full/multibenchmark_grid_campaign.json`
- Table 2 acceleration campaign: `running_waiting_for_full_gsm8k_linked_rows_and_external_artifacts` linked_complete=`0` blockers=`4`
- Table 2 manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/table2_acceleration_combinations/table2_acceleration_campaign.json`
- Dream-7B axis: `explicit_dream7b_axis_blockers_recorded` blockers=`4`
- Dream-7B manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/dream7b_table1_axis/dream7b_axis_campaign.json`
- Source parity audit: `evidence_bound_source_parity_blockers_ready` blockers=`4`
- Source parity report: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/source_parity_blocker_audit.json`
- GSM8K live shape risk audit: `postcompletion_shape_mismatch_requires_loop1_dag_repair` failing_metrics=`['accuracy_delta', 'prophet_avg_steps', 'step_speedup']` loop2_visible=`False`
- GSM8K risk audit report: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/gsm8k_live_shape_risk_audit.json`
- GSM8K protocol parity audit: `protocol_parity_partial_with_repair_nodes_encoded` findings=`{'prompt_template_parity': 'unknown', 'suffix_answer_region_parity': 'matches_released_eval_formula_but_not_full_paper_protocol', 'harness_parity': 'custom_fallback_support_only_until_equivalence_audit', 'generated_answer_extractor_parity': 'custom_extractor_not_exact_simple_evals_extractor'}` loop2_visible=`False`
- GSM8K protocol audit report: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/gsm8k_protocol_parity_audit.json`
- GSM8K Table 1 protocol-selection audit: `completed_table1_candidate_selection_repaired_primary_still_mismatch` primary_after_repair=`blocked_gsm8k_result_shape_mismatch` failed_metrics=`['accuracy_delta', 'prophet_avg_steps', 'step_speedup']` loop2_visible=`False`
- GSM8K Table 1 protocol-selection audit report: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/gsm8k_table1_protocol_selection_audit.json`
- Mean step reduction: `48.46702047005306`
- Mean seconds speedup: `1.2277952681409887`
- Convergence: `pending_full_artifacts_and_verifier_comparison`

This is a full-split GPU run, not reduced convergence evidence. It remains active until the verifier can compare final artifact shape to the paper evidence channels.
