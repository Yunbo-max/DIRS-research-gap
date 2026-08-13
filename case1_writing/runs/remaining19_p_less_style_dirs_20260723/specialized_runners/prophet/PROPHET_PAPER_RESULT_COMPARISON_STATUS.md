# Prophet Paper Result Comparison Status

- Updated: `2026-07-24T14:02:56Z`
- Status: `blocked_paper_result_comparison`
- Primary GSM8K status: `blocked_gsm8k_result_shape_mismatch`
- Samples: `1319/1319`
- Observed baseline accuracy: `79.68157695223654`
- Observed Prophet accuracy: `76.95223654283548`
- Observed Prophet avg steps: `207.53297952994694`
- Observed step speedup: `1.233538884180378`
- Trajectory status: `pass_close_to_paper_trajectory_shape`
- Remaining blockers: `['blocked_gsm8k_result_shape_mismatch', 'table1_threshold_dynamics_repair:running', 'table1_multibenchmark_grid:pending_or_blocked', 'table2_acceleration_combinations:explicit_external_artifact_blockers_recorded', 'table3_table4_ablation_grid:pending_or_blocked', 'dream7b_axis:explicit_external_artifact_blockers_recorded']`

## Explicit Debt

- `gsm8k_protocol_repair`: `ready_for_result_value_comparison` manifest=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/protocol_repair_full_gsm8k/protocol_repair_campaign.json`
- `table1_threshold_dynamics_repair`: `running` manifest=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/table1_threshold_repair_full_gsm8k/table1_threshold_repair_campaign.json`
- `table1_multibenchmark_grid`: `pending_or_blocked` manifest=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/multibenchmark_table1_full/multibenchmark_grid_campaign.json`
- `table2_acceleration_combinations`: `explicit_external_artifact_blockers_recorded` manifest=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/table2_acceleration_combinations/table2_acceleration_campaign.json`
- `table3_table4_ablation_grid`: `pending_or_blocked` manifest=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/ablation_grid_campaign.json`
- `dream7b_axis`: `explicit_external_artifact_blockers_recorded` manifest=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/dream7b_table1_axis/dream7b_axis_campaign.json`

This is a verifier-only paper-target comparison. It is not visible to the DAG-only author simulation.
