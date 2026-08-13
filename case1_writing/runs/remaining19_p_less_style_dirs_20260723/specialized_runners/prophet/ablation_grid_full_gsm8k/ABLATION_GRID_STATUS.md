# Prophet Full Ablation Grid Campaign

- Updated: `2026-07-24T14:02:54Z`
- Policy: full GSM8K split per runnable config; no reduced/proxy convergence.
- Runnable configs: `17`
- Explicit blockers: `2`

## GPU Inventory

- GPU `0` free=`8395` MiB used=`16169` MiB util=`99`%
- GPU `1` free=`8435` MiB used=`16129` MiB util=`98`%
- GPU `2` free=`7956` MiB used=`16608` MiB util=`99`%
- GPU `3` free=`8415` MiB used=`16149` MiB util=`100`%

## Config Statuses

- `table3a_static_L256_T16` status=`completed` rows=`1319` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table3a_static_L256_T16`
- `table3a_static_L256_T32` status=`completed` rows=`1319` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table3a_static_L256_T32`
- `table3a_static_L256_T64` status=`completed` rows=`1319` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table3a_static_L256_T64`
- `table3a_static_L256_T128` status=`completed` rows=`1319` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table3a_static_L256_T128`
- `table3a_prophet_L256_T256` status=`completed` rows=`1319` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table3a_prophet_L256_T256`
- `table3a_static_L128_T16` status=`completed` rows=`1319` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table3a_static_L128_T16`
- `table3a_static_L128_T32` status=`completed` rows=`1319` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table3a_static_L128_T32`
- `table3a_static_L128_T64` status=`completed` rows=`1319` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table3a_static_L128_T64`
- `table3a_static_L128_T128` status=`completed` rows=`1319` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table3a_static_L128_T128`
- `table3a_prophet_L128_T128` status=`completed` rows=`1319` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table3a_prophet_L128_T128`
- `table3b_remasking_random` status=`completed` rows=`2638` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table3b_remasking_random`
- `table3b_remasking_low_confidence` status=`completed` rows=`2638` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table3b_remasking_low_confidence`
- `table4_block_length_8` status=`stopped_partial_needs_resume` rows=`2620` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table4_block_length_8`
- `table4_block_length_16` status=`completed` rows=`2638` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table4_block_length_16`
- `table4_block_length_32` status=`stopped_partial_needs_resume` rows=`722` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table4_block_length_32`
- `table4_block_length_64` status=`stopped_partial_needs_resume` rows=`505` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table4_block_length_64`
- `table4_block_length_128` status=`stopped_partial_needs_resume` rows=`646` out=`/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/ablation_grid_full_gsm8k/table4_block_length_128`

## Explicit Blockers

- `table3b_remasking_top_k_margin`: `blocked_by_missing_official_top_k_margin_remasking_implementation`
- `dream7b_axis`: `blocked_until_dream7b_exact_runner_and_memory_budget_are_resolved`
