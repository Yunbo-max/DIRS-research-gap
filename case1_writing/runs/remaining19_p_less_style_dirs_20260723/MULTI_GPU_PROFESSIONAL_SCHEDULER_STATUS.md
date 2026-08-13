# Multi-GPU Professional Scheduler

- Updated: `2026-07-24T14:02:55Z`
- Status: `all_pool_gpus_claimed_by_active_one_paper_nodes`
- Paper: `Diffusion Language Models Know the Answer Before Decoding`
- Policy: one paper first; full-paper nodes only; no reduced/proxy convergence.
- GPU pool: `0,1,2,3`
- Launches this tick: `0`
- Cumulative launch history: `27`

## Active GPU Claims

- GPU `2` pid=`2714` config=`table1_threshold_relaxed_4_2_0p5` status=`running`
- GPU `0` pid=`2355` config=`table1_threshold_relaxed_6_4_2` status=`running`
- GPU `3` pid=`2903` config=`table1_threshold_relaxed_3_1p5_0p5` status=`running`
- GPU `1` pid=`2465` config=`table1_threshold_relaxed_5_3_1` status=`running`

## Launch Attempts


## Recent Launch History

- `prophet_ablation_grid_full_gsm8k` config=`table3b_remasking_random` gpu=`3` pid=`267608` source=`scheduler_launch_result`
- `prophet_ablation_grid_full_gsm8k` config=`table3b_remasking_low_confidence` gpu=`1` pid=`272915` source=`scheduler_launch_result`
- `prophet_ablation_grid_full_gsm8k` config=`table4_block_length_8` gpu=`0` pid=`278940` source=`scheduler_launch_result`
- `prophet_ablation_grid_full_gsm8k` config=`table4_block_length_16` gpu=`1` pid=`288071` source=`scheduler_launch_result`
- `prophet_ablation_grid_full_gsm8k` config=`table4_block_length_32` gpu=`3` pid=`296937` source=`scheduler_launch_result`
- `prophet_gsm8k_protocol_repair_full` config=`gsm8k_trajectory_prompt_constraint_L256_T256` gpu=`2` pid=`333224` source=`scheduler_launch_result`
- `prophet_ablation_grid_full_gsm8k` config=`table4_block_length_64` gpu=`2` pid=`333923` source=`scheduler_launch_result`
- `prophet_ablation_grid_full_gsm8k` config=`table4_block_length_128` gpu=`1` pid=`347595` source=`scheduler_launch_result`
- `prophet_table1_threshold_repair_full_gsm8k` config=`table1_threshold_relaxed_4_2_0p5` gpu=`2` pid=`2714` source=`reconstructed_from_active_status`
- `prophet_table1_threshold_repair_full_gsm8k` config=`table1_threshold_relaxed_6_4_2` gpu=`0` pid=`2355` source=`reconstructed_from_active_status`
- `prophet_table1_threshold_repair_full_gsm8k` config=`table1_threshold_relaxed_3_1p5_0p5` gpu=`3` pid=`2903` source=`reconstructed_from_active_status`
- `prophet_table1_threshold_repair_full_gsm8k` config=`table1_threshold_relaxed_5_3_1` gpu=`1` pid=`2465` source=`reconstructed_from_active_status`

## Skipped GPUs

- GPU `0` reason=`gpu_already_claimed_by_active_prophet_node`
- GPU `1` reason=`gpu_already_claimed_by_active_prophet_node`
- GPU `2` reason=`gpu_already_claimed_by_active_prophet_node`
- GPU `3` reason=`gpu_already_claimed_by_active_prophet_node`
