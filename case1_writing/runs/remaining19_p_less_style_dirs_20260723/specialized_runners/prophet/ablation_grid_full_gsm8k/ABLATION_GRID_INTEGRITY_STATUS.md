# Prophet Ablation Grid Integrity

- Updated: `2026-07-24T14:02:55Z`
- Status: `blocked_ablation_grid_integrity_failure`
- Configs checked: `17`
- Runnable config integrity blockers: `4`
- Manifest source-parity blockers: `2`
- Running configs: `0`
- Complete configs: `13`

## Manifest Source-Parity Blockers

- `table3b_remasking_top_k_margin` status=`blocked_by_missing_official_top_k_margin_remasking_implementation` role=`Table 3b remasking strategy compatibility`
- `dream7b_axis` status=`blocked_until_dream7b_exact_runner_and_memory_budget_are_resolved` role=`Dream-7B Table 1 axis`

## Configs

- `table3a_static_L256_T16` status=`pass_complete_jsonl_integrity` gpu=`0` pid=`59760` rows=`1319` complete=`1319/1319` reasons=`[]`
- `table3a_static_L256_T32` status=`pass_complete_jsonl_integrity` gpu=`1` pid=`59766` rows=`1319` complete=`1319/1319` reasons=`[]`
- `table3a_static_L256_T64` status=`pass_complete_jsonl_integrity` gpu=`2` pid=`59774` rows=`1319` complete=`1319/1319` reasons=`[]`
- `table3a_static_L256_T128` status=`pass_complete_jsonl_integrity` gpu=`0` pid=`85898` rows=`1319` complete=`1319/1319` reasons=`[]`
- `table3a_prophet_L256_T256` status=`pass_complete_jsonl_integrity` gpu=`1` pid=`88302` rows=`1319` complete=`1319/1319` reasons=`[]`
- `table3a_static_L128_T16` status=`pass_complete_jsonl_integrity` gpu=`2` pid=`155943` rows=`1319` complete=`1319/1319` reasons=`[]`
- `table3a_static_L128_T32` status=`pass_complete_jsonl_integrity` gpu=`2` pid=`173113` rows=`1319` complete=`1319/1319` reasons=`[]`
- `table3a_static_L128_T64` status=`pass_complete_jsonl_integrity` gpu=`3` pid=`182534` rows=`1319` complete=`1319/1319` reasons=`[]`
- `table3a_static_L128_T128` status=`pass_complete_jsonl_integrity` gpu=`1` pid=`234935` rows=`1319` complete=`1319/1319` reasons=`[]`
- `table3a_prophet_L128_T128` status=`pass_complete_jsonl_integrity` gpu=`0` pid=`267051` rows=`1319` complete=`1319/1319` reasons=`[]`
- `table3b_remasking_random` status=`pass_complete_jsonl_integrity` gpu=`3` pid=`267608` rows=`2638` complete=`1319/1319` reasons=`[]`
- `table3b_remasking_low_confidence` status=`pass_complete_jsonl_integrity` gpu=`1` pid=`272915` rows=`2638` complete=`1319/1319` reasons=`[]`
- `table4_block_length_8` status=`blocked_stopped_partial_artifact` gpu=`0` pid=`278940` rows=`2620` complete=`1310/1319` reasons=`['process_not_alive_before_full_config_completion']`
- `table4_block_length_16` status=`pass_complete_jsonl_integrity` gpu=`1` pid=`288071` rows=`2638` complete=`1319/1319` reasons=`[]`
- `table4_block_length_32` status=`blocked_stopped_partial_artifact` gpu=`3` pid=`296937` rows=`722` complete=`361/1319` reasons=`['process_not_alive_before_full_config_completion']`
- `table4_block_length_64` status=`blocked_stopped_partial_artifact` gpu=`2` pid=`333923` rows=`505` complete=`252/1319` reasons=`['process_not_alive_before_full_config_completion']`
- `table4_block_length_128` status=`blocked_stopped_partial_artifact` gpu=`1` pid=`347595` rows=`646` complete=`323/1319` reasons=`['process_not_alive_before_full_config_completion']`
