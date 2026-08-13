# Prophet Table 2 Acceleration Campaign

- Updated: `2026-07-23T14:07:40Z`
- Policy: full GSM8K and exact external-method artifacts only; no reduced/proxy convergence.
- Linked existing artifacts: `2`
- Linked existing complete: `0`
- Runnable configs: `0`
- Explicit blockers: `4`
- Launch: `{'launched': False, 'reason': 'no_runnable_configs_without_sdtt_fastdllm_external_artifacts'}`

## GPU Inventory

- GPU `0` free=`3586` MiB used=`20978` MiB util=`10`%
- GPU `1` free=`5933` MiB used=`18631` MiB util=`1`%
- GPU `2` free=`5372` MiB used=`19192` MiB util=`5`%
- GPU `3` free=`8515` MiB used=`16049` MiB util=`99`%

## Linked Existing Artifacts

- `table2_llada_teacher_full_step` complete=`False` samples=`566` rows=`1132`
- `table2_prophet_ours` complete=`False` samples=`566` rows=`1132`

## Explicit Blockers

- `sdtt_distilled_student_row`: `blocked_by_missing_sdtt_training_code_or_distilled_checkpoint`
- `sdtt_plus_prophet_row`: `blocked_by_missing_sdtt_checkpoint_and_prophet_integration_path`
- `fast_dllm_kv_cache_parallel_row`: `blocked_by_missing_fast_dllm_code_patch_and_speed_harness`
- `fast_dllm_plus_prophet_row`: `blocked_by_missing_fast_dllm_prophet_combined_runner`
