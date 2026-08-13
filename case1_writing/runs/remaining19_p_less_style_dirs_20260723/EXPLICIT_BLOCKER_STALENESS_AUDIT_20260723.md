# Explicit Blocker Staleness Audit

- Updated: `2026-07-24T14:02:56Z`
- Purpose: verify explicit blockers remain evidence-bound while the active non-reduced GPU run continues.
- Policy: this audit cannot accept a paper and cannot treat reduced/proxy evidence as convergence.
- Counts: evidence_bound=`10`, after_gpu_recheck=`8`, gpu_recheck=`0`, needs_repair=`0`, running=`1`

## Current GPU State

- GPU `0` NVIDIA GeForce RTX 4090 free=`8395` MiB used=`16169` MiB util=`99`%
- GPU `1` NVIDIA GeForce RTX 4090 free=`8435` MiB used=`16129` MiB util=`98`%
- GPU `2` NVIDIA GeForce RTX 4090 free=`7954` MiB used=`16610` MiB util=`99`%
- GPU `3` NVIDIA GeForce RTX 4090 free=`8415` MiB used=`16149` MiB util=`99`%

## Paper Statuses

- `CVPR2026_016_nuwa_class_specific_vit_pruning`: `explicit_blocker_evidence_bound_after_gpu_recheck` updates=`1` verifier_updates=`1` op_blockers=`1` tags=`['active_gpu_capacity', 'checkpoint_or_model', 'dataset']` weak=`[]`
- `CVPR2026_030_spark_vlm_articulated_reconstruction`: `explicit_blocker_evidence_bound` updates=`1` verifier_updates=`1` op_blockers=`2` tags=`['api_or_access', 'checkpoint_or_model', 'dataset', 'exact_hardware_class', 'source_release']` weak=`[]`
- `CVPR2026_052_seacache_spectral_evolution_cache`: `explicit_blocker_evidence_bound_after_gpu_recheck` updates=`1` verifier_updates=`1` op_blockers=`6` tags=`['active_gpu_capacity', 'checkpoint_or_model', 'exact_hardware_class']` weak=`[]`
- `CVPR2026_053_sencache_sensitivity_aware_caching`: `explicit_blocker_evidence_bound_after_gpu_recheck` updates=`1` verifier_updates=`1` op_blockers=`7` tags=`['active_gpu_capacity', 'checkpoint_or_model', 'exact_hardware_class']` weak=`[]`
- `CVPR2026_065_trellis2_native_compact_structured_latents`: `explicit_blocker_evidence_bound` updates=`1` verifier_updates=`1` op_blockers=`2` tags=`['checkpoint_or_model', 'dataset', 'exact_hardware_class', 'software_runtime']` weak=`[]`
- `CVPR2026_067_rdvq_differentiable_vq_rate_distortion`: `explicit_blocker_evidence_bound_after_gpu_recheck` updates=`1` verifier_updates=`1` op_blockers=`5` tags=`['active_gpu_capacity', 'checkpoint_or_model', 'dataset', 'exact_hardware_class']` weak=`[]`
- `CVPR2026_103_atoken_unified_visual_tokenizer`: `explicit_blocker_evidence_bound` updates=`1` verifier_updates=`1` op_blockers=`1` tags=`['checkpoint_or_model', 'dataset', 'exact_hardware_class', 'software_runtime']` weak=`[]`
- `ICLR2026_1J63FJYJKg_mrrope_mixed_radix_rope`: `explicit_blocker_evidence_bound` updates=`1` verifier_updates=`1` op_blockers=`6` tags=`['checkpoint_or_model', 'dataset', 'software_runtime']` weak=`[]`
- `ICLR2026_88ZLp7xYxw_prism_fmri_structured_text`: `explicit_blocker_evidence_bound_after_gpu_recheck` updates=`1` verifier_updates=`1` op_blockers=`1` tags=`['active_gpu_capacity', 'api_or_access', 'checkpoint_or_model', 'dataset', 'exact_hardware_class']` weak=`[]`
- `ICLR2026_EQhUvWH78U_rational_information_seeking_agents`: `explicit_blocker_evidence_bound` updates=`1` verifier_updates=`1` op_blockers=`1` tags=`['api_or_access', 'dataset', 'software_runtime']` weak=`[]`
- `ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding`: `running_not_explicit_blocker` updates=`4` verifier_updates=`1` op_blockers=`6` tags=`['dataset']` weak=`[]`
- `ICLR2026_h06l9w1clt_locality_parallel_decoding_ar_image`: `explicit_blocker_evidence_bound_after_gpu_recheck` updates=`1` verifier_updates=`1` op_blockers=`1` tags=`['active_gpu_capacity', 'dataset', 'exact_hardware_class']` weak=`[]`
- `ICLR2026_H6rDX4w6Al_flashvid_vllm_token_merging`: `explicit_blocker_evidence_bound` updates=`5` verifier_updates=`5` op_blockers=`6` tags=`['dataset', 'exact_hardware_class', 'software_runtime']` weak=`[]`
- `ICLR2026_JEYWpFGzvn_infotok_adaptive_video_tokenizer`: `explicit_blocker_evidence_bound_after_gpu_recheck` updates=`1` verifier_updates=`1` op_blockers=`1` tags=`['active_gpu_capacity', 'checkpoint_or_model', 'dataset', 'exact_hardware_class']` weak=`[]`
- `ICLR2026_LaVrNaBNwM_hsd_lossless_speculative_decoding`: `explicit_blocker_evidence_bound_after_gpu_recheck` updates=`1` verifier_updates=`1` op_blockers=`1` tags=`['active_gpu_capacity', 'exact_hardware_class']` weak=`[]`
- `ICLR2026_o29E01Q6bv_loongrl_long_context_reasoning`: `explicit_blocker_evidence_bound` updates=`1` verifier_updates=`1` op_blockers=`7` tags=`['checkpoint_or_model', 'dataset', 'exact_hardware_class']` weak=`[]`
- `ICLR2026_P5B97gZwRb_hyperparameter_trajectory_inference_clot`: `explicit_blocker_evidence_bound` updates=`1` verifier_updates=`1` op_blockers=`1` tags=`['exact_hardware_class']` weak=`[]`
- `ICLR2026_QMItTyQW92_dto_kd_dynamic_tradeoff_distillation`: `explicit_blocker_evidence_bound` updates=`1` verifier_updates=`1` op_blockers=`1` tags=`['checkpoint_or_model', 'dataset', 'exact_hardware_class']` weak=`[]`
- `ICLR2026_VdLEaGPYWT_sparserl_sparse_cuda_rl`: `explicit_blocker_evidence_bound` updates=`1` verifier_updates=`1` op_blockers=`0` tags=`['checkpoint_or_model', 'dataset', 'exact_hardware_class']` weak=`[]`
