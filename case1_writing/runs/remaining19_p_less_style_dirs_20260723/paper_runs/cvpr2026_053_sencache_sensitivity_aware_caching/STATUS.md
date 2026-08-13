# SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware Caching

- Paper id: `CVPR2026_053_sencache_sensitivity_aware_caching`
- Final status: `blocked_by_sensitivity_weights_model_data_runtime_hardware_and_result_grid_requirements_after_specialized_runner`
- Converged: `false`
- Semantic ready: `true`
- Professional ready: `false`
- DAG signature: `552080eadf9e917f`
- Specialized runner status: `blocked_by_sensitivity_weights_model_data_runtime_hardware_and_result_grid_requirements`
- Specialized status: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/sencache/SENCACHE_SPECIALIZED_STATUS.md`

## Checks

- `blind_contract`: `pass`
- `gap_semantic_match`: `pass`
- `method_gap_binding_match`: `pass`
- `reduced_proxy_rejection_gate`: `pass`
- `professional_artifact_package`: `blocked`
- `exact_artifact_debt_recorded`: `pass`

## Current Professional Blockers

- `paper_hardware_class`: Paper/DAG expects GH200 supplement latency or comparable high-end video-diffusion traces; visible GPUs are: NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090.
- `clean_gpu_slot_for_video_diffusion_grid`: Visible GPUs are memory-heavy or active; GPU 3 is occupied by Prophet. High-use rows: [{'index': '0', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16109, 'memory_free_mib': 8108, 'utilization_gpu_pct': 99}, {'index': '1', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 99}, {'index': '2', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16377, 'memory_free_mib': 7836, 'utilization_gpu_pct': 99}, {'index': '3', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 100}]
- `required_video_model_checkpoints_not_materialized`: Missing local loadable checkpoints: wan21_t2v_13b, cogvideox_15_5b, ltx_video_091
- `sensitivity_weights_or_calibration_outputs_missing`: Missing local sensitivity .npz weights or calibration outputs: sensitivity_wan21, sensitivity_cogvid, sensitivity_ltx
- `calibration_prompt_metric_artifacts_missing`: Missing local calibration/prompt/evaluator artifacts: mixkit_calibration_videos, vbench_full_prompt_set, t2v_compbench_70_prompts
- `video_diffusion_metric_runtime_missing`: Missing required runtime packages/imports: diffusers, lpips, vbench, wan
- `full_cached_uncached_result_grid_missing`: No raw cached/uncached Wan2.1, CogVideoX, and LTX outputs with VBench/LPIPS/PSNR/SSIM/latency/GFLOPs summaries were found.
