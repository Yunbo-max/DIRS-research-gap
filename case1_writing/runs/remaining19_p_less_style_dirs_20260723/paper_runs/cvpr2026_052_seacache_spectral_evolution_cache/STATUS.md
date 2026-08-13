# SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models

- Paper id: `CVPR2026_052_seacache_spectral_evolution_cache`
- Final status: `blocked_by_diffusion_model_data_hardware_runtime_and_script_parity_requirements_after_specialized_runner`
- Converged: `false`
- Semantic ready: `true`
- Professional ready: `false`
- DAG signature: `bc787c39d4c4ed03`
- Specialized runner status: `blocked_by_diffusion_model_data_hardware_runtime_and_script_parity_requirements`
- Specialized status: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/seacache/SEACACHE_SPECIALIZED_STATUS.md`

## Checks

- `blind_contract`: `pass`
- `gap_semantic_match`: `pass`
- `method_gap_binding_match`: `pass`
- `reduced_proxy_rejection_gate`: `pass`
- `professional_artifact_package`: `blocked`
- `exact_artifact_debt_recorded`: `pass`

## Current Professional Blockers

- `paper_hardware_class`: Paper/DAG expects Blackwell RTX PRO 6000 and/or A100 traces; visible GPUs are: NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090.
- `clean_gpu_slot_for_diffusion_grid`: Visible GPUs are already memory-heavy or active; GPU 3 is occupied by the Prophet full run. High-use rows: [{'index': '0', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16109, 'memory_free_mib': 8108, 'utilization_gpu_pct': 99}, {'index': '1', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 98}, {'index': '2', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16417, 'memory_free_mib': 7796, 'utilization_gpu_pct': 99}, {'index': '3', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 99}]
- `required_diffusion_checkpoints_not_materialized`: Missing local paper-scale model checkpoints: flux_dev, wan21_t2v_13b, wan21_t2v_14b, wan21_i2v_14b_720p, hunyuanvideo
- `prompt_metric_artifacts_not_materialized`: Missing local prompt/evaluator artifacts: drawbench_200_prompts, vbench_944_prompts_and_tooling, cyclereward_eval, compressedvqa_eval
- `diffusion_metric_runtime_missing`: Missing required runtime packages/imports: diffusers, calflops, vbench, lpips, wan, hyvideo
- `official_script_parity_for_full_grid`: SeaCache scripts need patch/wrapper before full grid: flux_prompt_grid_break, hunyuan_threshold_hardcoded
