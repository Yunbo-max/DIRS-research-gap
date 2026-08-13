# SenCache Specialized Runner Status

- Updated: 2026-07-24T13:32:00Z
- Paper: `SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware Caching`
- Status: `blocked_by_sensitivity_weights_model_data_runtime_hardware_and_result_grid_requirements`
- Professional package ready: `False`
- Repo files checked: `13`
- Compileall support check passed: `True`
- Model manifests checked: `3`
- Sensitivity weight manifests checked: `3`
- Dataset/metric artifacts checked: `3`
- Blocker count: `7`

## Artifact Paths
- Environment: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/sencache/environment.json`
- Official script manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/sencache/official_script_manifest.json`
- Model/data manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/sencache/model_data_manifest.json`
- Professional gate: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/sencache/professional_gate_result.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/sencache/sencache_specialized_verifier.json`

## Why This Is Not Converged
- This did not run the bundled one-line prompt files, syntax checks, or HF metadata checks as convergence evidence.
- The full SenCache paper shape requires sensitivity weights or calibration reruns, paper-scale video model checkpoints, prompt/evaluator suites, cached and uncached outputs, timing/GFLOPs traces, and metric tables.
- The DAG was updated so Loop 2 must satisfy those operational gates before the verifier can accept the research-gap simulation.

## Current Blockers
- `paper_hardware_class`: Paper/DAG expects GH200 supplement latency or comparable high-end video-diffusion traces; visible GPUs are: NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090.
- `clean_gpu_slot_for_video_diffusion_grid`: Visible GPUs are memory-heavy or active; GPU 3 is occupied by Prophet. High-use rows: [{'index': '0', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16109, 'memory_free_mib': 8108, 'utilization_gpu_pct': 99}, {'index': '1', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 99}, {'index': '2', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16377, 'memory_free_mib': 7836, 'utilization_gpu_pct': 99}, {'index': '3', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 100}]
- `required_video_model_checkpoints_not_materialized`: Missing local loadable checkpoints: wan21_t2v_13b, cogvideox_15_5b, ltx_video_091
- `sensitivity_weights_or_calibration_outputs_missing`: Missing local sensitivity .npz weights or calibration outputs: sensitivity_wan21, sensitivity_cogvid, sensitivity_ltx
- `calibration_prompt_metric_artifacts_missing`: Missing local calibration/prompt/evaluator artifacts: mixkit_calibration_videos, vbench_full_prompt_set, t2v_compbench_70_prompts
- `video_diffusion_metric_runtime_missing`: Missing required runtime packages/imports: diffusers, lpips, vbench, wan
- `full_cached_uncached_result_grid_missing`: No raw cached/uncached Wan2.1, CogVideoX, and LTX outputs with VBench/LPIPS/PSNR/SSIM/latency/GFLOPs summaries were found.
