# SeaCache Specialized Runner Status

- Updated: 2026-07-24T13:31:58Z
- Paper: `SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models`
- Status: `blocked_by_diffusion_model_data_hardware_runtime_and_script_parity_requirements`
- Professional package ready: `False`
- Repo files checked: `11`
- Compileall support check passed: `True`
- SEA filter unit support check passed: `True`
- Model manifests checked: `6`
- Dataset/metric artifacts checked: `4`
- Blocker count: `6`

## Artifact Paths
- Environment: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/seacache/environment.json`
- Official script manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/seacache/official_script_manifest.json`
- Model/data manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/seacache/model_data_manifest.json`
- Professional gate: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/seacache/professional_gate_result.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/seacache/seacache_specialized_verifier.json`

## Why This Is Not Converged
- This did not run a one-prompt FLUX image, syntax check, or SEA unit test as convergence evidence.
- The full SeaCache paper shape requires paper-scale checkpoints, prompt/evaluator suites, threshold sweeps, raw media, timing/TFLOPs traces, and metric tables.
- The DAG was updated so Loop 2 must satisfy those operational gates before the verifier can accept the research-gap simulation.

## Current Blockers
- `paper_hardware_class`: Paper/DAG expects Blackwell RTX PRO 6000 and/or A100 traces; visible GPUs are: NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090.
- `clean_gpu_slot_for_diffusion_grid`: Visible GPUs are already memory-heavy or active; GPU 3 is occupied by the Prophet full run. High-use rows: [{'index': '0', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16109, 'memory_free_mib': 8108, 'utilization_gpu_pct': 99}, {'index': '1', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 98}, {'index': '2', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16417, 'memory_free_mib': 7796, 'utilization_gpu_pct': 99}, {'index': '3', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 99}]
- `required_diffusion_checkpoints_not_materialized`: Missing local paper-scale model checkpoints: flux_dev, wan21_t2v_13b, wan21_t2v_14b, wan21_i2v_14b_720p, hunyuanvideo
- `prompt_metric_artifacts_not_materialized`: Missing local prompt/evaluator artifacts: drawbench_200_prompts, vbench_944_prompts_and_tooling, cyclereward_eval, compressedvqa_eval
- `diffusion_metric_runtime_missing`: Missing required runtime packages/imports: diffusers, calflops, vbench, lpips, wan, hyvideo
- `official_script_parity_for_full_grid`: SeaCache scripts need patch/wrapper before full grid: flux_prompt_grid_break, hunyuan_threshold_hardcoded
