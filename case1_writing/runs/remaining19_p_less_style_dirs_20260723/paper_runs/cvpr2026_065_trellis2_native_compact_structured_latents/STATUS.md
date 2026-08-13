# TRELLIS.2 Specialized Operational Gate

- Paper id: `CVPR2026_065_trellis2_native_compact_structured_latents`
- Title: `Native and Compact Structured Latents for 3D Generation`
- Updated: `2026-07-23T18:53:02Z`
- Final status: `blocked_by_weights_datasets_runtime_hardware_and_result_grid_requirements`
- Converged: `false`
- Professional ready: `false`
- DAG signature: `708adab776bbd98e`
- Discovered repo encoded: `/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/065_trellis2/TRELLIS.2`

## Blockers
- `h100_runtime_scaling_hardware_missing`: Paper/DAG includes H100 runtime scaling; visible GPUs are: NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090.
- `official_verified_a100_h100_environment_missing`: Official README says code verified on A100/H100; visible GPUs are: NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090.
- `clean_24gb_gpu_slot_missing`: No visible 24GB+ GPU has >=18GB free and <30% utilization for TRELLIS.2 inference. GPU rows: [{'index': '0', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 21343, 'memory_free_mib': 2873, 'utilization_gpu_pct': 15, 'temperature_c': 44, 'power_w': 64.75}, {'index': '1', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 18631, 'memory_free_mib': 5586, 'utilization_gpu_pct': 0, 'temperature_c': 39, 'power_w': 20.92}, {'index': '2', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 19037, 'memory_free_mib': 5177, 'utilization_gpu_pct': 17, 'temperature_c': 39, 'power_w': 20.16}, {'index': '3', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16049, 'memory_free_mib': 8168, 'utilization_gpu_pct': 99, 'temperature_c': 74, 'power_w': 249.36}]
- `trellis2_4b_weights_not_materialized`: Missing local HF model snapshots/checkpoints: trellis2_4b_hf_weights.
- `trellis2_datasets_or_preprocessed_artifacts_missing`: Missing local datasets/preprocessed artifacts: objaverse_xl, abo, hssd, texverse, toys4k, sketchfab_featured, nanobanana_prompts, pbr_dumps, dual_grid, ovoxel_vxz, shape_latents, pbr_latents, render_cond.
- `trellis2_runtime_extensions_missing`: Missing official runtime/extension imports: flash_attn, nvdiffrast, nvdiffrec, cumesh, o_voxel, flexgemm, imageio, trimesh, gradio, lpips, kornia, easydict.
- `full_3d_result_grid_and_userstudy_missing`: No complete verifier-comparable 3D generation/reconstruction/texturing/scaling/user-study result grid was found.

## Artifacts
- Gate: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/trellis2/professional_gate_result.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/trellis2/trellis2_specialized_verifier.json`
- Environment: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/trellis2/environment.json`
- Script manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/trellis2/official_script_manifest.json`
- Model/data manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/trellis2/model_data_manifest.json`

Examples, README timing, syntax checks, and model-card metadata remain support only.
