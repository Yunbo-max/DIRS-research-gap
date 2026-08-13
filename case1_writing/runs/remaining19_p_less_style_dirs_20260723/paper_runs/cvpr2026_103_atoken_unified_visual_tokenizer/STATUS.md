# AToken Specialized Operational Gate

- Paper id: `CVPR2026_103_atoken_unified_visual_tokenizer`
- Title: `AToken: A Unified Tokenizer for Vision`
- Updated: `2026-07-23T18:53:13Z`
- Final status: `blocked_by_checkpoints_datasets_runtime_hardware_and_benchmark_grid_requirements`
- Converged: `false`
- Professional ready: `false`
- DAG signature: `ae3bfe8ac298f3f1`

## Blockers
- `h100_training_scale_hardware_missing`: DAG records 64 H100 Stage 1 and 256 H100 Stages 2-3; visible H100 count is 0; visible GPUs: NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090.
- `clean_gpu_slot_for_official_atoken_inference_missing`: No visible GPU has >=18GB free and <30% utilization for official ATokenWrapper inference. GPU rows: [{'index': '0', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 21343, 'memory_free_mib': 2873, 'utilization_gpu_pct': 0, 'temperature_c': 44, 'power_w': 61.89}, {'index': '1', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 18631, 'memory_free_mib': 5586, 'utilization_gpu_pct': 0, 'temperature_c': 39, 'power_w': 20.57}, {'index': '2', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 19023, 'memory_free_mib': 5191, 'utilization_gpu_pct': 17, 'temperature_c': 39, 'power_w': 20.85}, {'index': '3', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16109, 'memory_free_mib': 8108, 'utilization_gpu_pct': 99, 'temperature_c': 74, 'power_w': 247.02}]
- `apple_checkpoints_not_materialized`: Official Apple checkpoint URLs exist but local .pt files are missing or empty: atoken_soc_continuous, atoken_sod_discrete, atoken_3d_decode_gs, atoken_soc_stage1, atoken_soc_stage2.
- `multimodal_training_eval_datasets_missing`: Missing local dataset/materialized benchmark artifacts: dfn_training, open_images_training, webvid_video_training, textvr_retrieval, panda70m_video_training, objaverse_3d, cap3d_3d_text, imagenet_classification, coco_retrieval_generation, davis_video_reconstruction, msrvtt_video_text.
- `atoken_runtime_dependencies_missing`: Missing imports needed for full official/video/GS path: diffusers, open_clip, imageio, webdataset, flash_attn, diff_gaussian_rasterization.
- `full_multitask_benchmark_grid_missing`: No complete verifier-comparable AToken tables/figures/raw outputs were found for reconstruction, retrieval, MLLM, generation, 3D, and scaling surfaces.

## Artifacts
- Gate: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/atoken/professional_gate_result.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/atoken/atoken_specialized_verifier.json`
- Environment: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/atoken/environment.json`
- Script manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/atoken/official_script_manifest.json`
- Model/data manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/atoken/model_data_manifest.json`

Reduced, lightweight, syntax-only, URL-only, and demo evidence remains support only.
