# NuWa Specialized Professional Gate

- Paper id: `CVPR2026_016_nuwa_class_specific_vit_pruning`
- Title: NuWa: Deriving Lightweight Class-Specific Vision Transformers for Edge Devices
- Status: `blocked_by_syntax_datasets_checkpoints_runtime_hardware_and_result_grid_requirements`
- Converged: `false`
- Professional ready: `false`
- DAG signature: `115952e11feaf2ad`
- Repo: `/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/016_nuwa/NuWa`

## Blockers

- `clean_rtx4090_for_runtime_trace_missing`: NuWa needs RTX 4090 runtime traces; no clean 4090 has >=12GB free and <30% util. GPUs: [{'index': '0', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16109, 'memory_free_mib': 8108, 'utilization_gpu_pct': 99, 'temperature_c': 79, 'power_w': 180.08}, {'index': '1', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 99, 'temperature_c': 87, 'power_w': 388.7}, {'index': '2', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16417, 'memory_free_mib': 7796, 'utilization_gpu_pct': 99, 'temperature_c': 78, 'power_w': 183.68}, {'index': '3', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 99, 'temperature_c': 80, 'power_w': 164.32}].
- `jetson_orin_nx_device_missing`: Paper/DAG expects Jetson Orin NX edge profiling; visible devices are NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090.
- `official_repo_python_syntax_errors`: Official NuWa repo has syntax errors before execution: method/get_anchor_model.py, method/get_clibration_data.py, method/utils.py
- `nuwa_runtime_dependencies_missing`: Missing imports needed for full recognition/detection/segmentation/runtime path: mmcv, mmdet, mmengine, fvcore, ptflops
- `nuwa_checkpoints_or_configs_missing`: Missing local model/checkpoint/config artifacts: timm_deit_pretrained_weights, vit_large_pretrained_weights, cifar_finetuned_vit_weights, mask_rcnn_swin_coco_checkpoint, mmdetection_swin_config
- `full_pruning_benchmark_grid_missing`: No complete verifier-comparable NuWa tables/figures/raw outputs were found for accuracy/pruning/compute/runtime/cost surfaces.

## Artifact Paths

- Professional gate: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/nuwa/professional_gate_result.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/nuwa/nuwa_specialized_verifier.json`
- Environment: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/nuwa/environment.json`
- Script manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/nuwa/official_script_manifest.json`
- Model/data manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/nuwa/model_data_manifest.json`

## Verifier Checks

- `blind_contract`: `pass`
- `repo_path_encoded`: `pass`
- `reduced_proxy_rejection_gate`: `pass`
- `professional_artifact_package`: `blocked`
- `result_shape_comparison_ready`: `blocked`
