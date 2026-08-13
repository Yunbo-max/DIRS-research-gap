# RDVQ Specialized Runner Status

- Updated: 2026-07-24T13:32:06Z
- Paper: `Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression`
- Status: `blocked_by_checkpoints_datasets_runtime_hardware_release_and_result_grid_requirements`
- Professional package ready: `False`
- Repo files checked: `19`
- Compileall support check passed: `True`
- Release validation support check passed: `False`
- Checkpoint manifests checked: `1`
- Dataset/metric artifacts checked: `7`
- Blocker count: `8`

## Artifact Paths
- Environment: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/rdvq/environment.json`
- Official script manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/rdvq/official_script_manifest.json`
- Model/data manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/rdvq/model_data_manifest.json`
- Professional gate: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/rdvq/professional_gate_result.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/rdvq/rdvq_specialized_verifier.json`

## Why This Is Not Converged
- This did not run a one-image debug probe, README asset comparison, or release-validation smoke as convergence evidence.
- The full RDVQ paper shape requires released checkpoints, Kodak/DIV2K/CLIC image folders, FID/KID references, estimated and real-bitstream outputs, prefix-transfer sweeps, timing/profile traces, and metric tables.
- The DAG was updated so Loop 2 must satisfy those operational gates before the verifier can accept the research-gap simulation.

## Current Blockers
- `highres_finetune_hardware_class`: Paper/DAG expects a single RTX Pro 6000 for high-resolution DF2K fine-tuning; visible GPUs are: NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090.
- `clean_gpu_slot_for_full_rd_grid`: No visible RTX 4090 has at least 12GB free and low utilization. GPU rows: [{'index': '0', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 17}, {'index': '1', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 99}, {'index': '2', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16377, 'memory_free_mib': 7836, 'utilization_gpu_pct': 99}, {'index': '3', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 99}]
- `rdvq_checkpoints_not_materialized`: Missing local released RDVQ checkpoint files: rdvq_testing_checkpoints
- `rd_curve_datasets_or_fid_refs_missing`: Missing local evaluation datasets/FID refs: kodak_image_folder, div2k_val_image_folder, clic2020_test_image_folder, fid_reference_tiles
- `training_and_highres_finetune_datasets_missing`: Missing training/fine-tuning datasets for full author reproduction: imagenet_training_images, openimage_training_images, df2k_highres_images
- `compression_metric_runtime_missing`: Missing required metric/codec runtime imports: cleanfid, pyiqa, pytorch_msssim, torchmetrics, compressai, ninja
- `release_validation_support_check_failed`: scripts/release_validate.sh did not complete cleanly; inspect environment.json for stdout/stderr.
- `full_estimated_and_real_bitstream_result_grid_missing`: No complete forward/Real metrics artifacts for Kodak, DIV2K, CLIC, prefix-transfer sweeps, and RD tables were found.
