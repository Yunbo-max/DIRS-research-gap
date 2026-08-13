# InfoTok Specialized Professional Gate

- Paper id: `ICLR2026_JEYWpFGzvn_infotok_adaptive_video_tokenizer`
- Title: InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression
- Status: `blocked_by_hf_artifacts_runtime_hardware_training_scale_and_result_grid_requirements`
- Converged: `false`
- Professional ready: `false`
- DAG signature: `15f3793bea1afd04`
- Repo: `/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/InfoTok`

## Blockers

- `h100_or_a100_80gb_runtime_missing`: README recommends H100-80GB/A100-80GB; visible GPUs are NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090.
- `paper_scale_32_h100_training_missing`: DAG records 32 H100 GPUs for paper-scale training; visible H100 count is 0.
- `clean_large_gpu_slot_missing`: No clean >=30GB free GPU is visible for official long-window inference. GPU rows: [{'index': '0', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 99, 'temperature_c': 79, 'power_w': 180.66}, {'index': '1', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16149, 'memory_free_mib': 8068, 'utilization_gpu_pct': 99, 'temperature_c': 87, 'power_w': 379.17}, {'index': '2', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16397, 'memory_free_mib': 7816, 'utilization_gpu_pct': 99, 'temperature_c': 78, 'power_w': 182.64}, {'index': '3', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16109, 'memory_free_mib': 8108, 'utilization_gpu_pct': 100, 'temperature_c': 80, 'power_w': 162.59}].
- `infotok_runtime_dependencies_missing`: Missing imports for full inference/training/evaluation path: diffusers, hydra, imageio, mediapy, megatron, omegaconf, pynvml, skimage, termcolor, apex, transformer_engine, token_bench
- `infotok_hf_artifacts_not_materialized`: Public HF artifacts are reachable but not locally materialized: infotok_flex_checkpoint, tokenbench_240p, davis_240p
- `full_reconstruction_metric_grid_missing`: No complete TokenBench/DAVIS metric JSON, FVD/LPIPS, latency, decoder-pass, ablation, and baseline table outputs were found.

## Artifact Paths

- Professional gate: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/infotok/professional_gate_result.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/infotok/infotok_specialized_verifier.json`
- Script manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/infotok/official_script_manifest.json`
- Environment: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/infotok/environment.json`
- Model/data manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/infotok/model_data_manifest.json`

## Verifier Checks

- `blind_contract`: `pass`
- `repo_path_encoded`: `pass`
- `reduced_proxy_rejection_gate`: `pass`
- `professional_artifact_package`: `blocked`
- `result_shape_comparison_ready`: `blocked`
