# LPD Specialized Professional Gate

- Paper id: `ICLR2026_h06l9w1clt_locality_parallel_decoding_ar_image`
- Title: Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation
- Status: `blocked_by_models_imagenet_geneval_runtime_hardware_and_full_image_generation_grid`
- Converged: `false`
- Professional ready: `false`
- DAG signature: `c57987405f3af25c`
- Repo: `/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/lpd`

## Blockers

- `a100_bf16_latency_trace_missing`: Paper expects ['A100 bf16 profiling', '8-GPU training/cache path', 'ImageNet 256/512 plus GenEval scoring']; visible GPUs are NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090.
- `eight_gpu_training_or_cache_trace_missing`: No clean large-memory professional GPU slot is visible. GPU rows: [{'index': '0', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 99, 'temperature_c': 79, 'power_w': 180.73}, {'index': '1', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 99, 'temperature_c': 86, 'power_w': 388.55}, {'index': '2', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16397, 'memory_free_mib': 7816, 'utilization_gpu_pct': 99, 'temperature_c': 77, 'power_w': 181.71}, {'index': '3', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16109, 'memory_free_mib': 8108, 'utilization_gpu_pct': 99, 'temperature_c': 80, 'power_w': 162.56}].
- `lpd_runtime_dependencies_missing`: Missing runtime imports: torch_fidelity, flash_attn
- `lpd_required_artifacts_missing`: Missing required paper artifacts: vqgan_tokenizer, lpd_l_256_ckpt, lpd_xl_256_ckpt, lpd_xxl_256_ckpt, lpd_l_512_ckpt, lpd_xl_512_ckpt, generated_orders, imagenet_cache, eval_outputs, geneval_outputs
- `lpd_full_result_grid_missing`: No complete verifier-comparable output grid found for: class-conditional ImageNet 256/512 50k-sample FID/IS/precision/recall; text-to-image GenEval score; latency and throughput profiling on A100 bf16; LPD-L/XL/XXL and 20/32/48 step settings; locality-order, mutual-visibility, group-size, and XL 256 ablations

## Artifact Paths

- Professional gate: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/lpd/professional_gate_result.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/lpd/lpd_specialized_verifier.json`
- Repo manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/lpd/repo_manifest.json`
- Environment: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/lpd/environment.json`
- Model/data manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/lpd/model_data_manifest.json`

## Verifier Checks

- `blind_contract`: `pass`
- `repo_path_encoded`: `pass`
- `reduced_proxy_rejection_gate`: `pass`
- `professional_artifact_package`: `blocked`
- `result_shape_comparison_ready`: `blocked`
