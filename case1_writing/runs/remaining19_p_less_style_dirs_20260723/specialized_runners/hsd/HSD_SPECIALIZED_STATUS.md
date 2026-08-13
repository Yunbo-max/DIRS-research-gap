# HSD Specialized Professional Gate

- Paper id: `ICLR2026_LaVrNaBNwM_hsd_lossless_speculative_decoding`
- Title: Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding
- Status: `blocked_by_model_data_hardware_transformers_patch_and_full_hsd_result_grid`
- Converged: `false`
- Professional ready: `false`
- DAG signature: `7bd8fcbb20b0f928`
- Repo: `/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/Hierarchical-Speculative-Decoding`

## Blockers

- `h20_96gb_or_h100_h200_hardware_missing`: Paper expects ['single NVIDIA H20 96GB for Qwen2.5 experiments', '8 H20 GPUs for Llama-3.1-70B extended evaluation', 'H100 80GB and H200 141GB for supplementary hardware comparisons']; visible GPUs are NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090.
- `llama70b_multigpu_h20_grid_missing`: No clean large-memory professional GPU slot is visible. GPU rows: [{'index': '0', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 99, 'temperature_c': 79, 'power_w': 179.99}, {'index': '1', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16149, 'memory_free_mib': 8068, 'utilization_gpu_pct': 99, 'temperature_c': 86, 'power_w': 384.59}, {'index': '2', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16397, 'memory_free_mib': 7816, 'utilization_gpu_pct': 99, 'temperature_c': 77, 'power_w': 182.2}, {'index': '3', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16169, 'memory_free_mib': 8048, 'utilization_gpu_pct': 99, 'temperature_c': 80, 'power_w': 156.04}].
- `hsd_runtime_dependencies_missing`: Missing runtime imports: auto_gptq, eagle
- `hsd_source_compile_or_patch_check_failed`: Repository compileall did not pass; see environment manifest.
- `hsd_required_artifacts_missing`: Missing required paper artifacts: gsm8k_tokenwise_result_jsonl, gsm8k_hsd_result_jsonl, gsm8k_multidraft_hsd_result_jsonl, humaneval_result_jsonl, cnndm_result_jsonl, table_metric_summary

## Artifact Paths

- Professional gate: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/hsd/professional_gate_result.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/hsd/hsd_specialized_verifier.json`
- Repo manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/hsd/repo_manifest.json`
- Environment: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/hsd/environment.json`
- Model/data manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/hsd/model_data_manifest.json`

## Verifier Checks

- `blind_contract`: `pass`
- `repo_path_encoded`: `pass`
- `reduced_proxy_rejection_gate`: `pass`
- `professional_artifact_package`: `blocked`
- `result_shape_comparison_ready`: `blocked`
