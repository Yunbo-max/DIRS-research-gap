# PRISM Specialized Professional Gate

- Paper id: `ICLR2026_88ZLp7xYxw_prism_fmri_structured_text`
- Title: Seeing Through the Brain: New Insights from Decoding Visual Stimuli with fMRI
- Status: `blocked_by_fmri_data_api_runtime_hardware_pipeline_artifacts_and_metric_grid`
- Converged: `false`
- Professional ready: `false`
- DAG signature: `4660083e62637f99`
- Repo: `/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/PRISM`

## Blockers

- `two_l40_48gb_hardware_missing`: README/DAG expect two NVIDIA L40 48GB GPUs; visible devices are NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090.
- `clean_large_fmri_generation_gpu_missing`: No clean >=45GB GPU is available for the paper-shaped PRISM reconstruction run. GPU rows: [{'index': '0', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16189, 'memory_free_mib': 8028, 'utilization_gpu_pct': 100, 'temperature_c': 79, 'power_w': 177.14}, {'index': '1', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 98, 'temperature_c': 86, 'power_w': 320.22}, {'index': '2', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16397, 'memory_free_mib': 7816, 'utilization_gpu_pct': 99, 'temperature_c': 77, 'power_w': 180.08}, {'index': '3', 'name': 'NVIDIA GeForce RTX 4090', 'memory_total_mib': 24564, 'memory_used_mib': 16129, 'memory_free_mib': 8088, 'utilization_gpu_pct': 99, 'temperature_c': 80, 'power_w': 157.15}].
- `prism_runtime_dependencies_missing`: Missing imports for the released PRISM pipeline: diffusers, xformers, langchain_openai, lpips, retry
- `prism_pipeline_artifacts_missing`: Missing PRISM pipeline artifacts: best_keyword_json, train_structured_npz, val_structured_npz, test_structured_npz, trained_checkpoint_cur_best, generated_output_dir, metric_output_dir
- `nsd_access_gated_fmri_data_missing`: No materialized NSD fMRI data for PRISM was found; README requires NSD terms/form access.
- `bold5000_benchmark_artifacts_missing`: No BOLD5000 benchmark artifacts found for PRISM verifier comparison.
- `generic_object_decoding_artifacts_missing`: No Generic Object Decoding benchmark artifacts found for PRISM verifier comparison.
- `coco_qa_artifacts_missing`: No PRISM-bound COCO image/caption/QA artifacts found for verifier comparison.
- `full_fmri_reconstruction_qa_metric_grid_missing`: No complete PixCorr/SSIM/LPIPS/CLIP/Inception/QA/CKA/CCA/generalization-gap result grid was found.

## Artifact Paths

- Professional gate: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prism/professional_gate_result.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prism/prism_specialized_verifier.json`
- Repo manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prism/repo_manifest.json`
- Environment: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prism/environment.json`
- Model/data manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prism/model_data_manifest.json`

## Verifier Checks

- `blind_contract`: `pass`
- `repo_path_encoded`: `pass`
- `reduced_proxy_rejection_gate`: `pass`
- `professional_artifact_package`: `blocked`
- `result_shape_comparison_ready`: `blocked`
