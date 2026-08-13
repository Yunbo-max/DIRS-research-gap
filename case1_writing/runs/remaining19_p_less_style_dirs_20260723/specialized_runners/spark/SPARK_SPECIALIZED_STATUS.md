# SPARK Specialized Professional Gate

- Paper id: `CVPR2026_030_spark_vlm_articulated_reconstruction`
- Title: SPARK: Sim-ready Part-level Articulated Reconstruction with VLM Knowledge
- Status: `blocked_by_placeholder_repo_unreleased_source_models_data_runtime_hardware_and_result_grid`
- Converged: `false`
- Professional ready: `false`
- DAG signature: `f68e4ca3f8faa6a5`
- Repo: `/tf/notebooks/cvpr2026_oral_paper_memory_141/repos/030_spark/SPARK`

## Blockers

- `official_repo_placeholder_no_executable_source`: Repository contains only ['LICENSE', 'README.md', 'assets/SPARK_teaser.png']; no executable source/config/script is released.
- `readme_declares_core_artifacts_unreleased`: README TODO marks these as unreleased: - [ ] Provide a HuggingFace🤗 demo.; - [ ] Release inference scripts and pretrained checkpoints.; - [ ] Release training code and data preprocessing scripts.; - [ ] Release preprocessed dataset.
- `four_h100_hardware_missing`: DAG expects 4 NVIDIA H100 GPUs; visible devices are NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090.
- `spark_3d_runtime_dependencies_missing`: Missing runtime imports for full 3D/simulation path: pytorch3d, trimesh, isaacsim, open3d, kaolin
- `spark_models_data_checkpoints_missing`: Missing released pretrained checkpoints, PartNet-Mobility/GAPartNet preprocessing, 100-image GAPartNet eval set, DINOv2/diffusion transformer configs, and VLM-generated part-reference artifacts.
- `spark_vlm_api_and_meshy_integration_missing`: No structural-reasoning VLM runner, part-reference image-generation runner, or Meshy texture synthesis integration/API artifacts are present.
- `spark_full_3d_result_grid_missing`: No verifier-comparable mesh/URDF/joint metric tables, ablations, Isaac Sim artifacts, raw outputs, or GPU traces are present.

## Artifact Paths

- Professional gate: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/spark/professional_gate_result.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/spark/spark_specialized_verifier.json`
- Repo manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/spark/repo_manifest.json`
- Environment: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/spark/environment.json`

## Verifier Checks

- `blind_contract`: `pass`
- `repo_path_encoded`: `pass`
- `placeholder_repo_rejection`: `pass`
- `professional_artifact_package`: `blocked`
- `result_shape_comparison_ready`: `blocked`
