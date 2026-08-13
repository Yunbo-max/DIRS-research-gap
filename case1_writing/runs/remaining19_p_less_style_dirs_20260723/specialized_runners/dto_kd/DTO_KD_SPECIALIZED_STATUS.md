# DTO-KD External Professional Gate

- Paper id: `ICLR2026_QMItTyQW92_dto_kd_dynamic_tradeoff_distillation`
- Title: DTO-KD: Dynamic Trade-off Optimization for Effective Knowledge Distillation
- Status: `blocked_by_no_official_source_repo_h100_data_checkpoints_and_full_kd_training_grid`
- Converged: `false`
- Professional ready: `false`
- GPU used: `false`; local RTX 4090 proxy is rejected because the paper requires four H100 plus source/data/checkpoints.
- DAG signature: `da619c1b5cec4ccc`

## Blockers

- `dto_kd_official_source_repository_missing`: OpenReview metadata, paper, supplement, status file, and local candidate repo scan do not provide an official DTO-KD source repo.
- `dto_kd_four_h100_hardware_missing`: Paper reports all experiments on four NVIDIA H100 GPUs; visible GPUs are NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090 | NVIDIA GeForce RTX 4090; H100 count=0.
- `dto_kd_required_datasets_missing`: Missing or unmaterialized dataset candidates: imagenet_1k, cifar100, coco2017
- `dto_kd_teacher_student_checkpoints_missing`: Missing teacher/student checkpoints or initialization artifacts: regnety_160_teacher, deit_tiny_student, deit_small_student, vidt_base_teacher, vidt_nano_tiny_small_students
- `dto_kd_full_training_grid_missing`: No ImageNet 300-epoch, CIFAR-100 six-pair, COCO 50-epoch, ablation, teacher-scale, or Figure 3/4 raw outputs exist under the runner result root.
- `dto_kd_result_shape_verifier_waiting_for_tables_figures`: Verifier needs outputs comparable to Table 1, Table 2, Table 3, Table 4, Table 5, Figure 3, and Figure 4 before accepting close result shape.
- `dto_kd_reimplementation_required_but_unvalidated`: Without official code, a faithful implementation from Algorithm 1 and Equations 8-18 must be created and validated before any experiment can count.

## Artifact Paths

- Professional gate: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/dto_kd/professional_gate_result.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/dto_kd/dto_kd_specialized_verifier.json`
- Source manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/dto_kd/source_manifest.json`
- Model/data manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/dto_kd/model_data_manifest.json`

## Verifier Checks

- `blind_contract`: `pass`
- `official_repo_source`: `blocked`
- `gpu_requirement_interpretation`: `blocked`
- `professional_artifact_package`: `blocked`
- `result_shape_comparison_ready`: `blocked`
