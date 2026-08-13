# LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts

- Paper id: `ICLR2026_o29E01Q6bv_loongrl_long_context_reasoning`
- Final status: `blocked_by_cluster_training_data_checkpoint_and_runtime_requirements_after_specialized_runner`
- Converged: `false`
- Semantic ready: `true`
- Professional ready: `false`
- DAG signature: `cd6905dea856e029`
- Specialized runner status: `blocked_by_cluster_training_data_checkpoint_and_runtime_requirements`
- Specialized status: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/loongrl/LOONGRL_SPECIALIZED_STATUS.md`

## Checks

- `blind_contract`: `pass`
- `gap_semantic_match`: `pass`
- `method_gap_binding_match`: `pass`
- `experiment_axis_match`: `pass`
- `reduced_proxy_rejection_gate`: `pass`
- `professional_artifact_package`: `blocked`
- `exact_artifact_debt_recorded`: `pass`

## Current Professional Blockers

- `loongrl_7b_training_gpu_topology`: DAG/paper-shaped 7B training expects 16 A100 GPUs; visible A100 count=0, total visible GPUs=4
- `loongrl_14b_training_gpu_topology`: DAG/paper-shaped 14B training expects 8 MI300X GPUs; visible MI300X count=0
- `official_longcontext_script_gpu_count`: official long-context GRPO scripts request trainer.n_gpus_per_node=8; visible GPUs=4
- `official_longcontext_training_data_paths_missing`: official GRPO scripts reference missing local parquet train files: /mnt/longcontext/models/siyuan/rl_datasets/rl_three/no_system/merged_data_deepscaler_openr1_130k_5000/train.parquet; /mnt/longcontext/models/siyuan/rl_datasets/rl_three/no_system/musique5000_seq8192/train.parquet; /mnt/longcontext/models/siyuan/rl_datasets/rl_three/no_system/hotpotqa5000_seq8192/train.parquet
- `trained_loongrl_checkpoints_missing`: no local LoongRL-7B or LoongRL-14B checkpoint directory was found for full benchmark evaluation
- `cluster_rl_runtime_packages_missing`: missing installed runtime packages needed by paper-shaped veRL/Ray/vLLM/SGLang flow: ray, vllm, sglang, flash_attn, verl
- `rocm_mi300x_runtime_unavailable`: MI300X/ROCm runtime path is not visible from rocm-smi, so the 14B MI300X paper route cannot run here
