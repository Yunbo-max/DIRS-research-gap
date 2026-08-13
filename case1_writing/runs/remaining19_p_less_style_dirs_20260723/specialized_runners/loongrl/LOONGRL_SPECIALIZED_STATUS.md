# LoongRL Specialized Runner Status

- Updated: 2026-07-23T18:53:37Z
- Paper: `LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts`
- Status: `blocked_by_cluster_training_data_checkpoint_and_runtime_requirements`
- Professional package ready: `False`
- Official scripts parsed: `11`
- Compileall support check passed: `True`
- HF model manifests checked: `10`
- HF dataset manifests checked: `1`
- Blocker count: `7`

## Artifact Paths
- Environment: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/loongrl/environment.json`
- Official script manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/loongrl/official_script_manifest.json`
- Model/data manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/loongrl/model_data_manifest.json`
- Professional gate: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/loongrl/professional_gate_result.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/loongrl/loongrl_specialized_verifier.json`

## Why This Is Not Converged
- This did not run a tiny GRPO job, one benchmark question, or repo import as convergence evidence.
- The full LoongRL paper shape requires cluster GRPO/evaluation artifacts, trained checkpoints, benchmark raw outputs, and compute traces.
- The DAG was updated so Loop 2 must satisfy those operational gates before the verifier can accept the research-gap simulation.

## Current Blockers
- `loongrl_7b_training_gpu_topology`: DAG/paper-shaped 7B training expects 16 A100 GPUs; visible A100 count=0, total visible GPUs=4
- `loongrl_14b_training_gpu_topology`: DAG/paper-shaped 14B training expects 8 MI300X GPUs; visible MI300X count=0
- `official_longcontext_script_gpu_count`: official long-context GRPO scripts request trainer.n_gpus_per_node=8; visible GPUs=4
- `official_longcontext_training_data_paths_missing`: official GRPO scripts reference missing local parquet train files: /mnt/longcontext/models/siyuan/rl_datasets/rl_three/no_system/merged_data_deepscaler_openr1_130k_5000/train.parquet; /mnt/longcontext/models/siyuan/rl_datasets/rl_three/no_system/musique5000_seq8192/train.parquet; /mnt/longcontext/models/siyuan/rl_datasets/rl_three/no_system/hotpotqa5000_seq8192/train.parquet
- `trained_loongrl_checkpoints_missing`: no local LoongRL-7B or LoongRL-14B checkpoint directory was found for full benchmark evaluation
- `cluster_rl_runtime_packages_missing`: missing installed runtime packages needed by paper-shaped veRL/Ray/vLLM/SGLang flow: ray, vllm, sglang, flash_attn, verl
- `rocm_mi300x_runtime_unavailable`: MI300X/ROCm runtime path is not visible from rocm-smi, so the 14B MI300X paper route cannot run here
