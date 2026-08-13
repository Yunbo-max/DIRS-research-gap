# SparseRL Specialized Runner Status

- Updated: 2026-07-23T18:59:25Z
- Paper: `Mastering Sparse CUDA Generation through Pretrained Models and Deep Reinforcement Learning`
- Status: `blocked_by_generated_policy_passatk_spmm_baselines_dataset_scale_and_v100_a100_grid`
- Convergence decision: `blocked_not_converged`
- CUDA rows: `12` correct / `18` attempted
- Physical GPU requested: `3`
- PyTorch mapping: CUDA_VISIBLE_DEVICES=3 makes the selected physical GPU appear as logical cuda:0 inside PyTorch; OOM messages may therefore say GPU 0 while referring to the selected visible device.
- Model policy attempted: `False`
- Model policy returncode: `None`
- Real CUDA convergence role: `support_only`

## Why Current GPU Evidence Cannot Converge
- deterministic CUDA rows use canonical hand-authored CSR/ELL/SELL kernels, not generated SparseRL policy samples
- the local matrix inventory contains six released sample Matrix Market files, not the paper-scale SuiteSparse and Deep Learning Matrix Collection grids
- the failed Qwen3-8B repo-minimal attempt is an OOM/runtime blocker, not pass@k policy evidence
- no SpMM executor, cuSPARSE/TVM-S/static-codegen baseline table, or V100/A100 paper-hardware trace has been produced

## Unresolved Professional Debt
- `generated_kernel_policy_grid`: unresolved - generated kernels from the pretrained/SFT/RL policy, not only hand-authored canonical kernels
- `pass_at_k_metrics`: unresolved - pass@1/pass@5/pass@1000 or equivalent sampling-grid measurements
- `paper_dataset_scale`: unresolved - SuiteSparse and Deep Learning Matrix Collection coverage, not only six released sample matrices
- `spmm_and_table_grid`: unresolved - SpMM executor and comparison rows in addition to SpMV
- `baseline_comparison`: unresolved - cuSPARSE/TVM-S/static LLM/codegen baselines with GFLOPS/TFLOPS/speedup
- `paper_hardware_match`: unresolved - paper-stated V100/A100 traces; current run records local RTX 4090 only

## Artifacts
- `environment.json`
- `matrix_inventory.json`
- `deterministic_cuda_kernel_eval.json`
- `model_policy_attempt.json`
- `sparserl_specialized_verifier.json`
