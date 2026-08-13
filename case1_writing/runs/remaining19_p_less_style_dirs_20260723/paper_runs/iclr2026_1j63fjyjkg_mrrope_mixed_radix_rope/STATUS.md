# MrRoPE: Mixed-radix Rotary Position Embedding

- Paper id: `ICLR2026_1J63FJYJKg_mrrope_mixed_radix_rope`
- Final status: `blocked_by_supplement_only_missing_datasets_models_runtime_after_specialized_runner`
- Converged: `false`
- Semantic ready: `true`
- Professional ready: `false`
- DAG signature: `b1f298ab51c78058`
- Specialized runner status: `blocked_by_supplement_only_missing_datasets_models_runtime`
- Specialized status: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/mrrope/MRROPE_SPECIALIZED_STATUS.md`

## Current Professional Blockers

- `source_distribution_supplement_only`: queue reports GitHub repository not found; only OpenReview supplement archive is available
- `local_long_context_datasets_missing`: required local dataset paths are missing: testset/pp-tokenized-llama3, testset/pp-tokenized-qwen2.5, testset/infity, testset/longbenchv2, testset/PaulGrahamEssays
- `local_model_checkpoints_missing`: supplement scripts expect missing local checkpoints: models/llama3.1-8b-ins, models/qwen2.5-3b-ins
- `flash_attention_2_missing`: supplement benchmark scripts use --flash-attention, but flash-attn is not installed
- `supplement_python_dependencies_missing`: missing supplement dependencies: jieba
- `no_idle_large_memory_gpu_for_128k_eval`: all visible GPUs have less than 16GiB free while another full GPU run is active; 128K model eval should wait for a clean device
