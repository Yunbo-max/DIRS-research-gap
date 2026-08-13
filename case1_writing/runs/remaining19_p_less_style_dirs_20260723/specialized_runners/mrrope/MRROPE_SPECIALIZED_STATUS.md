# MrRoPE Specialized Runner Status

- Updated: 2026-07-23T18:53:20Z
- Paper: `MrRoPE: Mixed-radix Rotary Position Embedding`
- Status: `blocked_by_supplement_only_missing_datasets_models_runtime`
- Professional package ready: `False`
- Supplement scripts parsed: `16`
- Blocker count: `6`

## Artifact Paths
- Environment: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/mrrope/environment.json`
- Script manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/mrrope/official_script_manifest.json`
- Model/data manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/mrrope/model_data_manifest.json`
- Professional gate: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/mrrope/professional_gate_result.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/mrrope/mrrope_specialized_verifier.json`

## Why This Is Not Converged
- This did not run one short-context probe or import check as convergence evidence.
- The full paper shape requires 128K Proofpile/NIAH/RULER/Infinite-Bench/LongBenchV2 artifacts and model/runtime traces.
- The DAG was updated so Loop 2 must satisfy these exact gates before verifier acceptance.

## Current Blockers
- `source_distribution_supplement_only`: queue reports GitHub repository not found; only OpenReview supplement archive is available
- `local_long_context_datasets_missing`: required local dataset paths are missing: testset/pp-tokenized-llama3, testset/pp-tokenized-qwen2.5, testset/infity, testset/longbenchv2, testset/PaulGrahamEssays
- `local_model_checkpoints_missing`: supplement scripts expect missing local checkpoints: models/llama3.1-8b-ins, models/qwen2.5-3b-ins
- `flash_attention_2_missing`: supplement benchmark scripts use --flash-attention, but flash-attn is not installed
- `supplement_python_dependencies_missing`: missing supplement dependencies: jieba
- `no_idle_large_memory_gpu_for_128k_eval`: all visible GPUs have less than 16GiB free while another full GPU run is active; 128K model eval should wait for a clean device
