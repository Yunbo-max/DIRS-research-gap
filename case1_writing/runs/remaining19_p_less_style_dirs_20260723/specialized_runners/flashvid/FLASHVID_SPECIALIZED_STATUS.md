# FlashVID Specialized Runner Status

- Updated: 2026-07-23T18:53:33Z
- Paper: `FlashVID: Efficient Video Large Language Models via Training-free Tree-based Spatiotemporal Token Merging`
- Status: `blocked_by_exact_professional_runtime_and_data_requirements`
- Convergence decision: `not_converged_explicit_professional_blockers_after_operational_preflight`
- Professional package ready: `False`
- Official scripts parsed: `12`
- FLOPs reproduction status: `pass`
- Blocker count: `6`

## Artifact Paths
- Environment: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/flashvid/environment.json`
- Official script manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/flashvid/official_script_manifest.json`
- Model/data manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/flashvid/model_data_manifest.json`
- FLOPs reproduction: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/flashvid/flops_reproduction.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/flashvid/flashvid_specialized_verifier.json`

## Why This Is Not Converged
- This gate did not run reduced VideoMME or one-video demos as convergence evidence.
- The full paper grid needs the official 8-process LMMs-Eval benchmark scripts, paper-compatible GPU class, exact runtime stack, and video dataset caches.
- Until those artifacts exist, the DAG is semantically plausible but operationally blocked.

## Current Blockers
- `main_grid_hardware_gpu_count`: official main scripts request --num_processes 8; visible GPUs=4
- `main_grid_hardware_gpu_class`: paper/DAG expects NVIDIA A800 80G for main experiments; visible GPUs are not A800 80G
- `efficiency_hardware_gpu_class`: paper/DAG expects a single NVIDIA A100 for efficiency traces; visible GPUs are not A100
- `transformers_version_exactness`: README badge requires Transformers 4.57; environment has 4.51.2
- `flash_attention_runtime`: official scripts request attn_implementation=flash_attention_2; flash-attn package is not installed
- `videomme_video_cache`: playground/bench_efficiency.py expects ~/.cache/huggingface/videomme/data videos; no populated cache was found
