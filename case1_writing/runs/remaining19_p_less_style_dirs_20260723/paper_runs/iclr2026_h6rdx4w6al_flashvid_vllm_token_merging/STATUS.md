# FlashVID DIRS Status

- Updated: 2026-07-23T09:49:13Z
- Status: `blocked_by_exact_professional_runtime_and_data_requirements_after_specialized_runner`
- Converged: `false`
- DAG iteration: `3`
- Specialized runner: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/flashvid/run_flashvid_professional_gate.py`
- Specialized verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/flashvid/flashvid_specialized_verifier.json`

## What The Runner Verified
- Parsed 12 official FlashVID scripts covering baseline, FlashVID, Qwen/LLaVA model families, fixed-token-budget, and efficiency paths.
- Confirmed model repositories are visible through Hugging Face metadata.
- Confirmed `assets/videomme.jsonl` exists with 2700 records and LMMs-Eval task configs exist.
- Reproduced the analytic FLOPs script: LLaVA-OneVision 48.82 TFLOPs and FlashVID 4.27 TFLOPs.

## Why This Paper Is Not Accepted Yet
- The main result grid needs the official 8-process LMMs-Eval scripts on paper-compatible A800 80G hardware.
- Efficiency traces need an A100-class run and a populated VideoMME local video cache.
- The current environment has four RTX 4090 GPUs, Transformers 4.51.2, and no `flash-attn`, so it cannot produce verifier-comparable paper tables under the strict non-reduced gate.

## Exact Blockers
- `main_grid_hardware_gpu_count`: official main scripts request --num_processes 8; visible GPUs=4
- `main_grid_hardware_gpu_class`: paper/DAG expects NVIDIA A800 80G for main experiments; visible GPUs are not A800 80G
- `efficiency_hardware_gpu_class`: paper/DAG expects a single NVIDIA A100 for efficiency traces; visible GPUs are not A100
- `transformers_version_exactness`: README badge requires Transformers 4.57; environment has 4.51.2
- `flash_attention_runtime`: official scripts request attn_implementation=flash_attention_2; flash-attn package is not installed
- `videomme_video_cache`: playground/bench_efficiency.py expects ~/.cache/huggingface/videomme/data videos; no populated cache was found
