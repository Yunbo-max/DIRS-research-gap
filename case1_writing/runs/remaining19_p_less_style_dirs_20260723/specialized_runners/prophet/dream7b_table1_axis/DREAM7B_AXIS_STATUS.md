# Prophet Dream-7B Axis Campaign

- Updated: `2026-07-23T14:07:40Z`
- Policy: exact Dream-7B full-grid artifacts only; no reduced/proxy convergence.
- Runnable configs: `0`
- Explicit blockers: `4`
- Launch: `{'launched': False, 'reason': 'no_runnable_configs_without_exact_dream7b_loader_and_generation_parity'}`

## Repository Evidence

- Repo: `/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/Prophet`
- Dream mentions: `1`
- `eval_llada.py registers model name llada_dist`
- `generate.py and generate_earlyexit.py examples instantiate GSAI-ML/LLaDA-8B-Instruct`
- `no eval_dream.py, Dream-specific generate wrapper, or Dream model identifier is present in the release`

## GPU Inventory

- GPU `0` free=`3586` MiB used=`20978` MiB util=`0`%
- GPU `1` free=`5933` MiB used=`18631` MiB util=`4`%
- GPU `2` free=`5360` MiB used=`19204` MiB util=`5`%
- GPU `3` free=`8515` MiB used=`16049` MiB util=`99`%

## Explicit Blockers

- `dream_model_identifier_and_loader`: `blocked_by_missing_exact_dream7b_model_identifier_and_loader`
- `dream_generation_function_parity`: `blocked_by_missing_dream_generate_and_earlyexit_parity_code`
- `dream_simple_evals_prompt_scorer_parity`: `blocked_by_missing_exact_simple_evals_prompt_and_answer_extractor_for_dream`
- `dream_table1_full_grid`: `blocked_until_dream_loader_prompt_scorer_and_gpu_budget_are_resolved`
