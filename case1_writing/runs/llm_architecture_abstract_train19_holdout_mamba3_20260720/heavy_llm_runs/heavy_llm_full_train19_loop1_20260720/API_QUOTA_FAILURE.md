# API Fallback Failure

Date: `2026-07-20`

Attempted command:

```bash
python /tf/notebooks/yunbo/DIRS/case1_writing/scripts/run_heavy_llm_abstract_simulation.py \
  --run-dir /tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_architecture_abstract_train19_holdout_mamba3_20260720 \
  --output-name heavy_llm_full_train19_loop1_20260720 \
  --max-loops 1 \
  --max-samples 0 \
  --mcts-rollouts 500 \
  --candidate-paths 4 \
  --stable-window 3 \
  --min-mean-score 0.88 \
  --seed 20260720
```

Result:

```text
openai.RateLimitError: Error code: 429
code: insufficient_quota
```

Interpretation:

```text
The unattended OpenAI API fallback cannot run the full 19-sample heavy pass in
this environment right now. Continue with Codex subagent batches instead.
```
