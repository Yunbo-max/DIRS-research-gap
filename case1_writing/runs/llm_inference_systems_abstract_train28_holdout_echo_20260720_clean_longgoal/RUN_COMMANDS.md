# LLM Inference / Systems / Token Efficiency Long-Goal Run

Run directory:

`/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_abstract_train28_holdout_echo_20260720_clean_longgoal`

Topic:

`/tf/notebooks/yunbo/DIRS/domain_topics/semantic_balanced_23_domains/llm_inference_systems_token_efficiency.md`

Held-out paper:

`ICML2026_71057_echo_elastic_speculative_decoding`

## Build Command

```bash
python /tf/notebooks/yunbo/DIRS/case1_writing/scripts/build_abstract_training_run.py \
  --domain-file /tf/notebooks/yunbo/DIRS/domain_topics/semantic_balanced_23_domains/llm_inference_systems_token_efficiency.md \
  --holdout-id ICML2026_71057_echo_elastic_speculative_decoding \
  --out-dir /tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_abstract_train28_holdout_echo_20260720_clean_longgoal \
  --force
```

## Single Convergence Pass

```bash
python /tf/notebooks/yunbo/DIRS/case1_writing/scripts/run_abstract_convergence_harness.py \
  --run-dir /tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_abstract_train28_holdout_echo_20260720_clean_longgoal \
  --max-loops 1000 \
  --min-loops 24 \
  --mcts-rollouts 5000 \
  --stable-window 10 \
  --seed 20260721
```

## 24-Hour Supervisor

The long-goal supervisor is running detached. It repeatedly launches the local deterministic convergence harness with new seeds, snapshots each iteration, and keeps going for 86400 seconds by default.

```bash
RUN_DIR=/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_abstract_train28_holdout_echo_20260720_clean_longgoal
setsid python "$RUN_DIR/run_longgoal_convergence_supervisor.py" \
  > "$RUN_DIR/longgoal_supervisor.nohup.log" 2>&1 < /dev/null &
```

Default environment:

```bash
DIRS_LONGGOAL_SECONDS=86400
DIRS_MIN_LOOPS=24
DIRS_MAX_LOOPS=1000
DIRS_MCTS_ROLLOUTS=5000
DIRS_STABLE_WINDOW=10
DIRS_BASE_SEED=20260720
DIRS_STOP_AFTER_CONVERGED=0
```

## Monitoring

```bash
RUN_DIR=/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_abstract_train28_holdout_echo_20260720_clean_longgoal
cat "$RUN_DIR/longgoal_supervisor.pid"
ps -p "$(cat "$RUN_DIR/longgoal_supervisor.pid")" -o pid,ppid,stat,etime,cmd
tail -n 20 "$RUN_DIR/longgoal_supervisor.events.jsonl"
python -m json.tool "$RUN_DIR/longgoal_supervisor.latest.json"
```

Snapshots are written under:

`longgoal_iterations/iter_*/`

## Blind-Holdout Rule

The harness does not read `holdout_private_after_generation.json`. The held-out ECHO abstract stays withheld until post-generation evaluation.

## Runtime Policy

This run uses the local deterministic DIRS harness. It avoids API calls, external model services, and A100-only assumptions, so it is suitable for the local RTX 4090 environment.
