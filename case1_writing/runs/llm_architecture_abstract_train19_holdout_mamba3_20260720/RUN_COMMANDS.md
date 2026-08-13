# DIRS Case 1 Run Commands

Date: `2026-07-20`

These commands reproduce the abstract-training split and the convergence
preflight for this run.

## Rebuild Training Split

```bash
python /tf/notebooks/yunbo/DIRS/case1_writing/scripts/build_abstract_training_run.py \
  --domain-file /tf/notebooks/yunbo/DIRS/domain_topics/semantic_balanced_23_domains/llm_architecture_attention_state_space_models.md \
  --out-dir /tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_architecture_abstract_train19_holdout_mamba3_20260720 \
  --holdout-chip-id ICLR2026_HwCvaJOiCj_mamba3
```

## Run Convergence Harness

```bash
python /tf/notebooks/yunbo/DIRS/case1_writing/scripts/run_abstract_convergence_harness.py \
  --run-dir /tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_architecture_abstract_train19_holdout_mamba3_20260720 \
  --max-loops 100 \
  --mcts-rollouts 500 \
  --stable-window 5 \
  --seed 20260720
```

## Run Heavy Subagent Simulation

Use this after the deterministic convergence harness passes. This is the real
editor/simulator/evaluator loop that checks whether the selected DAG can produce
training abstracts as prose, not just replay their structure.

Use the subagent protocol in `SUBAGENT_HEAVY_SIMULATION_PROTOCOL.md`. For this
run family, do not use API calls for the DIRS editor, simulator, or evaluator
roles.

```text
coordinator:
  main Codex thread

workers:
  editor subagent
  simulator subagent
  evaluator subagent
```

The completed subagent pass is recorded under:

```text
subagent_live_runs/
```

## Deprecated API Diagnostic

The Python API runner is retained only for diagnostic reproduction and must be
explicitly enabled with `DIRS_ALLOW_OPENAI_API=1`. It is not the default DIRS
runtime.

For a dry-run smoke test that does not call an API:

```bash
python /tf/notebooks/yunbo/DIRS/case1_writing/scripts/run_heavy_llm_abstract_simulation.py \
  --run-dir /tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_architecture_abstract_train19_holdout_mamba3_20260720 \
  --output-name heavy_llm_smoke_sample1 \
  --max-loops 1 \
  --max-samples 1 \
  --dry-run
```

## Blind Holdout Rule

Do not open this file until after blind generation and verification:

```text
holdout_private_after_generation.json
```

The public holdout inputs are:

```text
holdout_test_card.md
style_profile.json
node_support_scores.json
edge_support_scores.json
training_trace.json
the held-out chip facts
```
