# DIRS Long Goal Status

Date: `2026-07-20`

Goal: train DIRS Case 1 on one abstract-writing domain, hold out one chip, and
prepare the system for a long MCTS + dual-system writing run.

## Current State

```text
status: launched_preflight_and_ready_for_heavy_run
domain: LLM Architecture / Attention / State Space Models
training_chips: 19
holdout_chip: ICLR2026_HwCvaJOiCj_mamba3
holdout_title: Mamba-3: Improved Sequence Modeling using State Space Principles
```

The first smoke extraction was quarantined as:

```text
/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_architecture_abstract_train19_holdout_mamba3_20260720_bad_extraction
```

The clean run is:

```text
/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_architecture_abstract_train19_holdout_mamba3_20260720
```

## Preflight Result

```text
training abstracts extracted: 19 / 19
held-out original abstract: hidden in holdout_private_after_generation.json
training median target length: 201 words
target band: 166-236 words
```

The convergence harness ran with:

```text
max_loops: 100
mcts_rollouts_per_example: 500
stable_window: 5
```

It converged at loop `6`.

## Heavy 10-Hour Run

The 10-hour version uses the same split and must run after the lightweight
deterministic replay harness:

```text
Loop 1 editor:
  update the extended training DAG from result + chip + evaluator feedback

Loop 2 simulator:
  use MCTS over connected sub-DAGs to generate candidate abstracts

Evaluator:
  compare generated result against the training target during training
  produce feedback for Loop 1
```

For the held-out paper, the evaluator must not read
`holdout_private_after_generation.json` until after blind generation and
verification.

The runnable heavy-loop script is:

```text
/tf/notebooks/yunbo/DIRS/case1_writing/scripts/run_heavy_llm_abstract_simulation.py
```

The current heavy runner now keeps mutable shared DAG state, generates MCTS
candidate paths before each editor call, updates node/edge weights from
evaluator feedback, and supports blind held-out generation without reading
`holdout_private_after_generation.json`.

When subagents are available, use the coordinator/subagent protocol in:

```text
SUBAGENT_HEAVY_SIMULATION_PROTOCOL.md
```
