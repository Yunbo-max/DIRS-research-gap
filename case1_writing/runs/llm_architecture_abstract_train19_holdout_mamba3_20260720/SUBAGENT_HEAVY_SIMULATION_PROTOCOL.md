# Subagent Heavy Simulation Protocol

Date: `2026-07-20`

Purpose: define the default heavy DIRS runtime for abstract writing when Codex
can call subagents.

## Default Runtime

The deterministic MCTS replay harness is only a preflight. The real training
simulation should use subagents:

```text
Coordinator:
  owns the run state, files, loop counter, convergence checks, and blind rules

Loop 1 Editor Subagent:
  input = chip facts + current full DAG + prior feedback + node/edge support
  output = selected connected sub-DAG, budget plan, repair rule

Loop 2 Simulator Subagent:
  input = chip facts + selected connected sub-DAG + budget plan
  output = generated abstract text

Evaluator Subagent:
  input = generated abstract + training target abstract + chip facts
  output = score report + missing/unsupported claims + feedback for editor
```

The coordinator feeds the evaluator feedback into the next editor call. The
editor is the only role allowed to modify or repair the DAG proposal. The
simulator only writes from the selected DAG. The evaluator does not modify the
DAG; it produces feedback.

## Per-Sample Training Loop

For each training chip:

```text
for loop in 1..max_loops:
  editor_subagent.select_or_repair_dag(chip, current_dag, feedback)
  simulator_subagent.generate_abstract(chip, selected_dag)
  evaluator_subagent.compare(generated, training_abstract, chip)
  coordinator.store(event)
  if sample stable and verifier passes:
    stop sample or continue full-domain pass
```

For full-domain training:

```text
for loop in 1..max_loops:
  run the per-sample cycle across all training chips
  merge editor-selected paths into the shared full DAG
  stop when the shared full DAG is stable for 3-5 passes
```

## Held-Out Rule

During training, the evaluator may read the training target abstract. For the
held-out chip:

```text
allowed:
  chip facts
  style_profile.json
  node_support_scores.json
  edge_support_scores.json
  training_trace.json
  selected full DAG from convergence_report.json

forbidden before blind generation and verification:
  holdout_private_after_generation.json
```

## Why Subagents

Subagents better match DIRS than one monolithic LLM call:

```text
editor = graph repair and node selection
simulator = text/action execution from fixed graph
evaluator = result comparison and feedback
coordinator = memory, file state, convergence, and blind-boundary enforcement
```

This keeps the two loops separate:

```text
Loop 1 changes the graph.
Loop 2 simulates using the graph.
Evaluator compares outputs and gives feedback.
```

## API Boundary

Do not use hosted APIs for this heavy DIRS writing runtime when Codex subagents
are available. The API runner is only a diagnostic fallback and must be
explicitly opted into. For this run family, the intended engine is:

```text
main Codex coordinator + Codex subagents
```
