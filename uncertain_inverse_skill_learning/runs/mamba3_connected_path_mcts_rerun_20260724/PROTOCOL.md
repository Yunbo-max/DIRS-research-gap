# Mamba-3 Connected-Path DIRS-MCTS Rerun

Date: `2026-07-24`

Status: `strict_single_holdout_path_search`

## Correction Being Tested

MCTS must traverse a learned connected directed flow. It may not sample an
arbitrary node subset or combine macro-strategy names.

## Training And Holdout

Training graph source:

```text
yunbo/DIRS/case1_writing/runs/
llm_architecture_abstract_train19_holdout_mamba3_20260720/training_trace.json
```

Holdout:

```text
Mamba-3 public chip
```

The Mamba-3 expert abstract and paper text are forbidden to the writer and
evaluator.

## Graph Learning

Aggregate the 19 training examples only when every example is a complete
connected directed path. Save:

```text
learned_connected_dag.json
```

The first strict search space contains only complete paths observed in
training. Union-graph paths that are connected but never observed are retained
as hypotheses and excluded from this rerun.

## Writer Boundary

Use one fresh writer LLM. It may read only:

```text
Mamba-3 public chip
style_profile.json
this protocol
learned_connected_dag.json
```

It receives all six observed path specifications under the same evidence,
length, and effort budget. It writes one abstract per path plus claim
provenance and a read audit.

Forbidden:

```text
Mamba-3 expert abstract
Mamba-3 paper/full text
all earlier Mamba-3 generated abstracts and run folders
repository search
Internet
```

## Evaluator Boundary

Use one fresh evaluator LLM with no inherited conversation. It may read only:

```text
Mamba-3 public chip
style_profile.json
this protocol
an anonymized, deterministically shuffled draft packet
```

It must not see:

```text
path identifiers or path frequencies
writer self-scores
private anonymization mapping
expert abstract or full paper
earlier Mamba-3 runs
```

It produces hard factual checks, decomposed functional judgments, an overall
preference score, confidence, and a total ranking. Scores are an evaluator
outcome, not manually assigned path weights.

## MCTS

Construct a prefix tree from the six observed complete paths.

State:

```text
the complete selected prefix
```

Actions:

```text
only the next nodes that continue at least one observed training path
```

Prior:

```text
empirical conditional continuation frequency from the 19 training traces
```

Terminal reward:

```text
fresh evaluator overall preference score for that path's blind rollout
```

Every simulation saves:

```text
state prefix
valid outgoing frontier
selected next node
prior
visit count
Q before and after
terminal path and reward
backpropagation trace
```

## Comparisons

```text
exhaustive:
  evaluate all six observed paths and identify the reward oracle best

frequency-greedy:
  select the most frequent complete training path

random-valid:
  choose uniformly among the six complete observed paths

MCTS:
  search only through legal prefix continuations under limited budgets
```

## Claim Boundary

This rerun can test whether a prefix-constrained MCTS recovers the best cached
LLM rollout among six observed paths. It cannot establish:

```text
historical author cognition
open-vocabulary DAG discovery
cross-paper transfer
evaluator calibration
online MCTS with a newly sampled LLM rollout at every visit
general GFlowNet or DIRS convergence
```
