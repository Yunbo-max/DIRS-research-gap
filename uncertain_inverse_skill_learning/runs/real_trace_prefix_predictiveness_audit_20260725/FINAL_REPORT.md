# Real-Trace Prefix Predictiveness Audit

Date: `2026-07-25`

Status: `completed_observational_audit_insufficient_for_mcts_calibration`

## Direct Result

The saved 19-paper training run does not identify a reliable real-task prefix
value model. It should not be used to enable MCTS on writing tasks.

## Data

```text
papers:                          19
unique complete paths:            6
support per path:               1–6
paths executed per paper:         1
repeated rollouts per paper/path: 0
independent evaluator repeats:    0
```

All 19 drafts were repaired with the training target visible to the training
workflow. Final scores were tightly compressed:

```text
mean: 0.95563
standard deviation: 0.01188
range: 0.94–0.98
scores >= 0.94: 19/19
```

This ceiling limits the observable utility variation.

## Prefix Similarity

Across 171 paper pairs:

```text
correlation:
  shared prefix length versus final-score similarity = 0.1266

two-sided permutation p:
  0.0876
```

The direction is weakly positive but does not provide stable evidence that a
longer shared writing path predicts similar utility across papers.

## Path-Mean Prediction

Leave-one-case-out:

| Predictor | MAE |
|---|---:|
| global mean | 0.01056 |
| same-path mean with fallback | 0.00867 |

Point improvement: `0.00190`.

Post-hoc paired robustness:

```text
bootstrap 95% interval:
  [-0.00130, 0.00518]

one-sided sign-flip p:
  0.138
```

Leave-one-batch-out:

```text
point MAE improvement:
  0.00137

bootstrap 95% interval:
  [-0.00184, 0.00497]

one-sided sign-flip p:
  0.217
```

The intervals cross zero, so the apparent path-mean improvement is not robust
in this small dataset.

## Optional Node Associations

```text
C1 present minus absent:
  +0.00352, permutation p=0.533

M2 present minus absent:
  +0.01579, permutation p=0.011

E3 present minus absent:
  -0.01313, permutation p=0.058
```

The M2 association is statistically visible but not causal. M2 was selected
for theory- and mechanism-heavy papers, path choice used target knowledge, and
there is no same-paper M2 ablation. The negative E3 association further shows
why node presence must not be interpreted as a universal fixed reward.

## Identifiability Decision

The dataset cannot estimate:

```text
same-paper counterfactual path utility
execution variance conditional on paper and path
evaluator variance
within-paper deceptive branches
causal node or edge value
```

Decision:

```text
enable real-task MCTS:
  no

retain graph as a legal flow hypothesis:
  yes

collect balanced counterfactual multi-rollout data:
  required
```

## Required Next Dataset

For each validation paper, execute several complete legal paths under equal
evidence and writer budgets. Generate multiple independent artifacts per
paper/path, anonymize them, and obtain repeated blind judgments. Save prefix
checkpoints so prefix predictions can be compared with realized terminal
utility.
