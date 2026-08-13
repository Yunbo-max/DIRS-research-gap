# Real-Trace Prefix Predictiveness Audit

Date: `2026-07-25`

Status: `preregistered_observational_audit`

## Inputs

Use only the 19 non-Mamba-3 training traces and the five saved subagent batch
status tables from:

```text
yunbo/DIRS/case1_writing/runs/
llm_architecture_abstract_train19_holdout_mamba3_20260720/
```

Do not use the Mamba-3 holdout score to fit or test prefix-value associations.

## Available Observations

For every training paper:

```text
one selected complete DAG path
one initial writer/evaluator score
one repaired final score
unsupported-claim count after repair
```

There are no same-paper counterfactual paths and no repeated writer rollout
for the same paper/path.

## Analyses Fixed Before Running

```text
path-group support and score dispersion
leave-one-out path-mean prediction versus leave-one-out global mean
association between pairwise shared-prefix length and final-score similarity
score differences for optional C1, M2, and E3 node presence
score ceiling and repair-effect diagnostics
permutation uncertainty with 10,000 score-label permutations
```

Primary adequacy question:

```text
Does this dataset identify a contextual prefix-value model strongly enough to
justify enabling MCTS on real writing tasks?
```

## Non-Identifiable Quantities

The following cannot be estimated from one realized path per paper:

```text
within-paper counterfactual path utility
writer variance conditional on a fixed paper and path
evaluator variance conditional on a fixed artifact
probability that a below-average branch contains the best path
causal value of adding or deleting a node
```

## Interpretation

All associations are observational and cross-paper. Topic difficulty, source
quality, evaluator differences, and path choice are confounded. Even a strong
association would motivate a future intervention; it would not establish a
causal MCTS value model.

## Post-hoc Robustness Extension

After the first audit showed a small leave-one-case-out path-mean improvement,
add:

```text
paired bootstrap interval and sign-flip test for absolute-error improvement
leave-one-batch-out prediction to reduce same-batch evaluator contamination
the same leave-one-case-out comparison on unrepaired initial scores
```

These checks are explicitly post-hoc and cannot convert the observational
dataset into a causal counterfactual test.
