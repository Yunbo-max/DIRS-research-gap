# Real Counterfactual Multi-Rollout Protocol

## 1. Purpose

Real-task search policy selection requires within-task counterfactual evidence:

```text
same paper and evidence
different complete legal paths
multiple writer realizations per path
blind repeated evaluation
saved intermediate prefix states
```

Cross-paper one-path demonstrations cannot identify this.

## 2. Minimum Pilot

Use validation papers that were not used to learn the candidate graph or
search-policy parameters.

```text
validation papers:
  at least 5

legal complete paths per paper:
  all 6 current observed paths

writer rollouts per paper/path:
  at least 3

independent evaluator replicas per artifact:
  at least 2

minimum drafts:
  5 * 6 * 3 = 90

minimum scalar judgments:
  90 * 2 = 180
```

This is a pilot adequacy floor, not a universal sample-size theorem.
Power and evaluator disagreement from the pilot should determine the next
stage.

## 3. Isolation

Writer may read:

```text
public paper chip
frozen path node contracts
style and word-budget profile
claim provenance requirements
```

Writer must not read:

```text
expert abstract
full target paper text when the chip is the intended evidence boundary
other path rollouts for the same paper
evaluator results
path frequency or prior probability
```

Evaluator receives an anonymized artifact, the evidence chip, and a fixed
rubric. It must not see path identity, training support, writer identity,
expert abstract, or sibling rollout scores.

## 4. Rollout Record

```yaml
rollout:
  experiment_id:
  paper_id:
  evidence_hash:
  graph_hash:
  path_id:
  exact_path_nodes: []
  rollout_ordinal:
  writer_session_id:
  writer_version:
  sampling_config:
  prefix_checkpoints:
    - prefix_nodes: []
      evidence_bindings: []
      intermediate_plan:
      predicted_terminal_utility:
      uncertainty:
  final_artifact:
  claim_provenance: []
  hard_failures: []
```

## 5. Evaluation Record

```yaml
evaluation:
  anonymous_artifact_id:
  evaluator_replica:
  evaluator_version:
  factual_support:
  problem_gap:
  method_clarity:
  evaluation_function:
  result_salience:
  interpretation:
  boundedness:
  hard_failures: []
  holistic_utility:
  confidence:
```

Keep decomposed scores. Do not train a prefix value from one opaque scalar
without error attribution.

## 6. Identifiability Tests

Estimate separately:

```text
paper effect
path effect
paper-by-path interaction
writer variance
evaluator variance
hard-failure probability
```

Use a hierarchical model or paired mixed-effects analysis. The central
quantity for routing is not merely average path score:

\[
P\left(
\operatorname*{argmax}_{\pi} U(\pi,c)
\text{ lies under prefix }s
\mid h_s,c
\right).
\]

## 7. Prefix Audit

At each prefix depth:

1. Fit the prefix value model using calibration papers only.
2. Predict terminal mean, best-descendant utility, and uncertainty.
3. Evaluate calibration and rank association on untouched papers.
4. Measure how often the best final path lies below the prefix model's median.
5. Compare protected-coverage search with mean-backup MCTS.

Do not use the final test paper to select depth, backup target, prior,
threshold, or stopping rule.

## 8. Offline Policy Replay

After building a balanced rollout bank, compare policies on the same saved
reward streams:

```text
Sequential Halving
Top-Two Thompson
uniform allocation
frequency-greedy
frontier-constrained MCTS with mean backup
MCTS with calibrated best-descendant posterior
MCTS with protected branch coverage
```

Replay preserves paired fairness. A later online run is needed to measure
adaptive writer effects.

## 9. Acceptance Gate

Enable MCTS for a domain only if, on untouched validation papers:

```text
prefix predictions improve over context-only baselines
calibration intervals have acceptable coverage
deception rate is bounded
MCTS reduces regret or cost versus best-arm baselines
hard-failure rate does not increase
the result survives evaluator-replica and writer-seed sensitivity
```

All thresholds must be selected before the final holdout and saved with the
experiment version.
