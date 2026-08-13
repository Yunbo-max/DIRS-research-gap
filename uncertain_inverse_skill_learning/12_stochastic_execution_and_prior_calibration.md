# Stochastic Execution And Contextual Prior Calibration

## 1. Why One Reward Per Path Is Insufficient

A skill path constrains information flow, but it does not uniquely determine
the realized artifact. Different writer samples can vary in compression,
wording, evidence allocation, and claim strength. Different evaluators can
also disagree.

DIRS should model:

\[
U_{\pi,c}
=
\mathbb E[R\mid\pi,c],
\qquad
\sigma^2_{\pi,c}
=
\operatorname{Var}(R\mid\pi,c).
\]

The mean estimates expected quality; the variance can contain writer,
evaluator, and task ambiguity. These sources should be separated when the
experimental design permits it.

## 2. Four Uncertainties

```text
graph uncertainty:
  which nodes and dependencies form a valid reusable graph?

path uncertainty:
  which complete legal flow is suitable for the current context?

execution uncertainty:
  how does a writer realize the same selected path across samples?

evaluation uncertainty:
  how stable is utility under evaluator prompts, models, and human judgments?
```

Increasing MCTS visits addresses only uncertainty that is represented by
fresh observations. Repeating a cached scalar cannot measure execution or
evaluation variance.

## 3. Contextual Graph Prior

A useful factorization is:

\[
q(G,\pi,\theta\mid c,\mathcal D)
=
q_\omega(G\mid c,\mathcal D)
q_\psi(\pi\mid G,c,\mathcal D)
q_\eta(\theta\mid\pi,G,c,\mathcal D),
\]

where \(G\) is a valid graph snapshot, \(\pi\) a complete legal path, and
\(\theta\) execution/value parameters.

Training path frequency is evidence for \(q_\psi\), but context features,
transfer outcomes, failure history, and uncertainty are also required. A raw
frequency prior is unsafe under contextual distribution shift.

## 4. Online Evidence Protocol

For each selected terminal path:

1. Generate a fresh artifact from the same frozen evidence and path contract.
2. Record sampling parameters and writer version.
3. Verify factual provenance and hard constraints.
4. Anonymize the artifact independently of path identity.
5. Obtain paired or multi-evaluator judgments.
6. Update the terminal reward posterior.
7. Backpropagate the new observation through the exact traversed prefixes.

An auditable reward record is:

```yaml
rollout:
  episode_id:
  graph_hash:
  path_id:
  path_nodes: []
  rollout_ordinal:
  writer_version:
  writer_sampling:
  artifact_uri:
  provenance_uri:
  anonymization_id:
  evaluator_versions: []
  decomposed_scores: {}
  hard_failures: []
  scalar_utility:
  utility_rule_version:
```

## 5. Selection And Recommendation Are Different

The tree policy decides where to collect the next observation. The final
recommendation decides which path to deploy. They need not use the same rule.

```text
collection:
  contextual PUCT, Thompson sampling, Bayesian UCB, or information gain

recommendation:
  posterior mean, lower confidence bound, constrained Pareto choice, or
  probability of being best
```

The standard most-visited rule can remain dominated by a misspecified prior.
The recommendation rule must therefore be evaluated separately from the
collection policy.

## 6. Required Baselines

For a finite path set:

```text
frequency-greedy
uniform random-valid
even rollout allocation plus empirical-best selection
context-free bandit policy
MCTS with ablated structural/contextual priors
oracle best under the saved evaluation bank
```

MCTS should be claimed as beneficial only when it improves sample efficiency,
regret, hard-failure rate, or execution cost over these baselines on held-out
tasks.

## 7. Evidence From The 2026-07-25 Audit

The Mamba-3 stochastic simulation used six connected observed paths and nine
synthetic uncertainty settings. It found:

```text
raw frequency prior + most-visited recommendation:
  often remained locked to the most common training path

terminal empirical-Q recommendation:
  adapted to contextual utility shifts much better

even allocation:
  remained a strong baseline because only six paths existed

post-hoc prior smoothing:
  helped at intermediate mixing but failed to provide a universal setting

fully local-uniform action prior:
  was not path-uniform because of tree topology
```

This is algorithmic evidence about the controller, not evidence from new LLM
abstracts. It motivates learned contextual priors and repeated blind
executions; it does not validate a fixed smoothing coefficient.
