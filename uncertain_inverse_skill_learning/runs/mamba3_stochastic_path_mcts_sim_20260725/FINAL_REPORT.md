# Stochastic Connected-Path MCTS Simulation Report

Date: `2026-07-25`

Status: `completed_synthetic_uncertainty_audit`

## What Was Actually Run

The experiment froze the six complete connected paths learned from 19
training traces. A search action was legal only when it continued an observed
path from the complete current prefix.

The earlier blind evaluator scores were used as simulator centres. Across nine
uncertainty settings, each MCTS terminal visit received a newly sampled scalar
reward. The run contained:

```text
9 uncertainty scenarios
500 Monte Carlo episodes per scenario
192 maximum terminal samples per episode
5 recorded budget checkpoints
```

This is not a new writer/evaluator run. It tests controller behavior under
synthetic stochastic rewards.

## Main Result

Raw empirical training frequency was a poor substitute for a contextual
policy prior. Under contextual utility shifts, the standard most-visited MCTS
recommendation remained almost completely locked to `path_01`.

At medium uncertainty (`epistemic=0.02`, `rollout=0.03`) and budget 192:

| Policy | Best-path rate | Mean simple regret |
|---|---:|---:|
| frequency-greedy | 55.2% | 0.00825 |
| uniform allocation + Q | 88.4% | 0.00062 |
| MCTS most-visited | 55.4% | 0.00813 |
| MCTS terminal Q | 90.4% | 0.00047 |

At higher contextual uncertainty (`epistemic=0.05`,
`rollout=0.03`) and budget 192:

| Policy | Best-path rate | Mean simple regret |
|---|---:|---:|
| frequency-greedy | 40.8% | 0.02942 |
| uniform allocation + Q | 87.4% | 0.00045 |
| MCTS most-visited | 46.6% | 0.02154 |
| MCTS terminal Q | 85.0% | 0.00034 |

MCTS terminal-Q selection adapted far better than the visit-count rule, but it
did not consistently beat even allocation. With only six paths, even
allocation is cheap and difficult to improve upon.

## Fixed-Utility Control

When epistemic uncertainty was zero, `path_01` was defined by the simulator to
be the true best in every episode. Frequency-greedy and most-visited MCTS then
scored 100% even under high rollout noise. This is a construction of the
control scenario, not evidence that either method learned from stochastic
rollouts.

## Structural Validation

The saved audit traces were checked independently:

```text
audited simulations:       1,728
audited prefix decisions: 17,142
audited backpropagations: 17,142
illegal actions:               0
invalid terminals:             0
validation status:          pass
```

## Decision

Do not use raw training path frequency as the final contextual MCTS prior.
Do not use most-visited terminal path as the only recommendation rule.

The next real LLM experiment should:

1. Generate at least three independent artifacts per visited path.
2. Blind and pairwise-evaluate those artifacts.
3. Learn or validation-calibrate the contextual prior on non-target papers.
4. Report both visit-based and reward-posterior recommendations.
5. Retain even allocation as a baseline.

## Claim Boundary

The experiment supports a controller correction, not an abstract-quality
claim. Noise scales are sensitivity settings, reward centres come from one
earlier evaluator, and no fresh writer or evaluator was called.
