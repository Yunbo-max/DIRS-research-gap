# Connected-Path Search Policy Comparison

Date: `2026-07-25`

Status: `completed_synthetic_paired_comparison`

## Direct Answer

For this problem—six enumerable complete legal paths, stochastic terminal
rewards, and a fixed budget—standard MCTS was not the best search method.

The best overall point estimate was Top-Two Thompson Sampling. Sequential
Halving was a close, simpler, distribution-robust alternative.

## Experiment

```text
legal complete paths:       6
contextual uncertainty:     0.02 and 0.05
rollout noise:              0.03
paired episodes:            1,000 per scenario
budgets:                    12, 24, 48, 96, 192
total scenario-budget cells: 10
```

All methods used the same hidden utility and per-path reward streams inside
each episode.

## Overall Ordering

Descriptive mean simple regret across the ten scenario-budget cells:

| Policy | Mean regret |
|---|---:|
| Top-Two Thompson | 0.001792 |
| Sequential Halving | 0.001972 |
| Successive Rejects | 0.002293 |
| UCB1-Q | 0.002522 |
| Uniform allocation | 0.002563 |
| MCTS path-uniform Q | 0.002644 |
| MCTS empirical-prior Q | 0.003479 |
| MCTS path-uniform visits | 0.004032 |
| MCTS empirical-prior visits | 0.015863 |
| Frequency-greedy | 0.019944 |
| Random-valid | 0.055094 |

Top-Two Thompson had the lowest regret in 8 of 10 cells. Sequential Halving
had the lowest regret in the remaining 2.

This across-cell average is descriptive because budgets and uncertainty
settings are different experimental conditions; the paired comparisons below
are the stronger result.

## Budget 192

Moderate contextual uncertainty:

| Policy | Best-path rate | Mean regret |
|---|---:|---:|
| Top-Two Thompson | 90.8% | 0.000279 |
| Sequential Halving | 91.0% | 0.000307 |
| Successive Rejects | 90.5% | 0.000355 |
| MCTS empirical-prior Q | 89.2% | 0.000476 |
| MCTS path-uniform Q | 88.8% | 0.000510 |
| Uniform allocation | 87.7% | 0.000629 |

Higher contextual uncertainty:

| Policy | Best-path rate | Mean regret |
|---|---:|---:|
| Sequential Halving | 95.2% | 0.000154 |
| Top-Two Thompson | 94.9% | 0.000187 |
| Successive Rejects | 94.4% | 0.000241 |
| MCTS empirical-prior Q | 93.7% | 0.000297 |
| MCTS path-uniform Q | 93.5% | 0.000310 |
| Uniform allocation | 93.4% | 0.000338 |

## Paired Evidence

For Top-Two Thompson minus MCTS empirical-prior Q, the paired mean simple
regret difference was below zero with a normal-approximation 95% interval
entirely below zero in all 10 scenario-budget cells.

The same was true for Sequential Halving minus MCTS empirical-prior Q in all
10 cells.

Top-Two Thompson versus Sequential Halving was not uniformly resolved:

```text
Top-Two Thompson:
  lower point-estimate regret in 8/10 cells

Sequential Halving:
  lower point-estimate regret in 2/10 cells

paired intervals:
  often overlapped zero, so neither dominates in every regime
```

Top-Two Thompson was given the simulator rollout standard deviation. That is a
real informational advantage and must be estimated rather than assumed in an
LLM experiment.

## Why MCTS Lost

The current problem exposes only terminal rewards and has six enumerable
paths. MCTS therefore receives little benefit from shared prefixes:

```text
no informative intermediate node reward
no massive implicit terminal path set
no unequal path execution cost
no value network transferring information across related prefixes
```

Its visit-count recommendation also preserves prior-induced allocation bias.
The Q recommendation is substantially better, but the tree traversal still
spends samples less efficiently than methods designed for fixed-budget
best-arm identification.

## Recommended DIRS Router

```text
small enumerable legal path set:
  default to Sequential Halving

small enumerable set with calibrated reward posterior/noise:
  use Top-Two Thompson Sampling

large or implicit sub-DAG space:
  use frontier-constrained MCTS with progressive widening

large space plus stochastic terminal execution:
  use MCTS for structural expansion and a best-arm/posterior rule for leaf
  allocation and final recommendation
```

This preserves the learned DAG and connected-flow constraints without forcing
MCTS onto a problem where a simpler pure-exploration algorithm is better.

## Validation

```text
audited flat-policy samples: 1,920
audited MCTS simulations:      768
audited MCTS decisions:      7,619
illegal paths/actions:           0
validation status:            pass
```

## Claim Boundary

This result is a synthetic controller comparison. It does not yet establish
the same ordering with independently generated abstracts and blind human or
LLM evaluations.
