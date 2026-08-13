# Mamba-3 Stochastic Connected-Path MCTS Simulation

Date: `2026-07-25`

Status: `preregistered_algorithmic_simulation`

## Purpose

The earlier rerun used one cached LLM rollout for each of six observed paths.
This experiment asks a narrower question:

```text
If executing the same path can produce different rewards, does connected-path
MCTS identify the best latent path reliably under a limited rollout budget?
```

It does not claim to be a new LLM writer/evaluator experiment.

## Frozen Search Space

Use the graph and six complete paths learned from 19 non-Mamba-3 training
traces in:

```text
../mamba3_connected_path_mcts_rerun_20260724/learned_connected_dag.json
```

Every MCTS state is a complete selected prefix. Every action must be a legal
next node that continues at least one of the six observed paths. New nodes,
new edges, and disconnected node subsets are forbidden.

## Synthetic Reward Model

The six earlier blind evaluator scores are used only as simulator centres:

```text
latent path utility:
  clipped Gaussian(score centre, epistemic sigma)

each newly sampled terminal reward:
  clipped Gaussian(latent path utility, rollout sigma)
```

The search policy never reads the latent utility or score centre. It observes
only the newly sampled reward after completing a legal path.

Uncertainty is a sensitivity grid rather than a fitted claim:

```text
epistemic sigma:
  0.00, 0.02, 0.05

rollout sigma:
  0.01, 0.03, 0.06
```

Each of the nine scenarios contains 500 paired Monte Carlo episodes. Budgets
are 12, 24, 48, 96, and 192 terminal rollouts. The seed is fixed at 20260725.

## Compared Policies

```text
frequency_greedy:
  select the most frequent training path without holdout rollout evidence

random_valid:
  uniformly select one complete legal path

uniform_allocation_q:
  allocate rollouts evenly, then select the highest empirical path mean

mcts_visit:
  PUCT over legal prefixes, recommend the most-visited terminal path

mcts_terminal_q:
  use the same PUCT samples, recommend the highest empirical terminal mean
```

The same per-path reward streams are reused across search policies within an
episode where applicable, reducing avoidable Monte Carlo comparison noise.

## Metrics

```text
best-path identification rate
mean simple regret against the simulator oracle
selection counts
oracle-path distribution
full legal-frontier and backpropagation trace for episode 1
```

## Claim Boundary

This simulation can diagnose search behavior under controlled stochasticity.
It cannot establish writer diversity, evaluator calibration, human
preference, transfer to another paper, or superiority of MCTS in real online
LLM execution. A later real run must replace the scalar sampler with
independently generated artifacts and blind evaluations.
