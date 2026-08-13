# Fixed-Budget Connected-Path Search Policy Comparison

Date: `2026-07-25`

Status: `preregistered_synthetic_policy_comparison`

## Question

For six enumerable complete legal paths and stochastic terminal rewards, is
prefix-tree MCTS a better fixed-budget best-path identification method than
algorithms designed for pure exploration?

## Frozen Inputs

```text
graph:
  six complete paths observed in 19 non-Mamba-3 training traces

reward centres:
  scores from the earlier blind single-rollout evaluator

epistemic sigma:
  0.02 and 0.05

rollout sigma:
  0.03

episodes:
  1,000 paired episodes per scenario

budgets:
  12, 24, 48, 96, 192

seed:
  20260725
```

The simulator samples hidden contextual path utilities. Search policies see
only rewards obtained by executing complete legal paths.

## Policies Fixed Before Running

```text
frequency_greedy:
  most frequent training path; no new observations

random_valid:
  one uniformly selected complete legal path

uniform_allocation:
  equal terminal samples, then highest empirical mean

sequential_halving:
  equal samples among survivors, remove the lower half each round

successive_rejects:
  fixed-budget staged allocation and one-arm rejection each round

ucb1_q:
  flat complete-path UCB1 allocation, highest empirical mean recommendation

top_two_thompson:
  Gaussian top-two posterior sampling with beta=0.5 and known simulator
  rollout sigma, highest empirical mean recommendation

mcts_empirical_visit / mcts_empirical_q:
  prefix PUCT with empirical training-path-mass prior

mcts_path_uniform_visit / mcts_path_uniform_q:
  prefix PUCT with a prior uniform over complete paths, propagated to local
  actions by descendant path mass
```

MCTS uses `c_puct=2.0`. No parameter will be selected after viewing the target
results.

## Fairness

Within an episode, all adaptive policies draw the \(j\)-th observation of a
path from the same pre-generated per-path reward stream. The simulator truth
is hidden from every policy.

Every terminal arm corresponds to a connected path already observed in
training. Flat bandit policies select complete path identifiers; they cannot
construct arbitrary node subsets.

## Metrics

Primary:

```text
mean simple regret
```

Secondary:

```text
probability of identifying the simulator-best path
selection distribution
per-path allocation counts
paired episode-level simple-regret differences with normal-approximation
95% intervals
```

## Interpretation Boundary

This comparison tests the controller under synthetic scalar noise. It does not
test real writer diversity or evaluator disagreement.

For six enumerable leaf-reward paths, pure-exploration bandits are expected to
be strong. MCTS is justified only if tree structure, shared intermediate
values, unequal execution costs, or an implicit/large path space provides
additional information that flat complete-path algorithms cannot exploit.
