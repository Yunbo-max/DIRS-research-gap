# Tree-Search Scaling Boundary Test

Date: `2026-07-25`

Status: `preregistered_synthetic_scaling_test`

## Question

The six-path experiment favored fixed-budget best-arm methods. This test asks
when shared-prefix tree search becomes useful as the number of complete legal
paths grows.

## Valid DAG Family

Use full binary trees with:

```text
depths:
  4, 6, 8

complete connected paths:
  16, 64, 256

budget ratios:
  0.5K, 1K, 4K

episodes:
  300 per depth, landscape, and budget cell

terminal rollout noise:
  0.03

seed:
  20260725
```

Every action chooses one legal child of the current prefix. Every sampled
artifact arm is a complete root-to-leaf path. Arbitrary node subsets and
cross-branch jumps are impossible.

## Utility Landscapes

```text
hierarchical_smooth:
  early and intermediate prefix effects are shared by descendants; a prefix
  reward is therefore predictive of nearby terminal paths

independent_leaves:
  terminal utilities are independent; tree proximity carries no information

deceptive_needle:
  one globally best leaf is hidden inside a low-average root branch, while
  the other root branch has consistently good but suboptimal leaves
```

The third landscape explicitly tests whether average-value MCTS abandons a
branch containing a rare excellent path.

## Policies

```text
uniform_complete_path:
  sample complete paths uniformly; without replacement until all are seen

sequential_halving:
  fixed-budget elimination over all complete paths; applicable only when
  budget >= K

uct_mean_backup:
  legal-prefix UCT with mean reward backup

uct_max_backup:
  legal-prefix UCT with maximum observed reward backup
```

UCT uses the standard bounded-reward exploration coefficient
`sqrt(2)`. Final MCTS recommendation is the highest empirical terminal mean,
not the most-visited leaf.

## Metrics

```text
mean simple regret
best-leaf identification rate
top-5%-leaf selection rate
mean utility percentile of the selected leaf
paired episode-level simple-regret differences with normal-approximation
95% intervals
```

## Interpretation

Expected boundaries:

```text
smooth hierarchy:
  tree search may exploit predictive prefixes, especially when budget < K

independent leaves:
  tree structure should provide little or no advantage

deceptive needle:
  mean-backup MCTS may fail despite a valid connected tree
```

This is a controller simulation, not an LLM writing experiment. The numerical
landscapes are diagnostic constructions, not learned claims about papers.
