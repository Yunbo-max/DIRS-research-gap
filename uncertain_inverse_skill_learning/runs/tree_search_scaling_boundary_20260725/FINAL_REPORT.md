# Tree-Search Scaling Boundary Report

Date: `2026-07-25`

Status: `completed_paired_synthetic_boundary_test`

## Direct Result

MCTS became useful only when legal prefixes carried reliable information about
their descendant paths and the budget was too small to evaluate all complete
paths. A larger path count alone did not make MCTS better.

## Test Matrix

```text
binary DAG depths:       4, 6, 8
complete legal paths:   16, 64, 256
budget/path ratios:    0.5, 1, 4
utility landscapes:      smooth, independent, deceptive
episodes per cell:       300
total cells:              27
```

Every MCTS action was one legal binary extension of the current prefix. Every
other policy selected a complete root-to-leaf path.

## Low Budget: B = 0.5K

### Hierarchical smooth landscape

Mean simple regret:

| Paths | Uniform | UCT mean | UCT max | Paired conclusion |
|---:|---:|---:|---:|---|
| 16 | 0.02769 | 0.01878 | 0.01756 | both UCT variants better |
| 64 | 0.02155 | 0.01802 | 0.01781 | both UCT variants better |
| 256 | 0.01885 | 0.01757 | 0.01829 | intervals overlap zero |

UCT exploited shared prefix signal at 16 and 64 paths. Its point estimate
remained favorable at 256, but the paired 95% interval no longer excluded
zero.

### Independent leaves

No UCT-versus-uniform paired interval excluded zero at 16, 64, or 256 paths.
Tree structure provided no reliable information because sibling paths were
independent.

### Deceptive needle

UCT was significantly worse than uniform complete-path sampling at all three
sizes:

| Paths | Uniform regret | UCT mean | UCT max |
|---:|---:|---:|---:|
| 16 | 0.08226 | 0.09993 | 0.09910 |
| 64 | 0.07425 | 0.10689 | 0.09440 |
| 256 | 0.08082 | 0.10560 | 0.11496 |

The best path was hidden in a low-average branch. Mean backup abandoned that
branch; maximum-observed backup remained noise-sensitive and did not solve the
problem.

## Full Initial Coverage: B = K

UCT sometimes improved on one-sample-per-path selection by adaptively
resampling promising paths in smooth or benign landscapes. However, it remained
strongly worse in every deceptive-needle cell.

Therefore `B=K` is not by itself a sufficient routing criterion. Prefix-value
reliability is still required.

## Repeated Coverage: B = 4K

Sequential Halving had significantly lower paired regret than both UCT
variants for every smooth and independent landscape size.

Examples:

| Landscape | Paths | Sequential Halving | UCT mean | UCT max |
|---|---:|---:|---:|---:|
| smooth | 64 | 0.00127 | 0.00458 | 0.00418 |
| smooth | 256 | 0.00095 | 0.00532 | 0.00453 |
| independent | 64 | 0.00031 | 0.00191 | 0.00167 |
| independent | 256 | 0.00010 | 0.00212 | 0.00134 |

When all paths were enumerable and the budget supported repeated coverage,
fixed-budget elimination was better aligned with best-path identification.

## What This Changes

The prior routing rule is refined:

```text
small/enumerable and enough repeated coverage:
  Sequential Halving

enumerable with calibrated posterior:
  Top-Two Thompson remains a strong option

budget below path count + validated smooth prefix signal:
  frontier-constrained UCT/MCTS can help

independent or unknown prefix signal:
  MCTS has no demonstrated advantage

possible deceptive low-average branches:
  do not trust mean or raw-max backup; preserve explicit coverage or posterior
  probability that a branch contains the optimum
```

No fixed `B/K` threshold is accepted as universal. The values above are
diagnostic evidence; routing thresholds must be validation-selected.

## Structural Validation

```text
cells:                    27/27
audited uniform samples:  4,032
audited UCT simulations:  8,064
audited UCT decisions:   59,904
illegal actions/paths:        0
validation:                pass
```

## Claim Boundary

The graphs and utilities are synthetic. The experiment establishes a
controller failure/success boundary under the constructed landscapes, not
performance on real LLM writing traces.
