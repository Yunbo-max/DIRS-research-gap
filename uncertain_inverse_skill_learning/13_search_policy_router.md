# Search Policy Router

## 1. Do Not Equate DAG Execution With MCTS

The learned DAG defines legal information flow. It does not require every
legal-path selection problem to use MCTS.

```text
DAG:
  representation, dependency, connectivity, and execution validity

search policy:
  how limited observations are allocated among legal candidates
```

Replacing MCTS with another selector does not turn the DAG into a random node
combination. Every candidate must still be a connected, contract-valid flow.

## 2. Diagnose The Search Regime

Route using observable problem properties:

```text
K:
  number of enumerable complete legal paths

implicit:
  whether complete paths can be listed before search

intermediate_signal:
  whether partial DAG execution provides predictive rewards

shared_computation:
  whether evaluating a prefix can be reused by many descendants

cost_variation:
  whether path executions have materially different costs

reward_calibration:
  whether a reliable contextual reward posterior/noise model is available
```

These features determine which search family is appropriate.

## 3. Small Enumerable Path Set

When all legal paths can be enumerated and only terminal rewards are observed,
the problem is fixed-budget best-arm identification.

Recommended default:

```text
Sequential Halving
```

Reasons:

```text
simple and auditable
no learned value model required
does not need the reward variance to be known
systematically removes weak paths
directly optimizes terminal selection rather than tree visit counts
```

When the execution/evaluator reward posterior is calibrated, use:

```text
Top-Two Thompson Sampling
```

It concentrates observations on the two paths most likely to compete for
best, but its posterior assumptions and uncertainty estimates must be checked.

## 4. Large Or Implicit DAG

MCTS remains appropriate when:

```text
complete paths cannot be enumerated
the valid frontier is large
partial executions provide useful value estimates
shared prefixes reuse expensive computation
progressive widening is required
context gates reveal branches during execution
```

Use frontier-constrained MCTS to expand structure, but separate:

```text
tree collection policy:
  which prefix or frontier action to evaluate next

terminal allocation policy:
  which completed paths need repeated stochastic rollouts

deployment recommendation:
  which posterior path/sub-DAG should be executed
```

The most-visited terminal should not be the automatic deployment answer.

## 5. Hybrid DIRS Search

A general controller can be:

```text
1. Freeze or root-sample one valid graph snapshot.
2. Estimate whether complete legal paths are enumerable under the budget.
3. If enumerable and small:
     run Sequential Halving or Top-Two Thompson over legal complete paths.
4. Otherwise:
     run MCTS with typed legal-frontier expansion and progressive widening.
5. For repeated executions of promising terminal paths:
     use a best-arm posterior allocator.
6. Recommend using posterior utility and risk constraints, not raw visits.
7. Save router features, selected algorithm, observations, and counterfactual
   baseline outcomes.
```

## 6. Router Learning

The route should initially use transparent validation-selected thresholds.
After enough domains are available, learn:

\[
\rho(a\mid \phi(c,G,b)),
\]

where \(a\) is the search algorithm and \(\phi\) contains graph size,
branching, cost, context, uncertainty, and available intermediate signals.

The learned router is accepted only if it improves held-out regret or cost
without increasing hard failures. Algorithm names and thresholds are not
content skills and should not be inferred from the target holdout.

## 7. Current Evidence

The 2026-07-25 six-path synthetic comparison found Top-Two Thompson lowest in
8 of 10 scenario-budget cells and Sequential Halving lowest in 2. Both had
paired regret below empirical-prior MCTS-Q across all ten cells.

The subsequent 16/64/256-path tree test refined the large-space branch:

```text
budget < path count + smooth predictive prefixes:
  UCT improved over uniform sampling at 16 and 64 paths; the 256-path paired
  interval overlapped zero

independent leaves:
  UCT had no stable low-budget advantage

deceptive low-average branch:
  both mean and raw-max UCT were significantly worse at every tested size

budget = 4 * path count:
  Sequential Halving was significantly better than UCT on every smooth and
  independent size
```

Therefore a large path count is not a sufficient MCTS trigger. The router must
also test prefix-value predictiveness, deception risk, and available
coverage. Thresholds remain validation-selected rather than universal.
