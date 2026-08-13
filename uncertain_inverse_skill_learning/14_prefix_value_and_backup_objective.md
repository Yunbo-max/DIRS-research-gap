# Prefix Value Reliability And Backup Objective

## 1. The Hidden Assumption In DAG-MCTS

Tree search helps only when observations from a partial path predict useful
properties of its descendants. Connectivity alone does not provide this
signal.

For prefix \(s\), distinguish:

\[
\mu(s)
=
\mathbb E_{\pi\sim\rho(\cdot\mid s)}[U(\pi)]
\]

from:

\[
M(s)
=
\max_{\pi\succeq s} U(\pi).
\]

Mean-backup MCTS estimates \(\mu(s)\) under its rollout policy. Best-path
identification cares about \(M(s)\). A low-mean branch can still contain the
globally best path.

## 2. Why Raw Maximum Backup Is Not Enough

Replacing the mean by the maximum observed reward changes the objective but
introduces winner's-curse bias:

```text
more visits create more chances for a positive noise outlier
heteroscedastic writer/evaluator noise makes branches incomparable
one unsupported high-scoring artifact can dominate the backup
hard failures can be hidden by a single lucky rollout
```

The 2026-07-25 deceptive-tree test found that raw max backup did not reliably
recover the hidden best leaf and was sometimes worse than mean backup.

## 3. Required Prefix-Value Audit

Before enabling MCTS on a domain, estimate on non-target validation tasks:

```text
prefix mean predictiveness:
  association between early prefix estimates and descendant mean utility

prefix optimum predictiveness:
  association between prefix features/posteriors and best descendant utility

deception rate:
  frequency with which the best leaf lies under a below-median prefix

calibration:
  coverage of posterior intervals for descendant mean and maximum

cost reuse:
  computation saved when descendants share an executed prefix
```

If prefix optimum predictiveness is weak or deception rate is high, a tree
policy based on average rewards is not justified.

## 4. Better Backup Targets

Candidate learned backup targets include:

```text
posterior probability that the subtree contains the best feasible path
upper credible bound on best descendant utility
top-quantile descendant value
noise-corrected extreme-value posterior
risk-constrained probability of exceeding a deployment threshold
```

These are hypotheses to validate, not hard-coded universal rewards.

## 5. Coverage Safeguards

When deceptive branches are possible:

```text
reserve a minimum coverage budget across root/major branches
use root sampling over graph/value posterior uncertainty
apply Thompson-style branch exploration
do not prune solely from a low early sample mean
require confidence-based elimination
retain rollback and replay for pruned structures
```

Coverage rules must be calibrated to cost and risk. They exist to prevent an
uncertain controller from mistaking lack of evidence for negative evidence.

## 6. Updated Search Decision

```text
enumerable paths + adequate budget:
  pure-exploration best-arm identification

implicit paths + reliable smooth prefix signal:
  MCTS/UCT

implicit paths + uncertain prefix signal:
  posterior sampling plus protected coverage

deceptive or adversarial prefix signal:
  do not use mean-backup MCTS as the primary optimizer
```

## 7. Real-Trace Audit Result

The 2026-07-25 audit joined 19 saved writing paths with their initial and
repaired evaluator scores. Shared-prefix similarity was weak and the
leave-one-out path-mean advantage was not robust under paired bootstrap or
leave-one-batch-out checks.

The current real trace set contains one realized path per paper and therefore
cannot estimate within-paper counterfactual prefix value or deception rate.
DIRS must keep real-task MCTS disabled until balanced multi-path rollouts pass
the protocol in `15_real_counterfactual_rollout_protocol.md`.
