# Post-hoc Prior-Smoothing Diagnostic

Date: `2026-07-25`

Status: `post_hoc_failure_diagnosis_not_confirmatory_result`

## Motivation

The preregistered stochastic simulation showed that the standard
most-visited MCTS recommendation remained dominated by the empirical training
frequency prior. This diagnostic tests whether softening that prior changes
exploration. It was designed after observing the failure and therefore must
not be reported as a preregistered improvement.

## Controlled Change

At every legal prefix, replace the raw empirical action prior with:

\[
P_\eta(a\mid s)
=
(1-\eta)P_{\mathrm{freq}}(a\mid s)
+\eta P_{\mathrm{uniform}}(a\mid\mathcal F(s)).
\]

Sensitivity values:

```text
eta = 0.00, 0.25, 0.50, 1.00
```

All hard DAG-flow constraints remain unchanged. Only actions in the legal
frontier can receive probability.

For a focused diagnostic, use:

```text
epistemic sigma:
  0.02, 0.05

rollout sigma:
  0.03

episodes:
  300 per scenario and eta value

budgets:
  12, 24, 48, 96, 192
```

## Interpretation Rule

If prior smoothing helps, it shows that raw training frequency was
miscalibrated for this synthetic contextual shift. It does not establish that
one fixed smoothing coefficient is generally optimal. A real implementation
should learn or validation-select contextual priors on training/validation
tasks and keep the final held-out paper untouched.
