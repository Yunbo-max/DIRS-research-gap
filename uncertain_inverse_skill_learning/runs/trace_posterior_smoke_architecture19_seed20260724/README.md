# Uncertain Inverse Trace Optimization Smoke Test

Evidence status: `smoke_only_synthetic_proposal_noise`

This run does **not** establish method convergence or recovery of historical
expert behavior. It checks whether multi-hypothesis posterior aggregation and
validation-tuned hyperparameters work on controlled noisy versions of existing
DIRS proxy traces.

## Data

```text
source traces: 19
train: 11
validation: 4
test: 4
hypotheses per case: 12
seed: 20260724
```

## Optimized Hyperparameters

```json
{
  "temperature": 0.06,
  "complexity_penalty": 0.0,
  "graph_weight": 0.5,
  "validation_objective": 0.92509,
  "search_boundary_hits": [
    "complexity_penalty",
    "graph_weight"
  ]
}
```

## Held-Out Test

| Metric | Single hypothesis | Optimized multi-hypothesis | Delta |
|---|---:|---:|---:|
| MAP node F1 | 0.773529 | 0.964130 | +0.190601 |
| Expected node F1 | 0.773529 | 0.949497 | +0.175968 |
| Node Brier (lower is better) | 0.333333 | 0.030611 | -0.302722 |

## Interpretation Rule

Promote this only as a code-path smoke test. A real test must replace synthetic
proposals with independent LLM trace hypotheses, use artifact/tool evidence,
execute selected sub-DAGs blindly, and evaluate on partial-gold process traces
or expert outcomes.
