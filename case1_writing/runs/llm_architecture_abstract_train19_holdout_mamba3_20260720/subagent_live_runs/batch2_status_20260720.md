# Subagent Heavy Simulation Batch 2

Date: `2026-07-20`

Purpose: continue the heavy DIRS abstract-writing simulation with Codex
subagents after the API fallback failed due to insufficient quota.

Each subagent ran one training-sample mini-cycle:

```text
select connected DAG -> generate abstract -> evaluate against training target
-> repair once if needed -> return JSON feedback for the shared DAG
```

Held-out private metadata remained forbidden.

## Completed Samples

| # | Chip | Initial | Final | Unsupported Claims | Main Repair |
|---:|---|---:|---:|---:|---|
| 5 | `ICLR2026_kmK3WSCOCT_markov_laplace_mamba_icl` | 0.86 | 0.94 | 0 | Focused mechanism on convolution/estimator structure and kept only central quantitative anchors. |
| 6 | `ICLR2026_5C3LljOEGC_hatsolver_hierarchical_attention_groebner` | 0.84 | 0.94 | 0 | Added hierarchical attention, cost-analysis framing, and scale anchors without table-level detail. |
| 7 | `ICLR2026_pN261iTKvr_learning_to_segment_vrp` | 0.88 | 0.94 | 0 | Strengthened solver-family compatibility and kept result anchors supported by chip facts. |
| 8 | `ICLR2026_TLSUIyBIfs_length_generalization_bounds` | 0.88 | 0.95 | 0 | Restored theorem-regime distinctions: error notion, precision regime, layer count, and complexity variables. |

## Batch 2 Takeaways

```text
mean_initial_score: 0.865
mean_final_score: 0.9425
samples_repaired: 4 / 4
unsupported_claims_after_repair: 0
```

Shared DAG feedback:

```text
M2 is important for theory-heavy and efficiency-heavy abstracts.
E3 should allow qualitative anchors when exact constants are not abstract-level material.
C1 remains optional; several theory abstracts begin directly from the problem.
S1 should preserve bounded theoretical or solver-family scope rather than broad deployment claims.
```

