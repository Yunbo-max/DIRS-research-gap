# Subagent Heavy Simulation Batch 4

Date: `2026-07-20`

Purpose: continue the heavy DIRS abstract-writing simulation with Codex
subagents.

Each subagent ran:

```text
Loop 1 editor DAG selection -> Loop 2 simulator abstract -> evaluator feedback
-> one repair -> shared DAG feedback
```

Held-out private metadata remained forbidden.

## Completed Samples

| # | Chip | Initial | Final | Unsupported Claims After Repair | Main Repair |
|---:|---|---:|---:|---:|---|
| 13 | `ICML2026_71127_coevol_no_neural_operator` | 0.90 | 0.968 | 0 | Reduced result-table detail and used benchmark-count anchors for the neural-operator result. |
| 14 | `ICML2026_71039_discoformer_density_score_transformers` | 0.91 | 0.96 | 0 | Added sample-size/mode generalization and sharpened the attention-as-KDE theory claim. |
| 15 | `ICML2026_71192_context_parameter_updates` | 0.92 | 0.97 | 0 | Removed speculative prompt-to-adapter transfer and low-precision detail; kept exact context-as-weight-patch theory. |
| 16 | `ICML2026_71193_focus_dilution_attention` | 0.90 | 0.965 | 0 | Reduced diagnostic-metric detail and restored the stage-wise focus-dilution mechanism. |

## Batch 4 Takeaways

```text
mean_initial_score: 0.9075
mean_final_score: 0.96575
samples_repaired: 4 / 4
unsupported_claims_after_repair: 0
```

Shared DAG feedback:

```text
C1 is useful for nonstandard or broad-context papers: density/score estimation,
modern block variants, or broad Transformer-understanding gaps.

M2 is not a minor detail for theory abstracts. It often carries the central
contribution: predictor-corrector theory, KDE-attention equivalence,
controllability, or stage-wise gradient-flow analysis.

E3 should be renamed or interpreted as evidence_anchor, not only
quantitative_anchor. For these papers it can carry benchmark counts, theorem
scope, near-perfect matching, or stage decomposition.

S1 must suppress speculative transfer claims unless the original abstract makes
them part of the contribution.
```

