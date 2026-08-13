# Subagent Heavy Simulation Batch 1

Date: `2026-07-20`

Purpose: continue the heavy DIRS simulation after the OpenAI API fallback hit
quota. Each subagent performs one training-sample mini-cycle:

```text
select connected DAG -> generate abstract -> evaluate against training target
-> repair once if needed -> return JSON
```

Held-out private file remains forbidden.

## Completed Samples

| # | Chip | Initial | Final | Unsupported Claims | Main Repair |
|---:|---|---:|---:|---:|---|
| 1 | `CVPR2026_069_mdcs_moame_survival_prediction` | 0.86 | 0.94 | 0 | Added explicit scan directions, region/patch/gene levels, c-index and compute anchors. |
| 2 | `CVPR2026_136_segmote_token_moe_medical_segmentation` | 0.88 | 0.96 | 0 | Restored two bottlenecks, zero-shot/generalization framing, and removed excessive baseline inventory. |
| 3 | `ICLR2026_oZJFY2BQt2_tech_cotar_medts` | 0.88 | 0.95 | 0 | Foregrounded CoTAR, added APAVA 11.6% anchor, and separated mechanism/efficiency. |
| 4 | `ICLR2026_ZBj3Qp1bYg_ebt` | 0.86 | 0.94 | 0 | Added unsupervised-thinking question and 35%, 29%, 99% quantitative anchors. |

## Batch 1 Takeaways

```text
mean_initial_score: 0.87
mean_final_score: 0.9475
samples_repaired: 4 / 4
unsupported_claims_after_repair: 0
```

Shared DAG feedback:

```text
R1 -> G1 -> O1 -> M1 is stable across the batch.
M2 should remain optional but important for efficiency/theory-heavy chips.
E3 is valuable when exact numbers exist; it strongly improves alignment.
C1 should stay optional: useful for EBT/System-2 framing, unnecessary for many method abstracts.
```
