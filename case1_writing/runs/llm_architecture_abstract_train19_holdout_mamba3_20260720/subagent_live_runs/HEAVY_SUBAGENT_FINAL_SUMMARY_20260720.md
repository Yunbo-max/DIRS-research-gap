# Heavy Subagent Final Summary

Date: `2026-07-20`

Run:
`/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_architecture_abstract_train19_holdout_mamba3_20260720`

## Result

```text
training_samples_completed: 19 / 19
subagent_batches_completed: 5
macro_mean_final_score: 0.95666
unsupported_claims_after_repair: 0
holdout_private_file_opened: no
```

## Why The Run Used Subagents

The API-based fallback runner failed with an insufficient-quota error, so the
heavy pass was continued with Codex subagents. Each sample used the intended
dual-system loop:

```text
Loop 1 editor:
  chip + current shared DAG + evaluator feedback -> connected sample DAG

Loop 2 simulator:
  selected connected DAG + chip facts -> generated abstract

Evaluator:
  generated abstract + training reference/chip facts -> feedback and repair
```

## Main Learned Updates

```text
1. Keep a no-E3 route for concise, theory-heavy, or mechanistic abstracts:
   R1 -> G1 -> O1 -> M1 -> M2 -> E1 -> E2 -> I1 -> S1 -> P1

2. Interpret E3 as evidence_anchor, not only quantitative_anchor.
   It can carry a theorem scope, benchmark count, stage decomposition, or one
   compact result number.

3. Keep M2 high-priority for theory/mechanism papers.
   It often carries the actual contribution, such as controllability,
   predictor-corrector theory, WFA recurrence, KDE-attention equivalence, or
   numerical failure mechanism.

4. Use C1 only when the abstract needs broad domain context.
   It helps for nonstandard domains or broad Transformer-understanding gaps,
   but can be skipped when the problem-gap sentence already carries context.

5. Make S1 an active overclaim filter.
   It should remove unsupported deployment, large-scale LLM, speed, universality,
   or future-work claims unless the training abstract itself makes them central.
```

## Batch Files

```text
batch1_status_20260720.md
batch2_status_20260720.md
batch3_status_20260720.md
batch4_status_20260720.md
batch5_status_20260720.md
HEAVY_SUBAGENT_PROGRESS.md
```

