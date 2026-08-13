# Mamba-3 Strict Writer Baseline: Final Report

Date: `2026-07-24`

Status: `completed_over_constrained_baseline`

## Observed Result

Under the original preregistered protocol, the blind writer reached its local
stop rule at loop 5:

| Loop | Overall score | Delta | Structural change |
|---:|---:|---:|---|
| 1 | 0.875 | -- | initial candidate |
| 2 | 0.937 | +0.062 | add comparative MIMO quality strategy; prune detail |
| 3 | 0.959 | +0.022 | add compact qualitative capability validation |
| 4 | 0.966 | +0.007 | no graph edit; wording-only compression |
| 5 | 0.966 | +0.000 | no edit; exact confirmation |

For transitions 3-to-4 and 4-to-5, hard failures were zero, the typed graph
was unchanged, no structural edit was accepted, and score improvement was
below 0.01. The loop-5 abstract contains 183 words and is byte-identical to
loop 4. The controller independently verified identical typed graph content
and posterior weights.

## Useful Strictness Signal

In loop 2, feedback suggested a matched-quality state-size result, but the
public chip did not contain sufficient outcome evidence. The writer rejected
that factual addition rather than inventing it. This demonstrates the desired
factual evidence boundary.

## Methodological Correction

This is not evidence that strategy learning converged. The original protocol
over-constrained late-loop discovery:

```text
valid:
  facts in an abstract require evidence

invalid:
  every writing strategy must already be visible in the expert artifact
  or graph identity must be the final learning objective
```

A writer should be allowed to invent new selection, organization, compression,
verification, and repair strategies. A different abstract and a different DAG
may be equally good or better. New strategies should be judged by blind
functional utility, interventions, and transfer, not exact expert wording.

This run is therefore labelled an
`evidence_constrained_expert_feedback_assisted_local_stabilization` baseline.
It did not test autonomous post-plateau strategy discovery.

## Successor

The corrected protocol is
`PROTOCOL_V2_AUTONOMOUS_STRATEGY.md`. The general rule is specified in
`../../10_autonomous_strategy_discovery.md`.

Under v2, a one-paper improvement is only a `local_strategy_hypothesis`.
Promotion to `reusable_skill` requires blind improvement on separate papers.

## Claim Boundary

This run does not establish:

```text
recovery of the historical author process
one correct expert DAG
general DIRS convergence
autonomous strategy discovery
conditional GFlowNet convergence
cross-paper skill transfer
```
