# Mamba-3 Autonomous Strategy Discovery V2: Final Report

Date: `2026-07-24`

Status: `completed_single_case_blind_v2_pilot`

## Outcome

A fresh writer LLM, blind to the Mamba-3 expert abstract and the earlier
strict run, autonomously proposed six writing-strategy hypotheses across three
rounds and produced nine surface-distinct, evidence-grounded abstracts.

The run did not optimize lexical similarity or one reference DAG.

## Round Trace

| Round | Autonomous behavior | Controller decision |
|---:|---|---|
| 1 | Proposed S1-S4 and realized three distinct organizations | no winner; S2 remained a candidate pending counterfactual |
| 2 | Diagnosed hierarchy and numeric competition; invented S5 | S5 accepted as a low-to-moderate-confidence local strategy hypothesis |
| 3 | Spent reserved novelty budget; invented non-cosmetic S6 and an atomic control | S6 also accepted locally; S5 and S6 retained as non-dominated |

### S5

`salience_anchor_then_selective_bindings`

Place a system-level comparison early, retain a small number of
mechanism-adjacent diagnostic anchors, and state the SISO/MIMO non-dominance.
Its realization was preferred to a control that preserved the central facts
but detached bindings and delayed the aggregate result. Because that
intervention changed a bundle of operations, confidence remains limited.

### S6

`operating_point_decision_split`

Organize the abstract around the practical configuration choice: SISO as the
reported faster operating point and MIMO as the quality-for-latency operating
point, while separating shared-backbone evidence from mode-specific evidence.

The round-3 treatment and control were both 187 words and differed in one
15-word sentence; all other words, facts, numbers, order, and boundaries were
held fixed. The controller preferred the explicit role map for configuration
recovery. S6 did not replace S5: it served a different function.

## Why This Corrects The Earlier Error

The earlier baseline stopped when its selected graph remained unchanged. In
this v2 pilot, reserved exploration produced a new non-cosmetic strategy in
the final round. Therefore:

```text
unchanged MAP graph != exhausted strategy discovery
supported new strategy != copied expert trace
different good abstract != error
```

Facts stayed constrained by chip provenance, while procedural strategies were
allowed to earn induction support through blind realization and
counterfactual comparison.

## Loop Interpretation

The first locally supported autonomous strategy appeared after two rounds.
The third round did not demonstrate convergence; it discovered another
non-dominated candidate. The run stopped because the preregistered pilot budget
ended, not because no more useful strategies exist.

## Strictness And Limitations

```text
writer:
  fresh context and explicit allowlist

writer declarations:
  no hidden abstract, paper full text, old strict run, repository search,
  Internet, or expert-similarity objective

controller:
  used only public-evidence functional criteria for v2 feedback

not available:
  OS-level audit proving every file read
  a blinded human preference panel
  an evaluator with no prior exposure to the expert target
  cross-paper transfer
  statistical acceptance or calibration
  a trained conditional GFlowNet
```

The controller had encountered the expert target during the earlier baseline,
although it did not use that target in v2 evaluation. Thus the writer is
strictly blind by task isolation and declaration, but the full writer-evaluator
system is not a perfect double-blind experiment.

## Defensible Claim

This pilot demonstrates autonomous proposal and local functional support for
two writing-strategy hypotheses on one paper. It does not establish that either
is a reusable skill. Cross-paper blind replay is required for promotion.
