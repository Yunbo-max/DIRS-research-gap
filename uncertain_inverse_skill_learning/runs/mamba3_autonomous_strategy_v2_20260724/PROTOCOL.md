# Mamba-3 Autonomous Strategy Discovery Pilot

Date: `2026-07-24`

Status: `single_case_blind_v2_pilot`

## Question

Can one fresh blind writer LLM invent and test a useful writing strategy from a
public paper chip without seeing or trying to reproduce the expert abstract?

## Writer Inputs

The writer may read only:

```text
iclr2026_oral_paper_memory_fresh_248h/chips/ICLR2026_HwCvaJOiCj_mamba3.chip.json
yunbo/DIRS/case1_writing/runs/llm_architecture_abstract_train19_holdout_mamba3_20260720/style_profile.json
yunbo/DIRS/uncertain_inverse_skill_learning/04_learning_and_search_algorithms.md
yunbo/DIRS/uncertain_inverse_skill_learning/09_learned_policy_vs_hard_constraints.md
yunbo/DIRS/uncertain_inverse_skill_learning/10_autonomous_strategy_discovery.md
this protocol
prior-round files and controller feedback inside this run
```

## Forbidden Inputs

```text
the hidden Mamba-3 abstract
the Mamba-3 paper or OpenReview full text
the earlier mamba3_strict_writer_loop_20260724 run
other stored or generated Mamba-3 abstracts
repository-wide search
Internet search
```

The writer logs every file read.

## Round Outputs

Each `round_NN` contains:

```text
strategy_candidates.json
variants.md
selected_strategy.json
selected_abstract.md
writer_self_check.json
```

At least three strategy candidates and two surface-distinct abstracts are
required. Every strategy declares:

```text
proposal origin
operation contract
predicted effect
failure mode
falsification test
content-evidence boundary
status
```

## Evaluation Boundary

The controller does not use the hidden expert abstract. It evaluates:

```text
factual support against the public chip
functional role coverage
mechanism-result coherence
importance-weighted selection
bounded claims
clarity and compression
whether the proposed strategy has a measurable counterfactual effect
```

It may give outcome-level feedback but cannot prescribe a strategy edit.

## Pilot Schedule

```text
round 1:
  unconstrained autonomous proposals and variants

round 2:
  writer diagnoses functional feedback and proposes its own edit or replacement

round 3:
  reserved-novelty challenge plus an explicit counterfactual ablation
```

Graph or wording identity is not a stopping target.

## Acceptance

On this one paper, a proposal is accepted only as
`local_strategy_hypothesis` when:

```text
hard factual/leakage failures are zero
its selected realization is preferred under the functional rubric
the writer and controller can state a falsifiable counterfactual
the gain is not based on exact expert wording
```

No proposal may be called a reusable skill until tested on separate papers.

## Claim Boundary

This pilot cannot establish general learning, cross-paper transfer, calibrated
posterior convergence, or historical author-trace recovery.
