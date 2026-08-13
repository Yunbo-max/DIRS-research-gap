# Strict 19-Abstract Subagent Learning Report

Date: `2026-07-20`

Run:
`/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_architecture_abstract_train19_holdout_mamba3_20260720`

Domain:
`LLM Architecture / Attention / State Space Models`

## Runtime

This strict pass uses the DIRS runtime requested for heavy work:

```text
main Codex coordinator
  -> editor subagent
  -> simulator subagent
  -> evaluator subagent
  -> feedback to editor
```

The hosted API runner was not used as the execution engine. The old API script
is now guarded and requires explicit opt-in through `DIRS_ALLOW_OPENAI_API=1`.

## Completion

```text
training_samples: 19 / 19
subagent_batches: 5
independent_audit_subagents: 3
deterministic_preflight: converged at loop 6
MCTS_rollouts_per_example: 500
heldout_private_file_opened: no
active_subagents_remaining: none
```

## Score Summary

```text
mean_initial_score: 0.89000
mean_final_score: 0.95563
mean_improvement: 0.06563
min_final_score: 0.94
max_final_score: 0.98
unsupported_claims_after_repair: 0
```

Batch means:

```text
batch1 samples 1-4:   0.9475
batch2 samples 5-8:   0.9425
batch3 samples 9-12:  0.9625
batch4 samples 13-16: 0.96575
batch5 samples 17-19: 0.96133
```

## Auditor Verdict

Three read-only auditors checked the run without reading the held-out private
file.

```text
auditor_1 samples 1-7:  pass
auditor_2 samples 8-14: pass
auditor_3 samples 15-19 plus global status: pass_with_warnings
```

The warning is important: the batch files preserve summary-level
generation/evaluation/repair evidence, but not complete raw evaluator
transcripts for every sample. The next strict run should store per-sample JSON
transcripts in addition to markdown summaries.

## Learned Graph Policy

The strict pass confirms a stable abstract spine:

```text
R1_abstract_as_argument
  -> G1_problem_gap
  -> O1_named_method_or_object
  -> M1_architecture_or_mechanism
  -> E1_evaluation_setup
  -> E2_result_outcome
  -> I1_interpretation_or_tradeoff
  -> S1_bounded_takeaway
  -> P1_length_and_placement_prior
```

Optional branches:

```text
context branch:
  G1_problem_gap -> C1_domain_context -> O1_named_method_or_object

mechanism branch:
  M1_architecture_or_mechanism
    -> M2_efficiency_or_theory_detail
    -> E1_evaluation_setup

evidence branch:
  E2_result_outcome
    -> E3_evidence_anchor
    -> I1_interpretation_or_tradeoff

no-E3 branch:
  E2_result_outcome -> I1_interpretation_or_tradeoff
```

## Node Rules

```yaml
required_spine_nodes:
  - R1_abstract_as_argument
  - G1_problem_gap
  - O1_named_method_or_object
  - M1_architecture_or_mechanism
  - E1_evaluation_setup
  - E2_result_outcome
  - I1_interpretation_or_tradeoff
  - S1_bounded_takeaway
  - P1_length_and_placement_prior

C1_domain_context:
  select_when:
    - broad domain context is needed before naming the method
    - paper is outside the domain core
    - reader needs nonstandard task framing
  reject_when:
    - problem-gap sentence already provides enough context
    - adding context delays the named contribution

M2_efficiency_or_theory_detail:
  select_when:
    - theory, proof, bound, estimator, recurrence, efficiency, complexity,
      causal mechanism, numerical failure, or architecture-internal detail is
      central to the contribution
  reject_when:
    - it would add implementation detail without changing the abstract argument

E3_evidence_anchor:
  replaces_or_aliases: E3_quantitative_anchor
  allowed_forms:
    - one compact numeric result
    - theorem or regime scope
    - benchmark count or task breadth
    - stage decomposition
    - qualitative validation anchor
  reject_when:
    - it creates false precision
    - it turns the abstract into a result table
    - the chip does not support the claim

S1_bounded_takeaway:
  role: active overclaim filter
  remove_or_repair:
    - unsupported deployment claims
    - large-scale LLM readiness claims
    - speed or efficiency claims without measured support
    - universality claims beyond the training/evaluation family
    - future-work claims not present as paper contributions
```

## MCTS Selection Updates

```text
1. Start from the required spine, not the full graph.
2. Apply archetype gates before rollout:
   theory/formal
   mechanism/efficiency
   benchmark/system
   broad-context/nonstandard
3. Add C1 only when the chip needs context before the object.
4. Add M2 when the central contribution depends on theory/mechanism/efficiency.
5. Add E3 only when a compact supported evidence anchor improves the abstract.
6. Keep the low-support direct edges as valid escape routes:
   M1 -> E1
   E2 -> I1
7. Penalize table-like result dumps, unsupported numbers, and broad S1 claims.
```

## Strict Convergence Interpretation

The deterministic preflight converged, and the subagent pass repaired all 19
training samples to high final scores. The graph should be considered stable
for this domain/section pair, with one naming repair:

```text
rename or alias E3_quantitative_anchor -> E3_evidence_anchor
```

This is not a claim that every generated abstract equals the author abstract.
The learned claim is narrower and stronger:

```text
DIRS can select a connected, evidence-supported abstract DAG from chip facts,
generate a role-compatible abstract, evaluate failure modes, and repair the
path without using the held-out target.
```

## Next Strict Run Requirement

For the next run, save one JSON transcript per sample:

```text
sample_id.json:
  selected_nodes
  selected_edges
  generated_initial
  evaluator_report_initial
  repaired_nodes_or_constraints
  generated_final
  evaluator_report_final
  shared_dag_feedback
```

That will remove the main audit warning.

