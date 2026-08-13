# Mamba-3 Blind Writer Protocol V2: Autonomous Strategy Discovery

Date: `2026-07-24`

Status: `implemented_as_adjacent_v2_pilot`

Implementation:

```text
../mamba3_autonomous_strategy_v2_20260724/
```

## Objective

Test whether a blind writer can propose and validate a new writing strategy,
not whether it can reproduce one expert abstract or exact expert DAG.

## Information Boundary

The writer receives the same public chip, style profile, method documents, its
own prior candidates, and outcome-level evaluator feedback. It must not receive
the hidden abstract, paper full text, copied expert wording, an exact missing
sentence, or a prescribed target graph.

## Separate Provenance

```text
factual output nodes:
  require public-chip evidence paths

self-inferred strategy nodes:
  require proposal origin, predicted effect, falsification test, blind rollout
  evidence, and later transfer evidence
```

No strategy node is rejected merely because it is absent from the expert
artifact.

## One Round

1. Retain at least three structurally or operationally distinct strategy DAGs.
2. Execute at least two surface-distinct abstracts under matched evidence and
   budget.
3. Evaluate factual fidelity independently of writing utility.
4. Return only functional feedback and pairwise preferences.
5. Let the writer perform its own root-cause analysis.
6. Require at least one `self_inferred` proposal or an explicit, testable
   explanation for proposing none.
7. Run a frozen paired comparison between the incumbent and each valid new
   proposal.
8. Update posterior mass; do not force collapse to one DAG.

## Reward

Primary components:

```text
evidence fidelity
functional role coverage
mechanism-result coherence
importance-weighted result selection
bounded claims
clarity and compression
blind pairwise preference
counterfactual contribution
```

Forbidden primary rewards:

```text
exact string match
n-gram overlap
matching expert sentence order
matching one reconstructed expert DAG
```

## Promotion

On this one paper, a successful invention is labelled only
`local_strategy_hypothesis`. Promotion to `reusable_skill` requires preregistered
blind replay on other papers that did not motivate the proposal.

## Stop Rule

Minimum rounds: `3`.

Stop only after two consecutive challenge rounds where:

```text
hard factual and leakage failures are zero
held-out or cross-candidate functional utility improves by less than epsilon
posterior or Pareto-front movement is below its calibrated threshold
no self-inferred candidate passes the local acceptance test
at least one round used a reserved novelty budget
```

Graph identity and abstract identity are not required. Multiple high-utility
strategies may remain.
