# Gap Claim Verification

Date: `2026-07-20`

Purpose: decide whether a proposed research gap is valid, partially valid, or
unsupported after the evidence audit.

## Core DAG Path

```text
audited evidence
  -> direct coverage check
  -> near-miss analysis
  -> contradiction handling
  -> novelty boundary
  -> feasibility boundary
  -> verdict
  -> reframed gap
  -> next verification step
```

## Verdict Labels

```text
supported_gap:
  evidence suggests the gap is real within the stated scope

partial_gap:
  prior work covers part of the claim, but a narrower unresolved gap remains

already_solved:
  prior work substantially answers the claimed gap

wrong_framing:
  the issue is not a gap but a different problem, such as scale, evaluation,
  accessibility, or comparison quality

unverified:
  evidence is insufficient to make a confident claim
```

## Decision Nodes

```yaml
D1_direct_coverage:
  content_skill: identify sources that directly answer the gap claim
  action_skill: give them strongest weight

N1_near_miss_analysis:
  content_skill: identify close papers that differ in setting, metric, data, or assumption
  action_skill: turn near-misses into a precise novelty boundary

C1_contradiction_handling:
  content_skill: explain evidence that weakens the gap
  action_skill: update the verdict rather than arguing around it

B1_novelty_boundary:
  content_skill: state exactly what remains new
  action_skill: replace broad novelty with a narrow supported claim

F1_feasibility_boundary:
  content_skill: check whether the verified gap can be tested by available experiments
  action_skill: reject gaps that cannot support a concrete next step

V1_verdict:
  content_skill: choose supported_gap, partial_gap, already_solved, wrong_framing, or unverified
  action_skill: state confidence and evidence basis

R1_reframed_gap:
  content_skill: rewrite the gap into the strongest evidence-supported version
  action_skill: preserve novelty without exaggeration

X1_next_step:
  content_skill: name the next source check, experiment, or benchmark comparison
  action_skill: make verification actionable
```

## Good Output Shape

```text
original gap
verdict
confidence
evidence summary
what prior work already covers
what remains open
reframed gap
recommended experiment or source check
```

## Common Failures

```text
search only for supporting sources
ignore near-miss papers
claim novelty because wording differs
confuse engineering difficulty with research gap
state a strong gap when evidence is unavailable
```
