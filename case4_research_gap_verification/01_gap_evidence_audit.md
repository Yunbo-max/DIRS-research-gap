# Gap Evidence Audit

Date: `2026-07-20`

Purpose: collect and organize evidence relevant to a proposed research gap.

## Core DAG Path

```text
gap claim
  -> claim scope
  -> search questions
  -> source retrieval
  -> source screening
  -> evidence table
  -> contradiction list
  -> coverage summary
  -> uncertainty label
```

## Node Properties

```yaml
G1_gap_claim:
  content_skill: state the proposed gap exactly
  action_skill: separate the gap from the proposed solution

S1_scope_boundary:
  content_skill: define domain, task, model class, dataset, metric, and time window
  action_skill: narrow before searching

Q1_search_questions:
  content_skill: generate queries that could disprove the gap
  action_skill: search for counterevidence, not only support

R1_source_retrieval:
  content_skill: collect papers, code, benchmarks, docs, and leaderboards
  action_skill: prefer primary sources and official benchmark pages

F1_source_screening:
  content_skill: decide whether each source directly addresses, partially addresses, or misses the gap
  action_skill: quote or summarize only the relevant claim

T1_evidence_table:
  content_skill: align source coverage against the gap dimensions
  action_skill: make comparison inspectable

C1_contradiction_list:
  content_skill: record sources that weaken or refute the gap
  action_skill: do not hide inconvenient evidence

U1_uncertainty_label:
  content_skill: mark evidence as strong, partial, weak, or unavailable
  action_skill: distinguish unknown from false
```

## Evidence Table Shape

```text
source
year/date
claim or contribution
task/domain covered
method/evidence
does it address the gap?
remaining difference
confidence
```

## Verifier

```text
gap claim is scoped
search questions include disconfirmation queries
at least one source is checked against each scope dimension
near-miss sources are recorded
unsupported claims are labeled unknown
```
