# DIRS Case 4: Research Gap Verification

Date: `2026-07-20`

Purpose: task-specific protocol for applying DIRS to verify whether a proposed
research gap is real.

This case learns the human skill of checking a gap claim before building a
paper around it. The output is a verdict with evidence: supported gap, partial
gap, already-solved gap, wrong framing, or unverifiable gap.

## Current Targets

```text
01_gap_evidence_audit.md
02_gap_claim_verification.md
```

## Gap-Verification Specialization

```text
node = gap claim + prior-art evidence + contradiction check + verdict unit
edge = dependency from claim scope to search to evidence to decision
simulator = literature/source/tool search and comparison
verifier = source coverage, contradiction handling, novelty boundary, and claim strength
```

For gap verification, the two DIRS systems are:

```text
evidence system:
  target gap claim, papers, chips, citations, benchmarks, code, leaderboards,
  source spans, dates, and negative evidence

action system:
  search strategy, comparison table, contradiction handling, novelty boundary,
  verdict language, and next-step recommendation
```

## Tool-Calling Role

Gap verification usually requires tools:

```text
search:
  find prior work that might already solve the claimed gap

source reading:
  inspect abstracts, methods, experiments, appendices, code, and benchmark pages

comparison:
  align the claimed gap against what each source actually covers
```

If the system cannot access enough evidence, it should return `unverified`, not
pretend the gap is real.

## Expected Output

```text
gap claim
scope boundary
search protocol
candidate prior work
evidence table
contradictions or near-misses
novelty boundary
verdict
recommended reframing
next verification step
```

## Quality Standard

A good gap-verification output protects the research process from false novelty.
It should be willing to say that the original gap is weak, already solved, or
too broad, then propose a narrower gap that the evidence can actually support.
