# Retrospective: Over-Constrained Strategy Learning

Date: `2026-07-24`

The original strict protocol remains unchanged as an auditable baseline, but
its interpretation is narrowed.

## Identified Design Error

The run correctly isolated the writer from the hidden Mamba-3 abstract and
prevented unsupported factual claims. However, its late-loop freeze treated
unchanged graph structure as the desired endpoint and let expert-relative
feedback identify missing artifact functions. This is too conservative for
autonomous skill learning.

The intended distinction is:

```text
paper facts:
  must be supported by the public chip

writing strategies:
  may be newly inferred by the writer and accepted through blind functional
  improvement, intervention, and transfer tests
```

The generated abstract need not reproduce the expert abstract. Exact wording,
sentence order, and one target DAG are not ground truth.

## Valid Conclusion From This Run

This run measures local stabilization of an evidence-constrained,
expert-feedback-assisted writer. It does not test whether the writer can
autonomously discover new strategies after apparent convergence.

## Required Successor

Use `PROTOCOL_V2_AUTONOMOUS_STRATEGY.md`. The successor reserves exploration
budget for self-inferred strategy proposals, gives evaluators only functional
feedback, excludes lexical matching from the objective, and distinguishes a
one-case local strategy hypothesis from a cross-case reusable skill.
