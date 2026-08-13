# Subagent Heavy Simulation Batch 5

Date: `2026-07-20`

Purpose: finish the 19-sample heavy DIRS abstract-writing simulation with Codex
subagents.

Each subagent ran:

```text
Loop 1 editor DAG selection -> Loop 2 simulator abstract -> evaluator feedback
-> one repair -> shared DAG feedback
```

Held-out private metadata remained forbidden.

## Completed Samples

| # | Chip | Initial | Final | Unsupported Claims After Repair | Main Repair |
|---:|---|---:|---:|---:|---|
| 17 | `ICML2026_71133_rational_transductors` | 0.88 | 0.96 | 0 | Removed large-scale-LLM overclaim and kept WFA/parallel-scan theory plus synthetic validation. |
| 18 | `ICML2026_71098_secnet_event_cloud` | 0.91 | 0.96 | 0 | Tightened to the original qualitative evaluation breadth and omitted `E3`. |
| 19 | `ICLR2026_0jHyEKHDyx_low_precision_flash_attention_failure` | 0.91 | 0.964 | 0 | Removed unsupported speed-preservation claim and kept the arithmetic-causal failure explanation. |

## Batch 5 Takeaways

```text
mean_initial_score: 0.90
mean_final_score: 0.96133
samples_repaired: 3 / 3
unsupported_claims_after_repair: 0
```

Shared DAG feedback:

```text
The graph needs a clean no-E3 route:
R1 -> G1 -> O1 -> M1 -> M2 -> E1 -> E2 -> I1 -> S1 -> P1

This route fits theory-heavy, mechanistic, or concise abstracts whose original
style reports qualitative success rather than a headline number.

M2 is still essential on this no-E3 route because it carries the causal,
theoretical, or efficiency mechanism that makes the result interpretable.

S1 should actively remove broad readiness claims, such as implying large-scale
LLM deployment, universal event-vision performance, or measured speed retention
when the chip only supports a narrower claim.
```

