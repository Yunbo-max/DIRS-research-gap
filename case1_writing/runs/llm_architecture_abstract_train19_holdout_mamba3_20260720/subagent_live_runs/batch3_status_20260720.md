# Subagent Heavy Simulation Batch 3

Date: `2026-07-20`

Purpose: continue the heavy DIRS abstract-writing simulation with Codex
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
| 9 | `ICLR2026_FdkPOHlChS_softmax_transformers_turing_complete` | 0.95 | 0.98 | 0 | Removed excess implementation detail and skipped `E3` because the theorem abstract is qualitative/formal. |
| 10 | `CVPR2026_113_sat_structural_action_transformer` | 0.93 | 0.96 | 0 | Reduced table-level detail, kept one compact result anchor, and restored the structural action-token argument. |
| 11 | `ICLR2026_sSfep4udCb_tool_use_length_generalization_ssm` | 0.88 | 0.95 | 0 | Added supported train-to-test extrapolation anchors and clarified the theory/tool-use contrast. |
| 12 | `ICML2026_71083_any_order_gpt_mdm` | 0.88 | 0.96 | 0 | Added the temperature-annealing caveat and kept the formulation/architecture decoupling argument. |

## Batch 3 Takeaways

```text
mean_initial_score: 0.91
mean_final_score: 0.9625
samples_repaired: 4 / 4
unsupported_claims_after_repair: 0
```

Shared DAG feedback:

```text
Theory-heavy abstracts can use E2 -> I1 directly and skip E3 when the original
abstract style is theorem-scope rather than numeric-result focused.

M2 is high value when mechanism detail is inseparable from the contribution:
CoT C-RASP/RPE, SAT structural tokenization, tool-use SSM theory, or AO-GPT
target-position/cache-compatible generation.

E3 remains optional, not mandatory. Use one compact number when it improves the
abstract; reject it when it turns the abstract into a result table.

Risk caveats should be attached to the result sentence, not buried at the end:
for example temperature annealing in AO-GPT, code/reproducibility gaps in SAT,
or no-tool/single-turn failures in tool-use SSMs.
```

