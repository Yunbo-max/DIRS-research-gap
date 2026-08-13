# Ablation And Metric Protocol

Date: `2026-07-20`

Purpose: design the evidence grid that makes an experiment persuasive rather
than merely runnable.

## Core DAG Path

```text
main claim
  -> decomposition into mechanisms
  -> component ablations
  -> control ablations
  -> metric binding
  -> statistical reporting
  -> diagnostic slices
  -> cost and robustness checks
  -> interpretation rules
```

## Required Skill Nodes

```yaml
A1_mechanism_decomposition:
  content_skill: identify which components or assumptions drive the main claim
  action_skill: separate mechanism tests from leaderboard tests

A2_component_ablation:
  content_skill: remove or replace one component at a time
  action_skill: make each ablation answer one causal question

A3_control_ablation:
  content_skill: include sanity checks, randomization, oracle, or trivial baselines
  action_skill: prevent a strong number from hiding an invalid protocol

M1_primary_metric:
  content_skill: select one metric that directly measures success
  action_skill: define it before reporting any result

M2_secondary_diagnostics:
  content_skill: select metrics that explain why the primary metric changes
  action_skill: keep them diagnostic rather than replacing the main claim

S1_statistical_reporting:
  content_skill: specify seeds, confidence intervals, variance, or significance
  action_skill: state uncertainty in the result table plan

L1_slice_analysis:
  content_skill: identify subsets where the method should help or fail
  action_skill: bind slices to the claimed mechanism

C1_cost_check:
  content_skill: measure runtime, memory, data, annotation, or API cost when relevant
  action_skill: report cost beside quality when cost is part of the claim
  tool_skill: use logs, nvidia-smi, timing output, API call counts, or cost estimates when available

T1_tool_cost_attribution:
  content_skill: separate scientific method cost from infrastructure or API overhead
  action_skill: report what was measured, estimated, or left unknown
  tool_skill: inspect run logs, usage records, cached call counts, and hardware state without exposing secrets

I1_interpretation_rule:
  content_skill: predefine how each ablation outcome changes the claim
  action_skill: avoid post-hoc explanation after seeing results
  tool_skill: use verifier and run logs to distinguish metric failure from execution failure
```

## Metric Binding Rule

```text
Each metric must answer one of:
  Does the method work?
  Why does it work?
  When does it fail?
  What does it cost?
```

If a metric answers none of these, reject it or move it to an appendix.

## Good Output Shape

```text
claim tested
primary result table
ablation table
diagnostic slice table
cost table
tool/resource table
interpretation matrix
known limitations
```

## Common Failures

```text
result before metric
too many metrics with no claim binding
ablation removes several components at once
baseline chosen only because it is easy
no negative or sanity-control condition
cost omitted despite an efficiency claim
API/GPU feasibility assumed without a check
tool failure misread as scientific failure
```
