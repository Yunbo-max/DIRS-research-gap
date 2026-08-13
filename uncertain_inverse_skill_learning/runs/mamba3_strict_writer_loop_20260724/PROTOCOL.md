# Mamba-3 Strict LLM Writer Loop Protocol

Date: `2026-07-24`

Evidence status: `single_case_strict_llm_simulation`

This run tests how many feedback loops a blind writer LLM needs to stabilize a
local artifact-compatible trace and abstract for one paper. It does not
establish convergence of the cross-paper shared skill graph.

## Public Inputs For Writer

```text
../../../../iclr2026_oral_paper_memory_fresh_248h/chips/ICLR2026_HwCvaJOiCj_mamba3.chip.json
../../../../yunbo/DIRS/case1_writing/runs/llm_architecture_abstract_train19_holdout_mamba3_20260720/style_profile.json
../../04_learning_and_search_algorithms.md
../../09_learned_policy_vs_hard_constraints.md
```

Paths above are descriptive relative paths from the repository root; the
writer receives explicit repository-root paths in its task.

## Forbidden Inputs For Writer

```text
yunbo/DIRS/case1_writing/runs/llm_architecture_abstract_train19_holdout_mamba3_20260720/holdout_private_after_generation.json
iclr2026_oral_paper_memory_fresh_248h/text/ICLR2026_HwCvaJOiCj_openreview.txt
any existing abstract or generated answer for Mamba-3
```

The writer must not search the repository or Internet for Mamba-3. It may read
only the explicitly allowed files and feedback written into this run folder.

## Per-Loop Writer Output

```text
loop_NN/trace_hypotheses.json
loop_NN/selected_subdag.json
loop_NN/abstract.md
loop_NN/writer_self_check.json
```

At least three structurally distinct trace hypotheses must be retained until
feedback makes alternatives operationally dominated.

## Evaluator Boundary

The main controller may reveal the expert abstract only after loop 1 has been
saved. It returns structured feedback without copying expert phrases:

```text
representation errors
selection errors
execution errors
evidence errors
unsupported claims
missing artifact functions
quality subscores
```

## Stop Rule

Minimum loops: `3`

Maximum loops: `8`

Stop at the first loop where all are true for two consecutive transitions:

```text
no hard evidence or leakage failure
selected typed node/edge signature unchanged or operationally equivalent
no accepted structural graph edit
total evaluator score improves by less than 0.01
all remaining feedback is wording-only
```

The loop count is an observed result. It must not be forced to match a desired
number.

## Claim Boundary

Allowed:

```text
local trace/output stabilization for this one strict LLM writer case
```

Not allowed:

```text
general DIRS convergence
recovery of the historical author process
conditional GFlowNet training convergence
cross-paper skill learning
```
