# Abstract Writing Application

Date: `2026-07-20`

Purpose: apply the trained dual-system MCTS method to write an abstract from a
paper chip without using the original abstract during generation.

## Inputs

Required:

```text
target paper chip
selected domain file
domain node/edge support priors
domain style profile
target venue or paper type, if known
```

Optional after generation:

```text
original abstract for comparison only
```

## Procedure

```text
1. Read the chip.
2. Infer the paper signature.
3. Select the semantic domain from the domain split.
4. Estimate target length from domain style priors.
5. Score compatible abstract nodes.
6. Run deterministic MCTS replay as a preflight.
7. Run the heavy dual-system subagent simulation by default:
   editor subagent -> simulator subagent -> evaluator subagent -> feedback
   -> editor subagent.
8. Generate the held-out abstract only after the training loop is stable.
9. Verify coverage, order, length, and claim strength.
10. Repair the path or draft if verification fails.
11. Reveal the original abstract only for post-generation diagnosis.
```

The deterministic replay harness is only a substrate check. A real DIRS writing
run should include the heavy subagent simulation because prose quality is not
proven until the selected DAG is used to generate text and an evaluator compares
the result against the training target. Hosted APIs are not the default runtime
for this loop.

## Convergence And Stop Criteria

There are two different notions of convergence.

Training convergence means the shared DIRS graph has stabilized across a domain.
Use the general DISL criteria:

```text
same best graph retained for 3-5 full-domain passes
no meaningful node or edge edits
top node/edge support rankings are stable
MCTS selects same or equivalent validation sub-DAGs
validation verifier score changes by <= 0.005-0.01
replay cases still pass
```

Single-abstract inference does not retrain the shared graph. It stops when the
selected path and generated draft are stable enough:

```text
MCTS best sub-DAG unchanged for 3 repeated searches, or score gap is clear
selected sub-DAG is connected and follows edge direction
all factual claims are supported by the chip
no forbidden-domain nodes are selected
word count is within target band
verifier finds no no-jump, evidence, or overclaim failure
two repair attempts produce no material improvement
```

For a first smoke test:

```text
MCTS rollouts: 100-300
candidate drafts: 3-5
repair rounds: 1-2
accept if verifier passes and shortage map is empty or only minor
```

For a serious blind abstract test:

```text
MCTS rollouts: 1000-2000
candidate drafts: 8-16
repair rounds: up to 3
accept only if verifier passes, selected path is stable, and post-generation
comparison shows no major missing supported claim
```

For domain training, use the heavy dual-system subagent loop:

```text
Loop 1 editor:
  input = chip + current DAG + previous evaluator feedback
  output = selected connected sub-DAG + budget + repair rule

Loop 2 simulator:
  input = chip + selected connected sub-DAG
  output = generated section text

Evaluator:
  input = generated section + target training section + chip
  output = coverage/order/style/length score + feedback for Loop 1
```

In the preferred runtime these are separate Codex subagent roles, coordinated by
the main Codex thread.

The evaluator may read training targets during training. It must not read the
held-out original until after blind generation and verification.

## Default Abstract DAG Families

```text
R: section role and reader question
C: context or real setting
G: gap or missing condition
O: named object, method, dataset, model, or benchmark
M: mechanism or design details
E: evaluation, result, and evidence anchors
I: interpretation of evidence
S: bounded scope or takeaway
P: placement and style constraints
```

## Length Control

Use domain style first:

```text
target_words = median domain abstract length
```

If no domain style exists:

```text
short focused method: 120-180 words
normal method/benchmark: 180-230 words
large system/benchmark: 230-300 words
theory + simulation: 260-330 words
position/taxonomy: 140-220 words
```

The selector should add enough nodes to fill the length budget. If the draft is
too short, the fix is usually not more adjectives. The fix is to add the missing
compatible node family:

```text
missing setup -> add context/gap node
missing method detail -> add mechanism node
missing evidence -> add metric/result node
missing meaning -> add interpretation node
missing ending -> add scope/takeaway node
```

## Verifier

Reject or repair when:

```text
the selected nodes are not connected
the graph jumps over required dependencies
the abstract includes a result not in the chip
the abstract uses a generic high-frequency node with wrong semantics
the draft is far outside the target length band
the conclusion overclaims beyond evidence
the original abstract was used before generation
```

## Strict 19-Sample Update

The LLM Architecture 19-sample strict subagent run found that `E3` should be
interpreted as an evidence anchor, not only a quantitative anchor.

```text
old name:
  E3_quantitative_anchor

preferred alias:
  E3_evidence_anchor
```

Allowed `E3` forms:

```text
one compact numeric result
theorem or regime scope
benchmark count or task breadth
stage decomposition
qualitative validation anchor
```

The selector must also preserve the no-E3 route:

```text
E2_result_outcome -> I1_interpretation_or_tradeoff
```

This route is valid for concise, qualitative, theory-heavy, or mechanistic
abstracts whose training style does not use a headline number.

## Output Template

````markdown
# Abstract MCTS Writing Run

Date: `YYYY-MM-DD`

Paper:
`title`

Chip:
`path`

Domain source:
`path`

## Paper Signature

```yaml
paper_type:
domain_type:
method_object:
mechanism_type:
evidence_type:
result_type:
forbidden_types:
```

## Selected Connected Sub-DAG

```text
node_a -> node_b -> node_c
```

## Budget Plan

```yaml
target_words:
context_gap:
method_object:
mechanism:
evidence_result:
takeaway:
```

## Generated Abstract

...

## Blind Diagnostics

```yaml
connected_dag:
chip_coverage:
word_count:
forbidden_nodes:
shortage:
```

## Original Comparison

Only fill this section after generation.
````

## Good Historical Examples

```text
/tf/notebooks/yunbo/DIRS_method_sources_reference_20260720/tau2_bench_abstract_rewrite_from_dag_skill_20260708.md
/tf/notebooks/yunbo/DIRS_method_sources_reference_20260720/metamergen_abstract_blind_mcts_search_20260710.md
/tf/notebooks/yunbo/DIRS_method_sources_reference_20260720/sandbox_escape_abstract_typed_budgeted_subdag_chip_test_20260710.md
```
