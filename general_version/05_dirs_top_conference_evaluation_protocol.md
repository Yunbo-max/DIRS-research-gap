# DIRS Evaluation Protocol For Top-Conference Submission

Date: `2026-07-20`

Purpose: define how to evaluate DIRS as a research method for ICLR, ICML,
NeurIPS, ACL, or other top-tier venues.

## Evaluation Question

DIRS should be evaluated on whether it improves evidence-grounded artifact
generation, not merely whether its outputs sound fluent.

Primary question:

```text
Can a typed, connected, evidence-grounded skill DAG improve blind generation of
paper sections from chips compared with flat prompts, example retrieval, and
unstructured style transfer?
```

Secondary question:

```text
Which part matters most: content/style separation, edge constraints, MCTS
selection, support-rate priors, verifier repair, or domain topic routing?
```

## Data Protocol

Use the semantic domain split:

```text
/tf/notebooks/yunbo/DIRS/domain_topics/semantic_balanced_23_domains/INDEX.md
```

For each domain and section type:

```text
train papers:
  used to infer node/edge/style libraries

validation papers:
  used to tune verifier thresholds and repair policies

test papers:
  held out until final blind generation
```

The original target section for a test paper must be hidden during generation.
It can be revealed only after the generated section is saved.

## Tasks

Start with abstracts because they provide compact, high-signal tests of
argument structure. Then extend to full sections.

```text
Task 1: abstract generation from paper chips
Task 2: introduction rewrite from chips plus section facts
Task 3: method-section scaffold from source notes
Task 4: experiment/results-section rewrite from tables and result chips
Task 5: full-paper shortage diagnosis across sections
```

## Baselines

Compare against:

```text
Direct chip prompt:
  generate from chip with no learned graph

Style examples:
  retrieve same-domain examples and prompt directly

Flat skill list:
  provide learned nodes without edges

All-node DIRS:
  use every compatible node without MCTS selection

Greedy DIRS:
  select highest-scoring next node without tree search

Random connected DIRS:
  sample a connected sub-DAG without learned value scores

No-style DIRS:
  keep content nodes but remove style/action properties

No-content-binding DIRS:
  keep style/order but remove source evidence constraints

No-verifier DIRS-MCTS:
  select and generate without repair checks
```

## Metrics

Use a mixed evaluation because writing quality is not captured by one number.

```yaml
content_fidelity:
  checks:
    - all factual claims are supported by chip/source
    - no invented numbers, datasets, baselines, or methods
    - main contribution object is preserved

structure_quality:
  checks:
    - selected nodes are connected
    - dependency direction is valid
    - no result before metric/protocol
    - no interpretation before evidence
    - conclusion stays within evidence

style_fit:
  checks:
    - target section role is preserved
    - length follows domain/section prior
    - paragraph and sentence density match target type
    - rhetoric is domain-appropriate

comparison_to_expert:
  checks:
    - coverage of expert argument units
    - edit distance or semantic similarity after generation
    - human or LLM preference against baselines
    - noncopying score when original text is available

diagnostic_value:
  checks:
    - shortage map identifies missing evidence or weak sections
    - verifier catches unsupported claims
    - repair improves score without adding leakage
```

## Human Evaluation

For a top-conference submission, include expert or trained-rater judgments.
Recommended rubric:

```text
1. factual support
2. clarity of contribution
3. preservation of paper-specific novelty
4. section-role fit
5. ordering and coherence
6. appropriate strength of claims
7. usefulness for paper revision
```

Use pairwise comparisons:

```text
DIRS-MCTS vs direct chip prompt
DIRS-MCTS vs style-example prompt
DIRS-MCTS vs flat learned nodes
DIRS-MCTS vs greedy DIRS
```

## Automatic Verifier

The automatic verifier should produce both a scalar score and a structured
failure report:

```yaml
score:
  total:
  content_fidelity:
  edge_order:
  style_fit:
  length_fit:
  scope_control:
  noncopying:

failures:
  unsupported_claims: []
  missing_required_nodes: []
  broken_edges: []
  wrong_domain_nodes: []
  length_errors: []
  overclaims: []
```

The structured report matters because DIRS is a skill-learning method. A score
without a repair signal is not enough.

## Ablation Expectations

A convincing result should show:

```text
content binding reduces hallucinated claims
style properties improve section-role fit and length control
edge constraints reduce ordering jumps
support-rate priors improve domain-typical node selection
MCTS improves over greedy selection when several plausible paths exist
verifier repair improves final drafts without using the held-out original
```

If one component does not improve the result, report that honestly and narrow
the claim.

## Leakage Controls

For blind writing tests:

```text
1. freeze the chip before generation
2. log all files read by the generator
3. forbid reading the original held-out section until the draft is saved
4. store generation trace before comparison
5. run post-generation comparison in a separate step
6. flag suspicious phrase overlap
```

For cross-paper training:

```text
test paper source text may be used to build a chip only if the target section
text is excluded from the generation context
```

## Reporting Standard

Each experiment should report:

```text
domain
section type
number of train/validation/test papers
number of learned nodes and edges
support-rate distribution
MCTS rollout budget
early-stop criterion
generation model
verifier model or deterministic checks
cost and runtime
baseline scores
DIRS-MCTS scores
ablation scores
qualitative examples
failure cases
```

## Reviewer-Facing Claims

Strong claims DIRS can make if supported:

```text
DIRS improves blind section generation from structured paper chips.
DIRS reduces unsupported claims through node-level evidence binding.
DIRS reduces rhetorical jumps through edge-level dependency constraints.
DIRS-MCTS selects better paper-specific subgraphs than all-node or greedy use.
DIRS produces interpretable shortage maps that help diagnose weak drafts.
```

Claims to avoid unless separately proven:

```text
DIRS fully automates paper writing.
DIRS understands scientific novelty better than expert authors.
DIRS guarantees acceptance at any venue.
DIRS transfers to every domain without domain-specific training.
```

## Minimal Publishable Study

A minimal but credible first paper could be:

```text
data:
  3-5 semantic domains
  abstracts only
  held-out blind generation

methods:
  direct chip prompt
  style-example prompt
  flat learned-node prompt
  DIRS greedy
  DIRS-MCTS

metrics:
  factuality
  node coverage
  edge-order violations
  length/style fit
  human pairwise preference

analysis:
  support-rate calibration
  MCTS path examples
  no-jump violation examples
  failure cases where chip evidence is insufficient
```

## Stronger Follow-Up Study

The stronger version extends the same protocol to multiple section types:

```text
abstract
introduction
method
experimental setup
results
ablation
limitations
```

The key result should be not only better text, but better diagnosis:

```text
DIRS identifies what a draft is missing by locating missing nodes, broken
edges, unsupported claims, weak evidence bindings, and style mismatches.
```
