# DIRS Training Cycle

Date: `2026-07-20`

Purpose: specify DISL, the training procedure that learns reusable DIRS graphs
from expert artifacts.

## Goal

DISL learns a stable, reusable skill graph for a domain and artifact type:

```text
DISL = DAG-Inferred Skill Learning
DIRS = stable skill library produced by DISL
```

The training data are expert artifacts paired with source evidence:

```text
input:
  source evidence or chip x_i
  expert artifact y_i
  domain and artifact metadata m_i

output:
  domain-level graph G_domain,section
  node and edge support scores
  content/style property library
  verifier and replay set
```

For paper writing, `x_i` is a paper chip or source document and `y_i` is an
expert section. For other tasks, `x_i` may be a trace, environment state, code
repository, table, or problem statement.

## Mathematical View

DIRS can be understood as learning a reusable human skill. The skill may be
paper writing, source search, research-gap proposal, code repair, experiment
design, review, or another expert behavior.

For a skill family `tau`, the training set is:

```text
D_tau = {(x_i, y_i, m_i)} for i = 1..N
```

where:

```text
x_i = input situation, evidence, chip, question, codebase, trace, or source
y_i = expert artifact or action trace
m_i = metadata such as domain, role, section type, venue, or task type
```

DIRS assumes each expert output was produced through a hidden skill trace:

```text
z_i = (S_i, pi_i, b_i, h_i)
```

where:

```text
S_i = connected skill sub-DAG used by the expert
pi_i = topological order followed by the expert
b_i = budget allocation over skill nodes
h_i = bindings from skill nodes to evidence in x_i
```

Training learns a reusable graph:

```text
G* = argmax_G [
  sum_i max_{z_i subset G} R_train(x_i, y_i, z_i, G)
  + lambda_val * R_val(G)
  + lambda_stab * Stable(G)
  - lambda_size * Complexity(G)
]
```

At inference time, for a new input `(x*, m*)`, DIRS selects a connected sub-DAG:

```text
S* = argmax_{S subset G*} F(S; x*, m*, tau)
```

and then produces:

```text
y_hat* = Decode(x*, S*)
```

subject to:

```text
S* is connected and acyclic
S* follows learned edge directions
each selected content node is supported by x*
forbidden nodes and transitions are excluded
the output budget fits the target role
```

The full formulation is in:

```text
/tf/notebooks/yunbo/DIRS/general_version/06_dirs_mathematical_formulation.md
```

## Dual-System Training Signal

DIRS trains two coupled representations:

```text
content/evidence stream:
  what facts, claims, methods, metrics, results, equations, or actions are
  required and where they are grounded

style/action stream:
  how those units are ordered, compressed, phrased, scoped, formatted, or
  executed for the artifact role
```

The two streams are stored together at the node and edge level because writing
quality depends on both.

## Simulation Produces The Learning Signal

In DIRS, the graph is not only annotated. It is executed.

Each training loop must run the same basic procedure that inference will later
use:

```text
1. select a connected sub-DAG from the current graph
2. bind selected content nodes to evidence in x_i
3. allocate action/style budget over the selected nodes
4. simulate or generate y_hat_i by following the DAG order
5. verify y_hat_i against source support, edge order, role fit, and target y_i
6. use the verifier result to repair the graph
```

This is the dual-system simulation:

```text
content system:
  proposes what should be said, searched, tested, repaired, or executed

style/action system:
  proposes how it should be ordered, compressed, phrased, formatted, or
  operationalized

simulator:
  runs the selected DAG and produces an observable artifact or action trace

verifier:
  turns the simulated result into node, edge, budget, and style repair signals
```

Training and inference therefore share the same execution shape:

```text
select connected DAG -> execute/simulate -> verify
```

The difference is what happens after verification:

```text
training:
  verifier result updates G, node properties, edge properties, and replay cases

inference:
  verifier result selects, repairs, or rejects the candidate output
```

This is the key reason DIRS uses loops. A node is not considered learned only
because it was named. It becomes learned when simulation shows that selecting it
improves the artifact or action trace under the verifier, without breaking
replay cases.

## Per-Example DAGs And The Shared Full DAG

Training has two graph levels.

First, DIRS infers a small latent DAG for each expert example:

```text
example i:
  x_i, y_i -> inferred expert sub-DAG S_i^expert
```

This per-example DAG explains how one expert artifact or action trace works. It
is local and should not be treated as the whole skill.

Second, DIRS merges the per-example DAGs into a shared domain graph:

```text
G_t = Merge(S_1^expert, S_2^expert, ..., S_N^expert)
```

The merge step canonicalizes nodes and edges:

```text
same reusable skill -> one shared node with higher support
same dependency order -> one shared edge with higher support
paper-specific detail -> evidence binding or typed variant
wrong-domain pattern -> rejection rule
```

So after training on many examples, the shared graph is larger than any single
example:

```text
S_i^expert subset G_t
```

However, DIRS should not execute the entire shared graph at once. The full graph
contains alternative paths, optional expansions, domain variants, and rejection
rules. Running all of it would over-write, over-search, or over-act.

In the full dual-loop version, Loop 1 first proposes a fixed candidate graph
snapshot. Loop 2 then runs MCTS over that candidate snapshot:

```text
G'_t = Editor(G_t, x_i, y_i, f_i,t-1)
G_eval,t = G'_t
S_i,t^sim = MCTS_Select(G_eval,t, x_i, m_i, H_train)
y_hat_i,t = Simulate(x_i, S_i,t^sim)
```

In a simpler ablation, `G_eval,t` can be the current graph `G_t`. The important
constraint is that MCTS receives a fixed graph snapshot and does not commit graph
edits.

Then the verifier compares the simulated result to source support, task role,
and, after generation, the expert target:

```text
r_i,t = Verify(y_hat_i,t, x_i, y_i, S_i,t^sim)
```

The controller decides whether the candidate graph becomes the next shared
graph:

```text
G_{t+1} = Accept(G'_t, r_i,t, replay_t) ? G'_t : G_t
```

So the training answer is:

```text
yes, each example has its own inferred DAG;
yes, those DAGs are combined into a larger shared DAG;
yes, training simulates from a full DAG snapshot, usually the proposed G'_t;
but no, it should not execute every node in the full DAG.
It should select and simulate connected sub-DAGs from the full graph.
```

## Training State Persistence

Training does not restart from zero after each example or each loop.

The current graph is persistent:

```text
round 0 starts with G_0
Loop 1 proposes candidate G'_0 from G_0
Loop 2 evaluates G'_0 with MCTS
the controller accepts or rejects G'_0
round 1 starts with accepted graph G_1, not from zero
...
final output is G*
```

For each example, the system selects the best connected sub-DAG for that
example from the fixed graph being evaluated:

```text
S_i,t = MCTS_Select(G_eval,t, x_i, m_i, H_train)
```

But that selected sub-DAG is only a training rollout. It is not the final skill
library. The final skill library is the larger shared graph:

```text
G* = learned full DAG across examples, loops, verifier results, and repairs
```

So the relationship is:

```text
per-example selected DAG:
  local path used to simulate one case

shared full DAG:
  persistent learned skill graph updated across all cases

final DAG:
  the large shared graph G*, with node support, edge support, compatibility,
  rejection rules, style/action rules, and verifier rules
```

## Online Versus Batch Training

There are two valid ways to process examples.

In both modes, node selection should use the same search machinery as
inference. For the mature DIRS method, that means MCTS over the current shared
graph.

```text
training-time MCTS:
  input: current shared graph G_t, example x_i, metadata m_i, target role
  output: selected connected sub-DAG S_i,t
```

The training-time MCTS objective is:

```text
S_i,t = argmax_{S subset G_t} F_train(S; x_i, y_i, m_i)
```

where the rollout reward uses:

```text
before generation:
  evidence support, compatibility, edge direction, budget fit

after generation:
  verifier score, reconstruction gap to y_i, missing nodes, broken edges,
  unsupported claims, style/action mismatch
```

The original expert artifact `y_i` is used as a training target and post-rollout
scoring signal, not as text to copy into the generated candidate. For held-out
test examples, `y_i` must be hidden until after generation.

### Online Training

In online training, each example immediately updates the shared graph:

```text
start: G_0

sample 1:
  use MCTS to infer/select S_1 from G_0 and x_1
  simulate, verify, repair
  produce G_1

sample 2:
  use MCTS to infer/select S_2 from G_1 and x_2
  simulate, verify, repair
  produce G_2

sample 3:
  use MCTS to infer/select S_3 from G_2 and x_3
  simulate, verify, repair
  produce G_3
```

So if there are five samples, sample 2 does use the graph learned after sample
1. It does not start from none. But it uses `G_1` only as a prior and skill
library. It is not forced to reuse the exact sample-1 DAG.

For sample 2, the selector can:

```text
reuse sample-1 nodes if they fit x_2
reuse sample-1 edges if the dependency order fits x_2
add new nodes if x_2 needs a skill not in G_1
split a vague node into typed variants
reject sample-1 nodes if x_2 lacks supporting evidence
```

### Batch Training

For a cleaner research protocol, use batch updates:

```text
start: G_t

for all samples i = 1..N:
  use MCTS to infer/select S_i,t from G_t and x_i
  simulate y_hat_i,t
  verify y_hat_i,t

after all samples:
  merge all S_i,t and verifier results
  update G_t into G_{t+1}
```

In this version, sample 2 in the same pass does not use the immediate result of
sample 1. All samples use the same `G_t`, and the merge happens after the full
pass. This reduces order bias.

## Why Not Restart From Zero

DIRS does not restart from zero because the target is a reusable skill, not five
independent annotations.

If every sample starts from nothing:

```text
support counts never accumulate
edge frequencies never stabilize
style and length priors stay noisy
MCTS has no useful prior for selection
the system cannot learn which nodes transfer across examples
```

The point of training is to learn what repeats and what must be rejected:

```text
repeated across examples -> shared node or edge with higher support
works only for one type -> typed variant
fails under verifier -> rejection rule or repair case
```

Recommended default:

```text
use batch updates for publishable experiments
use online updates for quick exploratory debugging
```

## Dual-LLM DAG Feedback

For the strongest training setup, use separate roles. One model should not be
trusted to propose, grade, and accept its own graph edits.

The two loops should not be collapsed.

```text
Loop 1: DAG learning/editing loop
  modifies the shared DAG or proposes a graph diff

Loop 2: MCTS simulation loop
  uses a fixed DAG snapshot to select a sub-DAG and simulate an output
  does not modify the shared DAG
```

```text
LLM-A: graph proposer / editor
  takes the current shared DAG, chip/source input, previous feedback, and expert
  artifact during training; predicts or revises the inverse DAG and proposes
  node, edge, variant, verifier, or style/action repairs

LLM-B: graph critic / verifier
  compares generated results, verifier reports, and compact DAG summaries;
  identifies missing or wrong nodes, edges, evidence bindings, and style/action
  rules

controller:
  accepts a graph edit only if it improves verifier score or fixes a hard
  failure without breaking replay cases
```

### Step 0: Initial Inverse DAG Construction

For each training example, the editor first builds an initial inverse DAG from
the example's source input and final expert result:

```text
input:
  x_i = chip/source/evidence
  y_i = final expert artifact or action trace

output:
  S_i^expert = inferred expert DAG for this example
```

Across examples, these inferred DAGs are merged into the initial shared graph:

```text
G_0 = Merge(S_1^expert, ..., S_N^expert)
```

### Loop 1: DAG Learning / Editing

After Step 0, each training round updates the graph through feedback:

```text
input:
  current shared DAG G_t
  chip/source x_i
  final expert result y_i
  feedback f_i,t-1 from the eval agent
  previous selected/simulated DAG summary, if available

output:
  proposed inverse DAG revision or graph edit Delta_t
  candidate shared DAG G'_t
```

This loop is allowed to modify graph structure:

```text
add node
choose node modality: text/reasoning, tool-call, tool-router, verifier, or repair
split node
merge duplicate nodes
add edge
reverse edge
remove unsupported edge
change evidence binding
change content_skill
change style_skill
change action_skill
change tool_skill
change tool trigger, allowed tools, output contract, or failure policy
add rejection rule
change verifier rule
```

For downstream cases with tool use, Loop 1 decides whether a missing skill
should be represented as a text/reasoning node or as a tool-based node. For
example, if the feedback says "the experiment plan assumed GPU availability
without checking it," Loop 1 should add or strengthen a tool-call node such as
`G2_GPU_resource_tool`, not merely add a sentence telling the writer to mention
resources.

```text
text/reasoning node:
  transforms known evidence into a claim, decision, explanation, or plan step

tool-call node:
  obtains missing local or external state before a downstream decision is made

tool-router node:
  decides which tool-call nodes are required for this example

verifier node:
  checks whether selected nodes and produced outputs satisfy the task contract
```

### Loop 2: MCTS Simulation

The simulator is the MCTS executor. It evaluates a fixed graph snapshot, usually
the candidate graph `G'_t`:

```text
simulator:
  runs MCTS over fixed graph G'_t and x_i
  selects S_i,t^sim
  executes S_i,t^sim to produce y_hat_i,t
  emits a verifier report
```

This loop is not allowed to modify the shared DAG:

```text
allowed:
  choose nodes
  choose edges
  choose variants
  allocate budget
  generate or execute y_hat_i,t
  produce a verifier report

not allowed:
  add permanent nodes to G_t
  delete permanent nodes from G_t
  change support scores
  change compatibility rules
  accept graph repairs
```

### Eval Feedback Bridge

The eval agent receives the two final results or reports:

```text
input:
  generated result y_hat_i,t
  expert result y_i
  chip/source x_i
  simulator verifier report

output:
  feedback f_i,t
```

For output-level feedback, the eval agent does not need the full DAG. It can
compare the two final results directly:

```text
f_i,t = Eval(y_hat_i,t, y_i, x_i, report_i,t)
```

That feedback is then passed back into Loop 1:

```text
G'_t = Editor(G_t, x_i, y_i, f_i,t)
```

Feedback can operate at three levels:

```text
output-level feedback:
  compare y_hat_i,t with y_i or another target report/result
  no full DAG is required

selected-DAG feedback:
  compare S_i^expert with S_i,t^sim
  needs only compact selected sub-DAG summaries, not the entire G_t

full-graph acceptance:
  compare G_t with proposed G'_t
  needed only when deciding whether to update the persistent shared graph
```

So the critic does not always need the full DAG. If the goal is only to judge
the difference between two reports, the critic can use the two outputs plus
their verifier reports. If the goal is to repair the skill library, the critic
should also see the selected sub-DAGs and the proposed graph diff.

For graph repair, the critic should compare these graph objects:

```text
S_i^expert:
  inferred DAG explaining the expert artifact

S_i,t^sim:
  MCTS-selected DAG used to simulate the current example

G_t:
  current shared full DAG before repair

G'_t:
  proposed repaired shared full DAG
```

The comparison asks:

```text
1. What nodes in S_i^expert are missing from S_i,t^sim?
2. What nodes in S_i,t^sim are unsupported by x_i?
3. What edges are reversed, skipped, or missing?
4. Did G'_t add a reusable skill or only memorize one example?
5. Did G'_t improve this case while preserving replay cases?
6. Does the proposed edit change content skill, style skill, or both?
```

The critic should output a structured graph-diff report:

```yaml
dag_feedback:
  compared_graphs:
    expert_subdag: S_i^expert
    simulated_subdag: S_i,t^sim
    old_shared_graph: G_t
    proposed_shared_graph: G_prime_t
  missing_nodes:
    - node_id:
      reason:
      evidence_binding:
  unsupported_nodes:
    - node_id:
      reason:
  missing_edges:
    - from:
      to:
      no_jump_failure:
  wrong_edges:
    - from:
      to:
      repair: reverse | remove | make_optional
  over_specific_nodes:
    - node_id:
      repair: convert_to_evidence_binding | typed_variant | reject
  accepted_repairs:
    - repair_id:
      expected_effect:
  rejected_repairs:
    - repair_id:
      rejection_reason:
  replay_risk:
    status: low | medium | high
    notes:
```

This makes the training loop:

```text
Step 0:
  editor constructs initial inverse DAG from chip/source and final expert result

Loop 1:
  editor modifies or proposes the DAG using feedback

Loop 2:
  MCTS simulator uses the candidate DAG only; it does not modify the DAG

Eval:
  eval agent compares generated result with expert result and returns feedback

Back to Loop 1:
  editor uses feedback to revise the DAG

Controller:
  accepts or rejects the proposed graph update
```

The critic can use the expert artifact during training after generation, but not
during held-out inference. For held-out tests, the critic only checks the
generated artifact against source evidence, graph constraints, and verifier
rules until the output is saved.

## Node Schema

```yaml
node_id:
  node_type: text_reasoning | tool_call | tool_router | verifier | repair
  family: context | gap | object | mechanism | evidence | interpretation | scope | style
  title: short retrieval name
  content_skill: factual, argumentative, computational, or action role
  style_skill: phrasing, placement, compression, formatting, or execution behavior
  action_skill: decision, planning, execution, or control behavior when relevant
  tool_skill: tool-call behavior when node_type is tool_call or tool_router
  tool_contract:
    trigger: condition that requires the tool node
    allowed_tools: commands, APIs, browser/source lookup, or local inspections allowed
    output_contract: fields the tool node must return
    failure_policy: how to continue if the tool is unavailable or blocked
    effect_on_plan: downstream decisions affected by the tool observation
  evidence_binding:
    required_fields: chip/source fields that must exist
    optional_fields: evidence that strengthens the node
    forbidden_fields: evidence patterns that make the node invalid
  compatibility:
    positive_signatures: paper/task signatures where the node fits
    negative_signatures: signatures where the node must be rejected
  support:
    count: number of training artifacts using the node
    rate: count / domain_artifact_count
  budget_role: required | common | optional | expansion
  verifier_checks: checks for correct use
  common_failures: known misuse patterns
```

Example:

```yaml
E3_anchor_metric:
  family: evidence
  content_skill: preserve one central quantitative result when the chip supports it
  style_skill: place the number after method/protocol setup and before interpretation
  evidence_binding:
    required_fields: [reported_results]
    optional_fields: [metric_name, dataset_name, baseline_name]
  compatibility:
    positive_signatures: [empirical_method, benchmark, system]
    negative_signatures: [pure_position_without_results]
  budget_role: common
  verifier_checks:
    - metric appears after metric/protocol context
    - number is present in the chip
    - no extra unsupported numbers are introduced
  common_failures:
    - deleting the only concrete result
    - dumping too many numbers
    - reporting a number before the reader knows what was measured
```

## Edge Schema

```yaml
edge_id:
  from: upstream_node
  to: downstream_node
  content_dependency: why the downstream unit requires the upstream unit
  style_transition: how the artifact should bridge the units
  support:
    count: number of training artifacts using the edge
    rate: count / domain_artifact_count
  required_when: chip or section condition that makes the edge mandatory
  forbidden_when: condition that makes the transition misleading
  verifier_checks: checks for direction, dependency, and transition quality
  no_jump_failure: error caused by omitting the edge
```

Example:

```yaml
M_before_E:
  from: mechanism
  to: evidence
  content_dependency: a result is meaningful only after the mechanism, metric, or verifier is defined
  style_transition: move from design description into empirical evidence
  required_when: the artifact reports a metric, benchmark score, or ablation
  no_jump_failure: benchmark score floats without task or metric context
```

## Algorithm

```text
Algorithm: DISL training for one domain and section

Input:
  training artifacts D = {(x_i, y_i, m_i)}
  initial node families F
  verifier templates H
  maximum outer loops L

Initialize:
  infer S_i^expert = InverseDAG(x_i, y_i) for each training artifact
  shared graph G_0 = Merge({S_i^expert})
  replay set Q = empty
  mutation ledger U = empty

For loop t = 1..L:
  For each artifact i in D:
    1. infer paper/task signature s_i from x_i and m_i
    2. Loop 1 editor proposes G'_t from G_t, x_i, y_i, and feedback f_i,t-1
    3. Loop 2 MCTS simulator selects S_i,t^sim from fixed candidate graph G'_t
    4. simulator generates y_hat_i,t by executing S_i,t^sim on x_i
    5. verifier produces report_i,t for evidence, order, style, budget, and scope
    6. eval agent compares y_hat_i,t with y_i after generation
    7. eval agent returns feedback f_i,t to Loop 1
    8. controller accepts G'_t only if feedback/replay improve
    9. otherwise keep G_t and record the rejected edit

  Aggregate:
    update node support counts and rates
    update edge support counts and rates
    update compatibility and rejection rules
    update length and style priors
    update replay cases and accepted/rejected ledgers

  Stop early if graph, support scores, verifier rules, and style priors are stable.

Return:
  stable DIRS skill library G*
```

## No-Jump Rule

Every generated artifact must follow a connected directed path. The verifier
should reject:

```text
result before metric/protocol
mechanism before problem setup
interpretation before evidence
scope claim before support
unsupported content node
wrong-domain high-frequency node
disconnected node island
```

For paper abstracts, the default path family is:

```text
context/gap -> object -> mechanism/design -> evidence/result -> interpretation -> scope
```

For method sections, the default path family is:

```text
setup/notation -> representation -> algorithm -> implementation -> complexity/limits
```

## Cross-Paper Aggregation

For a domain with `N` papers:

```text
node_support_rate(v) = papers_using_node(v) / N
edge_support_rate(e) = papers_using_edge(e) / N
```

Support rate is a prior, not a command. A high-support node can still be
rejected when the held-out chip lacks evidence for it. A low-support node can
be selected when it is strongly supported by the chip and necessary for the
paper type.

## Repair Types

```yaml
node_repair:
  missing_node: add a reusable unit that repeatedly appears in expert artifacts
  overgeneral_node: split one vague node into typed variants
  unsupported_node: add negative compatibility or evidence-binding requirement
  weak_style: add placement, compression, or wording constraints

edge_repair:
  missing_dependency: add required edge to prevent a jump
  wrong_direction: reverse or remove an edge that caused bad order
  optional_transition: mark edge as useful but not mandatory
  forbidden_transition: prevent misleading rhetorical movement

verifier_repair:
  missing_check: add a deterministic or rubric check
  weak_threshold: tighten length, support, or order criterion
  leakage_gap: ensure expert text is hidden during generation
```

## Stability Criterion

A training run is stable when repeated passes produce no meaningful change to:

```text
node inventory
edge inventory
node support scores
edge support scores
evidence-binding rules
compatibility and rejection rules
style and length priors
verifier checks
replay outcomes
```

Recommended loop budgets:

```text
smoke test: 3-5 loops
exploratory run: 10-20 loops
serious domain run: 50-100 loops max
stop early: same best graph retained for 3-5 full-domain passes
```

Check convergence only after a full pass over the training domain, not after one
example. The best practical convergence test is:

```yaml
graph_edit_stability:
  pass_when: no nodes or edges are added, removed, split, or reversed for 3-5 passes

support_rank_stability:
  pass_when: top-k node and edge support rankings have high overlap across passes

selected_path_stability:
  pass_when: MCTS selects the same or equivalent sub-DAGs on validation examples

verifier_plateau:
  pass_when: validation verifier score changes by less than a small threshold

replay_stability:
  pass_when: old accepted cases still pass after new repairs

failure_stability:
  pass_when: no new unsupported-claim, no-jump, leakage, or overclaim failures appear

budget_style_stability:
  pass_when: length priors, section budgets, and style/action rules stop moving
```

Do not require byte-identical graph files. A graph may be stable even if labels
or comments are cleaned up. Require behavioral stability:

```text
same reusable nodes
same dependency structure
same selected paths on validation cases
same verifier decisions
same or better output quality
```

Stop training when:

```text
1. hard verifier checks pass,
2. validation score has plateaued,
3. selected sub-DAGs are stable,
4. replay cases do not regress,
5. new loops only make wording-level changes.
```

Continue training when:

```text
new examples still add useful nodes or edges
MCTS keeps selecting different structural paths
verifier failures reveal missing dependencies
support rankings are still moving
validation score is still improving
```

Stop and repair the method when:

```text
the graph keeps growing but validation score does not improve
new nodes are mostly paper-specific memorization
training score improves but held-out score drops
MCTS selects high-frequency nodes that lack evidence in the chip
```

## Train/Validation Discipline

To be credible as a research method, DISL should distinguish:

```text
training artifacts:
  used to infer and repair graph structure

validation artifacts:
  used to select repair moves and tune thresholds

held-out test artifacts:
  used only for final blind generation and post-generation comparison
```

The expert target for a held-out artifact must not be used during generation.
It may be used afterward for scoring, diagnosis, and replay creation.

## Output Artifacts

A complete training run should write:

```text
domain_skill_library.json
skill_graph.yaml
node_library.json
edge_library.json
node_support_scores.md
edge_support_scores.md
style_profile.md
length_prior.json
verifier.md
verifier_result.json
training_trace.jsonl
mutation_candidates.jsonl
accepted_updates.jsonl
rejected_updates.jsonl
replay_cases.jsonl
```

## Historical Evidence

```text
tau2 single-paper inverse DAG:
  /tf/notebooks/yunbo/DIRS_method_sources_reference_20260720/section_skill_learning_tau2_abstract_dag_converged_20260708.md

DIRS 14-paper rich run:
  /tf/notebooks/yunbo/DIRS_method_sources_reference_20260720/agent_interfaces_abstract_dirs_rich_14paper_run_20260708.md

agent-interface 14-paper stabilized library:
  /tf/notebooks/yunbo/DIRS_method_sources_reference_20260720/agent_interfaces_abstract_disl_100_loop_cross_paper_run_20260708.md

no-jump harness:
  /tf/notebooks/yunbo/DIRS_method_sources_reference_20260720/nojump_harness_agent_interfaces_20260708_cached/README.md

style profile:
  /tf/notebooks/yunbo/DIRS_method_sources_reference_20260720/agent_interfaces_abstract_total_style_profile_20260709.md
```
