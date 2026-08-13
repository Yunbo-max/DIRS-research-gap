# DIRS General Method

Date: `2026-07-20`

Purpose: define the task-independent version of DIRS in a form suitable for a
research-method description.

## Problem Setting

Given a collection of expert artifacts

```text
D = {(x_i, y_i, m_i)} for i = 1..N
```

where:

```text
x_i = source evidence, chip, trace, table, code, or observation set
y_i = expert artifact, such as an abstract, section, proof, answer, or workflow
m_i = metadata, such as domain, section type, paper type, or task type
```

DIRS learns a reusable skill library that can generate a new artifact `y*` from
a new evidence object `x*` without copying or seeing the held-out expert target
during generation.

## Central Hypothesis

Expert artifacts are not only collections of facts. They also encode a latent
dependency structure: some claims must be introduced before mechanisms, some
metrics before results, and some evidence before interpretation. DIRS represents
this latent structure as a directed skill graph.

## Skill Graph

For each domain and artifact type, DIRS learns:

```text
G = (V, E, R)
```

where:

```text
V = typed reusable skill nodes
E = directed dependency and transition edges
R = verifier/reward function for the artifact role
```

Each node stores both content and presentation/action properties:

```yaml
node:
  id: stable node identifier
  node_type: text_reasoning | tool_call | tool_router | verifier | repair
  family: context | gap | object | mechanism | evidence | interpretation | scope | style
  content_skill: what evidence, claim, computation, or precondition this unit carries
  style_skill: how the unit should be expressed, positioned, formatted, or executed
  action_skill: how the unit changes the plan, decision, workflow, or output
  tool_skill: how the unit calls tools when it must observe local or external state
  tool_contract:
    trigger: when a tool call is required
    allowed_tools: permitted commands, APIs, source lookups, or inspections
    output_contract: fields the tool node must produce
    failure_policy: what to do if the call fails or is unavailable
    effect_on_plan: downstream nodes changed by the tool result
  evidence_binding: chip fields, source spans, results, traces, or table entries that support it
  when_to_apply: trigger condition
  when_to_reject: incompatible signatures or missing evidence
  support_count: number of training artifacts using this node
  support_rate: support_count / domain_artifact_count
  budget_role: required | common | optional | expansion
  failure_modes: common ways the node is misused
```

Each edge stores both dependency and transition properties:

```yaml
edge:
  from: upstream_node
  to: downstream_node
  content_dependency: why the downstream unit requires the upstream unit
  style_transition: how the artifact should move between the two units
  required_when: condition that makes the dependency mandatory
  forbidden_when: condition that makes the edge misleading
  support_count: number of training artifacts using this edge
  support_rate: support_count / domain_artifact_count
  no_jump_failure: error caused by skipping this dependency
```

## Dual-System Decomposition

DIRS keeps source grounding separate from artifact form.

```text
content system:
  selects facts, claims, definitions, mechanisms, measurements, and results

style/action system:
  selects order, compression, paragraph shape, length, transitions, and scope
```

For tool-grounded tasks, DIRS adds a third operational stream:

```text
tool system:
  selects when to call tools, which tools are allowed, what observation is
  needed, how to handle failure, and how the result changes downstream nodes
```

This separation is important because two drafts can use the same facts but have
different quality. A strong draft must satisfy both:

```text
content correctness:
  every factual unit is supported by the chip or source

structural correctness:
  every unit appears in a valid dependency order for the section role

operational correctness:
  tool-dependent decisions are made only after the required tool observations,
  or are explicitly marked unknown or blocked
```

## Training Objective

DIRS training searches for a graph that explains expert artifacts while
remaining reusable:

```text
maximize:
  reconstruction_score(y_i | x_i, G)
  + transfer_score(heldout y_j | x_j, G)
  + graph_stability(G)

subject to:
  evidence support
  directed connectivity
  no-leakage constraints
  no-overclaim constraints
  bounded graph complexity
```

The learned graph should not memorize a single artifact. It should explain the
repeated section logic across many artifacts in a domain.

## Inference Objective

For a new input `x*`, DIRS selects a connected sub-DAG `S*` from the learned
graph:

```text
S* = argmax_S R(S, x*, target_role)
```

subject to:

```text
S is connected
S follows valid edge directions
every content node in S is supported by x*
forbidden nodes are excluded
the induced budget matches the target artifact
```

The artifact generator then writes by walking the selected path:

```text
selected sub-DAG -> budget plan -> draft -> verifier -> repair or accept
```

## What DIRS Is Not

DIRS is not:

```text
a single prompt template
a style-transfer pass over the original artifact
a bag of independent writing tips
a retrieval-only memory of old examples
```

DIRS is:

```text
a typed, evidence-grounded, directed skill graph with learned support rates,
compatibility rules, style/action properties, and verifier-backed repair.
```

## Expected Generalization

DIRS should generalize along three axes:

```text
same domain, new paper:
  use the domain graph and select a paper-specific sub-DAG

same paper, new section:
  reuse source facts but switch section-role reward and edge priorities

new domain:
  train a new domain graph while reusing task-independent node families,
  no-jump rules, and verifier schemas
```

## Auto-Research Skill Families

For auto-research, DIRS should learn several downstream skills from the same
general representation:

```text
writing:
  turn evidence into a bounded scientific artifact with correct section role,
  order, style, and claim strength

experiment design:
  turn a claim or question into a falsifiable protocol with baselines, metrics,
  ablations, controls, resource estimates, and failure interpretations

research question proposal:
  turn topic clusters or chips into grounded, novel, testable questions by
  connecting field patterns to unresolved uncertainties

research gap verification:
  test whether a proposed gap is real by searching for counterevidence,
  comparing prior work, identifying near-misses, and producing a verdict
```

The general graph machinery stays the same, but each downstream case changes
the node properties, simulator tools, verifier checks, and output schema.

## Output Contract

Every complete DIRS run should save:

```text
task_signature
source_evidence_manifest
learned_or_selected_nodes
learned_or_selected_edges
explicit_directed_path
node_property_scores
edge_property_scores
budget_plan
generated_artifact
verification_result
repair_trace
post_generation_comparison, when an expert target exists
```
