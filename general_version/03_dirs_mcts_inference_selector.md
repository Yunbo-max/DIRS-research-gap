# DIRS-MCTS Inference Selector

Date: `2026-07-20`

Purpose: define how a trained DIRS library is used at test time to select a
connected sub-DAG and generate a target artifact from source evidence only.

## Role Of MCTS

MCTS selects a graph slice from a fixed graph snapshot. It does not directly
modify the shared DAG. In training, the editor/controller modifies the DAG using
feedback produced after MCTS simulation. In inference, MCTS only selects and
verifies candidate outputs.

## Default Execution Engine

For heavy DIRS runs, the MCTS loop should be executed through Codex subagents
rather than hosted API calls:

```text
coordinator:
  keeps state, launches workers, records artifacts, enforces blind boundaries

editor subagent:
  proposes the connected sub-DAG from the current graph and feedback

simulator subagent:
  executes the selected DAG to produce text, plan, action trace, or artifact

evaluator subagent:
  scores the result and returns feedback; it does not modify the graph
```

Hosted model APIs are not the default runtime for DIRS itself. They are allowed
only when explicitly requested, or when a downstream task contains an API tool
node whose purpose is to test API feasibility as part of the task.

```text
DISL training:
  editor proposes DAG -> MCTS selects sub-DAG -> simulate -> evaluate -> repair graph

DIRS-MCTS inference:
  new chip -> select DAG -> simulate/generate -> verify -> accept or repair output
```

MCTS is needed because a rich domain graph should not be applied wholesale. A
new paper or task usually needs a subset of nodes, and the subset must remain
connected, ordered, evidence-supported, and length-compatible.

The inner loop is therefore shared:

```text
select connected DAG -> execute/simulate -> verify
```

The outer effect is different:

```text
training feedback updates the skill graph through the editor/controller
inference chooses the best output for the new input
```

Training-time MCTS and inference-time MCTS share the same search mechanics:

```text
state = partial connected sub-DAG
action = add node, add edge, choose variant, allocate budget, terminate
hard constraints = evidence support, edge direction, connectivity, budget
```

They differ in reward:

```text
training-time MCTS:
  can score against the expert artifact after generation and use failures to
  repair the shared graph, preferably through a separate proposer and critic

inference-time MCTS:
  cannot see the held-out expert artifact and uses only source support,
  verifier rules, role fit, and budget fit
```

In training, the critic should compare the expert sub-DAG, simulated sub-DAG,
old shared graph, and proposed repaired graph. In inference, the critic cannot
use the held-out expert artifact; it can only check the generated candidate
against source evidence and DIRS verifier rules.

## Blind Inference Protocol

During generation, the system may use:

```text
target chip or source evidence
domain skill library
node and edge support scores
style and length priors
section target
venue or artifact constraints, if provided
```

During generation, the system may not use:

```text
the original held-out section
phrases copied from the original section
post-hoc comparison text
```

The original section can be opened only after the draft is generated, and only
for evaluation or error analysis.

## Search State

```yaml
state:
  selected_nodes: ordered list of nodes
  selected_edges: directed edges connecting the nodes
  open_frontier: compatible next nodes reachable from the current path
  budget_used: words, paragraphs, or action steps already allocated
  evidence_used: chip/source bindings already consumed
  rejected_nodes: candidates removed by hard constraints
```

The state is valid only if:

```text
the selected nodes form one connected directed path or connected DAG
no edge violates learned dependency direction
no content node lacks evidence support
no forbidden node is selected
```

## Actions

```yaml
actions:
  add_node: append or branch to a compatible next node
  choose_variant: select a domain-specific version of a node
  allocate_budget: assign word/paragraph/action budget to a node family
  add_transition: select an edge and transition style
  terminate: stop path construction and move to generation
```

## Scoring

A practical node score is:

```text
node_score(v) =
  0.25 * support_prior(v)
  + 0.30 * type_compatibility(v, signature)
  + 0.30 * evidence_support(v, chip)
  + 0.10 * role_need(v, section_target)
  + 0.05 * budget_fit(v, target_budget)
  - forbidden_penalty(v, signature)
```

An edge score is:

```text
edge_score(u, v) =
  0.35 * edge_support_prior(u, v)
  + 0.30 * dependency_validity(u, v, chip)
  + 0.20 * transition_fit(u, v, section_target)
  + 0.15 * no_jump_value(u, v)
  - forbidden_transition_penalty(u, v)
```

The rollout reward combines structure, evidence, and artifact quality:

```text
R(path, draft) =
  chip_supported_coverage
  + connectedness
  + edge_order_validity
  + domain_compatibility
  + style_role_fit
  + target_length_fit
  + mechanism_evidence_balance
  + interpretation_after_evidence
  + bounded_scope
  + noncopying_score
  - unsupported_claim_penalty
  - disconnected_node_penalty
  - no_jump_violation_penalty
```

## Tree Policy

Use a standard UCT-style selection rule over partial paths:

```text
UCT(a) = Q(a) + c * sqrt(log N(parent) / (1 + N(a)))
```

where:

```text
Q(a): mean verifier reward for rollouts taking action a
N(a): visit count for action a
c: exploration constant
```

The action space should be pruned before UCT:

```text
remove unsupported content nodes
remove forbidden-domain nodes
remove edges that break dependency order
remove expansions that exceed the budget by a large margin
```

## Rollout Policy

Rollouts should be cheap and structured:

```text
1. start from required role/context nodes
2. add compatible gap/object/mechanism nodes
3. add evidence nodes only when metrics/results exist in the chip
4. add interpretation only after evidence nodes
5. add scope/takeaway nodes last
6. allocate length according to domain and section priors
```

For abstracts, the default path is:

```text
context/gap -> object -> mechanism/design -> evidence/result -> interpretation -> scope
```

For method sections:

```text
setup/notation -> representation -> algorithm -> implementation -> complexity/limitations
```

For experiments:

```text
question -> datasets/tasks -> baselines -> metrics -> protocol -> controls
```

For results:

```text
main table -> comparison -> ablation -> mechanism interpretation -> limitation
```

## Generation

Generation should use the selected path as an execution plan, not as optional
advice:

```text
for node in selected_path:
  realize the node using only its evidence binding
  obey the node's style/action property
  use the incoming edge transition
  stay within the assigned budget
```

The generator should never add a factual claim because it sounds natural. If a
claim is not bound to the chip/source, it must be omitted or marked as missing.

## Verifier

The verifier should report:

```yaml
connected_dag: pass | fail
edge_direction_valid: pass | fail
chip_evidence_support: pass | fail
unsupported_claims: list
forbidden_nodes: list
missing_required_nodes: list
length_fit: pass | fail
style_fit: pass | fail
result_after_metric: pass | fail
interpretation_after_evidence: pass | fail
scope_bounded: pass | fail
noncopying: pass | fail | not_applicable
repair_action: accept | repair_path | repair_draft | reject
```

## Stop Rule

Recommended budgets:

```text
quick abstract test: 2k-5k rollouts
serious blind abstract test: 20k rollouts
section long-run scaffold: 5k rollouts per section per outer loop
full domain audit: up to 100 outer loops with early stopping
```

Stop when:

```text
best verifier score no longer improves
selected path is unchanged across repeated searches
repairs no longer change structural nodes or edges
remaining changes are local wording only
draft passes all hard constraints
```

## Output Contract

Each inference run should write:

```text
paper_or_task_signature.yaml
selected_subdag.yaml
selected_path.txt
budget_plan.yaml
generated_draft.md
blind_reward_diagnostics.json
verifier_result.json
rejected_nodes.md
shortage_map.md
post_generation_comparison.md, if expert target exists
```

## Baseline Ablations

For a serious evaluation, compare DIRS-MCTS against:

```text
chip-only direct generation
flat prompt with domain style examples
all-nodes generation without selection
random connected sub-DAG
node selector without edge constraints
edge-constrained greedy selector
DIRS without style properties
DIRS without content evidence bindings
DIRS-MCTS without verifier repair
```

These ablations test whether quality comes from the graph, the dual-system node
properties, MCTS search, or the verifier.

## Historical References

```text
typed selector:
  /tf/notebooks/yunbo/DIRS_method_sources_reference_20260720/abstract_dag_selector_fix_domain_aware_budgeted_20260710.md

blind MCTS test:
  /tf/notebooks/yunbo/DIRS_method_sources_reference_20260720/metamergen_abstract_blind_mcts_search_20260710.md

section-level long run:
  /tf/notebooks/yunbo/DIRS_method_sources_reference_20260720/temporal_hetero_section_dual_system_longrun_README_20260712.md
```
