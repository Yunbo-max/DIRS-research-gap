# DIRS Case 2: Experiment Design

Date: `2026-07-20`

Purpose: task-specific protocol for applying DIRS to experiment design.

This case learns the human skill of turning a research claim, method idea, or
paper chip into a falsifiable experiment plan. The output is not prose polish;
it is a runnable evaluation design with hypotheses, controls, baselines,
metrics, ablations, leakage checks, and resource constraints.

## Current Targets

```text
01_experiment_plan_design.md
02_ablation_metric_protocol.md
```

## Experiment-Design Specialization

```text
decision node = scientific decision + operational constraint
tool node = external/local state check that may change the plan
edge = dependency between decisions and tool observations
simulator = tool-assisted dry run of design choices
verifier = falsifiability, fairness, leakage, feasibility, and interpretability
```

For experiment design, the two DIRS systems are:

```text
evidence system:
  paper chip, method claim, dataset facts, benchmark rules, baseline results,
  compute budget, available code, reported limitations

action system:
  hypothesis framing, control selection, metric choice, ablation order,
  execution plan, statistical reporting, failure analysis

tool system:
  shell commands, GPU checks, API checks, local code inspection, dataset
  inspection, benchmark lookup, smoke tests, and cost estimation
```

Important distinction:

```text
DIRS runtime:
  use Codex subagents for editor, simulator, and evaluator roles

experiment-design tool node:
  use API feasibility checks only when the proposed experiment itself depends
  on hosted models, paid APIs, or rate-limited services
```

So `API_feasibility_check` is not a request to run DIRS through an API. It is a
first-class node for experiments whose scientific plan depends on API access,
cost, or rate limits.

For Case 2, normal decision nodes and tool-calling nodes are both first-class
DAG nodes. A decision node may have a `tool_skill` field, but when a tool call
is required, DIRS should add an explicit tool node to the selected sub-DAG.

Decision-node schema:

```yaml
decision_node:
  content_skill: what scientific decision or evidence is needed
  action_skill: how to turn that evidence into an experiment-design decision
  tool_dependency: which tool nodes must be checked before this decision is trusted
```

Tool-node schema:

```yaml
tool_node:
  trigger: when this tool node is required
  tool_skill: what tool to call and what observation is needed
  allowed_tools: tools or commands allowed for this node
  output_contract: fields the simulator must return
  failure_policy: how to continue if the tool is unavailable or blocked
  effect_on_plan: how the observation changes the experiment design
```

## Tool-Calling Role

The simulator can call tools when the design requires external or local checks:

```text
local code search:
  verify available scripts, configs, datasets, checkpoints, and metrics

GPU/resource check:
  inspect available GPUs, memory, running jobs, disk, and expected runtime

API feasibility check:
  check model availability, credentials, pricing, rate limits, and a minimal
  smoke-test call when permitted

paper or benchmark lookup:
  confirm official protocols, leaderboard baselines, split definitions, and
  metric conventions

dataset inspection:
  check file existence, schema, split files, label distribution, and leakage
  boundaries

dry-run planning:
  estimate compute, memory, runtime, seeds, and expected artifact outputs
```

The simulator should not invent baseline numbers or benchmark rules. If a fact
is not in the chip, source, codebase, or checked external reference, the design
must mark it as unknown.

## Expected Output

```text
experiment question
hypothesis
variables and controls
datasets or environments
baselines
metrics
ablation table
execution order
resource estimate
tool-call plan
leakage and confound checks
expected result patterns
failure interpretation
minimum publishable evidence
```

## Quality Standard

A good DIRS experiment-design output is useful even if the expected result is
negative. It should make clear what would confirm the claim, what would falsify
it, and what additional evidence would be needed before making a strong paper
claim.
