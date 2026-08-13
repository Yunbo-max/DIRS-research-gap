# Experiment Plan Design

Date: `2026-07-20`

Purpose: construct a complete experiment plan from a claim, method idea, chip,
or research question.

## Core DAG Path

```text
problem claim
  -> testable hypothesis
  -> target task or setting
  -> tool-need router
  -> dataset/environment choice
  -> codebase protocol tool node
  -> benchmark rule tool node
  -> GPU resource tool node
  -> API feasibility tool node
  -> baseline set
  -> metric choice
  -> protocol constraints
  -> resource estimate
  -> smoke-test/dry-run tool node
  -> execution plan
  -> expected result pattern
  -> failure interpretation
  -> bounded claim
```

## First-Class Tool Nodes

For Case 2, tool calls are not merely annotations on ordinary nodes. They are
selectable DAG nodes. MCTS can include, skip, or prioritize them depending on
the target experiment and the unknowns in the input.

```text
decision nodes:
  choose hypothesis, task, baselines, metrics, protocol, interpretation, claim

tool nodes:
  inspect code, data, benchmark rules, GPU state, API feasibility, and dry-run
  status
```

Tool nodes should appear before the decision that depends on their observation:

```text
codebase_protocol_tool -> baseline_set
benchmark_rule_tool -> metric_choice
dataset_inspection_tool -> protocol_constraints
GPU_resource_tool -> execution_plan
API_feasibility_tool -> execution_plan
smoke_test_tool -> full_run_queue
```

Runtime distinction:

```text
subagent runtime:
  how DIRS runs editor/simulator/evaluator roles

API_feasibility_tool:
  a selected task node only when the experiment itself needs hosted models,
  paid APIs, rate limits, pricing, or credential checks
```

Do not select `API2_API_feasibility_tool` merely because the simulator is an
LLM. Select it only when the experiment plan would actually call or compare
hosted APIs.

The graph is a real DAG, not a single chain. Tool nodes may have fan-out:

```text
G2_GPU_resource_tool
  -> P1_protocol_constraints
  -> CR1_resource_estimate
  -> X1_execution_plan
  -> S1_bounded_claim

API2_API_feasibility_tool
  -> B1_baseline_set
  -> P1_protocol_constraints
  -> CR1_resource_estimate
  -> X1_execution_plan
  -> S1_bounded_claim

BM2_benchmark_rule_tool
  -> B1_baseline_set
  -> M1_metric_choice
  -> P1_protocol_constraints
  -> S1_bounded_claim
```

Decision nodes may also have multiple parents:

```text
X1_execution_plan depends on:
  C2_codebase_protocol_tool
  D2_dataset_inspection_tool
  G2_GPU_resource_tool
  API2_API_feasibility_tool
  DR2_smoke_test_tool
```

## Node Properties

```yaml
H1_problem_claim:
  content_skill: state the exact claim the experiment is meant to test
  action_skill: convert broad motivation into one measurable proposition
  tool_skill: no tool call unless the claim depends on unknown source facts

H2_testable_hypothesis:
  content_skill: define what outcome would support or reject the claim
  action_skill: write as a falsifiable prediction, not a hope
  tool_skill: no tool call; this is a reasoning node

T1_target_task_or_setting:
  content_skill: bind the experiment to a concrete task, dataset, simulator, or environment
  action_skill: avoid generic "evaluate on benchmarks" language
  tool_skill: inspect local docs or official benchmark pages if the setting is uncertain

D1_dataset_or_environment:
  content_skill: select data or environment supported by the problem setting
  action_skill: record split rules, scale, labels, and access assumptions
  tool_dependency: D2_dataset_inspection_tool when file/schema/split state is unknown

C1_codebase_protocol_check:
  content_skill: know which scripts, configs, evaluators, and checkpoints are actually available
  action_skill: adapt the experiment plan to runnable code rather than idealized code
  tool_dependency: C2_codebase_protocol_tool

G1_GPU_resource_check:
  content_skill: know whether the experiment is feasible on available hardware
  action_skill: decide whether to run full, subset, smoke-test, CPU-only, or simulation-only experiments
  tool_dependency: G2_GPU_resource_tool when local compute is required

API1_API_feasibility_check:
  content_skill: know model/API availability, authentication state, pricing, rate limits, and expected call volume
  action_skill: decide whether API experiments are feasible, need batching/caching, or should be replaced by local models
  tool_dependency: API2_API_feasibility_tool when hosted models or paid APIs are required

BM1_benchmark_rule_check:
  content_skill: know official split, metric, negative sampling, leaderboard, and submission rules
  action_skill: prevent an invalid or incomparable experiment design
  tool_dependency: BM2_benchmark_rule_tool when comparability or official protocol matters

B1_baseline_set:
  content_skill: choose relevant strong, simple, and domain-specific baselines
  action_skill: separate must-have baselines from stretch baselines
  tool_skill: inspect whether baseline code/configs exist; if not, mark as external or reproduction-risk

M1_metric_choice:
  content_skill: choose metrics that directly test the hypothesis
  action_skill: define primary metric before secondary diagnostics
  tool_skill: verify evaluator implementation or benchmark metric definition before finalizing the protocol

P1_protocol_constraints:
  content_skill: specify sampling, seeds, leakage boundaries, compute, and statistical tests
  action_skill: make reproduction-critical choices explicit
  tool_skill: check whether configs expose seeds, split files, evaluator flags, and logging outputs

CR1_resource_estimate:
  content_skill: estimate runtime, GPU memory, GPU-hours, storage, API call count, and monetary cost when relevant
  action_skill: distinguish measured, estimated, and unknown resource assumptions
  tool_dependency: G2_GPU_resource_tool, API2_API_feasibility_tool, and DR2_smoke_test_tool when applicable

DR1_smoke_test_or_dry_run:
  content_skill: know whether the planned command path can execute at minimal scale
  action_skill: decide whether the design is runnable or blocked before proposing full runs
  tool_dependency: DR2_smoke_test_tool when the command path is uncertain or expensive

X1_execution_plan:
  content_skill: order concrete runs, configs, and expected artifacts
  action_skill: write as a checklist that can be run
  tool_skill: convert verified commands and resource checks into a run queue

R1_expected_result_pattern:
  content_skill: state what result pattern would be meaningful
  action_skill: avoid claiming success before results exist
  tool_skill: no tool call unless using prior checked baselines to calibrate plausible effect sizes

F1_failure_interpretation:
  content_skill: explain how to read null, mixed, or negative results
  action_skill: preserve scientific value even when the method fails
  tool_skill: use logs, error messages, or failed smoke tests to separate scientific failure from infrastructure failure

S1_bounded_claim:
  content_skill: state what the experiment can and cannot establish
  action_skill: avoid overclaiming beyond protocol coverage
  tool_skill: bind the claim to actually checked data, tool, API, GPU, and benchmark constraints
```

## Tool Node Properties

```yaml
TR1_tool_need_router:
  node_type: tool_router
  content_skill: identify which parts of the experiment depend on unknown local or external state
  action_skill: choose the minimum necessary tool nodes before drafting the plan
  tool_skill: no external call; routes to code, data, benchmark, GPU, API, or dry-run nodes
  output_contract:
    - required_tool_nodes
    - skipped_tool_nodes
    - reason_for_each_skip

D2_dataset_inspection_tool:
  node_type: tool_call
  trigger: dataset files, schema, split, labels, or leakage boundaries are unknown
  tool_skill: inspect dataset paths, schema files, split files, and small samples when safe
  allowed_tools:
    - ls/find
    - rg
    - lightweight schema or row-count inspection
  output_contract:
    - dataset_available
    - split_available
    - schema_summary
    - leakage_risk
  failure_policy: mark dataset state unknown and design only smoke-test or placeholder protocol
  effect_on_plan: constrains dataset choice, split policy, and leakage checks

C2_codebase_protocol_tool:
  node_type: tool_call
  trigger: experiment should run in an existing repository
  tool_skill: inspect scripts, configs, evaluators, checkpoints, and documented commands
  allowed_tools:
    - rg
    - ls/find
    - config/help command inspection
  output_contract:
    - runnable_scripts
    - required_configs
    - evaluator_path
    - missing_implementation_pieces
  failure_policy: mark implementation blocked or propose design-only experiment
  effect_on_plan: changes baselines, run queue, and reproduction risk

BM2_benchmark_rule_tool:
  node_type: tool_call
  trigger: experiment claims comparability to a benchmark or leaderboard
  tool_skill: verify official split, metric, negative sampling, submission, and leaderboard rules
  allowed_tools:
    - local docs
    - official benchmark pages
    - paper/source lookup
  output_contract:
    - official_metric
    - split_rule
    - sampling_rule
    - comparability_constraints
  failure_policy: mark benchmark comparability unverified
  effect_on_plan: constrains metric choice, baseline set, and claim strength

G2_GPU_resource_tool:
  node_type: tool_call
  trigger: experiment uses local training, inference, evaluation, or large data processing
  tool_skill: inspect available GPU count, memory, running jobs, disk, and expected runtime limits
  allowed_tools:
    - nvidia-smi
    - disk and memory inspection
    - job/process inspection
  output_contract:
    - gpu_available
    - available_memory
    - running_jobs
    - bottleneck
    - full_subset_or_smoke_recommendation
  failure_policy: fall back to CPU/subset/smoke-test planning or mark compute blocked
  effect_on_plan: changes batch size, seed count, model size, subset size, and run schedule

API2_API_feasibility_tool:
  node_type: tool_call
  trigger: experiment depends on hosted models, paid APIs, or rate-limited services
  tool_skill: check model/API availability, credential presence, pricing, rate limits, and call volume
  allowed_tools:
    - environment-variable presence check without printing secrets
    - official docs lookup when needed
    - tiny smoke-test call only when explicitly permitted
  output_contract:
    - api_available
    - credential_state
    - rate_limit_or_cost_risk
    - batching_or_cache_policy
    - fallback_model
  failure_policy: mark API state unknown or blocked; propose local or smaller fallback
  effect_on_plan: changes model choice, sample size, caching, budget, and feasibility claim

DR2_smoke_test_tool:
  node_type: tool_call
  trigger: full experiment would be costly or command path is uncertain
  tool_skill: run a tiny dry run, import check, config validation, help command, or evaluator-only pass
  allowed_tools:
    - tiny command run
    - import check
    - config validation
    - evaluator no-training pass
  output_contract:
    - runnable_status
    - observed_error
    - minimum_successful_command
    - repair_or_fallback_plan
  failure_policy: do not propose full run until blocked command path is repaired or explicitly marked risky
  effect_on_plan: determines whether the execution plan is runnable, blocked, or design-only
```

## No-Jump Edges

```text
problem claim -> testable hypothesis:
  the reader must know what is being tested before seeing the protocol

testable hypothesis -> metric choice:
  metrics are justified by the hypothesis, not chosen after results

dataset/environment -> baseline set:
  baselines must be valid for the selected task and protocol

dataset/environment -> codebase and benchmark check:
  the plan must verify that the selected setting is locally or externally runnable

codebase and benchmark check -> GPU/API/resource check:
  resource estimates depend on the actual commands, models, and evaluators

GPU/API/resource check -> protocol constraints:
  seeds, subsets, batching, and cache policy depend on available resources

metric choice -> expected result pattern:
  result interpretation depends on the metric definition

smoke test or dry run -> execution plan:
  the full run queue should be based on a command path that can at least start

protocol constraints -> bounded claim:
  final claims must stay inside the evaluated setting
```

## Tool-Node Edges

```text
T1_target_task_or_setting -> TR1_tool_need_router:
  the task determines which local or external state must be checked

TR1_tool_need_router -> D2_dataset_inspection_tool:
  use when data availability, schema, splits, or leakage boundaries are unknown

TR1_tool_need_router -> C2_codebase_protocol_tool:
  use when the experiment should be implemented in a local or known repository

TR1_tool_need_router -> BM2_benchmark_rule_tool:
  use when official comparability is part of the claim

TR1_tool_need_router -> G2_GPU_resource_tool:
  use when local compute affects feasibility

TR1_tool_need_router -> API2_API_feasibility_tool:
  use when hosted model/API calls affect feasibility

C2_codebase_protocol_tool -> DR2_smoke_test_tool:
  dry runs depend on knowing the candidate command path

D2_dataset_inspection_tool -> P1_protocol_constraints:
  split and leakage decisions depend on inspected data state

BM2_benchmark_rule_tool -> M1_metric_choice:
  official metrics must be known before metric selection

G2_GPU_resource_tool -> X1_execution_plan:
  run scale and schedule depend on available compute

G2_GPU_resource_tool -> CR1_resource_estimate:
  runtime, batch size, memory, and GPU-hour estimates depend on checked hardware

G2_GPU_resource_tool -> P1_protocol_constraints:
  seeds, subset size, and batch size depend on available compute

G2_GPU_resource_tool -> S1_bounded_claim:
  feasibility claims must be bounded by checked hardware

API2_API_feasibility_tool -> X1_execution_plan:
  call volume, batching, caching, and fallback depend on API feasibility

API2_API_feasibility_tool -> CR1_resource_estimate:
  API cost and rate-limit estimates depend on checked API state

API2_API_feasibility_tool -> B1_baseline_set:
  hosted-model baselines depend on API access and cost

API2_API_feasibility_tool -> S1_bounded_claim:
  any API-based claim must disclose availability, cost, and rate-limit limits

DR2_smoke_test_tool -> X1_execution_plan:
  full execution plan should follow from a checked runnable path

DR2_smoke_test_tool -> CR1_resource_estimate:
  dry-run timing and errors calibrate the full-run resource estimate

DR2_smoke_test_tool -> F1_failure_interpretation:
  failed smoke tests should be interpreted as infrastructure blockers, not as
  scientific negative results
```

## Simulator

```text
1. read chip/source/code context
2. infer the claim to be tested
3. select a connected experiment-design sub-DAG
4. call local or external tools only for facts that require verification
5. inspect code, data, benchmark rules, GPU/API feasibility, or dry-run path as needed
6. draft the experiment plan
7. run verifier checks
8. repair missing controls, weak baselines, unsupported protocol claims, or infeasible tool assumptions
```

## Tool-Calling Nodes

```yaml
GPU_resource_check:
  node_id: G2_GPU_resource_tool
  trigger: experiment uses local training, inference, evaluation, or large data processing
  allowed_tools:
    - nvidia-smi
    - disk and memory inspection
    - job/process inspection
  output:
    - available GPU count and memory
    - likely bottleneck
    - full/subset/smoke-test recommendation

API_feasibility_check:
  node_id: API2_API_feasibility_tool
  trigger: experiment depends on hosted models, paid APIs, or rate-limited services
  allowed_tools:
    - environment-variable presence check without printing secrets
    - official docs lookup when needed
    - tiny smoke-test call only when permitted
  output:
    - available/unavailable/unknown API state
    - estimated call volume and cost risk
    - batching, caching, or fallback recommendation

codebase_protocol_check:
  node_id: C2_codebase_protocol_tool
  trigger: experiment should run in an existing repository
  allowed_tools:
    - rg
    - ls/find
    - config/help command inspection
  output:
    - runnable scripts
    - required configs
    - evaluator path
    - missing implementation pieces

benchmark_rule_check:
  node_id: BM2_benchmark_rule_tool
  trigger: experiment claims comparability to a benchmark or leaderboard
  allowed_tools:
    - local docs
    - official benchmark pages
    - paper/source lookup
  output:
    - split and metric rules
    - negative sampling or seed rules
    - comparability constraints

smoke_test_or_dry_run:
  node_id: DR2_smoke_test_tool
  trigger: full experiment would be costly or the command path is uncertain
  allowed_tools:
    - tiny command run
    - import check
    - config validation
    - evaluator no-training pass
  output:
    - runnable/blocked status
    - observed error
    - repair or fallback plan
```

## Verifier

```text
falsifiable hypothesis present
primary metric defined
baseline set is relevant
tool calls were made when feasibility depended on external/local state
tool outputs are summarized without exposing secrets
protocol avoids future leakage or data contamination
resource estimate is plausible
API/GPU/code/data assumptions are marked checked, unknown, or blocked
failure interpretation exists
claims are bounded by evidence
```
