# DIRS-TTA Training And Test-Time Adaptation Ablation Study

Date: `2026-07-28`

Purpose: define a large, controlled comparison suite for the proposed
test-time-adaptive extension of DIRS. The canonical machine-readable list is:

```text
dirs_tta_ablation_variants.csv
```

This document is an experiment specification, not a claim that every proposed
component is already implemented or validated.

For research-gap discovery on unseen topics, the state hierarchy, legal
self-supervision objectives, temporal cutoff, adaptation ladder, and
Chip-Memory interface are defined in:

```text
08_dirs_topic_test_time_learning_integration.md
```

## 1. Research Question

The main question is:

```text
At fixed task evidence and matched compute, does task-local adaptation of a
typed executable skill graph improve final utility, calibration, hard-failure
rate, and search efficiency over static DIRS, ordinary test-time search,
weight-based TTT, and compute-matched single-model refinement?
```

The study must separate five possible sources of improvement:

```text
representation:
  typed nodes, directed edges, ports, gates, evidence bindings, and macros

offline learning:
  inverse trace learning, contextual priors, replay, and graph governance

test-time adaptation:
  local posterior, gates, budget, recurrent state, local graph patches, or LoRA

search:
  greedy, best-arm identification, posterior sampling, or MCTS

system architecture:
  recurrent state, multi-agent communication, verification, memory, and routing
```

## 2. Canonical Full Configuration

`FULL-000` is the proposed full DIRS-TTA reference. It uses:

```yaml
persistent_state:
  graph: typed connected skill DAG
  representation:
    - content/style-action separation
    - typed node input/output ports
    - AND/OR dependency groups
    - context gates
    - evidence bindings
    - rejection rules
    - primitive nodes plus expandable macro motifs
  learning:
    - multi-hypothesis inverse trace posterior
    - context-conditioned node, edge, path, and graph priors
    - batch outer-loop updates
    - one attributable edit family per proposal
    - replay, rollback, provenance, and transfer gates

task_local_state:
  mutable:
    - path and graph posterior
    - node and edge utility estimates
    - active context gates
    - remaining budget
    - recurrent execution state
    - temporary graph patches
  forbidden:
    - reading the hidden target
    - mutating the persistent graph during a blind test episode

controller:
  search_policy_router:
    enumerable_paths: best-arm identification
    reliable_prefix_signal: posterior-aware MCTS
    uncertain_prefix_signal: posterior sampling with protected coverage
    simple_or_low_budget_case: greedy valid selection
  recommendation: risk-calibrated posterior utility
  stop: confidence, hard-constraint completion, and marginal-value-per-cost

execution:
  plan_plane: parallel heterogeneous skill proposals
  state_plane: sequential recurrent state carryover
  communication: sparse typed state plus latent/KV transfer when compatible
  verifier: deterministic hard checks plus an independent structured evaluator
  repair: typed failure-conditioned repair
  memory: ranked and diversified successful/failed traces with provenance
```

Weight-based TTT is not part of the default full configuration. It is an
optional comparison because many DIRS writing tasks do not provide legitimate
test-time supervision for parameter updates.

For topic-level research-gap discovery, instantiate the generic state as:

```text
global slow state:
  skill graph G, backbone, validator, hindsight editor, and router

topic-local state:
  topic latent u_T, path posterior, gates, budget, local graph patch, and
  optional temporary LoRA Delta_T

single-exploration state:
  current gap g_t, recurrent belief h_t, bound evidence, failure signature,
  and accumulated cost
```

The topic-local and exploration states must not write directly into the global
state during blind evaluation.

## 3. Three Evaluation Regimes

Every variant must declare the regime in which it is valid.

### F: Feedback-Only Blind Generation

Examples:

```text
paper writing from chips
research-gap verification
experiment-plan generation
question proposal
```

Allowed adaptation signals:

```text
source provenance
deterministic constraints
tool results
independent verifier reports
cross-view consistency
budget and execution feedback
```

Forbidden:

```text
the held-out expert artifact
post-generation comparison scores used during adaptation
pseudo-labels copied or paraphrased from the hidden target
```

### D: Demonstration-Supervised Test-Time Adaptation

Examples:

```text
ARC tasks with training input-output pairs
few-shot transformation tasks
tasks with an explicit support set and held-out query
```

The support demonstrations may be transformed or used leave-one-out. The query
target remains hidden.

### E: Executable-Feedback Adaptation

Examples:

```text
code repair with tests
program synthesis with an executor
interactive environments
tool-use tasks with observable state transitions
```

Execution feedback may guide adaptation, but the final hidden evaluation target
must not be exposed.

## 4. Temporal Topic Split For Gap Discovery

The central gap-discovery experiment must use:

```text
complete topic holdout
  + a preregistered time cutoff inside each held-out topic
  + adaptation using only pre-cutoff papers
  + frozen gap predictions before revealing post-cutoff work
```

Randomly splitting papers from the same topic is permitted only as a diagnostic
inflation control. It is not evidence of unseen-topic or future-gap discovery.

Legal topic-time supervision is restricted to:

```text
time-ordered leave-one-paper-out on pre-cutoff papers
masked pre-cutoff Chip fields or graph edges
support/refute/near-miss evidence classification
strict chronology, alias, identity, provenance, and graph consistency
```

Self-generated gaps and same-model novelty judgments are not ground truth.

## 5. Canonical Variant Catalog

The CSV contains `FULL-000` plus 121 controlled variants in ten families:

```text
BASE: external and compute-matched baselines
REP:  skill-graph representation
TRN:  offline graph/trace training and governance
TTA:  task-local test-time adaptation object
SEA:  search, allocation, backup, recommendation, and stopping
STA:  recurrent state, plan fusion, and communication
VER:  verifier, reward, and repair
MS:   memory and scheduling
ROB:  distribution shift, noise, and negative-transfer stress tests
```

Most rows change one factor relative to `FULL-000`. Standalone baselines change
the entire method and are marked as such. Do not interpret a comparison between
two rows that also differ in compute, evidence, generator, or verifier.

## 6. Minimum Main-Paper Panels

Running all variants is useful for diagnosis, but the main paper should organize
the results into interpretable panels.

### Panel A: Main Method

```text
FULL-000
BASE-001 direct evidence prompt
BASE-002 retrieved style examples
BASE-003 flat skill list
BASE-006 static greedy DIRS
BASE-007 compute-matched self-consistency
BASE-008 compute-matched self-refinement
BASE-009 compute-matched single recurrent model
TTA-001 no task-local adaptation
SEA-015 fixed MCTS without the policy router
```

### Panel B: What Learns At Test Time?

```text
TTA-001 no adaptation
TTA-002 terminal Q only
TTA-003 path posterior
TTA-004 node posterior
TTA-005 edge posterior
TTA-006 context gates
TTA-007 budget controller
TTA-008 local structural patch
TTA-009 latent task vector
TTA-010 prompt/prefix parameters
TTA-011 LoRA
TTA-012 full model parameters
TTA-016 posterior plus latent
TTA-017 posterior plus LoRA
TTA-018 graph plus latent plus LoRA
FULL-000 multi-object non-weight adaptation
```

### Panel C: Does MCTS Earn Its Complexity?

```text
SEA-001 exhaustive legal-path oracle, only when feasible
SEA-002 even allocation
SEA-003 Successive Rejects
SEA-004 Sequential Halving
SEA-005 Top-Two Thompson sampling
SEA-006 UCT
SEA-007 PUCT with empirical-frequency prior
SEA-008 PUCT with contextual prior
SEA-009 posterior root sampling
SEA-010 protected branch coverage
SEA-011 mean backup
SEA-012 raw maximum backup
SEA-013 top-quantile backup
SEA-014 calibrated best-descendant posterior
FULL-000 adaptive search-policy router
```

### Panel D: Advanced Architecture

```text
STA-001 no recurrent state
STA-002 text scratchpad
STA-003 continuous latent vector
STA-004 layer-wise KV cache
STA-005 typed graph state
STA-006 plan-only
STA-007 state-only
STA-008 parallel-plan averaging
STA-009 attention plan fusion
STA-010 candidate-selection fusion
STA-011 text communication
STA-012 latent/KV communication
FULL-000 parallel plan plus sequential state
```

### Panel E: Safety And Governance

```text
TRN-007 no executor feedback during training
TRN-008 no verifier feedback during training
TRN-009 no replay
TRN-010 no rollback/governance
VER-001 no verifier
VER-002 scalar verifier only
VER-009 generic repair instead of typed repair
VER-010 no hard factual-support gate
ROB-006 noisy verifier
ROB-008 task-local state carried across unrelated tasks
```

### Panel F: Topic-Level Adaptation Ladder

```text
TTA-001 frozen DIRS
BASE-002 frozen DIRS plus retrieval
TTA-009 topic latent
TTA-020 topic latent plus recurrent repair
TTA-021 topic latent plus recurrent repair plus hindsight editor
TTA-022 add temporary LoRA under legal supervision
TTA-023 add dynamic sparse routing
ROB-009 complete topic holdout plus temporal cutoff
ROB-010 random paper split, reported only as an inflation diagnostic
```

## 7. Required Factor Sweeps

The atomic variants isolate mechanisms. Run the following sweeps separately;
do not add every Cartesian product to the main ablation table.

```yaml
training_artifacts_per_domain: [2, 4, 8, 16, 32, all]
training_domains: [1, 3, 5, 10, all]
trace_hypotheses_per_artifact: [1, 2, 4, 8, 16]
accepted_graph_nodes: [16, 32, 64, 128, 256, full]
legal_terminal_paths: [4, 8, 16, 32, 64, 128, implicit]
test_time_adaptation_steps: [0, 1, 2, 4, 8, 16]
terminal_executions: [4, 8, 16, 32, 64, 128]
search_expansions: [32, 64, 128, 256, 512, 1024]
parallel_proposers: [1, 2, 4, 8]
recurrent_steps_per_node: [0, 1, 2, 4, 8]
memory_top_k: [0, 1, 2, 4, 8, 16]
state_transfer_budget_bytes: [0, 4096, 16384, 65536, 262144]
context_window_fraction: [0.25, 0.5, 0.75, 1.0]
verifier_samples_per_candidate: [1, 2, 4, 8]
prior_uniform_mixture: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
evidence_dropout_rate: [0.0, 0.1, 0.25, 0.5]
verifier_label_noise: [0.0, 0.05, 0.1, 0.2]
pre_cutoff_papers_per_topic: [4, 8, 16, 32, 64, all]
cutoff_horizon_years: [1, 2, 3, 5]
future_evaluation_horizon_years: [1, 2, 3, 5]
```

Use a fractional factorial or response-surface design for interactions. The
highest-priority interactions are:

```text
representation x test-time adaptation
test-time adaptation x feedback regime
search policy x prefix predictiveness
recurrent state x communication type
memory type x distribution shift
verifier type x repair type
scheduler x compute budget
LoRA TTT x number of demonstrations
topic latent x pre-cutoff corpus size
hindsight editor x failure-signature quality
temporal cutoff x future evaluation horizon
```

## 8. Compute Matching

Every comparison must declare one of these match groups:

```text
C-GEN:
  same generator, total generated tokens, and candidate count

C-EXEC:
  same number of terminal executions or tool calls

C-SEARCH:
  same terminal executions plus the same search-expansion ceiling

C-ADAPT:
  same adaptation FLOPs, steps, trainable parameter count, and demonstrations

C-MAS:
  same total model FLOPs/tokens/calls across all agents

C-TRAIN:
  same training artifacts, outer rounds, graph proposals, and verifier calls

C-STRESS:
  same trained checkpoint and controller; only the test condition changes
```

If exact FLOP matching is unavailable, report tokens, calls, wall-clock time,
peak memory, bytes transferred, and dollar cost separately. Do not call two
systems compute-matched solely because they have the same rollout count.

## 9. Metrics

### Task Utility

```text
task success or exact match, when defined
blind human pairwise preference
evidence-grounded content fidelity
structure and dependency validity
section/task-role fit
bounded claim strength
diagnostic usefulness
```

### Adaptation

```text
utility after each adaptation step
adaptation gain over the static initialization
area under the adaptation curve
steps to first valid solution
posterior entropy reduction
probability assigned to the selected legal path
local-patch acceptance and rollback rate
negative-transfer rate after task reset
```

### Search

```text
simple regret
cumulative regret
Oracle@K
SelectionAcc@K
best-path discovery rate
prefix-value calibration
deception rate
valid-frontier violation count
search expansions per accepted artifact
```

### Reliability

```text
unsupported-claim rate
hard-failure rate
no-jump violations
evidence-binding precision and recall
verifier false-positive and false-negative rates
expected calibration error
inter-evaluator disagreement
replay regression
```

### Topic-Level Gap Discovery

```text
pre-cutoff novelty and false-gap rate
evidence adequacy and near-miss discrimination
scope precision
future-uptake precision and recall
future support, refutation, and unresolved rates
time to first relevant post-cutoff work
expert-priority ranking correlation
future-information leakage count
```

### Efficiency

```text
total generated tokens
adaptation and inference FLOPs
LLM/agent calls
tool and executor calls
wall-clock latency
peak GPU and CPU memory
state/KV bytes transferred
active-agent ratio
energy or monetary cost, when measurable
```

Do not hide the result behind one composite score. Report a quality-cost Pareto
front. A scalar utility may be used for controller training only if all
coefficients are fixed on validation data before final testing.

## 10. Run Identity And Result Record

Use:

```text
<variant_id>__<regime>__<task>__<domain>__B<budget>__S<seed>
```

Example:

```text
TTA-003__F__abstract__reasoning_memory__B64__S20260728
```

Every run must save:

```yaml
run:
  run_id:
  variant_id:
  git_commit:
  config_hash:
  regime: F | D | E
  task:
  domain:
  split_manifest:
  topic_holdout_manifest:
  temporal_cutoff:
  future_evaluation_horizon:
  seed:
  evidence_manifest:
  hidden_target_hash:

method:
  graph_hash:
  generator_version:
  executor_version:
  verifier_versions: []
  trainable_objects: []
  trainable_parameter_count:
  search_policy:
  recommendation_policy:
  adaptation_steps:
  persistent_state_mutated: false

compute:
  generated_tokens:
  model_flops:
  model_calls:
  tool_calls:
  terminal_executions:
  search_expansions:
  wall_time_seconds:
  peak_gpu_bytes:
  peak_cpu_bytes:
  communication_bytes:

results:
  primary_utility:
  task_metrics: {}
  adaptation_curve: []
  frozen_gap_predictions_uri:
  post_cutoff_reveal_manifest:
  search_metrics: {}
  reliability_metrics: {}
  verifier_metrics: {}
  hard_failures: []
  local_graph_diff:
  replay_result:
```

## 11. Statistical Protocol

```text
unit of analysis:
  task/paper, not rollout

pairing:
  use the same tasks, evidence, generator checkpoint, budgets, and seeds

minimum repeats:
  at least 5 generation/search seeds per task for stochastic systems

intervals:
  paired bootstrap confidence intervals over tasks

hypothesis tests:
  paired randomization or permutation tests when distributions are irregular

multiple comparisons:
  Holm correction for the preregistered core panel
  Benjamini-Hochberg false-discovery control for the exploratory catalog

selection:
  tune thresholds and scalarization weights on validation only
  never select the reported variant after inspecting final test results
```

Report effect sizes and uncertainty even when a significance threshold is not
crossed. A failed or harmful component narrows the method claim and must remain
in the ablation table.

## 12. Recommended Execution Order

### Phase 0: Integrity

Run leakage, determinism, evidence-binding, persistent-state freeze, and
compute-accounting tests.

### Phase 1: Baselines And Static DIRS

Run `BASE-*`, `REP-*`, and `TTA-001`. This establishes whether the graph itself
has value before claiming test-time learning.

### Phase 2: Search Audit

Run `SEA-*` on balanced multi-path counterfactual executions. Enable real-task
MCTS only if prefix statistics or reusable computation justify tree search over
best-arm alternatives.

### Phase 3: Test-Time Adaptation

Run `TTA-*` separately under F, D, and E. Parameter-update variants are valid
only with legitimate support demonstrations or executable/self-supervised
feedback.

For topic-level gap discovery, run the adaptation ladder in order:

```text
topic latent
  -> recurrent gap/belief repair
  -> hindsight failure-to-repair editor
  -> temporary LoRA, when legal
  -> dynamic sparse routing
```

### Phase 4: State, Communication, Verification, And Memory

Run `STA-*`, `VER-*`, and `MS-*` under matched total system compute.

### Phase 5: Robustness And Scaling

Freeze all choices before running `ROB-*` and the final test split. Use the
factor sweeps to build quality-cost and adaptation-speed curves.

## 13. Interpretation Rules

```text
FULL-000 > TTA-001:
  evidence for task-local adaptation beyond static DIRS

TTA-003 > TTA-011 under F:
  graph-posterior adaptation is more suitable than weight TTT without labels

TTA-011 > TTA-003 under D:
  parameter adaptation benefits from legitimate demonstrations

FULL-000 > BASE-009 at matched FLOPs:
  evidence for structured multi-component adaptation beyond extra recurrence

FULL-000 > SEA-015:
  evidence for routing among search policies rather than always using MCTS

SEA-014 > SEA-011:
  evidence that best-descendant-aware backup better matches path selection

STA-012 > STA-011 at matched total bytes and FLOPs:
  evidence for latent rather than textual communication

VER-006 > VER-004:
  evidence that independent hybrid verification reduces self-judging bias

MS-005 > MS-003:
  evidence that ranked successful/failed memory is better than success-only

ROB-008 degrades strongly:
  evidence that task-local state must be reset or explicitly scoped

TTA-020 > TTA-009:
  evidence that recurrent candidate/belief repair adds value beyond topic encoding

TTA-021 > TTA-020:
  evidence that failure-conditioned hindsight policies transfer across topics

ROB-010 > ROB-009 by a large margin:
  evidence that random paper splits inflate apparent unseen-topic performance
```

No single comparison establishes the whole method. The final claim should name
only the components that improve held-out performance, reliability, or
efficiency under the preregistered protocol.
