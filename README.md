# DIRS: DAG-Inferred Reusable Skills

Date: `2026-07-25`

Standalone snapshot: `2026-08-13`

This repository is a standalone archive of the DIRS research experience. It
contains the DIRS-authored Python and shell code, Markdown documentation and run
reports, learned YAML graphs, verifier and control state, JSON/JSONL experiment
results, and historical execution logs. Downloaded model weights, caches, raw
PyTorch trajectory tensors, bytecode, and embedded third-party repositories are
intentionally excluded. See [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md) for the
exact packaging boundary.

The Chinese research-gap assessment that motivated this snapshot is available
at [docs/dirs_research_gap_conversation_2026-08-04.md](docs/dirs_research_gap_conversation_2026-08-04.md).
Historical scripts and reports may retain absolute paths from the original
execution environment; adjust those paths before rerunning experiments.

## Overview

DIRS is a method for learning reusable writing and task-execution skills from
expert artifacts. Instead of storing examples as raw demonstrations or
compressing them into a single prompt, DIRS infers the latent directed structure
that makes an artifact work:

```text
evidence -> rhetorical/action unit -> dependency order -> verifier -> repair
```

For paper writing, DIRS separates two systems that are usually mixed together:

```text
content system:
  paper chips, source facts, tables, equations, results, claims, constraints

style system:
  section role, discourse order, paragraph budget, evidence placement,
  claim strength, transition patterns, and length priors
```

The learned object is a connected directed skill graph. Nodes describe reusable
content/style units; edges describe allowed dependency order and transitions.
During training and inference, MCTS searches for an input-supported connected
sub-DAG, the simulator/generator executes that path, and the verifier scores the
result. Training sends that score back to the editor/controller to repair the
shared graph; MCTS itself does not modify the graph. Inference uses the score to
accept, repair, or reject the candidate output.

## Runtime Policy

For heavy DIRS training and inference, use Codex subagents as the default
execution engine. Do not use hosted model APIs for the DIRS editor, simulator,
or evaluator unless the user explicitly asks for an API run.

```text
default heavy runtime:
  coordinator in main Codex thread
  editor subagent
  simulator subagent
  evaluator subagent

not default:
  OpenAI/API batch runner for the DIRS roles
```

API checks may still appear as first-class tool nodes inside downstream tasks
such as experiment design. That means the experiment itself may depend on an API
baseline or hosted-model feasibility check. It does not mean DIRS should use API
calls to run its own training loop.

## Names

```text
DIRS      = DAG-Inferred Reusable Skills
DISL      = DAG-Inferred Skill Learning, the training procedure
DIRS-MCTS = DIRS skill library plus MCTS sub-DAG selection for training and inference
```

## Research Claim

DIRS is intended to test the following hypothesis:

```text
Expert writing skill can be represented more reliably as a typed,
evidence-grounded dependency graph than as a flat prompt or unstructured style
example. When the graph stores both content and style properties, a model can
generate stronger blind drafts from paper chips while reducing unsupported
claims, ordering jumps, and wrong-domain rhetorical moves.
```

## Core Contributions

1. `Dual-system skill representation`: separates source-backed content from
   style/action constraints.
2. `Inverse DAG learning`: infers nodes, edges, rewards, and repair rules from
   expert artifacts.
3. `Contextual probabilistic priors`: uses frequency as structural evidence
   while learning or calibrating which legal paths fit the current task.
4. `MCTS sub-DAG selection`: searches over compatible connected paths during
   both training simulation and held-out inference.
5. `No-jump verification`: rejects drafts that skip required dependencies,
   introduce unsupported facts, or make claims before evidence is established.
6. `Autonomous strategy discovery`: allows the learner to invent and test new
   procedural skills while keeping every generated factual claim grounded.

## Folder Structure

```text
general_version/
  Method specification, training loop, MCTS inference, representation patterns,
  and top-conference evaluation protocol.

uncertain_inverse_skill_learning/
  Detailed probabilistic extension for multi-hypothesis latent traces,
  context-conditioned skill-graph uncertainty, posterior-aware MCTS,
  outer-loop graph optimization, identifiability, replay, and rollback.

domain_topics/
  Semantic topic splits for oral papers. These files define which papers should
  be read together when learning domain priors.

case1_writing/
  Case 1 task adapter for applying DIRS-MCTS to scientific writing, including
  abstracts and full paper sections.

case2_experiment_design/
  Case 2 task adapter for designing falsifiable experiments, ablations,
  metrics, baselines, and execution protocols.

case3_research_question_proposal/
  Case 3 task adapter for proposing and ranking grounded, testable research
  questions from topic clusters, chips, or open-ended ideas.

case4_research_gap_verification/
  Case 4 task adapter for verifying whether a proposed research gap is real,
  partial, already solved, wrongly framed, or still unverified.
```

## Main Method Files

```text
general_version/README.md
general_version/01_dirs_general_method.md
general_version/02_dirs_training_cycle.md
general_version/03_dirs_mcts_inference_selector.md
general_version/04_dirs_skill_representation_patterns.md
general_version/05_dirs_top_conference_evaluation_protocol.md
general_version/06_dirs_mathematical_formulation.md
general_version/07_dirs_tta_ablation_study.md
general_version/dirs_tta_ablation_variants.csv
general_version/08_dirs_topic_test_time_learning_integration.md

uncertain_inverse_skill_learning/README.md
uncertain_inverse_skill_learning/01_problem_and_claim_boundary.md
uncertain_inverse_skill_learning/02_artifact_to_trace_posterior.md
uncertain_inverse_skill_learning/03_probabilistic_skill_graph.md
uncertain_inverse_skill_learning/04_learning_and_search_algorithms.md
uncertain_inverse_skill_learning/05_dual_loop_update_and_governance.md
uncertain_inverse_skill_learning/06_evaluation_and_identifiability.md
uncertain_inverse_skill_learning/07_existing_files_audit.md
uncertain_inverse_skill_learning/08_references.md
uncertain_inverse_skill_learning/09_learned_policy_vs_hard_constraints.md
uncertain_inverse_skill_learning/10_autonomous_strategy_discovery.md
uncertain_inverse_skill_learning/11_dag_flow_constrained_mcts.md
uncertain_inverse_skill_learning/12_stochastic_execution_and_prior_calibration.md
uncertain_inverse_skill_learning/13_search_policy_router.md
uncertain_inverse_skill_learning/14_prefix_value_and_backup_objective.md
uncertain_inverse_skill_learning/15_real_counterfactual_rollout_protocol.md
```

## Domain Source

```text
domain_topics/01_domain_topic_paper_splits.md
domain_topics/02_training_topic_routing.md
domain_topics/semantic_balanced_23_domains/INDEX.md
```

## Case Adapters

```text
case1_writing/01_abstract_writing.md
case1_writing/02_section_writing.md

case2_experiment_design/01_experiment_plan_design.md
case2_experiment_design/02_ablation_metric_protocol.md

case3_research_question_proposal/01_question_generation.md
case3_research_question_proposal/02_question_ranking.md

case4_research_gap_verification/01_gap_evidence_audit.md
case4_research_gap_verification/02_gap_claim_verification.md
```

## Reference Archive

Historical runs, scripts, support-score files, no-jump harness outputs, and
copied artifacts used to derive the current method are kept outside the active
DIRS package:

```text
/tf/notebooks/yunbo/DIRS_method_sources_reference_20260720
```

Cleanup and consistency audit notes are kept under:

```text
/tf/notebooks/yunbo/DIRS_method_sources_reference_20260720/maintenance
```

## Minimal Training-To-Inference Pipeline

```text
1. Select a semantic domain and downstream case adapter.
2. Read chips, sources, traces, or expert artifacts for the training cases.
3. Infer per-example task DAGs.
4. Aggregate node support, edge support, content bindings, and style priors.
5. Run repair loops until the shared graph stabilizes.
6. Hide the original text for a held-out chip.
7. Use MCTS to select a connected sub-DAG.
8. Simulate or generate the target artifact from supported evidence only,
   preferably through subagent roles for heavy runs.
9. Verify support, order, tool-use validity, budget, and no-leakage constraints.
10. Compare with the expert target only after generation when a target exists.
```

## Expected Outputs

A serious DIRS run should produce:

```text
domain_skill_library.json
skill_graph.yaml
node_library.json
edge_library.json
node_support_scores.md
edge_support_scores.md
style_profile.md
length_prior.json
mcts_policy.yaml
verifier_result.json
training_trace.jsonl
replay_cases.jsonl
accepted_updates.jsonl
rejected_updates.jsonl
```

## Current Status

This folder is the current method package. It contains the latest general DIRS
definition, the training cycle, the MCTS inference selector, external
skill-representation lessons, domain topic splits, and four downstream case
adapters.

The connected-path cached-rollout test and stochastic controller simulation
now emit legal-frontier, reward, backpropagation, baseline, and validation
artifacts. The stochastic audit found that raw trace frequency can
over-constrain MCTS under contextual shift, so contextual policy priors and
terminal reward posteriors are now separated explicitly.

The next implementation milestone is a real multi-rollout test that generates
several independent artifacts per visited path, anonymizes them, obtains blind
paired judgments, and calibrates the contextual prior on non-target validation
papers before evaluating a final holdout.

The selector is now conditional: small enumerable legal-path sets use
Sequential Halving by default or Top-Two Thompson when reward uncertainty is
calibrated; MCTS is reserved for large or implicit sub-DAG spaces where
prefix-level structure can reduce search cost. MCTS additionally requires
validation that prefix values predict the desired descendant objective; raw
mean or maximum backup is unsafe when rare optimal paths can occur inside
low-average branches.

The available 19-paper real training trace contains only one path per paper
and does not pass the prefix-value identifiability gate. Real-task MCTS remains
disabled pending balanced same-paper counterfactual multi-rollout evaluation.

For filesystem cleanup and consistency audit notes, use the external reference
archive's `maintenance/` folder.
