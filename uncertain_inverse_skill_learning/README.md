# DIRS Uncertain Inverse Skill Learning

Date: `2026-07-25`

Status: detailed general-method extension to `../general_version`.

## Purpose

This directory resolves a central ambiguity in learning skills from expert
artifacts:

```text
one final artifact can be compatible with many latent production traces
```

DIRS therefore should not claim to recover a unique historical expert thought
process from a paper, answer, code patch, or other final product. It should
infer a posterior over artifact-compatible information flows, distill their
cross-case regularities into a persistent probabilistic skill graph, and learn
an executable strategy that reproduces expert-level outcomes on new cases.

## Core Claim

```text
DIRS infers multiple evidence-consistent latent skill traces from expert
artifacts, learns a context-conditioned posterior over a shared typed skill
graph, and uses fixed-snapshot search to select and execute a task-specific
sub-DAG. Persistent graph edits are made only by an outer loop and are accepted
only after typed error attribution, paired validation, replay, and rollback
checks.
```

## Reading Order

```text
01_problem_and_claim_boundary.md
  Defines what is observed, latent, and learnable, and separates historical,
  explanatory, and operational traces.

02_artifact_to_trace_posterior.md
  Defines multi-hypothesis trace extraction, evidence provenance, forward
  replay, counterfactual testing, and posterior weighting.

03_probabilistic_skill_graph.md
  Defines canonical skill nodes, semantic and execution graphs, contextual
  node/edge uncertainty, posterior aggregation, and graph invariants.

04_learning_and_search_algorithms.md
  Assigns distinct roles to conditional GFlowNet or posterior sampling, ES or
  bounded graph editing, and fixed-snapshot MCTS.

05_dual_loop_update_and_governance.md
  Defines outer-loop graph updates, inner-loop execution, typed errors,
  acceptance tests, diffs, versions, and rollback.

06_evaluation_and_identifiability.md
  Defines experiments for trace uncertainty, graph calibration, intervention,
  held-out execution, ablations, and defensible scientific claims.

07_existing_files_audit.md
  Records the file-by-file review of ../general_version and the precise
  extension or correction made here.

08_references.md
  Lists the closest prior methods and which technical component is borrowed.

09_learned_policy_vs_hard_constraints.md
  Separates legitimate logical/safety constraints from node inventories,
  weights, thresholds, routes, and utilities that must be learned from data.

10_autonomous_strategy_discovery.md
  Separates factual grounding from strategy induction and defines how an agent
  may invent, test, retain, transfer, or reject new writing strategies without
  copying an expert artifact.

11_dag_flow_constrained_mcts.md
  Defines frontier-constrained node selection, typed information flow,
  macro-to-subgraph expansion, and the required evidence for a real MCTS run.

12_stochastic_execution_and_prior_calibration.md
  Separates graph, path, execution, and evaluator uncertainty; defines online
  rollout evidence, contextual prior calibration, and robust recommendation
  rules discovered by the stochastic connected-path audit.

13_search_policy_router.md
  Routes small enumerable legal-path problems to pure-exploration methods and
  reserves MCTS for large or implicit DAGs with useful structural expansion.

14_prefix_value_and_backup_objective.md
  Separates average-descendant value from best-descendant value, defines the
  prefix reliability audit, and addresses deceptive branches and noisy max
  backup.

15_real_counterfactual_rollout_protocol.md
  Defines the balanced same-paper multi-path, multi-writer, multi-evaluator
  dataset required to identify real prefix value and compare search policies.
```

## Three Separate Uncertainties

DIRS must not collapse all uncertainty into one node confidence:

```text
trace uncertainty:
  which latent information flow could explain one artifact?

graph uncertainty:
  which reusable nodes and dependencies belong in the shared skill model?

selection/execution uncertainty:
  which sub-DAG should be used for this task, and how reliably will the
  executor realize it?
```

## Recommended First Implementation

The first publishable implementation should minimize unnecessary machinery:

```text
trace posterior:
  diverse LLM proposals + hard constraints + verifier/replay weighting

shared graph:
  posterior node/edge support with context gates and explicit provenance

outer-loop optimization:
  bounded edit proposals or a small evolutionary population

inner-loop optimization:
  posterior-sampled fixed-snapshot MCTS

acceptance:
  paired validation + hard-failure non-regression + replay + complexity bound
```

A conditional DAG-GFlowNet is a stronger later implementation when calibrated,
multimodal posterior sampling becomes a primary contribution.

## No Hand-Coded Skill Policy

The schemas in this directory are a representation language, not a fixed skill
program. DIRS must learn node proposals, node equivalence, contextual
activation, execution dependencies, utility weights, search priors, and
acceptance thresholds from training/validation evidence. Only logical,
provenance, permission, and leakage constraints remain hard.

Factual claims and learned strategies use different provenance. A factual
claim in an output must be supported by task evidence. A newly proposed
strategy such as comparative compression, evidence allocation, or
claim-ordering need not appear verbatim in the source or expert artifact. It
may be retained when blind execution, intervention, and held-out transfer show
that it improves utility without increasing unsupported claims. The target is
functional strategy learning, not exact reconstruction of an expert's wording
or one supposedly unique DAG.
