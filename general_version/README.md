# DIRS General Version

Date: `2026-07-20`

Purpose: professional, task-independent specification of DIRS.

## Reading Order

```text
01_dirs_general_method.md
  Defines DIRS, the dual-system representation, the skill graph, and the
  training/inference objectives.

02_dirs_training_cycle.md
  Defines DISL, the iterative procedure for inducing and stabilizing a DIRS
  graph from expert artifacts.

03_dirs_mcts_inference_selector.md
  Defines test-time sub-DAG selection with MCTS, blind generation constraints,
  verifier outputs, and ablations.

04_dirs_skill_representation_patterns.md
  Summarizes lessons from skill-evolving systems and maps them to the DIRS
  file/schema design.

05_dirs_top_conference_evaluation_protocol.md
  Defines baselines, metrics, leakage controls, reporting standards, and
  minimal publishable experiments for ICLR/ICML/NeurIPS-style review.

06_dirs_mathematical_formulation.md
  Defines DIRS as general human-skill learning: latent skill traces, node and
  edge support estimation, constrained sub-DAG inference, MCTS search, and blind
  generation.

07_dirs_tta_ablation_study.md
  Defines the proposed DIRS-TTA full configuration, 122 controlled training and
  test-time variants, compute-matching rules, factor sweeps, metrics, statistical
  tests, and the required execution order.

dirs_tta_ablation_variants.csv
  Machine-readable registry for the full configuration and all ablation
  variants.

08_dirs_topic_test_time_learning_integration.md
  Defines the global/topic/exploration state hierarchy, legitimate test-time
  supervision, topic latent, recurrent repair, hindsight editor, temporary
  LoRA, temporal topic holdout, and persistence rules.
```

## Core Method Sentence

```text
DIRS learns reusable skills by inferring a typed dependency graph from expert
artifacts, storing both content and style/action properties at each node and
edge, then using MCTS to select, simulate, verify, and repair
evidence-supported connected sub-DAGs. Training uses verifier feedback to update
the shared graph; inference uses verifier feedback to choose the best output for
a new task.
```

## Recommended Citation Name

```text
DIRS: DAG-Inferred Reusable Skills
```

Use `DISL` only when referring specifically to the training algorithm.
