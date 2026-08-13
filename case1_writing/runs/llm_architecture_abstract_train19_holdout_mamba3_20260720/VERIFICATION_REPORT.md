# DIRS Case 1 Verification Report

Date: `2026-07-20`

## Objective Check

```text
domain_selected: yes
domain: LLM Architecture / Attention / State Space Models
total_domain_chips: 20
training_split: 19 chips
holdout_split: 1 chip
holdout_chip: ICLR2026_HwCvaJOiCj_mamba3
training_abstracts_extracted: 19 / 19
training_artifacts_generated: yes
convergence_harness_run: yes
prepared_for_longer_multiloop_run: yes
heavy_llm_runner_added: yes
subagent_protocol_added: yes
```

## Artifact Evidence

```text
manifest.json
training_trace.json
style_profile.json
node_support_scores.json
edge_support_scores.json
convergence_report.json
convergence_trace.jsonl
longrun_config.json
SUBAGENT_HEAVY_SIMULATION_PROTOCOL.md
heavy_llm_runs/
holdout_test_card.md
holdout_private_after_generation.json
RUN_COMMANDS.md
```

## Blind-Holdout Evidence

The public manifest records the held-out source fields as hidden:

```yaml
abstract_source_path: hidden_until_after_generation
abstract_word_count: null
source_paths_tried:
  - hidden_until_after_generation
post_generation_only:
  stored_in: holdout_private_after_generation.json
```

The held-out original metadata is isolated in:

```text
holdout_private_after_generation.json
```

This file must stay unread until after blind generation and verification.

## Convergence Evidence

```text
max_loops: 100
mcts_rollouts_per_example: 500
stable_window: 5
completed_loops: 6
converged: true
converged_at_loop: 6
final_mean_replay_score: 0.977003
final_min_replay_score: 0.966213
```

## Current Status

The lightweight deterministic DIRS + MCTS preflight is complete and the run is
ready for the heavier LLM-editor / MCTS-simulator / evaluator loop described in
`longrun_config.json`.
