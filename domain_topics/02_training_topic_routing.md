# DIRS Training Topic Routing

Date: `2026-07-20`

Purpose: tell DIRS which paper-topic files and chip paths to read when training
domain writing skills.

This file is operational. When a user asks DIRS to train on a topic, the system
should route through this table, load the exact listed domain file, and build the
training set from the chip paths inside that file.

## Canonical Topic Split

Use this topic split:

```text
/tf/notebooks/yunbo/DIRS/domain_topics/semantic_balanced_23_domains/INDEX.md
```

This index contains:

```text
502 oral papers
23 semantic domains
one assignment per local oral-paper chip
```

Each domain file contains:

```text
paper title
conference bucket
exact chip path
skill notes to extract
```

## Training Rule

When the user asks to train DIRS:

```text
1. Read semantic_balanced_23_domains/INDEX.md.
2. Select the matching domain file.
3. Read every paper title and chip path in that domain file.
4. Load those chips as the domain training set.
5. Extract node content_skill and style_skill.
6. Extract edge content_dependency and style_transition.
7. Compute node/edge support frequency across the domain.
8. Build or update the domain DIRS skill library.
```

Do not guess the training papers from memory. Use the chip paths listed in the
selected domain file.

## Held-Out Rule

When the user asks DIRS to test or write for a specific paper:

```text
1. Identify the target paper chip.
2. Select the nearest domain file.
3. Remove the target paper from the training set if it appears there.
4. Train or load priors from the remaining papers only.
5. Generate from the target chip without reading the original target section.
6. Reveal the original section only after generation for comparison.
```

Report the held-out status in the run artifact:

```yaml
target_chip:
domain_file:
domain_paper_count:
training_paper_count_after_holdout:
target_seen_during_training: false
original_section_used_before_generation: false
```

## Node-Scoring Rule

For each domain, estimate node and edge importance from both frequency and
compatibility:

```text
node_score =
  support_rate
  + evidence_fit
  + section_role_fit
  + paper_signature_fit
  - forbidden_domain_penalty
  - unsupported_claim_penalty
```

High support alone is not enough. A common node should still be rejected when the
target chip lacks the evidence needed to support it.

## Domain Routing Table

| Domain | Papers | File |
|---|---:|---|
| Controllable Image / Video / Motion Generation | 30 | `semantic_balanced_23_domains/controllable_image_video_motion_generation.md` |
| 3D World Models / Pose / Robotics | 29 | `semantic_balanced_23_domains/3d_world_models_pose_robotics.md` |
| LLM Inference / Systems / Token Efficiency | 29 | `semantic_balanced_23_domains/llm_inference_systems_token_efficiency.md` |
| Safety / Multimodal / VLM Robustness | 29 | `semantic_balanced_23_domains/safety_multimodal_vlm_robustness.md` |
| Training Data / Dynamics / Distillation | 29 | `semantic_balanced_23_domains/training_data_dynamics_distillation.md` |
| Diffusion Sampling / Score-Based Modeling | 28 | `semantic_balanced_23_domains/diffusion_sampling_score_based_modeling.md` |
| Optimization / Theory / Statistical Learning | 28 | `semantic_balanced_23_domains/optimization_theory_statistical_learning.md` |
| Benchmark Design / Metrics / Evaluation Protocols | 27 | `semantic_balanced_23_domains/benchmark_design_metrics_evaluation_protocols.md` |
| 3D Geometry / Reconstruction / Registration | 26 | `semantic_balanced_23_domains/3d_geometry_reconstruction_registration.md` |
| LLM Evaluation / Behavior / Language Understanding | 26 | `semantic_balanced_23_domains/llm_evaluation_behavior_language_understanding.md` |
| Reasoning Methods / Cognitive and Step Models | 26 | `semantic_balanced_23_domains/reasoning_methods_cognitive_and_step_models.md` |
| Fine-tuning / LoRA / Adaptation / Continual Learning | 22 | `semantic_balanced_23_domains/fine_tuning_lora_adaptation_continual_learning.md` |
| Vision Segmentation / Medical and Bio Recognition | 22 | `semantic_balanced_23_domains/vision_segmentation_medical_and_bio_recognition.md` |
| Planning / Long-Horizon / Game Agents | 21 | `semantic_balanced_23_domains/planning_long_horizon_game_agents.md` |
| LLM Architecture / Attention / State Space Models | 20 | `semantic_balanced_23_domains/llm_architecture_attention_state_space_models.md` |
| LLM Training / Pretraining / Midtraining | 16 | `semantic_balanced_23_domains/llm_training_pretraining_midtraining.md` |
| AI for Science - Physics / Climate / Materials / Imaging | 15 | `semantic_balanced_23_domains/ai_for_science_physics_climate_materials_imaging.md` |
| Preference Alignment / RLHF / DPO / Rewards | 15 | `semantic_balanced_23_domains/preference_alignment_rlhf_dpo_rewards.md` |
| Agent Interfaces / Conversational / Social Behavior | 14 | `semantic_balanced_23_domains/agent_interfaces_conversational_social_behavior.md` |
| Gaussian Splatting / NeRF / Radiance Fields | 14 | `semantic_balanced_23_domains/gaussian_splatting_nerf_radiance_fields.md` |
| Graphs / Causality / Structured Models | 13 | `semantic_balanced_23_domains/graphs_causality_structured_models.md` |
| Speech / Audio / Translation / Text NLP | 12 | `semantic_balanced_23_domains/speech_audio_translation_text_nlp.md` |
| Reasoning - Memory / RAG / Retrieval | 11 | `semantic_balanced_23_domains/reasoning_memory_rag_retrieval.md` |

## Example

If asked:

```text
train DIRS on agent interfaces abstracts
```

Read:

```text
semantic_balanced_23_domains/agent_interfaces_conversational_social_behavior.md
```

Then train on all `14` chip paths listed inside that file.

If asked:

```text
train DIRS across all oral-paper domains
```

Read all `23` domain files and all `502` chip paths listed in them.
