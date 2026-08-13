# Full-Paper Section Gap Learning Report

Date: `2026-07-22`

Domain: `LLM Inference / Systems / Token Efficiency`

Source run used for paper selection:
`/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_abstract_train28_holdout_echo_20260720_clean_longgoal`

Selection rule: first 20 training chip paths from the completed abstract longgoal
`training_trace.json`. The prior private holdout file was not read.

Generated evidence table:
`paper_section_evidence_table.json`

## Evidence Scope

This pass corrects the earlier abstract-only interpretation. It uses full-paper
memory chips and local paper artifacts rather than only abstract DAG traces.

Coverage across the 20 selected papers:

| Evidence channel | Count |
|---|---:|
| Local text artifact present | 20 / 20 |
| Local PDF artifact present | 20 / 20 |
| Introduction covered | 20 / 20 |
| Method covered | 20 / 20 |
| Experiments covered | 20 / 20 |
| Results covered | 20 / 20 |
| Limitations covered | 20 / 20 |
| Appendix or supplement covered | 20 / 20 |
| Explicit related work covered | 14 / 20 |

The six papers without an explicit related-work field still contain prior-work
or prior-limitation evidence inside the introduction, method framing, or
problem-gap fields.

## What Was Learned

Across full papers, the research gap is not a single abstract sentence. It is a
paper-scale evidence structure:

```text
introduction pressure
  -> prior method family
  -> unsolved constraint or failure mode
  -> method necessity
  -> mechanism that directly attacks the gap
  -> experiment protocol aligned to the gap
  -> result showing the constraint improved
  -> ablation showing why the mechanism, not luck, caused the result
  -> limitation/future-work boundary
```

The key learned skill is to identify a gap as a constraint mismatch:

```text
Existing systems can do X,
but deployment or scaling requires Y,
and existing assumptions fail under Z.
Therefore a new mechanism is needed,
and the experiments must test exactly Y and Z.
```

## Reusable Section DAG

### Introduction

Nodes:

- `intro.1.deployment_pressure`: Establish why the problem matters now: speed,
  memory, latency, long context, multimodal token count, compression rate,
  correctness, serving cost, or hardware efficiency.
- `intro.2.prior_family_map`: Name the current family: speculative decoding,
  KV cache compression, token merging, diffusion caching, visual tokenization,
  quantization, sparse CUDA generation, long-context RL, or sampling.
- `intro.3.failure_mode`: State the exact way existing methods break:
  static thresholds, tokenwise early rejection, fixed-rate token budgets,
  reconstruction/semantic fragmentation, hard VQ assignment, heuristic cache
  reuse, likelihood-only code generation, or retuning cost.
- `intro.4.gap_claim`: Convert the failure into a bounded research gap.
- `intro.5.method_need`: Explain why a new mechanism is needed instead of only
  more data, more compute, or another benchmark.
- `intro.6.contributions`: Bind the proposed object, mechanism, evaluation
  axes, and main outcome.

Edges:

```text
intro.1.deployment_pressure -> intro.2.prior_family_map
intro.2.prior_family_map -> intro.3.failure_mode
intro.3.failure_mode -> intro.4.gap_claim
intro.4.gap_claim -> intro.5.method_need
intro.5.method_need -> intro.6.contributions
```

### Related Work

Nodes:

- `related.1.closest_baselines`: Identify the closest systems, not generic
  background.
- `related.2.axis_of_difference`: Separate the new paper along the relevant
  axis: losslessness, adaptivity, hardware fit, semantic preservation,
  modality unification, rate control, or training-free deployment.
- `related.3.near_miss_handling`: Admit when prior work partially solves the
  problem, then narrow the remaining gap.
- `related.4.no_strawman_check`: Avoid claiming the entire area is missing when
  the real novelty is a specific constraint combination.

Edges:

```text
intro.4.gap_claim -> related.1.closest_baselines
related.1.closest_baselines -> related.2.axis_of_difference
related.2.axis_of_difference -> related.3.near_miss_handling
related.3.near_miss_handling -> related.4.no_strawman_check
```

### Method

Nodes:

- `method.1.object_definition`: Define the actual object being introduced:
  tokenizer, cache policy, verifier, sampler, pruning framework, kernel
  generator, trajectory surrogate, or reasoning/data pipeline.
- `method.2.mechanism_delta`: State the change relative to the baseline family.
- `method.3.constraint_binding`: Tie each design choice to the gap constraint:
  latency, memory, exactness, rate-distortion, quality, stability, hardware,
  or data efficiency.
- `method.4.operational_steps`: Make the method executable as a pipeline,
  algorithm, update rule, gating rule, or architecture.
- `method.5.theory_or_invariant`: Include theory only when the paper uses it:
  losslessness, rate objective, optimal transport geometry, sensitivity
  estimate, or position-scaling view.

Edges:

```text
intro.5.method_need -> method.1.object_definition
method.1.object_definition -> method.2.mechanism_delta
method.2.mechanism_delta -> method.3.constraint_binding
method.3.constraint_binding -> method.4.operational_steps
method.4.operational_steps -> method.5.theory_or_invariant
```

### Experiments

Nodes:

- `experiments.1.axis_match`: Each experiment must correspond to a gap axis:
  speed, memory, rate, quality, exactness, robustness, generality, or hardware.
- `experiments.2.baseline_strength`: Compare against the closest strong
  baseline family, not only weak defaults.
- `experiments.3.metric_pairing`: Pair efficiency metrics with correctness or
  quality metrics. Examples: tokens/sec plus accuracy, BPP plus PSNR/LPIPS,
  latency plus FID, memory plus task score, speedup plus exactness.
- `experiments.4.scale_or_stress`: Include stress dimensions such as longer
  draft length, larger targets, higher temperature, longer context, higher
  token compression, larger models, or different hardware.
- `experiments.5.ablation_or_control`: Test the claimed mechanism, not just the
  final method.

Edges:

```text
method.3.constraint_binding -> experiments.1.axis_match
experiments.1.axis_match -> experiments.2.baseline_strength
experiments.2.baseline_strength -> experiments.3.metric_pairing
experiments.3.metric_pairing -> experiments.4.scale_or_stress
experiments.4.scale_or_stress -> experiments.5.ablation_or_control
```

### Results And Discussion

Nodes:

- `results.1.primary_table_read`: Report the main table as evidence for the
  original gap, not as detached scoreboard numbers.
- `results.2.tradeoff_interpretation`: Interpret whether the method improves
  the target constraint without losing the paired metric.
- `results.3.exception_scan`: Name saturation, weaker settings, hardware limits,
  or quality regressions when present.
- `results.4.mechanism_attribution`: Use ablations to argue why the mechanism
  explains the gain.
- `results.5.scope_boundary`: Preserve limitations and avoid universal claims.

Edges:

```text
experiments.5.ablation_or_control -> results.1.primary_table_read
results.1.primary_table_read -> results.2.tradeoff_interpretation
results.2.tradeoff_interpretation -> results.3.exception_scan
results.3.exception_scan -> results.4.mechanism_attribution
results.4.mechanism_attribution -> results.5.scope_boundary
```

## Gap Taxonomy Learned From The 20 Papers

1. Fragmentation/unification gaps:
   tokenizers or representations solve one mode, task, or objective but not the
   unified interface needed by downstream systems.

2. Constraint-mismatch gaps:
   existing algorithms work in principle but fail under latency, memory,
   hardware, long-context, or high-concurrency constraints.

3. Static-policy gaps:
   fixed thresholds, fixed token budgets, fixed pruning, or hand-tuned weights
   cannot adapt to sample, timestep, class, temperature, or deployment state.

4. Exactness-versus-speed gaps:
   faster decoding or verification often sacrifices distributional correctness;
   strong papers specify which invariant must remain exact.

5. Objective-misalignment gaps:
   the training or generation objective does not reward the property the system
   actually needs, such as runtime speed, rate, error, or task-specific utility.

6. Mechanism-specificity gaps:
   a result is not enough; the paper must show that the specific mechanism
   explains the improvement through ablation or controlled comparison.

## Experiment Pattern Learned

The experiment section should not be written as a generic benchmark list. It
should be a verification plan for the gap.

```text
gap axis -> baseline family -> metric pair -> stress setting -> ablation
```

Common pairings across the papers:

- Speed or latency with task quality.
- Token reduction with accuracy or reconstruction fidelity.
- Compression rate with distortion/perceptual quality.
- Cache reuse with output quality and runtime.
- Decoding acceleration with distributional exactness or accuracy.
- Hardware/code generation with compile success, correctness, and speedup.
- Long-context efficiency with held-out length generalization.

## Anti-Rewards Learned

- Do not call a topic gap new when prior work already covers the broad topic.
  The real gap is usually a narrower constraint combination.
- Do not write "efficient" without naming what is saved: tokens, memory, FLOPs,
  wall-clock latency, target calls, bandwidth, GPU time, or dollars.
- Do not let results float away from the introduction. The main table must
  answer the original gap claim.
- Do not claim a mechanism is responsible unless an ablation, sensitivity test,
  control, or theorem supports it.
- Do not use only one metric when the claimed gap is a tradeoff.
- Do not hide limitations; in these papers, limitations often define the next
  publishable gap.

## Paper-Level Reading Notes

| # | Paper | Learned gap move |
|---:|---|---|
| 1 | AToken | Fragmented visual tokenization becomes a unification gap across reconstruction, semantics, image, video, and 3D. |
| 2 | RDVQ | Hard VQ assignment blocks rate optimization, so the gap is differentiable rate-distortion control. |
| 3 | TRELLIS.2 | 3D generation needs compact native latents that preserve topology and PBR materials. |
| 4 | NuWa | Edge deployment needs class-specific pruning, not all-class compression. |
| 5 | SeaCache | Diffusion caching needs a signal-aware reuse metric rather than raw feature distance. |
| 6 | SenCache | Cache reuse needs sample/timestep sensitivity, not heuristic skip rules. |
| 7 | Prophet DLM | DLM decoding wastes refinement steps after answers stabilize. |
| 8 | DTO-KD | Distillation needs dynamic objective weighting because task and teacher gradients conflict. |
| 9 | FlashVID | Video LLM token compression must be training-free and spatiotemporally adaptive. |
| 10 | HTI/CLOT | Deployed models need new hyperparameter behavior without retraining every anchor. |
| 11 | InfoTok | Long videos need adaptive token budgets instead of fixed-rate tokenization. |
| 12 | LPD | Flat-token autoregressive image generation needs parallel decoding while retaining compatibility. |
| 13 | LoongRL | Long-context RL needs data and training structure that induces retrieval plus reasoning. |
| 14 | SparseRL | Sparse CUDA generation needs runtime-aware rewards, not next-token likelihood alone. |
| 15 | MrRoPE | RoPE extension needs a unified scaling view and controlled extrapolation behavior. |
| 16 | HSD | Speculative decoding needs longer-prefix acceptance while preserving exact target distribution. |
| 17 | p-less Sampling | LLM decoding needs robust truncation without task/temperature-specific hyperparameters. |
| 18 | PRISM | fMRI reconstruction needs structured object/attribute/relation text rather than holistic latents. |
| 19 | Rational agents | Information-seeking agents need uncertainty-aware question/act decisions. |
| 20 | SPARK | Single-image 3D reconstruction needs simulation-ready articulated structure, not just plausible meshes. |

## Bottom Line

The full-paper lesson is stronger than the abstract-only lesson:

```text
A research gap is valid only when the introduction names it,
the related-work boundary protects it,
the method directly targets it,
the experiment protocol measures it,
the ablations attribute it,
and the limitations bound it.
```

For future LLM inference or systems papers, the safest gap-finding procedure is
therefore:

```text
1. Identify the deployment constraint.
2. Map the closest baseline family.
3. Locate the exact failure mode under that constraint.
4. State the new mechanism needed to remove or reduce that failure.
5. Require experiments that pair efficiency with correctness/quality.
6. Require ablations or theory that isolate the mechanism.
7. Preserve the remaining limitation as the next gap, not as hidden weakness.
```

