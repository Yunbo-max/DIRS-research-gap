# Subagent Heavy Simulation Chain

Date: `2026-07-20`

Sample:

```text
ICML2026_71083_any_order_gpt_mdm
```

Purpose: one live DIRS heavy simulation using Codex subagents as the editor,
simulator, evaluator, repair editor, repair simulator, and repair evaluator.

The held-out private file was not opened.

## Chain

```text
coordinator
  -> editor subagent
  -> simulator subagent
  -> evaluator subagent
  -> repair editor subagent
  -> repair simulator subagent
  -> repair evaluator subagent
```

## Initial Editor DAG

```text
R1_abstract_as_argument
G1_problem_gap
O1_named_method_or_object
M1_architecture_or_mechanism
M2_efficiency_or_theory_detail
E1_evaluation_setup
E2_result_outcome
E3_quantitative_anchor
I1_interpretation_or_tradeoff
S1_bounded_takeaway
P1_length_and_placement_prior
```

## Initial Generated Abstract

```text
Comparisons between autoregressive LLMs and masked diffusion language models often conflate generative formulation with architecture: standard AR systems are causal decoder-only, whereas MDMs are typically full-attention encoder-only, and uniform any-order training is both slow and context-order sensitive in decoder-only models. We present AO-GPT, a decoder-only masked diffusion language model that realizes masked diffusion as any-order autoregression inside a GPT-style causal decoder. AO-GPT samples token orders, shuffles tokens and positions accordingly, conditions each prediction on the target position through adaptive LayerNorm, and trains with shifted cross entropy. Its Random_CL training uses 90% random orders and 10% left-to-right orders, while KV-cache-compatible diffusion-style unmasking with a parallel attention mask supports efficient generation; Lemma 1 gives O(n) decoder-only AO-AR generation complexity. We evaluate on OpenWebText with 1024-token contexts using GPT-2 Small and Medium shapes, comparing GPT-2, sigma-GPT, SEDD, and RADD on zero-shot perplexity, any-order perplexity, generation perplexity, and wall-clock generation time. Uniform any-order training converges more slowly, but the 10% left-to-right mixture improves convergence and final loss, and context-order ensembling substantially narrows the decoder-only any-order gap. At sequence length 1024 and batch size 32, AO-GPT is about 25x faster than SEDD; with top-p 0.95 and temperature 0.7 at 1024 steps, generation perplexity is 4.611 versus SEDD's 5.051. Overall, AO-GPT reveals a speed-quality tradeoff rather than uniform dominance, with ensembling improving likelihood at added evaluation cost.
```

## Initial Evaluator Feedback

```yaml
overall_score: 0.85
main_issues:
  - metric-scope overreach in the evaluation sentence
  - merged uniform any-order training slowness with context-order sensitivity
  - missing simpler encoder-only MDM conditional space vs larger decoder-only AO-GPT conditional space
  - missing temperature-annealing contrast
feedback_for_editor: >
  Revise the evaluation sentence to separate zero-shot/any-order baselines from
  AO-GPT-vs-SEDD generation comparisons, and add the conditional-space and
  temperature-annealing points.
```

## Repair Editor Note

The repair editor kept the same node path but returned one malformed edge
`P1_length_and_placement_prior -> R1_abstract_as_argument`. The coordinator
rejected that edge and used the valid final edge:

```text
S1_bounded_takeaway -> P1_length_and_placement_prior
```

This is a useful validation example: editor outputs must be checked against the
current DAG before simulation.

## Repaired Generated Abstract

```text
Masked diffusion language models (MDMs) promise non-left-to-right generation, but comparisons to autoregressive LLMs often entangle the modeling formulation with the architecture: standard AR models use causal decoders, whereas most MDMs use full-attention encoders. This paper studies AO-GPT, a decoder-only MDM obtained by viewing masked diffusion training as any-order autoregression inside a GPT-style transformer. The design shuffles tokens by sampled orders, injects target-position information through adaptive LayerNorm, trains with a 90% random-order and 10% left-to-right order mixture, and supports parallel unmasking with a KV-cache-compatible attention mask and two-stage masked-token sampling. The analysis separates two difficulties: uniform any-order language training converges more slowly than left-to-right training, while decoder-only any-order models also suffer context-order sensitivity because, unlike encoder-only MDMs with a simpler conditional probability space over observed token sets, AO-GPT must model a larger space of conditionals over ordered contexts. On OpenWebText-trained GPT-2 Small and Medium settings, zero-shot perplexity on LAMBADA, WikiText2, PTB, WikiText103, and 1BW shows competitive but not uniformly best behavior; context-order ensembling narrows the decoder-only any-order perplexity gap. Separately, in AO-GPT-Medium versus SEDD-Medium generation at length 1024, AO-GPT is reported about 25x faster. Without annealing its GPT-2-Large generation perplexity is much worse, but with top-p 0.95 and temperature 0.7 it becomes comparable or better on that metric, framing AO-GPT as a bounded speed-quality trade-off rather than a dominance claim.
```

## Repair Evaluator Result

```yaml
overall_score: 0.94
coverage_score: 0.95
order_score: 0.96
style_score: 0.92
length_score: 0.94
unsupported_claims: []
remaining_minor_feedback:
  - make the two comparison goals more explicit
  - clarify that GPT-2-Large is the generation-perplexity judge, not an AO-GPT-Large model
```

## Takeaway

This confirms the intended DIRS heavy loop:

```text
editor selects/repairs DAG
simulator writes from fixed DAG
evaluator compares result and returns feedback
coordinator validates DAG edges and feeds feedback to the next editor pass
```
