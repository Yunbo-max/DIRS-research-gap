# DIRS Abstract Training Run

Date: `2026-07-20`

Domain: `LLM Inference / Systems / Token Efficiency`

## Split

- Total chips: `29`
- Training papers: `28`
- Training papers with extracted abstracts: `28`
- Held-out chip: `ICML2026_71057_echo_elastic_speculative_decoding`
- Held-out title: `ECHO: Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios`

Blind rule: do not read the held-out original abstract until after generation.

## Style Prior

```json
{
  "domain": "LLM Inference / Systems / Token Efficiency",
  "train_paper_count": 28,
  "holdout_paper_count": 1,
  "abstract_word_count": {
    "min": 75,
    "median": 163.5,
    "mean": 163.36,
    "max": 263
  },
  "abstract_sentence_count": {
    "median": 8.0,
    "mean": 7.36
  },
  "recommended_target_words": 164,
  "recommended_band": [
    128,
    198
  ]
}
```

## Node Support

| id | support_count | support_rate |
| --- | --- | --- |
| E1_evaluation_setup | 28 | 1.0 |
| E2_result_outcome | 28 | 1.0 |
| G1_problem_gap | 28 | 1.0 |
| I1_interpretation_or_tradeoff | 28 | 1.0 |
| M1_architecture_or_mechanism | 28 | 1.0 |
| O1_named_method_or_object | 28 | 1.0 |
| P1_length_and_placement_prior | 28 | 1.0 |
| R1_abstract_as_argument | 28 | 1.0 |
| S1_bounded_takeaway | 28 | 1.0 |
| E3_quantitative_anchor | 21 | 0.75 |
| M2_efficiency_or_theory_detail | 13 | 0.4643 |
| C1_domain_context | 7 | 0.25 |

## Edge Support

| id | support_count | support_rate |
| --- | --- | --- |
| E1_evaluation_setup->E2_result_outcome | 28 | 1.0 |
| I1_interpretation_or_tradeoff->S1_bounded_takeaway | 28 | 1.0 |
| O1_named_method_or_object->M1_architecture_or_mechanism | 28 | 1.0 |
| R1_abstract_as_argument->G1_problem_gap | 28 | 1.0 |
| S1_bounded_takeaway->P1_length_and_placement_prior | 28 | 1.0 |
| E2_result_outcome->E3_quantitative_anchor | 21 | 0.75 |
| E3_quantitative_anchor->I1_interpretation_or_tradeoff | 21 | 0.75 |
| G1_problem_gap->O1_named_method_or_object | 21 | 0.75 |
| M1_architecture_or_mechanism->E1_evaluation_setup | 15 | 0.5357 |
| M1_architecture_or_mechanism->M2_efficiency_or_theory_detail | 13 | 0.4643 |
| M2_efficiency_or_theory_detail->E1_evaluation_setup | 13 | 0.4643 |
| C1_domain_context->O1_named_method_or_object | 7 | 0.25 |
| E2_result_outcome->I1_interpretation_or_tradeoff | 7 | 0.25 |
| G1_problem_gap->C1_domain_context | 7 | 0.25 |

## Held-Out Test Card

```yaml
chip_id: ICML2026_71057_echo_elastic_speculative_decoding
title: ECHO: Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios
chip_path: /tf/notebooks/icml2026_oral_paper_memory_fresh_24h/chips/ICML2026_71057_echo_elastic_speculative_decoding.chip.json
target_words_from_training_median: 164
target_band: [128, 198]
original_abstract: hidden_until_after_generation
```

## Files

```text
manifest.json
style_profile.json
node_support_scores.json
edge_support_scores.json
training_trace.json
holdout_test_card.md
holdout_private_after_generation.json
```
