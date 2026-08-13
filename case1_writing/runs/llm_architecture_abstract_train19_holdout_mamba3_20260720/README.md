# DIRS Abstract Training Run

Date: `2026-07-20`

Domain: `LLM Architecture / Attention / State Space Models`

## Split

- Total chips: `20`
- Training papers: `19`
- Training papers with extracted abstracts: `19`
- Held-out chip: `ICLR2026_HwCvaJOiCj_mamba3`
- Held-out title: `Mamba-3: Improved Sequence Modeling using State Space Principles`

Blind rule: do not read the held-out original abstract until after generation.

## Style Prior

```json
{
  "domain": "LLM Architecture / Attention / State Space Models",
  "train_paper_count": 19,
  "holdout_paper_count": 1,
  "abstract_word_count": {
    "min": 114,
    "median": 201,
    "mean": 236.63,
    "max": 446
  },
  "abstract_sentence_count": {
    "median": 8,
    "mean": 10.47
  },
  "recommended_target_words": 201,
  "recommended_band": [
    166,
    236
  ]
}
```

## Node Support

| id | support_count | support_rate |
| --- | --- | --- |
| E1_evaluation_setup | 19 | 1.0 |
| E2_result_outcome | 19 | 1.0 |
| G1_problem_gap | 19 | 1.0 |
| I1_interpretation_or_tradeoff | 19 | 1.0 |
| M1_architecture_or_mechanism | 19 | 1.0 |
| O1_named_method_or_object | 19 | 1.0 |
| P1_length_and_placement_prior | 19 | 1.0 |
| R1_abstract_as_argument | 19 | 1.0 |
| S1_bounded_takeaway | 19 | 1.0 |
| E3_quantitative_anchor | 15 | 0.7895 |
| M2_efficiency_or_theory_detail | 14 | 0.7368 |
| C1_domain_context | 7 | 0.3684 |

## Edge Support

| id | support_count | support_rate |
| --- | --- | --- |
| E1_evaluation_setup->E2_result_outcome | 19 | 1.0 |
| I1_interpretation_or_tradeoff->S1_bounded_takeaway | 19 | 1.0 |
| O1_named_method_or_object->M1_architecture_or_mechanism | 19 | 1.0 |
| R1_abstract_as_argument->G1_problem_gap | 19 | 1.0 |
| S1_bounded_takeaway->P1_length_and_placement_prior | 19 | 1.0 |
| E2_result_outcome->E3_quantitative_anchor | 15 | 0.7895 |
| E3_quantitative_anchor->I1_interpretation_or_tradeoff | 15 | 0.7895 |
| M1_architecture_or_mechanism->M2_efficiency_or_theory_detail | 14 | 0.7368 |
| M2_efficiency_or_theory_detail->E1_evaluation_setup | 14 | 0.7368 |
| G1_problem_gap->O1_named_method_or_object | 12 | 0.6316 |
| C1_domain_context->O1_named_method_or_object | 7 | 0.3684 |
| G1_problem_gap->C1_domain_context | 7 | 0.3684 |
| M1_architecture_or_mechanism->E1_evaluation_setup | 5 | 0.2632 |
| E2_result_outcome->I1_interpretation_or_tradeoff | 4 | 0.2105 |

## Held-Out Test Card

```yaml
chip_id: ICLR2026_HwCvaJOiCj_mamba3
title: Mamba-3: Improved Sequence Modeling using State Space Principles
chip_path: /tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/chips/ICLR2026_HwCvaJOiCj_mamba3.chip.json
target_words_from_training_median: 201
target_band: [166, 236]
original_abstract: hidden_until_after_generation
```

## Files

```text
LONGGOAL_STATUS.md
CONVERGENCE_REPORT.md
VERIFICATION_REPORT.md
RUN_COMMANDS.md
SUBAGENT_HEAVY_SIMULATION_PROTOCOL.md
subagent_live_runs/
manifest.json
style_profile.json
node_support_scores.json
edge_support_scores.json
training_trace.json
convergence_report.json
convergence_trace.jsonl
longrun_config.json
holdout_test_card.md
holdout_private_after_generation.json
```
