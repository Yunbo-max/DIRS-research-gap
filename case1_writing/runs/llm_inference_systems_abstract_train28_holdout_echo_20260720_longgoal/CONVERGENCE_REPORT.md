# DIRS Abstract Convergence Report

Date: `2026-07-20`

Training examples: `28`
Domain: `LLM Inference / Systems / Token Efficiency`
Held-out: `ICML2026_71057_echo_elastic_speculative_decoding`
Completed loops: `24`
Minimum loops before early stop: `24`
MCTS rollouts per example: `5000`
Converged: `True`
Converged at loop: `24`
Final mean replay score: `0.981805`
Final min replay score: `0.965033`

## Final Selected Full DAG Nodes

```text
C1_domain_context
E1_evaluation_setup
E2_result_outcome
E3_quantitative_anchor
G1_problem_gap
I1_interpretation_or_tradeoff
M1_architecture_or_mechanism
M2_efficiency_or_theory_detail
O1_named_method_or_object
P1_length_and_placement_prior
R1_abstract_as_argument
S1_bounded_takeaway
```

## Final Selected Full DAG Edges

```text
C1_domain_context->O1_named_method_or_object
E1_evaluation_setup->E2_result_outcome
E2_result_outcome->E3_quantitative_anchor
E2_result_outcome->I1_interpretation_or_tradeoff
E3_quantitative_anchor->I1_interpretation_or_tradeoff
G1_problem_gap->C1_domain_context
G1_problem_gap->O1_named_method_or_object
I1_interpretation_or_tradeoff->S1_bounded_takeaway
M1_architecture_or_mechanism->E1_evaluation_setup
M1_architecture_or_mechanism->M2_efficiency_or_theory_detail
M2_efficiency_or_theory_detail->E1_evaluation_setup
O1_named_method_or_object->M1_architecture_or_mechanism
R1_abstract_as_argument->G1_problem_gap
S1_bounded_takeaway->P1_length_and_placement_prior
```

## Blind Rule

held-out original remains in holdout_private_after_generation.json and is not read by this harness
