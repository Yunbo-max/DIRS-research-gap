# Round 01 Blind Abstract Variants

All variants were written from the permitted chip and style profile. They deliberately follow different information paths; none targets hidden wording.

## Variant A — Result-first Pareto ledger

Strategy: `S1_result_first_pareto_ledger`  
Word count: 197

Linear-time sequence models promise constant recurrent state, but they often trade away language quality and can remain inefficient at GPU decode. Mamba-3 attacks both problems through three coupled changes: an exponential-trapezoidal state-space recurrence with a learned trap gate and B/C biases, a data-dependent complex-valued state update, and a rank-4 multi-input multi-output (MIMO) path designed to raise arithmetic intensity. Under a shared 100B-token FineWeb-Edu protocol with 2K contexts, models span 180M to 1.5B parameters. At 1.5B, Mamba-3 MIMO reaches 10.24 perplexity and 57.6 average downstream accuracy, compared with 10.51 and 55.4 for the Transformer, 10.47 and 55.7 for Mamba-2, and 10.45 and 55.8 for Gated DeltaNet. Component tests connect the design to distinct benefits: the complex update attains 100% parity accuracy and sharply improves modular-arithmetic tracking over Mamba-2, while the full bias-and-trapezoidal formulation improves 440M perplexity over its ablations. On H100 in BF16 with state size 128, the SISO and MIMO kernels require 0.156 and 0.179 ms/token, versus 0.203 for Mamba-2 and 0.257 for Gated DeltaNet. Pure Mamba-3 remains uneven on difficult real-world retrieval, where hybrid attention improves recall at additional attention cost. These results position Mamba-3 as a quality-efficiency tradeoff rather than an unqualified replacement for attention.

## Variant B — Mechanism-ablation staircase

Strategy: `S2_mechanism_ablation_staircase`  
Word count: 204

Can a fixed-state recurrent model improve capability without surrendering its inference advantage? Mamba-3 builds an affirmative, but bounded, case by changing the state-space update at three levels. First, exponential-trapezoidal discretization, a learned trap gate, and B/C biases enrich the recurrence; at 440M parameters the complete design obtains 15.72 perplexity, versus 16.68 without both bias and trap, while adding a short convolution yields 15.85. Second, data-dependent complex rotations expand state dynamics. On formal languages, Mamba-3 scores 100.0 on parity, 98.51 on arithmetic without brackets, and 87.75 with brackets, compared with 0.9, 47.81, and 0.88 for Mamba-2. Third, a rank-4 MIMO formulation increases recurrent arithmetic intensity. Across models trained on 100B FineWeb-Edu tokens, its 1.5B configuration improves from the SISO model's 10.35 perplexity and 56.4 average downstream accuracy to 10.24 and 57.6. Specialized H100 kernels preserve a favorable decode tradeoff: at BF16 state size 128, MIMO runs at 0.179 ms/token, faster than Mamba-2 at 0.203, although SISO is faster still at 0.156. Evaluation also reveals the fixed-state boundary: pure Mamba-3 is weaker on some semi-structured and unstructured retrieval tasks, and sparse attention helps at added cost. The evidence therefore supports mechanism-specific gains in state tracking, language modeling, and decode efficiency, while leaving direct 7B scaling untested.

## Variant C — Evaluation mosaic with scope gate

Strategy: `S3_evaluation_mosaic_with_scope_gate`  
Word count: 210

Efficiency alone does not resolve the central weakness of recurrent language models: a compact state can be fast yet discard information or underperform attention on quality. Mamba-3 treats this as a multi-axis design problem. Its state-space block combines an exponential-trapezoidal recurrence, data-dependent complex rotations, and either SISO or rank-4 MIMO input-output structure; the latter spends more computation per state load to improve arithmetic intensity. Evidence is deliberately broad. Models from 180M to 1.5B are trained on 100B FineWeb-Edu tokens with 2K context. At 1.5B, MIMO records 10.24 perplexity and 57.6 average downstream accuracy, outperforming the reported Transformer, Mamba-2, and Gated DeltaNet results under the shared protocol. Formal-language tests isolate a complementary strength: Mamba-3 reaches 100.0 parity accuracy and 87.75 on bracketed modular arithmetic, whereas Mamba-2 obtains 0.9 and 0.88. Inference measurements expose a tradeoff rather than a single winner. For BF16 H100 kernels at state size 128, SISO is fastest at 0.156 ms/token; MIMO takes 0.179, still below Mamba-2's 0.203 and Gated DeltaNet's 0.257. At a 16,384-token end-to-end setting, SISO also leads the recurrent variants reported. Yet pure Mamba-3 remains mixed on real-world retrieval; adding sparse NoPE attention improves retrieval while partially restoring attention costs. Mamba-3 thus advances the quality-latency frontier, with hybridization and scaling beyond 1.5B remaining empirical questions.

## Deliberate multimodality

- Variant A leads with the system-level comparison and compresses component evidence into one bridge.
- Variant B orders the abstract by three mechanism-to-test bindings and delays synthesis.
- Variant C organizes evidence by evaluation function and makes the scope gate part of the evaluation sequence.
- `S4_boundary_first_contrast_loop` is retained as an unexecuted candidate for a matched counterfactual in a later protocol round; it is not silently discarded.
