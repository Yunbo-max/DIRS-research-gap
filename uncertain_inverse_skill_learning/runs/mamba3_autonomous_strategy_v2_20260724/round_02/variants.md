# Round 02 Blind Abstract Variants

No variant uses hidden wording or expert similarity. Variants A and B deliberately preserve the same main factual units and numeric anchors; B disrupts the proposed ordering and adjacency operations. Variant C executes a separate boundary-first branch.

## Variant A — Autonomous repair: salience anchor plus selective bindings

Strategy: `S5_salience_anchor_then_selective_bindings`  
Word count: 196

Linear-time recurrent models offer constant state, yet often sacrifice language quality and can remain memory-bound during GPU decoding. Mamba-3 addresses this joint gap with an exponential-trapezoidal recurrence using a learned trap gate and B/C biases, data-dependent complex state dynamics, and a rank-4 MIMO path that raises arithmetic intensity. Under shared 100B-token FineWeb-Edu training, its 1.5B MIMO model outperforms the reported Transformer, Mamba-2, and Gated DeltaNet baselines, reaching 10.24 perplexity and 57.6 average downstream accuracy. The supporting evidence is organized around selected mechanism-specific checks. Removing bias and trapezoidal discretization worsens 440M perplexity from 15.72 to 16.68. On formal state tracking, the complex-update model reaches 100% parity accuracy versus 0.9% for Mamba-2 and also strongly improves modular arithmetic. MIMO improves quality over SISO; on H100 in BF16 at state size 128, its 0.179 ms/token kernel remains faster than Mamba-2's 0.203, although SISO is faster at 0.156. This tradeoff makes MIMO a quality-efficiency choice, not a universal latency winner. Pure Mamba-3 remains weak on some real-world retrieval settings; sparse attention helps but restores attention costs. Overall, Mamba-3 advances recurrent language modeling through linked improvements in recurrence, state dynamics, and hardware use, while retrieval and scaling beyond 1.5B remain open.

## Variant B — Matched counterfactual: detached bindings and delayed salience

Control: `A1_binding_detached_and_salience_delayed`  
Word count: 212

Linear-time recurrent models offer constant state, yet often sacrifice language quality and can remain memory-bound during GPU decoding. Mamba-3 combines an exponential-trapezoidal recurrence with a learned trap gate and B/C biases, data-dependent complex state dynamics, and a rank-4 MIMO path intended to increase arithmetic intensity. Models are trained on 100B FineWeb-Edu tokens, with reported scales up to 1.5B. The evaluation spans language modeling, formal state tracking, retrieval, and H100 latency. In BF16 at state size 128, MIMO runs at 0.179 ms/token, compared with 0.203 for Mamba-2; SISO is faster at 0.156. At 440M, the full configuration obtains 15.72 perplexity, while removing bias and trapezoidal discretization gives 16.68. Pure Mamba-3 remains weak on some real-world retrieval settings, and sparse attention improves retrieval while adding attention costs. At 1.5B, MIMO improves quality over SISO and reaches 10.24 perplexity and 57.6 average downstream accuracy, outperforming the reported Transformer, Mamba-2, and Gated DeltaNet baselines. Formal-task accuracy reaches 100% on parity versus 0.9% for Mamba-2 and also improves strongly on modular arithmetic. Together, these measurements show gains in quality, diagnostic state tracking, and decode efficiency, alongside unresolved retrieval. The architecture therefore offers a promising recurrent tradeoff, but the current evidence does not establish which component matters most for the system-level gains or whether results scale beyond 1.5B.

## Variant C — Retained multimodal branch: boundary-first contrast

Strategy: `S4_boundary_first_contrast_loop`  
Word count: 181

Fixed-state recurrence offers constant memory, but its compact state can miss sparse evidence; pure Mamba-3 remains weaker on some semi-structured and unstructured retrieval tasks. Hybrid attention improves retrieval, although it reintroduces attention compute and memory costs. Against that boundary, Mamba-3 redesigns the recurrent core with exponential-trapezoidal discretization and a learned trap gate, data-dependent complex rotations, and rank-4 MIMO projections that raise arithmetic intensity. The design is tested from 180M to 1.5B parameters under a shared 100B-token FineWeb-Edu protocol. At 1.5B, MIMO achieves 10.24 perplexity and 57.6 average downstream accuracy, ahead of the reported Transformer, Mamba-2, and Gated DeltaNet baselines. Mechanism-oriented tests add context: the full recurrence obtains 15.72 perplexity at 440M versus 16.68 without bias and trap, while Mamba-3 reaches 100% parity accuracy compared with 0.9% for Mamba-2. Efficiency is also conditional. On H100 in BF16 with state size 128, SISO is fastest at 0.156 ms/token; MIMO takes 0.179, still below Mamba-2's 0.203. Mamba-3 therefore improves the recurrent quality-efficiency frontier without eliminating the information-retention problem that motivates hybrid designs. The reported evidence ends at 1.5B, so performance at 7B remains unproven.

## Matched comparison contract

- A and B use the same problem, architecture, training-scale, aggregate-quality, recurrence-ablation, formal-state, latency, retrieval, hybrid-cost, and scale-boundary factual units.
- A uses an early aggregate anchor and places each selected diagnostic next to its mechanism role.
- B lists mechanisms once, presents evaluation blocks in a different order, and delays the aggregate result; it does not invent contrary pairings.
- The 16-word length difference remains within the target band and reflects connective text needed to keep the detached control grammatical.
- C is not part of the causal contrast. It preserves posterior mass for a substantially different limitation-led organization.
