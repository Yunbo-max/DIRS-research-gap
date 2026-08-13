# Round 03 Reserved-Novelty Variants

All three abstracts are blind to hidden wording and use only chip-grounded content. Variants A and B differ in exactly one 15-word sentence. Variant C retains the locally supported S5 branch with its procedural meta-discourse removed.

## Variant A — New strategy: operating-point decision split

Strategy: `S6_operating_point_decision_split`  
Word count: 187

State-space language models can decode with constant recurrent state, yet their efficiency often comes with weaker capability, and memory-bound updates can still underuse GPUs. Mamba-3 modifies the recurrence with exponential-trapezoidal discretization, a learned trap gate and B/C biases, adds data-dependent complex state dynamics, and offers SISO or rank-4 MIMO input-output structure. Under shared 100B-token FineWeb-Edu training, the 1.5B MIMO model reaches 10.24 perplexity and 57.6 average downstream accuracy, exceeding the reported Transformer, Mamba-2, and Gated DeltaNet baselines. Its quality advantage comes with a mode choice: on H100 in BF16 at state size 128, SISO runs at 0.156 ms/token, while MIMO takes 0.179, still faster than Mamba-2 at 0.203. Thus SISO is the speed-oriented operating point, whereas MIMO spends some latency for better quality. Other tests probe the shared recurrent design: removing bias and trapezoidal discretization worsens 440M perplexity from 15.72 to 16.68, and Mamba-3 reaches 100% parity accuracy versus 0.9% for Mamba-2. Pure Mamba-3 nevertheless remains weak on some real-world retrieval tasks; sparse attention helps but adds attention-related cost. Mamba-3 therefore advances the recurrent quality-efficiency frontier within one design, while retrieval and scaling beyond 1.5B remain open.

## Variant B — Narrow counterfactual: directional role map removed

Control: `A2_remove_directional_mode_role_map`  
Word count: 187

State-space language models can decode with constant recurrent state, yet their efficiency often comes with weaker capability, and memory-bound updates can still underuse GPUs. Mamba-3 modifies the recurrence with exponential-trapezoidal discretization, a learned trap gate and B/C biases, adds data-dependent complex state dynamics, and offers SISO or rank-4 MIMO input-output structure. Under shared 100B-token FineWeb-Edu training, the 1.5B MIMO model reaches 10.24 perplexity and 57.6 average downstream accuracy, exceeding the reported Transformer, Mamba-2, and Gated DeltaNet baselines. Its quality advantage comes with a mode choice: on H100 in BF16 at state size 128, SISO runs at 0.156 ms/token, while MIMO takes 0.179, still faster than Mamba-2 at 0.203. Thus SISO and MIMO are alternative configurations with different quality and latency under these evaluations. Other tests probe the shared recurrent design: removing bias and trapezoidal discretization worsens 440M perplexity from 15.72 to 16.68, and Mamba-3 reaches 100% parity accuracy versus 0.9% for Mamba-2. Pure Mamba-3 nevertheless remains weak on some real-world retrieval tasks; sparse attention helps but adds attention-related cost. Mamba-3 therefore advances the recurrent quality-efficiency frontier within one design, while retrieval and scaling beyond 1.5B remain open.

## Variant C — Retained non-dominated branch: cleaned S5

Strategy: `S5_salience_anchor_then_selective_bindings`  
Status: `local_strategy_hypothesis`  
Word count: 187

Linear-time recurrent models offer constant state, yet often sacrifice language quality and can remain memory-bound during GPU decoding. Mamba-3 addresses this joint gap with an exponential-trapezoidal recurrence using a learned trap gate and B/C biases, data-dependent complex state dynamics, and a rank-4 MIMO path that raises arithmetic intensity. Under shared 100B-token FineWeb-Edu training, its 1.5B MIMO model outperforms the reported Transformer, Mamba-2, and Gated DeltaNet baselines, reaching 10.24 perplexity and 57.6 average downstream accuracy. Removing bias and trapezoidal discretization worsens 440M perplexity from 15.72 to 16.68. On formal state tracking, the complex-update model reaches 100% parity accuracy versus 0.9% for Mamba-2 and also strongly improves modular arithmetic. MIMO improves quality over SISO; on H100 in BF16 at state size 128, its 0.179 ms/token kernel remains faster than Mamba-2's 0.203, although SISO is faster at 0.156. This tradeoff makes MIMO a quality-efficiency choice, not a universal latency winner. Pure Mamba-3 remains weak on some real-world retrieval settings; sparse attention helps but restores attention costs. Overall, Mamba-3 advances recurrent language modeling through linked improvements in recurrence, state dynamics, and hardware use, while retrieval and scaling beyond 1.5B remain open.

## Challenge interpretation

- A and B have identical facts, measurements, sentence order, prose, and 187-word length except for the single role-mapping sentence.
- A explicitly assigns the reported operating roles; B reports only that the configurations differ.
- Correct configuration-role recovery is the primary test. Factual support, general clarity, and coverage are guardrails rather than expected treatment effects.
- C is not part of the narrow causal contrast. It preserves the stronger component-binding function of S5 and removes the round-02 meta-discourse sentence.
