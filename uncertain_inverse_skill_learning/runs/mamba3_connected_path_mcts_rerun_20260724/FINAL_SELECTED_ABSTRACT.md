# Final Connected-Path Abstract

Selected path: `path_01`

Selection method: `prefix-constrained cached-rollout MCTS`

Blind evaluator reward: `0.965`

Word count: `214`

Transformer language models deliver strong quality, but quadratic attention, growing KV caches, and the weaker capability of many linear alternatives leave a persistent quality–efficiency gap. We introduce Mamba-3, an inference-oriented state-space sequence model designed to close this gap. It replaces simpler updates with an exponential-trapezoidal recurrence and learned trap gate, adds data-dependent complex rotations for richer state dynamics, and supports a rank-4 multi-input multi-output formulation. The resulting Llama-style stack combines the recurrent mixer with SwiGLU blocks, normalized and biased B/C projections, and fused implementations. MIMO increases arithmetic intensity in memory-bound decoding, while SISO provides the leanest latency point. We evaluate models from 180M to 1.5B parameters after shared 100B-token FineWeb-Edu pretraining, alongside retrieval, formal state-tracking, and H100 inference tests against Transformers, Mamba-2, and Gated DeltaNet. Mamba-3 improves language-model quality, formal state tracking, and BF16 recurrent decoding, although pure recurrence remains uneven on difficult retrieval. At 1.5B, MIMO attains 10.24 perplexity and 57.6 average accuracy; at BF16 state size 128, SISO takes 0.156 ms per token versus 0.203 for Mamba-2. These results expose a practical tradeoff: MIMO favors quality, SISO favors speed, and hybrid attention can recover sparse-fact retrieval at added attention cost. Thus, Mamba-3 offers a promising efficient sequence backbone, with evidence currently bounded to models no larger than 1.5B and specialized GPU kernels.
