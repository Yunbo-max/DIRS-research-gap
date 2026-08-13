# DIRS Strategy Combination Examples

These six examples are blind compositions of strategy nodes `S1`–`S6`. They are not ranked by expert similarity, do not imply a trained GFlowNet, and do not promote any composition to a reusable skill.

> Representation correction: these are macro-strategy composition stress
> tests, not valid DIRS-MCTS samples. `S1`–`S6` are subgraph motifs rather than
> atomic skill nodes. A valid search must first expand motifs into the typed
> primitive flow in `../flow_graph/primitive_skill_dag.json`, then select only
> nodes on the dependency-satisfied frontier. The prose examples remain useful
> as output ablations, but their macro arrows are not evidence of MCTS search.

## C01 — Sparse Pareto capsule

- Category: sparse minimal subgraph
- Strategy nodes: `S1_result_first_pareto_ledger`
- Typed edges: none; a single-node DAG
- Context gate: short architecture abstract where aggregate quality, latency, and one decisive boundary receive nearly all of the information budget
- Selection reason: test whether one strategy node can preserve the central quality-efficiency tradeoff without local diagnostic branches
- Possible failure: the method and mechanism validation may be too compressed
- Word count: 166

Transformer language models provide strong quality, but attention incurs quadratic prefill compute and a KV cache that grows with context. Mamba-3 targets the resulting quality-efficiency gap with a recurrent state-space design combining exponential-trapezoidal discretization, data-dependent complex state updates, and SISO or rank-4 MIMO structure. Under a shared 100B-token FineWeb-Edu protocol, the 1.5B MIMO model obtains 10.24 perplexity and 57.6 average downstream accuracy, compared with 10.51 and 55.4 for the Transformer and lower accuracy for Mamba-2 and Gated DeltaNet. Efficiency results expose a tradeoff: on H100 in BF16 at state size 128, SISO runs at 0.156 ms/token and MIMO at 0.179, both below Mamba-2 at 0.203. MIMO improves quality over SISO, whereas SISO is the faster Mamba-3 mode in this setting. Pure Mamba-3 is nevertheless weaker on some semi-structured and unstructured retrieval tasks. Adding sparse attention improves retrieval but reintroduces attention-related cost. Mamba-3 therefore advances the reported recurrent quality-latency frontier without establishing a universal replacement for attention or performance beyond the tested 1.5B scale in these experiments.

## C02 — Boundary-gated mechanism staircase

- Category: dual-strategy composition
- Strategy nodes: `S4_boundary_first_contrast_loop`, `S2_mechanism_ablation_staircase`
- Typed edge: `S4 --SCOPE_GATES--> S2`
- Context gate: activate when early calibration about fixed-state retrieval is required, but component-to-test attribution remains a primary function
- Selection reason: constrain the interpretation before spending the remaining budget on three ordered mechanism-result bindings
- Possible failure: the negative opening and three local steps can delay the aggregate contribution
- Word count: 176

A fixed recurrent state provides constant memory, but it can lose sparse evidence: pure Mamba-3 remains weak on some semi-structured and unstructured retrieval tasks, while hybrid attention helps at added attention cost. Within that boundary, Mamba-3 strengthens the recurrent core in three steps. Exponential-trapezoidal discretization, a learned trap gate, and B/C biases improve the local recurrence; at 440M, removing bias and trap worsens perplexity from 15.72 to 16.68. Data-dependent complex rotations expand state dynamics; Mamba-3 reaches 100% parity accuracy and 87.75 on bracketed modular arithmetic, compared with 0.9 and 0.88 for Mamba-2. Rank-4 MIMO projections then increase arithmetic intensity and improve the 1.5B SISO result from 10.35 to 10.24 perplexity and from 56.4 to 57.6 average downstream accuracy. The hardware tradeoff remains explicit: at BF16 state size 128 on H100, MIMO takes 0.179 ms/token, faster than Mamba-2 at 0.203 but slower than SISO at 0.156. These component-linked results support stronger state tracking, language quality, and recurrent decode efficiency. They do not remove the retrieval boundary or demonstrate that the gains transfer beyond the reported 1.5B scale.

## C03 — Operating-point evaluation matrix

- Category: dual-strategy composition
- Strategy nodes: `S6_operating_point_decision_split`, `S3_evaluation_mosaic_with_scope_gate`
- Typed edge: `S6 --ROLE_ASSIGNMENT_FEEDS--> S3`
- Context gate: activate when the reader must choose a quality- or speed-oriented configuration and still distinguish what language, formal-state, and retrieval evaluations establish
- Selection reason: make the configuration decision the entry point, then assign every other evaluation a separate diagnostic role
- Possible failure: decision framing can crowd out architectural novelty
- Word count: 169

Which Mamba-3 configuration is preferable depends on the objective. Both modes share exponential-trapezoidal recurrence, a learned trap gate and B/C biases, and data-dependent complex state dynamics; rank-4 MIMO additionally raises arithmetic intensity. Under 100B-token FineWeb-Edu training, the 1.5B MIMO model reaches 10.24 perplexity and 57.6 average downstream accuracy, improving over SISO at 10.35 and 56.4 and exceeding the reported Transformer, Mamba-2, and Gated DeltaNet baselines. H100 kernel measurements reverse the within-family ordering for speed: in BF16 at state size 128, SISO requires 0.156 ms/token and MIMO 0.179. MIMO is therefore the reported quality-oriented choice, whereas SISO is the speed-oriented choice in this setting. Other evaluations test different functions of the shared design. Formal languages probe state dynamics, where Mamba-3 scores 100% on parity versus 0.9 for Mamba-2. Retrieval probes the fixed-state boundary: pure Mamba-3 is mixed on difficult real-world tasks, and sparse attention improves retrieval while adding attention cost. The evidence supports distinct quality and latency operating points, not one universally dominant configuration, and reported scaling stops at 1.5B.

## C04 — Pareto-led selective binding

- Category: dual-strategy composition
- Strategy nodes: `S1_result_first_pareto_ledger`, `S5_salience_anchor_then_selective_bindings`
- Typed edge: `S1 --LEDGER_SEEDS_ANCHOR--> S5`
- Context gate: activate when the main comparative outcome must be immediately visible, but at least two local mechanism checks must remain recoverable
- Selection reason: use S1 to choose the headline Pareto facts and S5 to attach only the most discriminative local evidence
- Possible failure: result-first framing can make mechanisms look post hoc
- Word count: 175

At 1.5B parameters, Mamba-3 MIMO reaches 10.24 FineWeb-Edu perplexity and 57.6 average downstream accuracy, outperforming the reported Transformer, Mamba-2, and Gated DeltaNet baselines under shared 100B-token training. This aggregate gain comes from an inference-first state-space architecture rather than attention: Mamba-3 combines exponential-trapezoidal discretization with a learned trap gate and B/C biases, data-dependent complex state updates, and rank-4 MIMO projections that increase arithmetic intensity. Selective diagnostics distinguish the roles of these changes. Removing bias and trap degrades 440M perplexity from 15.72 to 16.68, while the complex-update model reaches 100% parity accuracy versus 0.9 for Mamba-2 and strongly improves modular arithmetic. MIMO improves quality over SISO but is not the fastest Mamba-3 mode. On H100 in BF16 at state size 128, MIMO runs at 0.179 ms/token, below Mamba-2 at 0.203 but above SISO at 0.156. The recurrent state also imposes a boundary: pure Mamba-3 remains weaker on some real-world retrieval tasks, and sparse attention improves retrieval only by adding attention-related cost. The results support linked quality and efficiency gains while leaving retrieval and transfer beyond 1.5B unresolved.

## C05 — Boundary-scoped diagnostic audit

- Category: triple-strategy composition
- Strategy nodes: `S4_boundary_first_contrast_loop`, `S3_evaluation_mosaic_with_scope_gate`, `S2_mechanism_ablation_staircase`
- Typed edges: `S4 --SCOPE_GATES--> S3`; `S3 --EVAL_ROLE_ORDERS--> S2`
- Context gate: activate for readers who require both early risk calibration and a diagnostic account of which evaluation tests which architectural function
- Selection reason: set the retrieval boundary, allocate nonredundant evaluation roles, then order component evidence by those roles
- Possible failure: the abstract may read like an audit checklist rather than a unified contribution
- Word count: 180

Mamba-3 is best assessed as a set of diagnostic claims rather than by one aggregate score. Retrieval first defines the scope: its fixed recurrent state remains weak on some semi-structured and unstructured tasks, and hybrid attention improves retrieval while restoring attention-related costs. Recurrence ablations then test local sequence modeling. At 440M, the full exponential-trapezoidal design with a learned trap gate and B/C biases achieves 15.72 perplexity, versus 16.68 without bias and trap. Formal languages test the data-dependent complex state update: Mamba-3 obtains 100% parity accuracy, 98.51 without-bracket arithmetic accuracy, and 87.75 with brackets, far above the corresponding Mamba-2 results. Language modeling measures system quality under shared 100B-token training; at 1.5B, rank-4 MIMO reaches 10.24 perplexity and 57.6 average downstream accuracy, ahead of Transformer, Mamba-2, and Gated DeltaNet. Finally, H100 kernels test the intended efficiency tradeoff. In BF16 at state size 128, SISO runs at 0.156 ms/token and MIMO at 0.179, versus 0.203 for Mamba-2. Together, the evaluations bind different mechanisms to different functions while showing that no single metric establishes universal superiority. Direct 7B scaling and complete retrieval remain unproven.

## C06 — Context-gated quality, speed, or retrieval fork

- Category: conditional-branch composition
- Strategy nodes: `S5_salience_anchor_then_selective_bindings`, `S6_operating_point_decision_split`, `S4_boundary_first_contrast_loop`
- Typed edges: `S5 --IF_DEPLOYMENT_CHOICE_SALIENT--> S6`; `S5 --IF_RETRIEVAL_RISK_SALIENT--> S4`; both branches `--JOINS_AT--> bounded synthesis
- Context gate: activate S6 for speed/quality configuration choice, S4 for sparse-evidence risk, or both for serving contexts where latency and retrieval are simultaneously binding
- Realized path here: both branches active
- Selection reason: retain a common early contribution anchor while allocating the remaining budget conditionally instead of applying every strategy unconditionally
- Possible failure: activating both branches can erase the sparsity benefit of conditional planning
- Word count: 179

Mamba-3 addresses the capability and decode costs of linear sequence models with exponential-trapezoidal recurrence, data-dependent complex state dynamics, and SISO or rank-4 MIMO structure. Under shared 100B-token FineWeb-Edu training, its 1.5B MIMO configuration reaches 10.24 perplexity and 57.6 average downstream accuracy, ahead of the reported Transformer, Mamba-2, and Gated DeltaNet baselines. The appropriate path then depends on context. For latency-sensitive H100 serving in BF16 at state size 128, SISO is the faster Mamba-3 option at 0.156 ms/token. When quality is prioritized, MIMO improves over SISO and runs at 0.179 ms/token, still below Mamba-2 at 0.203. When preserving sparse facts is the binding requirement, neither recurrent mode fully resolves the problem: pure Mamba-3 remains weaker on some real-world retrieval tasks, and sparse attention improves retrieval while reintroducing attention cost. The shared recurrence is also supported by component tests. Removing bias and trapezoidal discretization worsens 440M perplexity from 15.72 to 16.68, and Mamba-3 reaches 100% parity accuracy versus 0.9 for Mamba-2. Thus the architecture offers context-dependent quality, speed, and hybrid-retrieval paths rather than one universal winner; performance beyond 1.5B remains untested.
