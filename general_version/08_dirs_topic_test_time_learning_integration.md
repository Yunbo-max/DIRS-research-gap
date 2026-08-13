# DIRS Topic-Level Test-Time Learning Integration

Date: `2026-07-28`

Purpose: define how DIRS may learn on an unseen research topic without leaking
future papers, treating self-generated gaps as truth, or contaminating the
persistent skill graph.

This is the canonical local integration of the topic-level test-time learning
design. It complements:

```text
03_dirs_mcts_inference_selector.md
05_dirs_top_conference_evaluation_protocol.md
07_dirs_tta_ablation_study.md
dirs_tta_ablation_variants.csv
```

## 1. Correct Problem Classification

The original DIRS pipeline is:

```text
training topics
  -> infer a reusable global skill graph
  -> freeze the accepted graph

unseen topic
  -> retrieve compatible skills
  -> search and execute a connected sub-DAG
  -> verify and select an output
```

This is meta-learning plus test-time inference. It becomes test-time learning
only when observations from the unseen topic update a task-local state that
changes later decisions on that same topic.

```text
test-time search:
  collect candidates while the task policy and beliefs remain fixed

test-time adaptation:
  update topic beliefs, path posterior, state, gates, or budget from legitimate
  observations

test-time training:
  update temporary trainable parameters such as a topic latent, prefix, or LoRA
```

Do not call static sub-DAG selection `test-time training`.

## 2. Three State Levels

### 2.1 Global Slow State

\[
\mathcal M_{\mathrm{global}}
=
(G,\Theta_{\mathrm{backbone}},H_{\mathrm{validator}},
\phi_{\mathrm{editor}},\phi_{\mathrm{router}}).
\]

It contains:

```text
accepted reusable skill graph G
backbone and shared adapters
validator suite and calibration
meta-learned hindsight editor
meta-learned routing and stopping policy
replay set, posterior priors, provenance, and rollback history
```

It is frozen for the entire blind test topic.

### 2.2 Topic-Local State

\[
\mathcal A_T
=
(u_T,q_T(G,\pi),\gamma_T,b_T,\Delta_T,\Delta G_T).
\]

It contains:

```text
u_T:
  low-dimensional topic latent

q_T(G, pi):
  posterior over valid graph snapshots and executable skill paths

gamma_T:
  topic-specific context gates and applicability beliefs

b_T:
  remaining search, execution, communication, and adaptation budget

Delta_T:
  optional temporary prefix, adapter, or LoRA parameters

Delta G_T:
  temporary typed graph hypotheses or repairs
```

The entire topic-local state must be reset or archived as non-persistent
evidence after the topic. It may not directly modify the global state.

### 2.3 Single-Exploration State

\[
\mathcal E_{T,k}
=
(g_k,h_k,\mathcal B_k,e_k,f_k,c_k).
\]

It contains:

```text
g_k:
  current candidate research gap or research relation

h_k:
  recurrent belief about terminology, evidence structure, prior coverage,
  contradictions, and unresolved uncertainty

B_k:
  bound evidence and executed node outputs

e_k:
  new search, retrieval, tool, or execution observation

f_k:
  typed failure signature

c_k:
  cumulative compute and information cost
```

`u_T`, `h_k`, `Delta_T`, and `Delta G_T` are not reusable knowledge merely
because they improved one topic.

## 3. Highest-Priority Adaptation Objects

### 3.1 Topic Latent

The first trainable test-time object should be a compact topic latent:

\[
u_T^*
=
\arg\min_u
\left[
\mathcal L_{\mathrm{legal}}(u;\mathcal D_T^{<c})
+\lambda_u\|u-u_0\|_2^2
+\lambda_H\mathcal H(q_T)
\right].
\]

It should capture:

```text
domain terminology and aliases
common evidence and evaluation relations
which methods and failure modes tend to co-occur
which graph motifs are applicable
which claims require stronger support
```

It must not encode the hidden future gap target.

### 3.2 Recurrent Gap And Belief Repair

Maintain separate candidate and belief states:

\[
(g_{k+1},h_{k+1})
=
F_\theta(g_k,h_k,e_k,f_k,u_T).
\]

The loop is:

```text
propose a bounded gap
  -> search for prior work and counterevidence
  -> update the topic belief
  -> classify the failure
  -> edit, split, narrow, defer, or reject the gap
  -> decide whether another observation is worth its cost
```

This is not free-form agent debate. Each iteration must bind new evidence,
record the state transition, and state why the candidate changed.

### 3.3 Hindsight Failure-To-Repair Editor

Training should preserve the relation:

```text
failure context
  -> chosen repair/search action
  -> resulting evidence and state
  -> utility and cost change
  -> whether the action transferred to another topic
```

Canonical record:

```yaml
hindsight_transition:
  source_topic:
  state_before:
  candidate_gap_before:
  failure_signature:
  available_actions: []
  selected_action:
  state_after:
  candidate_gap_after:
  new_evidence: []
  utility_before:
  utility_after:
  cost:
  hard_failures_before: []
  hard_failures_after: []
  transfer_topics: []
  provenance:
```

The editor learns:

\[
\phi^*
=
\arg\max_\phi
\mathbb E
\left[
\Delta U
-\lambda_C C
-\lambda_F N_{\mathrm{hard\ failure}}
\right].
\]

It is trained across past topics. On the blind topic it selects local actions;
it does not learn from the hidden future result.

### 3.4 Temporary LoRA

Temporary LoRA is optional and lower priority. It is permitted only when:

```text
the topic contains legitimate pre-cutoff supervision
the final query or future target remains hidden
the adapter has a strict parameter and step budget
early stopping uses pre-cutoff support cases
the adapter is reset after the topic
the frozen base and topic-latent alternatives are reported
```

\[
\Delta_T^*
=
\arg\min_{\Delta\in\mathcal C}
\mathcal L_{\mathrm{legal}}
(\Theta_{\mathrm{backbone}}+\Delta;\mathcal D_T^{<c}).
\]

Self-generated future-gap statements are not labels.

### 3.5 Dynamic Multi-Agent Routing

Dynamic routing is added only after the single-controller latent, recurrent, and
hindsight variants are validated. The router chooses among typed operations:

```text
RETRIEVE
PROPOSE
SEARCH_COUNTEREXAMPLE
CHECK_COVERAGE
VERIFY_EVIDENCE
NARROW
SPLIT
REJECT
DESIGN_EXPERIMENT
STOP
```

Its objective includes marginal information value and total system cost:

\[
a_k
=
\arg\max_a
\mathbb E[
\Delta U_a
+\beta I_a
-\lambda_T C_{\mathrm{token},a}
-\lambda_L C_{\mathrm{latency},a}
-\lambda_C C_{\mathrm{comm},a}
].
\]

## 4. Legitimate Test-Time Supervision

ARC-like tasks have input-output demonstrations. An unseen research topic does
not provide a correct future gap. DIRS must therefore adapt only from
pre-cutoff, independently checkable objectives.

### 4.1 Time-Ordered Leave-One-Paper-Out

For a pre-cutoff paper \(p_j\), condition on earlier permitted papers and
predict a masked structured property of \(p_j\):

\[
\mathcal L_{\mathrm{LOPO}}
=
\sum_{j:t_j<c}
\ell(
\operatorname{Predict}(\mathcal D_T^{<t_j}),
\operatorname{MaskTarget}(p_j)
).
\]

The target can be a method family, evidence relation, evaluation role, or
failure mode. Do not train on post-cutoff papers.

### 4.2 Masked Chip Field Or Graph Edge

Mask one legitimate pre-cutoff Chip-Memory field or typed edge:

```text
Method
Gap
Evaluation
Result
footprint relation
SUPPLIES/VALIDATES/SCOPES/COMPARES/GATES/REPAIRS edge
```

Train reconstruction or contrastive ranking without exposing the final
evaluation gap.

### 4.3 Evidence Support, Refutation, And Near-Miss Classification

Construct labelled pairs from cited or manually verified pre-cutoff evidence:

```text
supports
refutes
near-miss
insufficient
wrong scope
already solved before cutoff
```

This provides a safer objective than using a generated novelty claim as truth.

### 4.4 Strict Consistency Objectives

Permitted deterministic or weakly supervised signals include:

```text
chronological order
canonical entity aliases
topic and paper identity consistency
node and edge type constraints
citation-to-evidence binding
graph connectivity and acyclicity
required-input and output-port compatibility
```

### 4.5 Combined Legal Objective

\[
\mathcal L_{\mathrm{legal}}
=
\lambda_1\mathcal L_{\mathrm{LOPO}}
+\lambda_2\mathcal L_{\mathrm{masked\ chip}}
+\lambda_3\mathcal L_{\mathrm{evidence}}
+\lambda_4\mathcal L_{\mathrm{consistency}}.
\]

All coefficients and stopping thresholds are selected without the final test
topics or post-cutoff results.

## 5. Forbidden Pseudo-Supervision

```text
generate a gap and train the model to reproduce it
ask the same generator whether its gap is novel and use that answer as truth
use post-cutoff paper titles, abstracts, citations, or graph nodes during adaptation
retrieve a future paper and then hide only its title
use expert final rankings to tune the test-topic adapter
write topic-local discoveries directly into the persistent graph
select a checkpoint after observing future-gap performance
```

An LLM judge may produce an uncertain feature or proposal. It cannot create
ground truth merely by confidence.

## 6. Temporal Topic-Holdout Protocol

Random paper splits are invalid for the central research-gap discovery claim.

### 6.1 Outer Split

```text
training:
  complete topics used for global graph, editor, router, and validator learning

validation:
  complete held-out topics used for method choices and thresholds

test:
  untouched complete topics used once for final evaluation
```

No paper from a final test topic may influence the global graph or adapter
hyperparameters unless the study explicitly reports a transductive protocol.

### 6.2 Inner Temporal Cutoff

For each validation or test topic choose a preregistered cutoff \(c_T\):

```text
adaptation corpus:
  only papers and evidence with timestamp < c_T

adaptation:
  learn u_T, h_T, q_T, local gates, local patches, and optional Delta_T

freeze:
  candidate gaps, rankings, evidence, confidence, and all topic-local state

reveal:
  post-cutoff papers and independent expert judgments only after freezing
```

Formally:

\[
\hat{\mathcal G}_T
=
\operatorname{Freeze}
\left[
\operatorname{DIRS\text{-}TTA}
(\mathcal D_T^{<c_T},\mathcal M_{\mathrm{global}})
\right],
\]

\[
\operatorname{Score}
(\hat{\mathcal G}_T,\mathcal D_T^{\ge c_T},Y_T^{\mathrm{expert}})
\quad\text{only after freeze}.
\]

### 6.3 What Post-Cutoff Papers Can Measure

Post-cutoff literature is not a perfect oracle that an earlier gap was correct.
Use it as one independent outcome family:

```text
future uptake:
  later work explicitly studies the predicted missing condition or relation

future support:
  later evidence supports the gap's bounded premise

future refutation:
  later work shows the predicted gap premise was wrong

already solved:
  pre-cutoff literature already addressed it

unresolved:
  no decisive later evidence
```

Expert evaluation should remain separate from temporal outcome matching.

## 7. Required Adaptation Ladder

Run in this order:

```text
0. frozen DIRS plus retrieval
1. topic latent
2. topic latent plus recurrent candidate/belief repair
3. topic latent plus recurrent repair plus hindsight editor
4. add temporary LoRA when legal supervision exists
5. add budget-aware dynamic multi-agent routing
```

Do not attribute a gain at level 5 to multi-agent routing if level 2 or level 3
was not compute-matched.

## 8. Required Comparisons

```text
frozen DIRS
frozen DIRS plus retrieval
topic-latent DIRS
recurrent-refinement DIRS
hindsight-editor DIRS
temporary-LoRA DIRS
dynamic-router DIRS
compute-matched single recurrent model
direct retrieval and synthesis
random-paper-split diagnostic, reported only as an inflation control
```

The full registry and additional mechanism ablations are in
`dirs_tta_ablation_variants.csv`.

## 9. Gap-Discovery Metrics

### Validity At Cutoff

```text
pre-cutoff novelty:
  no sufficiently matching solution found before cutoff

evidence adequacy:
  every premise is supported or explicitly uncertain

scope precision:
  the gap states the missing condition, population, mechanism, or evaluation

near-miss discrimination:
  related work is distinguished from a full solution

false-gap rate:
  fraction already solved by pre-cutoff work
```

### Temporal Outcomes

```text
future-uptake precision and recall
future support, refutation, and unresolved rates
time to first relevant post-cutoff work
ranking correlation with independent expert priority
```

### Adaptation

```text
gain after learning u_T
gain after each recurrent repair step
repair success conditioned on failure type
hindsight-editor transfer gain
temporary-LoRA gain and overfitting gap
dynamic-router marginal gain per compute
```

### Safety

```text
future-information leakage
topic-state contamination
unsupported novelty claims
validator calibration under topic shift
negative transfer across topics
```

## 10. Chip-Memory Interface

Chip-Memory supplies structured observations. It does not supply truth labels
for future novelty.

Each reusable operation should record:

```yaml
operation:
  name:
  input_types: []
  output_types: []
  preconditions: []
  applicable_topics: []
  failure_signatures: []
  counterexamples: []
  expected_information_gain:
  token_cost:
  latency_cost:
  verifier_requirements: []
  provenance:
```

Paper chips remain content/evidence memory. DIRS remains the learned
skill-control system. Topic-local latents and recurrent states remain ephemeral.

## 11. Promotion And Persistence

After a blind topic finishes:

```text
discard or archive u_T, h_T, Delta_T, and Delta G_T as topic-local traces
do not merge them into G
extract candidate hindsight transitions with provenance
test each proposed reusable operation on separate topics
run replay and calibration checks
promote only transferable operations through the governed outer loop
```

A future paper confirming one predicted gap does not by itself justify a
persistent skill-graph edit. The reusable object is the operation or repair
policy that transferred, not the specific future fact.

## 12. Minimal Implementation Order

```text
M0:
  temporal topic manifests and leakage checker

M1:
  topic latent with masked-Chip and evidence-relation objectives

M2:
  recurrent gap/belief state and typed failure transitions

M3:
  hindsight transition dataset and editor

M4:
  temporary LoRA with strict support/query separation

M5:
  dynamic routing and multi-agent execution
```

Do not begin M4 or M5 before M1-M3 have compute-matched results. This ordering
keeps the central test-time learning claim identifiable.
