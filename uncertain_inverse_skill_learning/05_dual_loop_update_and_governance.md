# Dual-Loop Update And Governance

## 1. State

At outer round \(t\), store:

\[
\mathcal M_t=(G_t,q_t,\theta_t,\psi_t,H_t,Q_t,L_t),
\]

where:

```text
G_t: accepted persistent graph version
q_t: trace and graph posterior state
theta_t: contextual utility and selection parameters
psi_t: executor configuration
H_t: verifier suite
Q_t: replay set
L_t: mutation, acceptance, and rollback ledger
```

## 2. Outer Loop

Only the outer loop may propose changes to persistent state:

\[
\widetilde{\mathcal M}_t
=
\operatorname{Propose}(\mathcal M_t,\mathcal D,F_{<t}).
\]

Before evaluation, freeze:

```text
graph version
posterior parameters
executor version
verifier version
model and prompt versions
random seeds and budgets
```

Changing several components in one proposal makes attribution impossible.
Prefer one edit family per candidate unless a dependency requires an atomic
multi-edit.

The proposal source is not restricted to diagnosed omissions from an expert
artifact. The outer loop must also preserve an autonomous exploration channel
for self-inferred strategies. These proposals are hypotheses, not accepted
knowledge: they become persistent only after blind execution, intervention,
and transfer validation.

## 3. Inner Loop

For paired case \(i\):

\[
G_{i,t}^{(k)}
\sim
q_{\widetilde{\mathcal M}_t}(G\mid c_i),
\]

\[
S_{i,t}^{(k)}
=
\operatorname{MCTSSelect}
(G_{i,t}^{(k)},x_i,m_i,\widetilde\theta_t),
\]

\[
\hat y_{i,t}^{(k)}
=
\operatorname{Execute}
(x_i,S_{i,t}^{(k)},\widetilde\psi_t),
\]

\[
(r_{i,t}^{(k)},e_{i,t}^{(k)})
=
\operatorname{Evaluate}
(\hat y_{i,t}^{(k)},y_i,x_i,S_{i,t}^{(k)},\widetilde H_t).
\]

The expert target is hidden from selection and execution. It is revealed only
for post-generation training feedback.

## 4. Typed Root-Cause Attribution

```text
trace-inference error:
  candidate explanatory traces omit or hallucinate necessary operations

representation error:
  shared graph lacks or misstates a reusable node, edge, condition, contract,
  evidence binding, validator, or repair rule

strategy opportunity:
  no current component is erroneous, but a self-inferred operation or
  dependency is predicted to improve utility, robustness, or efficiency

selection error:
  MCTS chooses a poor sub-DAG from an adequate frozen graph

execution error:
  the executor fails to follow an adequate selected sub-DAG

evaluation error:
  the verifier is noisy, biased, leaky, or rewards a proxy

data/evidence error:
  source evidence is missing, wrong, or insufficient to support the target
```

Authorization:

```text
trace-inference error:
  update proposal distribution or extraction checks

representation error:
  may authorize persistent graph edits

strategy opportunity:
  may authorize a candidate edit in an exploratory branch; it cannot bypass
  acceptance, replay, provenance, or transfer tests

selection error:
  update priors, utility, widening, or MCTS budget

execution error:
  update executor, prompt, tool handling, or node realization

evaluation error:
  calibrate verifier on independent cases

data/evidence error:
  repair evidence or mark the task underdetermined
```

Representation errors and explicitly labelled strategy opportunities may
authorize candidate structural changes. Neither automatically authorizes
promotion to the persistent graph.

## 5. Permitted Persistent Edits

```text
ADD_NODE
DELETE_NODE
SPLIT_NODE
MERGE_NODES
RETYPE_NODE
ADD_EDGE
DELETE_EDGE
REVERSE_EDGE
CONDITION_EDGE
ADD_DEPENDENCY_GROUP
UPDATE_PRECONDITION
UPDATE_REJECTION_RULE
UPDATE_EVIDENCE_BINDING
UPDATE_COST_OR_BUDGET
UPDATE_VERIFIER
UPDATE_REPAIR_RULE
UPDATE_CONTEXT_GATE
```

## 6. Edit Record

```yaml
graph_edit:
  operation_id:
  parent_version:
  proposal_origin: expert_inferred | self_inferred | failure_induced | crossover
  operation_type:
  targets: []
  evidence_cases: []
  induction_evidence: []
  transfer_cases: []
  contradicting_cases: []
  typed_root_causes: []
  posterior_before: {}
  expected_effect:
  replay_risk:
  validation_plan:
  exact_diff:
  rollback_diff:
```

## 7. Acceptance

Compare proposal and incumbent under identical:

```text
cases
graph-sampling count
model and prompt versions
MCTS rollout budget
tool budget
random seeds
verifier version
```

Let:

\[
\Delta_i
=
J_i(\widetilde{\mathcal M}_t)
-J_i(\mathcal M_t).
\]

Accept atomically only if:

\[
\widehat{\Delta}>\delta_{\min},
\qquad
\operatorname{LCB}_{1-\alpha}(\Delta)>0,
\]

\[
\operatorname{HardFailure}(\widetilde{\mathcal M}_t)
\le
\operatorname{HardFailure}(\mathcal M_t),
\]

\[
\operatorname{Replay}(\widetilde{\mathcal M}_t)
\ge
\operatorname{Replay}(\mathcal M_t)-\epsilon_Q,
\]

and:

```text
all hard graph invariants pass
complexity remains within budget
calibration does not materially regress
the improvement is not confined to the motivating case
```

For a one-case experiment, a successful self-inferred edit may be retained only
as a `local_strategy_hypothesis`. Promotion to `reusable_skill` requires
improvement on separate transfer cases. Similarity to expert wording, sentence
order, or a single reconstructed expert DAG is never an acceptance gate.

\(\delta_{\min}\), \(\alpha\), \(\epsilon_Q\), complexity budgets, and
calibration tolerances must be preregistered from validation data or operational
risk requirements. They may not be changed after inspecting test outcomes.

Use paired bootstrap or a paired randomization test when reward distributions
are non-normal or sample sizes are small.

## 8. Posterior Updates Are Also Versioned

A graph can keep the same MAP structure while uncertainty changes. Record:

```text
node posterior shifts
edge posterior shifts
context-gate shifts
alternative-graph mass
posterior entropy
calibration metrics
```

A large confidence update requires provenance even when no structural diff
occurs.

## 9. Rollback

Every accepted version stores:

```text
parent identifier
exact forward diff
exact inverse diff
evaluation manifest
replay results
posterior snapshot or reproducible update inputs
executor and verifier versions
```

Rollback must restore both graph structure and associated contracts,
posteriors, gates, validators, and policies.

## 10. Stop Criteria

Stop only after:

```text
structural stability:
  no accepted structural edit for K full passes

behavioral stability:
  equivalent sub-DAG choices and outputs on validation cases

posterior stability:
  small node/edge posterior and entropy changes

calibration stability:
  no improvement in held-out calibration

replay stability:
  no material regression
```

Do not stop merely because labels or Markdown wording stop changing.
Also do not equate a frozen MAP graph with exhausted discovery. Before stopping
training, run a challenge pass that allocates nonzero proposal budget to novel
strategies. Stop only when no candidate passes the preregistered utility and
transfer gates and posterior or Pareto-front changes are small.

The edit vocabulary is hard because it defines auditable state transitions;
which edit to make, where to apply it, and its expected value are learned from
data and feedback.
