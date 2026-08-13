# Learning And Search Algorithms

## 1. Do Not Give One Algorithm Three Jobs

DIRS contains three different combinatorial problems:

```text
posterior inference:
  retain multiple plausible latent traces or graphs

persistent structure optimization:
  improve the shared skill library across cases

task-time planning:
  select one useful sub-DAG for a particular input
```

MCTS, evolutionary search, and GFlowNet have different strengths and should not
be treated as interchangeable labels.

## 2. Posterior Inference

### Minimum Implementation

Use diverse LLM proposals followed by:

```text
hard graph validation
evidence scoring
blind forward replay
counterfactual tests
posterior normalization
```

This is easy to audit and sufficient for a first paper.

### Conditional DAG-GFlowNet Option

When the goal is calibrated multimodal sampling, learn:

\[
P_\phi(G\mid x,y,m)
\propto
R(G;x,y,m).
\]

Construct a graph sequentially:

```text
start with typed candidate nodes
add an allowed node or execution edge
reject cycle-producing actions
stop and score the completed graph
```

The reward combines evidence, replay, context, and complexity. A GFlowNet is
better matched than best-path search when several structurally different graphs
should retain probability mass.

## 3. Outer-Loop Structure Optimization

The outer loop may use a bounded LLM editor, evolutionary search, or both.
Its candidates need not be copied from an inferred expert trace. A dedicated
exploration channel may propose new procedural nodes, edges, gates, or repair
rules from self-critique, counterfactual failures, uncertainty, and predicted
information gain.

### Edit Genome

```yaml
candidate_edit_set:
  parent_version:
  proposal_origin: expert_inferred | self_inferred | failure_induced | crossover
  edits:
    - operation: ADD_NODE | DELETE_NODE | SPLIT_NODE | MERGE_NODES
      targets: []
    - operation: ADD_EDGE | DELETE_EDGE | REVERSE_EDGE | CONDITION_EDGE
      targets: []
    - operation: UPDATE_EVIDENCE_BINDING | UPDATE_VERIFIER | UPDATE_REPAIR_RULE
      targets: []
  motivating_cases: []
  induction_evidence: []
  transfer_test_cases: []
  predicted_effects: []
```

For `self_inferred` proposals, `induction_evidence` replaces any requirement
that the expert explicitly used or described the strategy. It does not replace
evidence for factual claims produced during execution.

### Evolutionary Population

Maintain multiple candidate graph versions:

\[
\mathcal G_t=\{G_t^{(1)},\ldots,G_t^{(M)}\}.
\]

Mutation uses bounded edits. Crossover is permitted only at contract-compatible
subgraphs and must revalidate acyclicity and dependencies.

Multiobjective fitness:

\[
F(G')=
\Delta J_{\mathrm{val}}
-\lambda_C C(G')
-\lambda_H H(G')
-\lambda_R R_{\mathrm{regress}}(G')
-\lambda_K \operatorname{ECE}(G').
\]

Here:

```text
C: graph complexity and execution cost
H: hard-failure rate
R_regress: replay regression
ECE: posterior or confidence calibration error
```

Keep a Pareto frontier when quality, cost, and risk conflict.

The \(\lambda\) coefficients above are notation for learned or
validation-selected trade-offs. The default implementation should retain the
Pareto vector and let a preregistered deployment constraint choose among
non-dominated graphs. It must not hand-tune weights after viewing test results.

## 4. Inner-Loop Fixed-Snapshot MCTS

Sample or choose a frozen graph snapshot:

\[
G^{(k)}\sim q_\omega(G\mid\mathcal D,c).
\]

MCTS state:

\[
s_t=(S_t,F_t,b_t,h_t,c),
\]

where:

```text
S_t: partial selected sub-DAG
F_t: valid frontier after prerequisites and context gates
b_t: remaining budget
h_t: current evidence bindings and produced intermediate artifacts
c: task context
```

Actions:

```text
ADD_NODE
SATISFY_DEPENDENCY_GROUP
CHOOSE_ALTERNATIVE
ALLOCATE_BUDGET
CALL_TOOL
STOP
```

Permanent graph edits are not actions.

## 5. Uncertainty-Aware Tree Policy

During training:

\[
a^*=
\arg\max_a
\left[
Q(s,a)
+c_{\mathrm{puct}}P(a\mid s,c)
\frac{\sqrt{N(s)}}{1+N(s,a)}
+\beta U_{\mathrm{epi}}(s,a)
\right].
\]

`U_epi` encourages exploration of uncertain but plausible nodes or edges.

The prior \(P(a\mid s,c)\), exploration scale, widening schedule, and
uncertainty coefficient are learned or selected on training/validation
episodes. Node family names do not receive manually privileged scores.

During risk-sensitive deployment:

\[
a^*=
\arg\max_a
\left[
Q(s,a)
+\mathrm{PUCT}(s,a)
-\rho U_{\mathrm{risk}}(s,a)
\right].
\]

Training may explore epistemic uncertainty; deployment should not blindly
reward uncertainty.

## 6. Rollout Reward

\[
R=
w_1R_{\mathrm{evidence}}
+w_2R_{\mathrm{contract}}
+w_3R_{\mathrm{task}}
+w_4R_{\mathrm{quality}}
+w_5R_{\mathrm{transfer}}
-w_6C
-w_7H.
\]

The reward report must remain decomposed. A scalar alone cannot support error
attribution.

## 7. Posterior Sampling In MCTS

Two useful variants:

```text
root sampling:
  sample one G at the start of a rollout and keep it fixed

Thompson-style selection:
  sample contextual node/edge parameters, then plan under that sampled model
```

Do not independently sample every edge without checking acyclicity and
dependency groups. Use complete valid graph samples or a validity-preserving
construction policy.

## 8. Progressive Widening

A large skill graph can expose too many actions. Expand the frontier gradually:

\[
|\mathcal A_{\mathrm{expanded}}(s)|
\le
kN(s)^\alpha,
\qquad 0<\alpha<1.
\]

Candidate priority can use:

```text
context gate
evidence availability
dependency readiness
posterior support
expected information gain during training
cost
```

These are learned features of a proposal policy. The list declares observable
inputs, not a fixed weighted formula.

## 9. Recommended Algorithm

```text
Input:
  expert cases D
  initial skill ontology and contracts
  validation and replay cases

1. Generate K candidate traces per expert case.
2. Reject invalid candidates and score the rest by evidence and blind replay.
3. Form q_i(z) for each case.
4. Canonicalize skills by contract and build an initial graph posterior.
5. Outer loop proposes bounded graph versions.
6. Freeze each candidate version.
7. Inner MCTS selects and executes task-specific sub-DAGs on paired cases.
8. Attribute errors to representation, selection, execution, or evaluation.
9. Also allow exploration-motivated strategy proposals; accept persistent
   edits only when blind paired validation and transfer gates pass.
10. Update posterior support, provenance, replay, version, and rollback data.
11. Stop after behavioral and posterior stability.
```

## 10. Role Of Conditional Diffusion

A conditional graph diffusion model could generate \(G\mid c\), but it adds a
second complex generative model and does not automatically enforce contracts,
acyclicity, or calibrated posterior mass. It should be a later ablation, not a
required component of the base method.

For the initial method:

```text
GFlowNet or explicit proposal posterior:
  uncertainty-preserving graph generation

ES or bounded editor:
  persistent graph improvement

MCTS:
  task-conditioned sub-DAG planning
```

## 11. Learned Components

```text
node discovery:
  open-vocabulary proposals from artifacts and failures

node canonicalization:
  calibrated contract-equivalence model plus replay checks

edge discovery:
  posterior inference from information transfer, process evidence, and
  deletion/reversal interventions

context gates:
  supervised or preference-learned p(v|c) and p(e|c)

utility:
  inverse/preference learning from expert artifacts and paired executions

proposal policy:
  learned from accepted and rejected graph edits

MCTS prior and value:
  amortized from previous rollouts and held-out verifier outcomes

stopping and acceptance:
  validation-calibrated statistical decision rules

autonomous strategy proposals:
  learned from self-critique, unsuccessful rollouts, uncertainty, and the
  measured effects of prior accepted or rejected inventions
```

No abstract-writing route such as `gap -> method -> result` is a universal
hard-coded policy. Such routes may emerge as high-posterior structures in a
domain and remain conditional and revisable.

## 12. Path Frequency Is Not Contextual Utility

Training support estimates how often a path occurred under the training case
distribution:

\[
\hat P_{\mathrm{train}}(\pi).
\]

It does not directly estimate either:

\[
P(\pi\mid c_{\mathrm{new}})
\quad\text{or}\quad
\mathbb E[U\mid\pi,c_{\mathrm{new}}].
\]

Using raw trace frequency as a strong PUCT prior can lock the search onto a
common path when the new context favors another route. DIRS should keep three
quantities separate:

```text
structural support:
  whether a node, edge, or complete path is plausible and executable

contextual policy prior:
  which legal continuation is promising for the current task

execution value posterior:
  the reward distribution observed when the selected path is executed
```

The contextual prior should be learned or calibrated on training/validation
tasks. Frequency may be one input feature, but not an unexamined substitute
for contextual utility.

## 13. Recommendation Under Stochastic Execution

When each path can produce multiple writer realizations, a terminal path has a
reward distribution rather than one cached score:

\[
R_{\pi,j}\sim p(R\mid \pi,c,\theta_{\mathrm{writer}},
\theta_{\mathrm{evaluator}}).
\]

Saving only visit counts can preserve a biased prior even after rollout
evidence arrives. A stochastic run must report at least:

```text
terminal sample count
empirical or posterior mean utility
uncertainty interval
hard-failure probability
visit-count recommendation
value-posterior recommendation
simple-regret and best-path-identification baselines
```

Neither visit count nor posterior mean is universally correct. The deployment
rule should be preregistered and validation-calibrated, with a risk-sensitive
lower confidence bound when hard failures matter.

For a small finite path set, even allocation is a required baseline. MCTS is
justified only when shared prefixes, unequal costs, a large frontier, or
contextual value estimates let it use samples more efficiently.

## 14. Search-Algorithm Routing

MCTS is not part of the definition of a valid learned DAG. Choose the
controller according to the search regime:

```text
small enumerable complete-path set, terminal rewards only:
  Sequential Halving by default

small enumerable set with calibrated reward posterior:
  Top-Two Thompson Sampling

large or implicit path space with useful partial values:
  frontier-constrained MCTS plus progressive widening

large structural space with stochastic terminal executions:
  MCTS for structure; best-arm posterior allocation for repeated leaves
```

The 2026-07-25 six-path paired simulation found lower simple regret for both
Top-Two Thompson and Sequential Halving than empirical-prior MCTS-Q in every
tested scenario-budget cell. This supports routing by problem structure
rather than declaring one universal optimizer.
