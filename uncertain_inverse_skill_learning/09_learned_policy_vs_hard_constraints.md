# Learned Policy Versus Hard Constraints

## 1. Principle

Intelligence in DIRS must come from learning which skills exist, when they
apply, how they depend on each other, and which strategy succeeds. A manually
written route with manually chosen weights is a structured prompt, not inverse
skill learning.

The method therefore separates:

```text
hard meta-constraints:
  conditions required for logical validity, evidence integrity, permissions,
  reproducibility, and leakage prevention

learned skill policy:
  domain skills, graph structure, contextual activation, utility, search
  priorities, repair choices, and stopping behavior
```

## 2. Legitimate Hard Constraints

These do not encode a domain solution:

```text
execution graph must be acyclic
declared input/output schemas must type-check
factual actions require provenance
held-out targets cannot enter selection or execution
tool calls must obey allowed permissions
required outputs cannot be left dangling
graph versions require exact diffs and rollback
training, validation, and test boundaries must be respected
```

They are analogous to a programming language type system or an experimental
protocol.

`factual actions require provenance` does not mean that every procedural
strategy must already occur in an expert artifact. DIRS must distinguish:

```text
content provenance:
  evidence supporting a factual claim in the generated artifact

induction provenance:
  why a new strategy was proposed and which blind rollouts, interventions, or
  transfer cases support keeping it
```

A self-inferred strategy may have no expert-trace citation. It is valid as a
candidate when it declares induction provenance; any facts it produces still
need content provenance.

## 3. Components That Must Be Learned

```text
skill node identities and granularity
node aliases and canonical equivalence
node inclusion probability
edge existence and direction
conditional dependency activation
AND/OR dependency groups
evidence-to-skill bindings
expert utility or preference parameters
graph-edit proposal distribution
MCTS policy prior and value estimate
exploration and widening schedules
executor choice and realization policy
verifier calibration
acceptance and stopping thresholds
self-inferred strategy generation and novelty allocation
```

A human-written value may initialize learning, but it must be labelled as a
prior, compared with non-informative and learned alternatives, and updated
using training evidence.

## 4. Open-Vocabulary Node Discovery

Do not begin with a closed list such as:

```text
context, gap, method, result, interpretation
```

as the only possible graph. Instead:

1. Segment artifact claims and operations.
2. Propose skill contracts in open vocabulary.
3. Cluster only when preconditions, input/output behavior, validators, and
   interventions support equivalence.
4. Split a node when one contract hides incompatible conditions or outcomes.
5. Add new nodes from repeated unexplained failures.
6. Retire nodes that lack held-out utility.

Named families may be metadata learned or assigned after discovery. They must
not determine selection by themselves.

## 5. Learned Edge Structure

Co-occurrence or text order is only a feature. An edge posterior uses:

```text
explicit information transfer
observed process order when available
cross-case recurrence
downstream precondition satisfaction
deletion effect
reversal effect
alternative-path performance
context
```

The system learns:

\[
p(e,\operatorname{direction},\operatorname{condition}\mid
\mathcal D,\operatorname{interventions}).
\]

It does not promote `method -> result` merely because that route was authored
in a template.

## 6. Learned Utility

Use paired expert or verifier preferences:

\[
\Pr(z_a\succ z_b\mid c)
=
\sigma(U_\theta(z_a,c)-U_\theta(z_b,c)).
\]

Features may include evidence, validity, quality, cost, and risk, but their
weights are learned. A nonlinear contextual utility is allowed when supported
by data. Report uncertainty over \(\theta\) and test reward-model robustness.

## 7. Learned Search

MCTS supplies a general exploration mechanism, not task intelligence. Its
policy and value should be amortized:

\[
P_\eta(a\mid s,c),
\qquad
V_\xi(s,c),
\]

from accepted rollouts, rejected rollouts, replay failures, and posterior graph
samples. Search constants are selected by nested validation or an online
bandit, never by viewing final test performance.

## 8. Learned Acceptance

Graph acceptance remains statistically constrained, but thresholds come from:

```text
validation calibration
desired false-accept rate
operational hard-failure tolerance
predeclared resource budget
```

The evaluator does not get to modify thresholds to accept its own proposal.
Nor may it require lexical agreement with one expert artifact. Learned utility
should combine blind functional judgments, evidence fidelity, preferences,
transfer, cost, and risk.

## 9. Required Ablations

```text
fixed hand-written node inventory vs open-vocabulary discovery
manual route templates vs learned edge posterior
manual weights vs learned preference utility
fixed MCTS prior vs amortized learned prior/value
point thresholds vs calibrated statistical acceptance
uninformative prior vs human prior vs empirical-Bayes prior
expert-trace-only proposals vs autonomous strategy discovery
reference-similarity reward vs functional preference reward
```

If a hand-coded version performs equally well, the claim must be narrowed:
the experiment has demonstrated a useful workflow, not learned intelligence.

## 10. Audit Record

Every run declares:

```yaml
method_audit:
  hard_constraints: []
  human_priors: []
  learned_components: []
  validation_selected_parameters: []
  test_visible_decisions: []
  unresolved_manual_rules: []
```

`test_visible_decisions` must be empty before final evaluation.
