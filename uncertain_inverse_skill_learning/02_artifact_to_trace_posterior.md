# Artifact-To-Trace Posterior

## 1. Objective

Given \((x_i,y_i,m_i,a_i)\), produce a diverse, weighted set:

\[
\mathcal P_i=
\{(z_i^{(k)},w_i^{(k)})\}_{k=1}^{K},
\qquad
\sum_k w_i^{(k)}=1.
\]

The output is a posterior approximation, not a list of unverified LLM stories.

## 2. Stage A: Artifact Decomposition

Split the artifact into addressable units:

```yaml
artifact_unit:
  unit_id:
  span:
  role: definition | claim | method | result | interpretation | limitation
  entities: []
  quantities: []
  cited_sources: []
  direct_evidence: []
  unresolved_support: []
```

The segmentation must preserve claim boundaries. Sentence segmentation alone
is insufficient when one sentence contains several claims with different
support.

## 3. Stage B: Backward Information-Flow Reconstruction

For every artifact unit, ask:

```text
What input or intermediate result is required to justify this unit?
Which operation transforms that input into the unit?
What validator would reject an invalid transformation?
Which dependencies are mandatory, optional, or merely stylistic?
```

This produces an artifact-specific graph:

```text
final claim
  <- interpretation operation
  <- evaluated result
  <- metric and comparison protocol
  <- experiment design
  <- hypothesis or question
  <- source evidence
```

Backward reconstruction identifies required support. It does not assert the
expert's historical chronological order.

## 4. Stage C: Diverse Trace Proposals

Generate candidates using controlled diversity:

```text
proposal family 1:
  minimum sufficient trace

proposal family 2:
  reliability-oriented trace with verification and robustness steps

proposal family 3:
  alternative ordering allowed by partial independence

proposal family 4:
  process-evidence-first trace based on logs, code, or revisions

proposal family 5:
  counter-hypothesis that explains the artifact with different dependencies

proposal family 6:
  self-inferred strategy that is absent from the observed artifact trace but
  is predicted to improve blind execution
```

Use different seeds, prompts, or models when possible. Deduplicate candidates
by typed graph isomorphism or contract-aware edit distance, not textual
similarity alone.

Proposal family 6 is essential for learning rather than imitation. The agent
may invent an operation such as contrastive result selection, mechanism-result
alignment, uncertainty-aware claim budgeting, or a new repair step. Such a
strategy does not need a sentence-level citation to the expert artifact,
because it is a hypothesis about how to act. It must instead declare its
induction provenance, predicted effect, applicable context, failure modes, and
an intervention or replay test.

## 5. Stage D: Hard Validation

Reject a candidate before scoring if it violates:

```text
acyclicity of execution dependencies
missing evidence contract for factual nodes
dangling required outputs
unbound artifact claims
forbidden target leakage
invalid tool preconditions
task or budget impossibility
```

Do not reject a procedural strategy merely because the expert artifact does
not explicitly name it. Evidence contracts are mandatory for factual content
nodes. Self-inferred strategy nodes use an induction-and-validation contract;
their generated factual outputs remain subject to ordinary evidence checks.

## 6. Stage E: Candidate Scoring

For surviving trace \(z\):

\[
\log \widetilde w(z)=
\lambda_A A(z,y)
+\lambda_E E(z,x,a)
+\lambda_C C(z,m)
+\lambda_F F(z;x)
+\lambda_I I(z)
-\lambda_\Omega\Omega(z).
\]

Terms mean:

```text
A:
  artifact coverage and role alignment

E:
  evidence and auxiliary-process consistency

C:
  contextual compatibility

F:
  forward replay performance when executing from source evidence only

I:
  intervention support from deletions, reversals, or substitutions

Omega:
  complexity, redundancy, unsupported specificity, and memorization
```

The coefficients \(\lambda\) are not hand-written domain rules. They are
learned from process-supervised calibration cases, pairwise expert
preferences, or nested validation. If data are insufficient to identify
separate coefficients, use a Pareto set and report the unresolved trade-off
instead of choosing weights that favor the desired result.

Normalize:

\[
w^{(k)}
=
\frac{\exp(\log\widetilde w^{(k)}/T)}
{\sum_j\exp(\log\widetilde w^{(j)}/T)}.
\]

Temperature \(T\) must be calibrated on cases with stronger process evidence.
It must not be selected on the reported test set.

Artifact fit is functional, not lexical. It may score role coverage,
information selection, dependency satisfaction, and communicative utility,
but must not reward reproducing the expert's exact phrases or sentence order.

## 7. Forward Replay

The strongest artifact-only check is prospective replay:

```text
input to executor:
  x_i, m_i, candidate trace z_i^(k)

hidden:
  expert wording y_i

output:
  generated artifact y_hat_i^(k)
```

Only after generation compare \(\hat y_i^{(k)}\) with \(y_i\). Score:

```text
evidence fidelity
coverage of expert argument units
dependency satisfaction
task-role fit
noncopying
cost and failure rate
```

A trace that sounds plausible but cannot guide blind reconstruction should lose
posterior mass.

The expert artifact is one high-quality realization, not the unique output
label. A different abstract may receive higher utility when it is
evidence-faithful, fulfills the same task functions, and communicates more
effectively. Exact-match, n-gram overlap, and a requirement to reproduce the
expert DAG are invalid primary objectives.

## 8. Counterfactual Trace Tests

For node \(v\) and edge \(e\), evaluate:

\[
\Delta_v=J(z)-J(z\setminus v),
\qquad
\Delta_e=J(z)-J(z\setminus e).
\]

Also test:

```text
edge reversal
alternative predecessor
node replacement with a contract-equivalent variant
validator removal
evidence-binding corruption
```

Interpretation:

```text
large stable delta:
  likely operationally important

small delta:
  optional, redundant, or poorly measured

delta only on one artifact:
  case-specific rather than reusable
```

These are functional interventions on an execution strategy, not proof of
human cognitive causality.

## 9. Posterior Record

```yaml
trace_hypothesis:
  trace_id:
  case_id:
  proposal_family:
  nodes: []
  execution_edges: []
  semantic_relations: []
  topological_orders: []
  evidence_bindings: {}
  observability_labels: {}
  hard_checks: {}
  scores:
    artifact_fit:
    evidence_fit:
    context_fit:
    forward_replay:
    intervention_support:
    complexity:
  posterior_weight:
  unresolved_ambiguities: []
```

## 10. EM-Style Training View

An approximate E-step infers:

\[
q_i(z)
\approx
p(z\mid x_i,y_i,m_i,a_i,G_t,\theta_t).
\]

The M-step updates the shared model:

\[
(G_{t+1},\theta_{t+1})
=
\arg\max_{G,\theta}
\sum_i
\mathbb E_{z\sim q_i}
[\log p(y_i,z\mid x_i,m_i,G,\theta)]
-\lambda\Omega(G).
\]

In implementation, the M-step may be a bounded outer-loop graph editor or an
evolutionary population rather than a differentiable optimizer.

## 11. Failure Controls

```text
post-hoc rationalization:
  require evidence binding and blind forward replay

proposal collapse:
  require structural diversity and report posterior entropy

LLM self-confirmation:
  separate proposal, deterministic checks, execution, and acceptance

target leakage:
  hide expert wording during replay and log all inputs

false certainty:
  retain multiple hypotheses and unresolved ambiguity labels

expert imitation:
  exclude lexical matching from the primary reward and accept functionally
  equivalent or superior realizations

blocked self-discovery:
  reserve proposal mass for self-inferred strategies and test them by blind
  replay rather than requiring prior appearance in an expert trace
```
