# Existing `general_version` File Audit

Date reviewed: `2026-07-24`

All seven files in `../general_version` were read before creating this
extension. The original files remain unchanged.

## `README.md`

Existing contribution:

```text
defines the reading order and a concise DIRS method sentence
```

Gap:

```text
says DIRS infers a typed dependency graph, but does not state that multiple
latent traces and graphs can explain one artifact
```

Extension:

```text
the core learned object is a posterior over artifact-compatible traces and
context-conditioned graph structure
```

## `01_dirs_general_method.md`

Existing contribution:

```text
defines expert artifacts, typed nodes and edges, content/style/tool streams,
training and inference objectives, and output contracts
```

Gaps:

```text
G=(V,E,R) is effectively a point graph
support is not separated into existence, selection, and execution reliability
semantic KG relations and executable DAG relations are not separated
```

Extension:

```text
introduces semantic graph K, executable conditional DAG G_c, canonical skill
contracts, graph posterior q(G|D), and three uncertainty types
```

## `02_dirs_training_cycle.md`

Existing contribution:

```text
defines per-example DAGs, a persistent shared graph, batch versus online
updates, dual LLM roles, fixed-snapshot MCTS, schemas, replay, and convergence
```

Critical gaps:

```text
uses S_i^expert = InverseDAG(x_i,y_i) as a single inferred expert graph
uses max over z_i, encouraging early commitment
uses hard node/edge support counts
does not distinguish historical, explanatory, and operational traces
does not explicitly separate trace-inference, representation, selection,
execution, evaluation, and data errors
```

Extension:

```text
replaces one inverse DAG with q_i(z)
uses soft posterior aggregation or EM-style learning
adds evidence levels, diverse proposals, replay weighting, typed error
authorization, posterior versioning, and calibration
```

## `03_dirs_mcts_inference_selector.md`

Existing contribution:

```text
correctly states that MCTS selects from a fixed graph and does not modify the
persistent graph; defines search state, actions, UCT, rollout, and blind use
```

Gaps:

```text
uses a deterministic graph and fixed hand-weighted scores
does not propagate graph posterior uncertainty
does not distinguish exploratory uncertainty bonuses from deployment risk
does not address action explosion in a large cross-paper graph
```

Extension:

```text
adds root/Thompson posterior sampling, context gates, uncertainty-aware PUCT,
risk-sensitive deployment, dependency-group actions, and progressive widening
```

## `04_dirs_skill_representation_patterns.md`

Existing contribution:

```text
defines structured skill packages, JSON skill banks, executable skills,
validation loops, node/edge records, and trained-run artifacts
```

Gaps:

```text
canonicalization is not defined by input/output/precondition contracts
support records are point estimates
semantic relationships and execution dependencies share one conceptual graph
```

Extension:

```text
defines node contract (P,I,A,O,V,F,C), semantic relation types, execution
dependency types, posterior fields, provenance, contextual gates, and
alternative graphs
```

## `05_dirs_top_conference_evaluation_protocol.md`

Existing contribution:

```text
defines blind splits, baselines, writing metrics, human evaluation, leakage
controls, ablations, and reporting
```

Gaps:

```text
does not evaluate trace posterior coverage, graph calibration,
non-identifiability, or causal/functional edge interventions
does not compare K=1 against multi-hypothesis trace learning
```

Extension:

```text
adds partial-gold process cases, credible-set coverage, Brier/ECE/NLL metrics,
intervention tests, ambiguity reports, posterior ablations, and operational
equivalence classes
```

## `06_dirs_mathematical_formulation.md`

Existing contribution:

```text
provides the most complete original formalization of z_i, G, support,
simulation, proposer-critic repair, inference, MCTS, and blind generation
```

Critical mathematical gaps:

```text
z_i* is a single argmax
G* uses sum_i max_z rather than marginalizing trace uncertainty
S_i^expert is named as if it were observed or uniquely identifiable
Beta smoothing is applied to hard inferred memberships
there is no posterior over graphs or contextual edge activation
```

Extension:

\[
q_i(z)
\approx
p(z\mid x_i,y_i,m_i,a_i,G,\theta),
\]

\[
q_\omega(G\mid\mathcal D),
\]

\[
\sum_i\log\sum_z
p(y_i\mid x_i,z)p(z\mid G,m_i)
\]

replace premature point estimates. The new formulation also assigns posterior
inference, persistent structure optimization, and task-time planning to
separate algorithms.

## Compatibility Decision

The new directory is an extension, not a silent rewrite:

```text
general_version:
  retains the original task-independent DIRS specification

uncertain_inverse_skill_learning:
  supplies the probabilistic latent-trace and graph-posterior formulation
  needed for a stronger research claim
```

Future integration should update the original files only after the terminology,
algorithm choice, and first empirical implementation are fixed.

