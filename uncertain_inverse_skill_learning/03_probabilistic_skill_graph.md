# Probabilistic Shared Skill Graph

## 1. Two Related Graphs

A general skill system needs more relations than an execution DAG can contain.
Define:

\[
K=(V,E_{\mathrm{sem}})
\]

as a typed semantic skill graph, and:

\[
G_c=(V_c,E_{\mathrm{exec},c})
\]

as a context-specific executable DAG.

Semantic relations may include:

```text
SIMILAR_TO
SPECIALIZES
ALTERNATIVE_TO
CONTRADICTS
COMPOSES_WITH
REPAIRS
VALIDATES
```

Execution relations include:

```text
REQUIRES
PRODUCES_FOR
CONDITIONALLY_REQUIRES
```

Only the execution projection must be acyclic. Calling the entire structure
only a knowledge graph would hide the operational semantics; calling every
relation a DAG edge would incorrectly forbid useful symmetric or cyclic
semantic relations.

## 2. Canonical Skill Contract

Each reusable node is:

\[
v=(P,I,A,O,V,F,C).
\]

```yaml
skill_node:
  stable_id:
  title:
  preconditions: []
  input_schema: {}
  operation:
  output_schema: {}
  validators: []
  failure_modes: []
  cost_model: {}
  evidence_contract: {}
  provenance_cases: []
  aliases: []
```

Node canonicalization must compare contracts. Textual or embedding similarity
is only a candidate generator.

## 3. Context

Let:

\[
c=(d,t,r,e,b,o,q),
\]

where:

```text
d: domain or topic
t: task family
r: artifact role
e: evidence regime
b: budget and resources
o: output contract
q: risk or quality requirement
```

DIRS learns:

\[
p(v\in S\mid c,\mathcal D),
\qquad
p(u\rightarrow v\mid c,\mathcal D).
\]

An edge can be strongly supported for empirical papers and inactive for
theoretical papers. A global scalar support rate cannot represent this.

## 4. Separate Uncertainty Variables

For node \(v\):

\[
\rho_v^{\mathrm{exist}}
=p(v\in V\mid\mathcal D)
\]

describes global membership.

\[
\rho_v^{\mathrm{select}}(c)
=p(v\in S\mid c,G,\mathcal D)
\]

describes local relevance.

\[
\rho_v^{\mathrm{execute}}(c,\psi)
=p(\operatorname{success}\mid v,c,\psi)
\]

describes executor reliability.

For edge \(e=(u,v)\):

\[
\rho_e^{\mathrm{exist}},
\quad
\rho_e^{\mathrm{active}}(c),
\quad
\rho_e^{\mathrm{causal\_utility}}
\]

must also be separated.

## 5. Posterior Aggregation Across Cases

Given trace posterior \(q_i(z)\), use soft counts:

\[
\widetilde n_v
=
\sum_i\Pr_{z\sim q_i}[v\in z],
\]

\[
\widetilde n_e
=
\sum_i\Pr_{z\sim q_i}[e\in z].
\]

With a Beta prior:

\[
\rho_v
=
\frac{\alpha_v+\widetilde n_v}
{\alpha_v+\beta_v+N}.
\]

This improves on hard counts from one selected inverse DAG, but still assumes
exchangeable cases. Context-conditioned models should replace the scalar when
enough data exist:

\[
\rho_v(c)=\sigma(f_v(c)),
\qquad
\rho_e(c)=\sigma(f_e(c)).
\]

The prior parameters \((\alpha_v,\beta_v)\), contextual functions, and
activation thresholds are estimated by empirical Bayes or validation
calibration. They are not fixed by node names or a manually authored list of
"required" skills.

## 6. Dependency Evidence

An execution edge should not be added merely because two units occur in order.
Record:

```yaml
execution_edge:
  from:
  to:
  relation: REQUIRES | PRODUCES_FOR | CONDITIONALLY_REQUIRES
  condition:
  information_transferred:
  downstream_precondition_satisfied:
  supporting_cases: []
  contradicting_cases: []
  deletion_effect:
  reversal_effect:
  posterior:
```

Evidence strength:

```text
weak:
  textual order or co-occurrence

medium:
  repeated order plus explicit information transfer

strong:
  deletion/reversal causes replay failure across cases
```

## 7. Alternative Paths And Hyperdependencies

Some operations require several predecessors:

```text
dataset definition AND metric definition AND baseline configuration
  -> matched result comparison
```

A collection of independent binary edges can falsely suggest that any one
predecessor is sufficient. Represent an AND-precondition explicitly:

```yaml
dependency_group:
  group_id:
  type: all_of | any_of | k_of_n
  members: []
  target:
  condition:
```

The execution engine compiles the active dependency groups into a task-specific
DAG.

## 8. Graph Posterior

The learned object is:

\[
q_\omega(G\mid\mathcal D),
\]

not only one adjacency matrix. Useful summaries include:

```text
MAP graph
posterior node inclusion
posterior edge inclusion
context-conditioned inclusion
credible alternative subgraphs
posterior entropy
calibration error
```

The deployed accepted graph can remain versioned and deterministic while
training retains posterior alternatives.

The initial node schema is open-vocabulary. A human may provide only the
generic contract fields; candidate node identities and aliases come from
artifact decomposition, process evidence, counterexamples, and failed
execution. A fixed domain node inventory is an ablation, not the full method.

## 9. Hard Invariants

Every executable sample must satisfy:

```text
stable unique identifiers
acyclic execution projection
valid input/output contracts
no dangling required outputs
reachable terminal output
evidence contract for factual operations
declared tool permissions and failure behavior
no held-out target exposure
bounded complexity and budget
```

Semantic relations are validated separately and do not automatically authorize
execution.

## 10. Storage Layout

```text
graph_version/
  manifest.yaml
  semantic_skill_graph.yaml
  execution_graph_map.yaml
  node_posteriors.json
  edge_posteriors.json
  context_gates.json
  dependency_groups.yaml
  provenance.jsonl
  alternative_graphs/
  validators/
  replay_cases.jsonl
  accepted_diff.json
  rollback_diff.json
```
