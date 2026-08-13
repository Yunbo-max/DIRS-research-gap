# Problem Definition And Claim Boundary

## 1. Observed Data

For task family \(\tau\), let:

\[
\mathcal D_\tau=\{(x_i,y_i,m_i,a_i)\}_{i=1}^{N},
\]

where:

```text
x_i:
  source evidence available to a learner, such as a chip, references, tables,
  code, task state, or problem statement

y_i:
  expert final artifact or observed action trace

m_i:
  context, including domain, topic, task role, audience, budget, evidence
  regime, risk level, and output contract

a_i:
  optional auxiliary process evidence, such as code history, experiment logs,
  revisions, reviewer response, tool traces, timestamps, or author annotation
```

The common case is that `y_i` is observed but most of the expert production
process is not.

## 2. Three Trace Objects

The word `expert trace` is ambiguous. DIRS must distinguish:

\[
z_i^{\mathrm{hist}},
\qquad
z_i^{\mathrm{expl}},
\qquad
z_i^{\mathrm{oper}}.
\]

### Historical Trace

```text
z_hist:
  what the expert actually did, including detours, failed experiments,
  revisions, private reasoning, and checks
```

It is recoverable only to the degree that direct process evidence exists.

### Explanatory Trace

```text
z_expl:
  an evidence-consistent information flow sufficient to explain how the
  observed artifact could be produced
```

Multiple explanatory traces can be valid for the same artifact.

### Operational Trace

```text
z_oper:
  an executable strategy that can produce a strong artifact on a new case
  under evidence, budget, and safety constraints
```

DIRS primarily learns `z_oper`, using `z_expl` as latent supervision and
`z_hist` when direct observations are available.

## 3. Claim Boundary

With final artifacts alone, DIRS may claim:

```text
infer artifact-compatible latent skill traces
distill recurrent information dependencies
learn an executable strategy that transfers to held-out tasks
```

It may not claim:

```text
recover the unique true thought process of the expert
identify causal cognitive dependencies from text alone
prove that an inferred ordering was the author's historical ordering
```

Preferred paper language:

> We infer a posterior over artifact-compatible latent skill traces and
> distill recurrent, intervention-tested structure into a reusable execution
> policy.

## 4. Local Trace Representation

A candidate trace is:

\[
z_i=(S_i,\pi_i,b_i,h_i,\kappa_i,\ell_i),
\]

where:

```text
S_i:
  selected typed sub-DAG

pi_i:
  one valid topological execution order, not necessarily the unique order

b_i:
  budget allocation over nodes

h_i:
  evidence and artifact-span bindings

kappa_i:
  contextual activation and edge conditions

ell_i:
  provenance labels indicating observed, strongly implied, weakly inferred,
  or speculative elements
```

## 5. Artifact-Compatible Trace Set

Rather than assuming one trace:

\[
\mathcal Z_i=
\{z:\operatorname{Compatible}(z,x_i,y_i,m_i,a_i)=1\}.
\]

Compatibility requires:

```text
all factual operations have evidence bindings
all artifact claims have plausible upstream support
the trace is connected where information must flow
the execution projection is acyclic
the trace respects task and budget constraints
the trace can account for the artifact without copying it as an instruction
```

DIRS approximates:

\[
q_\phi(z_i\mid x_i,y_i,m_i,a_i),
\]

instead of returning only:

\[
\hat z_i=\operatorname{InverseDAG}(x_i,y_i).
\]

## 6. What Is A Skill?

DIRS first extracts artifact-specific content flow and only then abstracts
reusable skills:

\[
C_i
\xrightarrow{\operatorname{Abstract}}
S_i.
\]

Example:

```text
artifact-specific content:
  compare Method A and Method B on Table 2 under the same dataset and metric

reusable skill:
  perform a matched-setting baseline comparison before attributing improvement
```

The permanent graph stores the reusable skill. Paper-specific entities,
numbers, and claims remain in evidence bindings and case records.

## 7. Common, Simple, And Effective Traces

These are distinct objectives:

```text
common trace:
  recurrent across expert cases

simple trace:
  lowest-complexity explanation of an artifact

effective trace:
  highest expected held-out utility under cost and risk
```

A shortest explanatory trace can omit invisible but essential checks. DIRS
therefore uses common structure as a prior and held-out execution as the final
criterion:

\[
z^*=
\arg\max_z
\mathbb E[R(z,c)]
-\lambda_C C(z)
-\lambda_R \operatorname{Risk}(z).
\]

## 8. Evidence Levels

Every inferred operation receives an observability label:

```text
L0:
  final artifact only

L1:
  artifact plus citations, tables, and appendix

L2:
  plus code, configurations, and released outputs

L3:
  plus experiment logs, version history, failed attempts, and reviews

L4:
  direct action trace, author annotation, or interview
```

Confidence must not be interpreted without its evidence level.

