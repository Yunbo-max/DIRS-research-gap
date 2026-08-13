# Evaluation And Identifiability

## 1. Questions To Answer

DIRS evaluation must separately test:

```text
Q1:
  Does the trace posterior contain evidence-consistent explanations?

Q2:
  Are posterior probabilities calibrated?

Q3:
  Does cross-case aggregation recover reusable skills rather than content?

Q4:
  Do learned dependencies matter under intervention?

Q5:
  Does posterior-aware sub-DAG selection improve held-out execution?

Q6:
  Does the dual-loop update rule prevent misattributed graph changes?

Q7:
  Can the learner invent useful strategies not explicitly recoverable from an
  expert artifact?

Q8:
  Are outputs judged by functional quality rather than imitation of expert
  wording or one reference DAG?
```

## 2. Partial Gold Process Dataset

Construct a calibration subset with increasing observability:

```text
artifact only
artifact + appendix and code
artifact + logs and revisions
direct trace or author annotation
```

Hide the richer evidence during inference, then evaluate whether posterior mass
increases around traces compatible with the revealed process evidence.

Do not require exact historical sequence when only partial orders are known.

## 3. Trace Metrics

```text
artifact coverage:
  supported artifact units reachable from the trace

evidence precision:
  selected factual operations with valid evidence

partial-order precision/recall:
  agreement with observed process constraints

posterior coverage:
  whether a compatible trace appears in a high-posterior credible set

diversity:
  contract-aware graph edit distance among high-weight traces

forward replay:
  blind reconstruction quality and failure rate

functional equivalence:
  role coverage and task utility without requiring lexical or structural match

self-discovery yield:
  accepted self-inferred proposals divided by valid tested proposals

strategy transfer:
  paired utility of self-inferred nodes on cases that did not motivate them

complexity:
  nodes, edges, dependencies, and execution cost
```

Top-1 trace accuracy alone would reward premature posterior collapse.
Reference-string similarity alone would reward imitation rather than skill
learning and must not be a primary success metric.

## 4. Graph Metrics

```text
node inclusion calibration
edge inclusion calibration
context-gate calibration
expected calibration error
Brier score
negative log likelihood where gold labels exist
posterior entropy
credible-set coverage
graph edit distance to partial gold
```

For observational artifacts, report equivalence or uncertainty rather than
forcing one directed edge label.

## 5. Intervention Tests

For high-confidence nodes and edges:

```text
delete node
delete edge
reverse edge
corrupt evidence binding
remove validator
replace with contract-equivalent alternative
```

Measure paired changes in:

```text
hard-failure rate
evidence fidelity
output quality
cost
sub-DAG stability
```

A dependency claim is stronger when deletion or reversal causes predictable,
repeatable failures across held-out cases.

## 6. Held-Out Execution Protocol

For every test case:

```text
freeze x_test and m_test
hide y_test
freeze graph/posterior/executor/verifier versions
sample or choose graph snapshot
run MCTS and save selected sub-DAG
generate and save output
only then reveal y_test
evaluate and archive all inputs
```

## 7. Baselines

```text
direct generation from x
retrieved expert examples
flat skill list
single LLM-inferred trace
MAP graph without uncertainty
all-nodes graph
greedy sub-DAG selection
random valid sub-DAG
MCTS without posterior sampling
posterior sampling without MCTS
ES graph optimization without inverse traces
DIRS without intervention-tested edges
DIRS without typed error attribution
```

Closest-system comparisons should include task-appropriate implementations of:

```text
automatic workflow optimization
task-graph learning from demonstrations
persistent skill-library maintenance
trajectory-mined skill/tool graphs
```

## 8. Essential Ablations

```text
K=1 versus K>1 trace hypotheses
hard counts versus soft posterior counts
global support versus context-conditioned gates
semantic graph only versus executable dependencies
point-estimate graph versus graph posterior
MCTS versus greedy and ES at task time
ES/bounded editing versus uncontrolled LLM graph rewriting
with versus without replay and rollback
with versus without error-type gating
self-inferred strategy proposals versus expert-trace-only proposals
functional/preference reward versus lexical-reference reward
with versus without a final novel-strategy challenge pass
```

## 9. Non-Identifiability Reporting

For each ambiguous case, report:

```yaml
ambiguity:
  indistinguishable_trace_ids: []
  shared_required_structure: []
  disputed_nodes: []
  disputed_edges: []
  evidence_needed_to_resolve: []
  operationally_equivalent: true | false
```

If two traces produce equivalent behavior under all available interventions,
DIRS may retain them as an operational equivalence class.

## 10. Success Criteria

The method succeeds without recovering historical thought exactly if:

```text
high-posterior traces are evidence-consistent
credible sets cover richer observed process evidence
shared skills transfer across papers and topics
high-confidence dependencies survive intervention tests
posterior-aware MCTS improves held-out utility or risk
accepted graph updates improve paired validation without replay regression
self-inferred strategies improve blind held-out execution without factual
support regression
multiple surface-distinct outputs can receive equivalent high utility
```

## 11. Defensible Claims

Strong but supportable:

> DIRS learns a posterior over artifact-compatible latent skill traces and
> distills recurrent, context-conditioned dependencies into a reusable
> execution graph.

> Separating persistent graph optimization from fixed-snapshot posterior-aware
> MCTS improves attribution, auditability, and held-out execution.

Avoid:

```text
the recovered DAG is the expert's true mental process
edge confidence proves cognitive causality
MCTS itself learns the permanent graph
one learned graph is correct for every task context
```
