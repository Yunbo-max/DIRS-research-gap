# DAG-Flow-Constrained MCTS

## 1. Correction

DIRS must not select a bag of strategy labels and connect them afterward.
Node selection is a typed information-flow process:

```text
wrong:
  sample {S1, S4, S6} and invent arrows between them

correct:
  freeze a primitive skill graph
  begin at source evidence and task context
  expose only dependency-satisfied frontier nodes
  expand a connected partial sub-DAG
  stop only after a verified artifact is reachable
```

High-level strategies such as result-first selection or boundary-first
calibration are graph motifs. They expand into primitive operations and
constraints; they are not necessarily atomic nodes.

## 2. Three Different Objects

```text
primitive skill node:
  one typed operation with explicit inputs, outputs, preconditions, and
  rejection rules

macro strategy motif:
  a reusable partial-order pattern over several primitive nodes

task sub-DAG:
  one connected, context-valid execution graph selected from a frozen graph
```

Confusing these objects creates arbitrary combinations that have names but no
executable flow.

## 3. Frozen Graph Before Search

At the beginning of one MCTS episode, sample or choose:

\[
G^{(k)} \sim q_\omega(G\mid c,\mathcal D).
\]

Freeze:

```text
node contracts
typed edges
AND/OR dependency groups
context gates
evidence bindings
executor and verifier versions
search budget
```

MCTS may select a sub-DAG inside \(G^{(k)}\). It may not invent persistent
nodes or edges during the episode.

## 4. Search State

\[
s_t=(V_t,E_t,B_t,A_t,h_t,c),
\]

where:

```text
V_t, E_t:
  selected connected partial sub-DAG

B_t:
  bound typed artifacts produced so far

A_t:
  currently active context gates and dependency alternatives

h_t:
  execution results, verifier observations, and remaining information budget

c:
  task, audience, length, evidence regime, and deployment context
```

## 5. Valid Frontier

A node \(v\) enters the frontier only when:

```text
every required input type has a bound producer
at least one alternative in each OR dependency group is satisfied
all mandatory members of each active AND group are satisfied
its context gate is true or still unresolved
adding it preserves acyclicity
it can still reach a required terminal output
its cost fits the remaining budget
```

Formally:

\[
\mathcal F(s_t)=
\{v\notin V_t:
\operatorname{DepsSatisfied}(v,B_t,A_t)
\land
\operatorname{Acyclic}(V_t\cup\{v\})
\land
\operatorname{ReachableToGoal}(v)
\}.
\]

MCTS actions are limited to:

```text
ADD_FRONTIER_NODE(v)
CHOOSE_OR_BRANCH(group, option)
ACTIVATE_CONTEXT_GATE(g)
ALLOCATE_BUDGET(node, amount)
EXECUTE_READY_NODE(v)
STOP
```

`ADD_ARBITRARY_STRATEGY` is not an action.

## 6. Typed Edge Meaning

An edge must transport or constrain something:

```yaml
edge:
  source:
  target:
  relation: SUPPLIES | VALIDATES | SCOPES | COMPARES | GATES | REPAIRS
  output_port:
  input_port:
  condition:
  required: true | false
```

Rhetorical adjacency alone is metadata unless it changes an input contract,
selection decision, or realization constraint.

## 7. Macro Expansion

Before MCTS, expand an activated macro into primitive constraints:

```text
operating-point decision motif:
  aggregate-quality comparison
  + matched latency comparison
  -> operating-role map
  -> bounded synthesis

boundary-first motif:
  limitation evidence
  -> scope boundary
  -> claim-strength gate
  -> bounded synthesis
```

Two macros may compose only if their expanded primitive graphs have compatible
ports, noncontradictory gates, and a valid join. Macro-name co-occurrence is
insufficient.

## 8. Selection, Execution, And Backpropagation

For each simulation:

1. Traverse using PUCT among valid frontier actions only.
2. Expand one valid action.
3. Execute ready nodes or complete the partial graph using a rollout policy.
4. Generate the artifact only from bound evidence and selected operations.
5. Score decomposed functional utility and hard failures.
6. Backpropagate reward along the actual selected action path.

\[
a^*=\arg\max_{a\in\mathcal F(s)}
\left[
Q(s,a)+
cP(a\mid s,c)\frac{\sqrt{N(s)}}{1+N(s,a)}
\beta U_{\mathrm{epi}}(s,a)
\right].
\]

The prior, value, and trade-offs must be learned or validation-selected. The
frontier constraint is hard because it enforces executability, not domain
preference.

## 9. Stop Legality

`STOP` is valid only when:

```text
all required artifact functions are reachable and realized
every selected factual claim has evidence
all selected nodes lie on a source-to-output path
the verifier has executed
no hard failure remains
```

An unfinished or disconnected collection of good nodes is not a candidate.

## 10. Required MCTS Evidence

A run may be called `DIRS-MCTS` only if it saves:

```text
frozen graph identifier and hash
root state
valid frontier at every expanded state
selected actions
visit counts N(s,a)
Q values and decomposed rollout rewards
rollout artifacts
hard-failure records
backpropagation trace
terminal selected sub-DAG
comparison with greedy and random-valid baselines
```

Without these artifacts, a set of generated DAGs is proposal sampling or an
ablation, not MCTS.

## 11. Interaction With GFlowNet

A conditional GFlowNet may sample complete valid graph snapshots or macro
expansions. It must use validity-preserving construction actions. After root
sampling one graph snapshot, MCTS selects and executes a sub-DAG inside that
frozen snapshot.

```text
GFlowNet/posterior:
  uncertainty over valid graph snapshots

MCTS:
  frontier-constrained task-time execution inside one snapshot

outer loop:
  persistent graph learning and edits
```

None of these components should be replaced by random macro-name combination.

## 12. Local Uniformity Is Not Uniform Over Complete Paths

In a branching prefix tree, assigning equal probability to each local action
does not generally assign equal probability to each terminal path. A branch
with fewer later alternatives can receive more terminal mass than a branch
with many descendants.

If the intended prior is uniform over valid complete paths, compute local
action mass from descendant terminal support:

\[
P(a\mid s)
=
\frac{|\Pi(s\mathbin{\|}a)|}
{\sum_{a'\in\mathcal F(s)}|\Pi(s\mathbin{\|}a')|},
\]

where \(\Pi(s\mathbin{\|}a)\) is the set of valid complete paths continuing
through action \(a\). For weighted path priors, replace the count by total
descendant posterior mass.

The implementation must state whether its prior is:

```text
uniform over local legal actions
uniform over valid complete paths
empirical training trace frequency
context-conditioned learned path probability
posterior sample from graph or policy uncertainty
```

These priors are not interchangeable.

## 13. Multiple Rollouts Per Terminal Path

One cached artifact per path confounds path quality with one writer draw.
Online or replay-bank MCTS should index terminal executions:

\[
(\pi,j,\theta_w,\theta_e,R_{\pi,j}),
\]

and save the generated artifact, writer/evaluator versions, decomposed reward,
and factual failure record for every \(j\). Reusing the same cached scalar on
every visit is acceptable for a deterministic search audit, but not as
evidence that MCTS handled execution uncertainty.

## 14. When Not To Use MCTS

If all complete legal paths are enumerable, only terminal reward is observed,
and path execution costs are comparable, the selection problem is a finite
best-arm identification problem. Sequential Halving, Successive Rejects, or a
calibrated Top-Two Thompson policy may allocate samples more efficiently than
tree search.

Using one of these policies does not weaken DAG constraints:

```text
candidate arms:
  complete connected paths validated against the frozen DAG

forbidden:
  arbitrary node subsets, new edges, cyclic paths, or unsatisfied contracts
```

MCTS should earn its complexity through an implicit path space, useful
intermediate values, reusable prefix computation, unequal costs, or dynamic
frontier revelation.

## 15. Mean Backup Does Not Optimize The Best Descendant

Mean reward backup estimates the value of the current rollout distribution
below a prefix. It can reject a low-average branch that contains a rare
high-utility path. Raw maximum backup is not a general correction because
noise and unequal visit counts bias observed maxima.

Before using MCTS, report validation evidence that prefix statistics predict
the desired terminal objective. When the goal is best-path identification,
consider a calibrated posterior for best-descendant utility or probability of
containing the optimum, with explicit branch-coverage safeguards.
