# Autonomous Strategy Discovery

## 1. Correction

DIRS must not confuse two different requirements:

```text
factual grounding:
  output claims must be supported by the task's available evidence

strategy grounding:
  a proposed way of selecting, ordering, compressing, checking, or presenting
  information must earn support through successful execution
```

Requiring every strategy node to be directly visible in an expert artifact
would reduce inverse skill learning to imitation. The expert product is one
demonstration of high utility, not a unique trace, unique DAG, or unique
wording target.

## 2. What The Agent May Invent

The agent may autonomously propose:

```text
new open-vocabulary strategy nodes
new conditional dependencies or alternative branches
new information-selection and compression rules
new claim-budgeting or uncertainty-handling operations
new verification and repair operations
new context gates and stopping rules
```

It may derive these proposals from:

```text
self-critique of its own outputs
failure clusters
counterfactual deletions or substitutions
uncertain graph regions
differences among successful traces
expected information gain
novel recombinations of contract-compatible skills
```

## 3. What It May Not Invent

Autonomous learning is not permission to fabricate content. The agent may not
invent:

```text
paper results, measurements, datasets, settings, or citations
causal claims unsupported by the available evidence
hidden expert wording presented as observed evidence
transfer claims based on the motivating case alone
```

A strategy can be novel while every factual realization remains traceable.

## 4. Candidate Contract

```yaml
self_inferred_strategy:
  proposal_id:
  operation_contract:
    preconditions: []
    inputs: []
    transformation:
    outputs: []
    rejection_rule:
  proposal_origin:
    type: self_critique | failure_induced | uncertainty | recombination
    observations: []
  predicted_effects: []
  applicable_context:
  possible_failures: []
  content_evidence_required_at_execution: []
  local_tests: []
  transfer_tests: []
  status: candidate | local_strategy_hypothesis | reusable_skill | rejected
```

## 5. Discovery And Selection Loop

For each outer round:

1. Sample several existing trace or graph hypotheses.
2. Execute them blindly from source evidence.
3. Let the agent inspect its own outputs, verifier decompositions, and
   counterfactuals, but not hidden target wording.
4. Reserve nonzero proposal mass for new strategies and novel
   contract-compatible combinations.
5. Generate candidate edits with predicted effects and falsification tests.
6. Freeze each candidate and compare it with the incumbent under matched
   budgets and seeds.
7. Retain a candidate locally if it improves task utility without factual or
   hard-failure regression.
8. Promote it to a reusable skill only after improvement on separate cases.

MCTS can explore which candidate strategy to apply within a frozen graph. ES
or an outer graph editor can propose persistent recombinations. A conditional
GFlowNet can learn a multimodal proposal distribution when enough accepted and
rejected graph trajectories exist.

## 6. Acceptance Levels

### Candidate

The operation is valid, explicit, and testable. No claim of usefulness is made.

### Local Strategy Hypothesis

On the motivating case:

```text
hard failures remain zero
all output facts remain supported
blind functional utility improves
the improvement survives a direct counterfactual comparison
```

This is the strongest conclusion available from a single paper.

### Reusable Skill

In addition:

```text
paired improvement appears on held-out papers or topics
the lower confidence bound exceeds the calibrated acceptance margin
replay and calibration do not materially regress
the effect is not explained by copying or target leakage
complexity remains within budget
```

## 7. Functional Rather Than Exact Matching

Several abstracts can be equally good while using different:

```text
sentence boundaries
claim order
surface terminology
numeric compression
method/result grouping
argument paths through the DAG
```

Evaluation should therefore prioritize:

```text
factual support
problem-method-result function coverage
dependency coherence
importance-weighted information selection
bounded claims
clarity and compression
blind pairwise preference
transfer and robustness
```

Lexical overlap with an expert abstract may be reported as a diagnostic, but
it must not control graph learning or stopping.

## 8. Evaluator Information Boundary

To allow genuine self-discovery, feedback should describe outcome failures,
not prescribe the hidden expert's solution:

```text
allowed:
  "the main result is not sufficiently comparative"
  "the mechanism and validation are weakly connected"
  decomposed scores, factual failures, uncertainty, and pairwise preference

not allowed:
  copied expert phrases
  the exact missing sentence
  an instruction to reproduce the expert's DAG
```

The agent, not the evaluator, diagnoses which strategy edit to propose.

## 9. Stop Rule

Do not require one exact DAG or one exact abstract. Stop an optimization stage
only when:

```text
hard failures are controlled
held-out functional utility has plateaued under uncertainty
posterior or Pareto-front movement is small
no novel candidate passes the acceptance test
a final challenge round with reserved exploration budget also fails to improve
```

Alternative high-utility graphs should retain posterior mass when available.

## 10. Scientific Claim Boundary

A one-paper run can show autonomous local strategy discovery. It cannot show
that the new node is a general skill. Reusability requires separate papers,
topics, writers, and evaluators with preregistered transfer tests.
