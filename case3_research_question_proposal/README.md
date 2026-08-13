# DIRS Case 3: Research Question Proposal

Date: `2026-07-20`

Purpose: task-specific protocol for applying DIRS to propose research
questions.

This case learns the human skill of turning a topic area, paper cluster, chip
set, or user curiosity into a small set of sharp research questions. The output
should be novel enough to matter, grounded enough to be credible, and testable
enough to become an experiment.

## Current Targets

```text
01_question_generation.md
02_question_ranking.md
```

## Research-Question Specialization

```text
node = field observation + unresolved uncertainty + testable question unit
edge = reasoning path from known work to proposed question
simulator = literature/chip scan plus candidate-question rollout
verifier = novelty, grounding, testability, scope, and value
```

For research-question proposal, the two DIRS systems are:

```text
evidence system:
  topic cluster, recent papers, chips, benchmarks, known limitations, negative
  results, unexplained phenomena

action system:
  question framing, mechanism targeting, scope control, comparison axis,
  feasibility reasoning, ranking and selection
```

## Auto-Research Skills To Learn

```text
question decomposition
topic trend extraction
prior-art contrast
under-tested mechanism detection
benchmark or data gap spotting
failure-mode transformation into a question
question narrowing
novelty and feasibility ranking
```

## Expected Output

```text
domain summary
known pattern
unresolved uncertainty
candidate research questions
evidence supporting each question
possible experiment for each question
novelty risk
feasibility risk
ranked recommendation
```

## Quality Standard

A good DIRS question is not just interesting. It must connect to evidence,
identify what is unknown, and imply a possible test. If the question cannot be
tested or falsified, the verifier should mark it as a weak idea rather than a
research question.
