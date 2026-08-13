# Research Question Ranking

Date: `2026-07-20`

Purpose: choose which generated research questions are worth pursuing.

## Core DAG Path

```text
candidate question
  -> evidence grounding
  -> novelty pressure
  -> feasibility
  -> expected insight
  -> experiment path
  -> risk assessment
  -> ranking decision
```

## Ranking Criteria

```yaml
G_grounding:
  content_skill: identify the papers, chips, observations, or failures that motivate the question
  action_skill: reject questions that sound good but have no evidence base

N_novelty:
  content_skill: estimate whether the exact question has already been answered
  action_skill: distinguish new mechanism, new setting, and new phrasing

F_feasibility:
  content_skill: assess whether the question can be tested with available data, tools, and time
  action_skill: penalize questions that require a new field-scale infrastructure

I_expected_insight:
  content_skill: identify what would be learned if the question is answered
  action_skill: favor questions whose answer would change method choice or theory

E_experiment_path:
  content_skill: sketch one experiment, analysis, proof, or benchmark that could answer the question
  action_skill: require a concrete first test

R_risk:
  content_skill: name novelty, data, implementation, evaluation, and interpretation risks
  action_skill: prefer bounded risk over hidden risk

D_decision:
  content_skill: rank, merge, split, or reject the question
  action_skill: give a reasoned recommendation, not only a score
```

## Scoring Rubric

```text
0-2 evidence grounding
0-2 novelty
0-2 feasibility
0-2 expected insight
0-2 experiment path clarity
0-2 scope control
```

Recommended actions:

```text
10-12:
  pursue now

7-9:
  keep, but narrow or verify novelty

4-6:
  merge with a stronger question or use as a diagnostic subquestion

0-3:
  reject
```

## Common Failures

```text
question is only a topic
question is already answered by a known paper
question cannot be tested
question is too broad for one project
question lacks a mechanism
question is interesting but not important
```
