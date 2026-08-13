# Research Question Generation

Date: `2026-07-20`

Purpose: generate candidate research questions from a domain, paper set, chip
set, or open-ended idea.

## Core DAG Path

```text
domain topic
  -> recent pattern
  -> unresolved uncertainty
  -> missing condition
  -> mechanism hypothesis
  -> target object
  -> research question
  -> possible evidence
  -> bounded scope
```

## Node Properties

```yaml
T1_domain_topic:
  content_skill: define the specific research area or subdomain
  action_skill: avoid starting from a vague field name

P1_recent_pattern:
  content_skill: summarize what recent work commonly does or assumes
  action_skill: use a compact field-level pattern, not a paper-by-paper list

U1_unresolved_uncertainty:
  content_skill: identify what remains unknown, unstable, or under-explained
  action_skill: state uncertainty as a question pressure

C1_missing_condition:
  content_skill: name a setting, data regime, user behavior, failure mode, or constraint missing from prior work
  action_skill: make the condition concrete enough to test

M1_mechanism_hypothesis:
  content_skill: propose why the missing condition might matter
  action_skill: phrase as a mechanism, not only as a benchmark gap

O1_target_object:
  content_skill: specify the model, system, dataset, task, or theory object to study
  action_skill: keep the object narrow enough for one project

Q1_research_question:
  content_skill: formulate a direct question connecting uncertainty, condition, and object
  action_skill: make the question answerable by an experiment or analysis

E1_possible_evidence:
  content_skill: suggest what evidence would answer the question
  action_skill: connect the question to measurements, data, or proofs

S1_bounded_scope:
  content_skill: define what the question excludes
  action_skill: prevent the question from becoming a whole field
```

## Question Templates

```text
When [missing condition] holds, does [method/system] still [desired behavior],
and which [mechanism] explains the change?

Can [new measurement or intervention] distinguish [mechanism A] from
[mechanism B] in [specific task/domain]?

What fails first when [assumption from prior work] is relaxed in
[concrete setting]?
```

## Verifier

```text
grounded in at least one observed field pattern
states an unresolved uncertainty
contains a concrete missing condition
names an object of study
implies a feasible test
bounded enough for a paper or workshop project
```
