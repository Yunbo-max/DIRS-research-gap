# DIRS Case 1: Writing

Date: `2026-07-20`

Purpose: task-specific protocol for applying DIRS to scientific writing.

This folder is Case 1 of the DIRS downstream-task adapters. The general DIRS
method defines how to learn and search over reusable skill graphs; this case
defines how those graphs are used when the artifact is a paper abstract,
introduction, method section, experiment section, result section, or
related-work discussion.

## Current Targets

```text
01_abstract_writing.md
02_section_writing.md
```

## Writing Specialization

```text
node = content_skill + style_skill
edge = content_dependency + style_transition
global style = section-level length/order/tone profile
MCTS = connected sub-DAG selection for the target chip
```

For writing, the content system and style system must be trained separately and
used jointly:

```text
content system:
  paper chip, source facts, tables, claims, methods, results, limitations

style system:
  section role, rhetorical order, paragraph budget, length prior, claim strength
```

This prevents a common failure mode: selecting a node because it is stylistically
common even when the target chip does not support it.

## Default Heavy Runtime

For Case 1 writing, heavy DIRS runs should use Codex subagents, not hosted API
calls, for the editor/simulator/evaluator roles.

```text
main Codex coordinator:
  owns files, run state, blind rules, convergence, and batch progress

editor subagent:
  selects or repairs the connected writing DAG

simulator subagent:
  writes only from chip facts and the selected DAG

evaluator subagent:
  compares generated text to training target/chip facts and returns feedback
```

The old Python API runner is a diagnostic fallback only. It should not be used
for normal DIRS writing experiments when subagents are available.

## Research-Use Contract

DIRS writing runs should be reported as blind generation whenever possible.

```text
allowed before generation:
  chip facts
  domain skill library
  node and edge support rates
  section style and length priors

not allowed before generation:
  original target section text
  exact original sentence order
  post-hoc comparison notes
```

The original section can be revealed only after generation, and only for
diagnosis, scoring, and repair proposals.

## Expected Output

Every writing run should leave an auditable trail:

```text
paper signature
selected domain prior
selected connected sub-DAG
rejected nodes and why they were rejected
budget plan
generated text
verifier report
post-generation comparison, if original text is available
shortage map and repair moves
```

## Quality Standard

For an ICLR-style paper about DIRS itself, the writing target should demonstrate
more than attractive prose. It should show that the system:

```text
1. preserves source-supported facts,
2. chooses different DAG paths for different paper types,
3. follows edge order without rhetorical jumps,
4. controls length without seeing the original target section,
5. improves after rejection sampling or MCTS,
6. generalizes to held-out papers and domains.
```

The useful claim is not that DIRS writes the same abstract as the author. The
useful claim is that DIRS can infer reusable section structure, select a
compatible path from source facts, and produce a bounded scientific section that
matches expert rhetorical function under blind conditions.
