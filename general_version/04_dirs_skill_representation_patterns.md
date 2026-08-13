# DIRS Skill Representation Patterns

Date: `2026-07-20`

Purpose: record how skill-evolving systems write skills down, and translate
those patterns into the DIRS representation for writing and other tasks.

## Question

When DIRS learns a writing skill, should a node only store content? No. The
node must store both:

```text
content property: what evidence, claim, mechanism, result, or action it carries
style/action property: how that unit should be phrased, positioned, formatted,
or executed
```

The same applies to edges:

```text
content dependency: why one unit depends on another
style/action transition: how the artifact should move from one unit to another
```

This is the main lesson from inspecting skill-evolution systems: successful
systems write skills as structured objects with trigger conditions, execution
rules, validation, and replay/evolution state. They do not rely on one loose
prompt.

## Sources Inspected

```text
Voyager:
  https://github.com/MineDojo/Voyager
  local inspection:
    /tmp/dirs_skill_repo_inspection_20260720/Voyager/voyager/agents/skill.py
    /tmp/dirs_skill_repo_inspection_20260720/Voyager/skill_library/README.md
    /tmp/dirs_skill_repo_inspection_20260720/Voyager/skill_library/trial1/skill/skills.json

EvoSkill:
  https://github.com/sentient-agi/EvoSkill
  local inspection:
    /tmp/dirs_skill_repo_inspection_20260720/EvoSkill/README.md
    /tmp/dirs_skill_repo_inspection_20260720/EvoSkill/docs/architecture.md
    /tmp/dirs_skill_repo_inspection_20260720/EvoSkill/src/loop/runner.py
    /tmp/dirs_skill_repo_inspection_20260720/EvoSkill/src/schemas/skill_proposer.py
    /tmp/dirs_skill_repo_inspection_20260720/EvoSkill/src/agent_profiles/skill_generator/prompt.py

CoEvoSkills:
  https://github.com/Zhang-Henry/CoEvoSkills
  local inspection:
    /tmp/dirs_skill_repo_inspection_20260720/CoEvoSkills/README.md
  note: public repository says full code is coming soon, but describes the
  skill-generator plus surrogate-verifier loop and multi-file skill packages.

SkillRL:
  https://github.com/aiming-lab/SkillRL
  local inspection:
    /tmp/dirs_skill_repo_inspection_20260720/SkillRL/README.md
    /tmp/dirs_skill_repo_inspection_20260720/SkillRL/skill_generation/search.py
    /tmp/dirs_skill_repo_inspection_20260720/SkillRL/memory_data/search/claude_style_skills_search.json
    /tmp/dirs_skill_repo_inspection_20260720/SkillRL/memory_data/alfworld/claude_style_skills.json

Agent0 / Agent0-VL:
  https://github.com/aiming-lab/Agent0
  local inspection:
    /tf/notebooks/icml2026_oral_paper_memory_fresh_24h/repos/Agent0/README.md
    /tf/notebooks/icml2026_oral_paper_memory_fresh_24h/repos/Agent0/Agent0-VL/README.md

Local skill package examples:
  /tf/notebooks/icml2026_oral_paper_memory_fresh_24h/repos/CVE-Factory/skills/cve-test-generator/SKILL.md
  /tf/notebooks/icml2026_oral_paper_memory_fresh_24h/repos/CVE-Factory/skills/cheat-detect/SKILL.md

Local practice-memory evolution templates:
  /tf/notebooks/opsd_research_memory_skills_project_20260611/skills/general-practice-memory/SKILL.md
  /tf/notebooks/opsd_research_memory_skills_project_20260611/skills/general-practice-memory/references/evolution_templates.md
```

## Pattern 1: Skill As A Markdown Package

Observed in local `SKILL.md` examples and EvoSkill-generated skill targets.

```text
skill-name/
  SKILL.md
  references/
  scripts/
  assets/
```

The key properties are:

```text
frontmatter:
  name
  description

body:
  when to use
  when not to use
  ordered workflow
  route table or selection rule
  required output format
  self-check or verifier
```

DIRS implication:

```text
DIRS should keep one human-readable method file, but the real learned skill
should also exist as structured node, edge, verifier, replay, and policy files.
```

## Pattern 2: Skill As A JSON Skill Bank

Observed in SkillRL.

```json
{
  "general_skills": [
    {
      "skill_id": "gen_001",
      "title": "Short name",
      "principle": "Reusable action rule",
      "when_to_apply": "Trigger condition"
    }
  ],
  "task_specific_skills": {
    "task_type": [
      {
        "skill_id": "task_001",
        "title": "Short name",
        "principle": "Task-specific rule",
        "when_to_apply": "Trigger condition"
      }
    ]
  },
  "common_mistakes": [
    {
      "mistake_id": "err_001",
      "description": "Failure mode",
      "why_it_happens": "Cause",
      "how_to_avoid": "Preventive rule"
    }
  ]
}
```

DIRS implication:

```text
global nodes = reusable across a domain or section
typed nodes = only active for compatible paper signatures
mistake nodes = negative compatibility and verifier checks
```

## Pattern 3: Skill As Executable Code Plus Retrieval Metadata

Observed in Voyager.

```text
skill/
  code/*.js
  description/*.txt
  skills.json
  vectordb/
```

Each skill has:

```json
{
  "skill_name": {
    "code": "executable function",
    "description": "natural-language retrieval summary"
  }
}
```

DIRS implication:

```text
node text alone is not enough.
Each node needs a short retrieval summary, evidence bindings, and a
machine-readable representation so MCTS can select it without loading every
long method note.
```

## Pattern 4: Skill Evolution As Propose, Generate, Validate, Keep

Observed in EvoSkill and local practice-memory templates.

```text
1. run current agent/artifact on cases
2. collect failures
3. propose create/edit mutation
4. generate the skill or prompt change
5. validate on held-out cases
6. keep, discard, or mark needs_replay
7. record feedback for later iterations
```

DIRS implication:

```text
training cannot stop after a nice-looking DAG.
It should save mutation proposals, validation results, rejected attempts,
accepted updates, and replay obligations.
```

## Pattern 5: Skill Evolution Needs A Separate Verifier

Observed in CoEvoSkills and Agent0/Agent0-VL.

```text
generator:
  proposes or revises the skill/artifact

verifier:
  checks correctness, feedback, reward, test assertions, or process quality

oracle or held-out check:
  prevents leakage and prevents the generator from grading itself too softly
```

DIRS implication:

```text
DIRS generation and DIRS verification should remain separate roles.
For blind writing tests, the original section can be used only after generation
for comparison, never as a generation input.
```

## Recommended DIRS Package Layout

A complete DIRS skill package should be written as both a readable method and
a machine-readable library:

```text
DIRS/
  README.md
  domain_topics/
    01_domain_topic_paper_splits.md
    02_training_topic_routing.md
    semantic_balanced_23_domains/
  general_version/
    README.md
    01_dirs_general_method.md
    02_dirs_training_cycle.md
    03_dirs_mcts_inference_selector.md
    04_dirs_skill_representation_patterns.md
    05_dirs_top_conference_evaluation_protocol.md
    06_dirs_mathematical_formulation.md
  case1_writing/
    README.md
    01_abstract_writing.md
    02_section_writing.md
  case2_experiment_design/
    README.md
    01_experiment_plan_design.md
    02_ablation_metric_protocol.md
  case3_research_question_proposal/
    README.md
    01_question_generation.md
    02_question_ranking.md
  case4_research_gap_verification/
    README.md
    01_gap_evidence_audit.md
    02_gap_claim_verification.md
```

Historical run artifacts, support scores, harnesses, and scripts are reference
material, not the active method package. Keep them outside the package:

```text
/tf/notebooks/yunbo/DIRS_method_sources_reference_20260720
```

For a trained domain, the run output should be:

```text
domain_skill_library.json
skill_graph.yaml
node_support_scores.md
node_support_scores.json
edge_support_scores.md
edge_support_scores.json
style_profile.md
length_prior.json
mcts_policy.yaml
verifier.md or verifier.py
replay_cases.jsonl
training_trace.jsonl
mutation_candidates.jsonl
accepted_updates.jsonl
rejected_updates.jsonl
```

## DIRS Node Record

```yaml
node_id:
  family: context | gap | object | mechanism | evidence | interpretation | scope | style
  title: short retrieval title
  content_skill: factual, argumentative, computational, or action role
  style_skill: compression, phrasing, placement, format, or execution behavior
  evidence_binding: chip fields or source spans that can support the node
  when_to_apply: trigger condition
  positive_compatibility: signatures where the node is valid
  negative_compatibility: signatures where the node must be rejected
  support_count: number of domain papers or cases using this node
  support_rate: support_count / domain_case_count
  word_budget_role: short | medium | long | optional
  verifier_checks: checks that prove the node was used correctly
  common_failures: ways this node is often misused
```

## DIRS Edge Record

```yaml
edge_id:
  from: upstream_node
  to: downstream_node
  content_dependency: why the downstream unit depends on the upstream unit
  style_transition: how the prose or action should bridge the two units
  support_count: number of domain papers or cases using this transition
  support_rate: support_count / domain_case_count
  required_when: chip or section condition that makes the edge mandatory
  forbidden_when: condition that makes the transition misleading
  no_jump_failure: what goes wrong if this edge is skipped
  verifier_checks: checks that the edge direction and transition are respected
```

## DIRS Writing-Specific Style State

Node-level style is necessary but not sufficient. The system also needs a
whole-section style profile:

```yaml
style_profile:
  target_word_count_distribution: section/domain prior
  paragraph_count_distribution: section/domain prior
  sentence_density: compact | moderate | expansive
  result_position: early | middle | near_end
  metric_density: none | one_anchor | multiple
  contribution_rhythm: single_named_object | list_of_components | theorem_result_pair
  transition_patterns: common edge phrasing
  forbidden_style_moves:
    - opening with unsupported hype
    - reporting a result before the metric exists
    - adding a number not present in the chip
    - copying original abstract phrasing during blind generation
```

This answers the earlier length problem: if the original abstract is hidden,
target length should come from domain and section priors, not from the original
text.

## DIRS Inference Rule

At test time, DIRS should not load every learned skill. It should select a
connected sub-DAG:

```text
chip facts
  -> paper signature
  -> compatible node candidates
  -> connected MCTS path
  -> budget allocation
  -> draft
  -> verifier
  -> repair or accept
```

The selected path must satisfy:

```text
no unsupported content node
no forbidden-domain node
no disconnected node island
no result before method/metric
no interpretation before evidence
no broad final claim outside evidence
```

## Practical Upgrade To DIRS

The current DIRS folder already has the right high-level pieces. The next
implementation upgrade is to make every training or inference run emit these
files automatically:

```text
skill_graph.yaml
node_library.json
edge_library.json
mcts_policy.yaml
verifier_result.json
replay_cases.jsonl
accepted_updates.jsonl
```

That would make DIRS closer to the strongest observed systems:

```text
SKILL.md package discipline
+ SkillRL-style triggerable JSON skills
+ Voyager-style retrieval summaries
+ EvoSkill-style mutation frontier and validation
+ CoEvoSkills/Agent0-style separated verifier/reward role
+ DIRS-specific connected DAG and content/style edge constraints
```
