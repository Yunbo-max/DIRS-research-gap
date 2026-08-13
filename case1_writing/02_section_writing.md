# Section Writing Application

Date: `2026-07-20`

Purpose: apply the trained dual-system MCTS method to rewrite or draft full
paper sections, not only abstracts.

## Inputs

Required:

```text
target paper chip or local advisory
source PDF/text, if available
target section list
domain writing prior
section role prior
```

Optional:

```text
external citations for related work or positioning
original section text for post-generation comparison
```

## Section-Level Pipeline

```text
1. Convert local paper source into a chip.
2. Map source paragraphs/tables/results to section facts.
3. Select the domain style prior.
4. Build a section-specific typed DAG.
5. Run MCTS for each section.
6. Generate draft text from the selected path.
7. Verify each section against source coverage and section role.
8. Produce a shortage map.
9. Repair missing facts, missing citations, weak claims, or bad ordering.
10. Repeat until section DAGs and shortage map are stable.
```

## Section Roles

```text
abstract:
  compress full argument into context, gap, object, mechanism, result, takeaway

introduction:
  expand motivation, problem importance, gap, contribution, and evidence preview

related_work:
  position against closest methods and explain why the new problem is different

method:
  define objects, notation, algorithm, architecture, and implementation choices

experimental_setup:
  define datasets, splits, metrics, baselines, protocol, seeds, and leakage checks

results:
  report main comparisons, numeric anchors, tradeoffs, and safe interpretation

ablation_analysis:
  test whether the claimed mechanism explains the result

limitations:
  state what is not proven, what remains fragile, and what evidence is missing
```

## Section-Specific Verifier

Each section needs a different reward:

```yaml
abstract:
  reward: argument completeness, length fit, no overclaim

introduction:
  reward: reader motivation, problem gap, contribution clarity

related_work:
  reward: closest baselines, non-strawman distinctions, citation need

method:
  reward: reproducible mechanism, notation, no missing algorithmic step

experimental_setup:
  reward: protocol details, metrics, datasets, baselines, no leakage

results:
  reward: table-grounded comparisons, numeric anchors, scoped claims

ablation_analysis:
  reward: mechanism-test alignment, not just extra numbers

limitations:
  reward: honest scope, threat model, missing experiment list
```

## Shortage Map

For each section, report:

```yaml
source_words:
rewrite_words:
score:
missing_facts:
missing_edges:
unsupported_claims:
style_mismatch:
remaining_shortage:
repair_move:
```

## Long-Run Defaults

Use:

```text
100 outer loops max
5000 MCTS rollouts per section
early stop when selected DAG and shortage map are stable
```

For quick exploratory runs:

```text
10-20 outer loops
1000-2000 MCTS rollouts per section
```

## Good Historical Example

The best current section-level reference is:

```text
/tf/notebooks/yunbo/DIRS_method_sources_reference_20260720/temporal_hetero_section_dual_system_longrun_README_20260712.md
```

That run demonstrates:

```text
content system: PDF/advisory -> local paper chip
style system: 14-paper learned priors -> connected section DAG
optimizer: 100 loops x 8 sections
inner_search: 5000 MCTS rollouts per section
verifier: coverage, role, metric/protocol anchors, no-leakage guard
```
