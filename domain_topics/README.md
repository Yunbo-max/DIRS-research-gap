# DIRS Domain Topics

Date: `2026-07-20`

Purpose: canonical domain splits and chip-routing files used by DIRS training
and inference.

This folder defines the source population for learning reusable section skills.
DIRS should not choose training papers from memory. It should route through this
folder, select the relevant semantic domain, and then read the chip paths listed
inside that domain file.

## Canonical Split

Use the semantic-balanced 23-domain split as the default source:

```text
semantic_balanced_23_domains/INDEX.md
```

It contains:

```text
502 oral papers
23 semantic domains
one assignment per local paper chip
```

The split was copied from:

```text
/tf/notebooks/yunbo/mcts_abstract_learning_source_semantic_balanced_20260720
```

## Routing Files

```text
01_domain_topic_paper_splits.md:
  compact overview of the topic split and why this split is preferred

02_training_topic_routing.md:
  operational routing table from user request to domain file and chip paths

semantic_balanced_23_domains/:
  full domain files with paper titles, conference buckets, and chip paths
```

## How DIRS Uses The Split

The topic split is used before both training and inference:

```text
training:
  domain request -> domain file -> paper chips -> node and edge support scores

inference:
  new chip -> paper signature -> nearest domain -> node/style support prior
```

When asked to train on a domain, first read `02_training_topic_routing.md`, then
read the chosen domain file under `semantic_balanced_23_domains/`. Each domain
file contains paper titles and exact chip paths.

## Leakage Rule

For held-out evaluation, the target paper must be excluded from the domain
training split before computing support rates, style priors, length priors, or
MCTS rollout preferences.

```text
allowed:
  train on neighboring papers in the same domain

not allowed:
  train on the target paper's original section and then call the generation blind
```

## Why This Split Is Preferred

The semantic-balanced split is the strongest current choice for DIRS because it
has enough papers per domain to estimate stable node and style frequencies while
keeping domains meaningful:

```text
too broad:
  generic cross-field priors blur mechanism and evidence styles

too narrow:
  tiny clusters overfit to one paper's rhetoric

semantic-balanced:
  enough local similarity for writing style, enough diversity for generalization
```
