# DIRS Domain Topic Paper Splits

Date: `2026-07-20`

Purpose: define the domain source used by the DIRS writing-skill learner.

## Best Current Split

Use:

```text
/tf/notebooks/yunbo/DIRS/domain_topics/semantic_balanced_23_domains
```

Main index:

```text
/tf/notebooks/yunbo/DIRS/domain_topics/semantic_balanced_23_domains/INDEX.md
```

Provenance source:

```text
/tf/notebooks/yunbo/mcts_abstract_learning_source_semantic_balanced_20260720
```

Original clustering source:

```text
/tf/notebooks/yunbo/oral_paper_topic_learning_clusters_final_20260708/INDEX.md
```

This split has:

```text
502 total oral papers
23 semantic learning domains
CVPR: 141
ICML: 137
ICLR: 224
domain size range: 11-30 papers
```

## Why This Split

The final semantic-balanced split is the best source for MCTS writing skills
because:

```text
domain labels are semantic, not arbitrary Part 1 / Part 2 labels
each cluster is large enough to estimate style and node frequency
clusters are not so large that domain-specific rhetoric gets washed out
every oral-paper chip is assigned exactly once
```

## Historical Splits

Earlier folders are useful for audit but not the default source:

```text
/tf/notebooks/experiments/artifact_to_dag_inverse_skill/topic_cluster_skill_learning_20260708/all_oral_paper_learning_clusters
  target size about 20, but many artificial Part labels and tiny leftovers

/tf/notebooks/experiments/artifact_to_dag_inverse_skill/topic_cluster_skill_learning_20260708/all_oral_paper_learning_clusters_balanced
  32 clusters, roughly 10-30 papers, still uses some Part labels

/tf/notebooks/experiments/artifact_to_dag_inverse_skill/topic_cluster_skill_learning_20260708/all_oral_paper_learning_clusters_semantic
  30 semantic clusters, but some clusters are very small

/tf/notebooks/experiments/artifact_to_dag_inverse_skill/topic_cluster_skill_learning_20260708/all_oral_paper_learning_clusters_semantic_balanced
  final experiment version, copied into DIRS as the canonical source
```

## Domain Selection Rule

For a new paper chip:

```text
1. infer paper domain from title, abstract-like facts, method object, evidence type
2. match it to one of the 23 semantic domain files
3. use that domain as the main style/node prior
4. smooth with global priors from all 502 papers
5. allow domain-new nodes only when the chip strongly supports them
```

Do not select a domain only by one keyword. Use the whole paper signature:

```yaml
paper_type: benchmark | method | theory | system | data | analysis
domain_type: task or research area
method_object: named method, benchmark, dataset, model, or framework
mechanism_type: key technical mechanisms
evidence_type: experiments, theorems, human study, benchmark table, ablation
result_type: what the paper proves or shows
forbidden_types: things not present in the chip
```

## Output Of This Layer

The domain layer should produce:

```yaml
domain_file: selected semantic-balanced cluster file
domain_paper_count: number of papers in that file
global_source_count: 502
style_prior_source: selected domain + global smoothing
node_support_source: selected domain + all-paper prior
held_out_target_removed: true_or_false
```
