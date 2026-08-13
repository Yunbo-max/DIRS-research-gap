# Technical References And Method Boundaries

This is a focused method map, not a claim that DIRS is identical to any listed
work.

## Learning Task Structure From Demonstrations

### Neural Task Graphs

```text
Huang et al., Neural Task Graphs: Generalizing to Unseen Tasks From a Single
Video Demonstration, CVPR 2019.
```

https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Neural_Task_Graphs_Generalizing_to_Unseen_Tasks_From_a_Single_CVPR_2019_paper.html

Borrow:

```text
explicit compositional task graph as an intermediate representation
```

Difference:

```text
DIRS learns persistent cross-artifact skills, posterior uncertainty, and
verifier-governed graph revisions
```

### Differentiable Task Graph Learning

```text
Seminara, Farinella, and Furnari, Differentiable Task Graph Learning,
NeurIPS 2024.
```

https://proceedings.neurips.cc/paper_files/paper/2024/hash/6d19163eaec3b0f0accbe462a0139466-Abstract-Conference.html

Borrow:

```text
direct optimization of task-graph dependencies from demonstration sequences
```

Difference:

```text
DIRS has latent artifact-compatible traces, LLM/tool execution, persistent
skills, and dual-loop updates
```

### CircuitHTN

```text
Chen et al., Learning Hierarchical Task Networks with Preferences from
Unannotated Demonstrations, CoRL 2020.
```

https://corlconf.github.io/corl2020/paper_351/

Borrow:

```text
learning task structure and execution preference from unannotated
demonstrations
```

## Inverse Specification Learning

```text
Vazquez-Chanlatte et al., Learning Task Specifications from Demonstrations,
NeurIPS 2018.
```

https://people.eecs.berkeley.edu/~sseshia/pubs/b2hd-vazquez-neurips18.html

Borrow:

```text
maximum-entropy demonstration likelihood and compositional specifications
```

Boundary:

```text
DIRS must acknowledge reward/trace ambiguity rather than call one inferred
trace the true expert behavior
```

## Bayesian DAG Structure Learning

### DAGs With NO TEARS

https://proceedings.neurips.cc/paper/2018/hash/e347c51419ffb23ca3fd5050202f9c3d-Abstract.html

Borrow:

```text
exact smooth acyclicity characterization when continuous adjacency
optimization is appropriate
```

### BayesDAG

https://proceedings.neurips.cc/paper_files/paper/2023/hash/05cf28e3d3c9a179d789c55270fe6f72-Abstract-Conference.html

Borrow:

```text
posterior inference over DAGs rather than only a point graph
```

### ProDAG

https://proceedings.neurips.cc/paper_files/paper/2025/hash/ee42c13f231836e914930925f950fc62-Abstract-Conference.html

Borrow:

```text
uncertainty quantification with distributions supported directly on sparse
acyclic graphs
```

Boundary:

```text
skill dependencies are operational contracts, not automatically causal
variables in a structural equation model
```

## GFlowNets For Graph And Latent-Structure Posteriors

### DAG-GFlowNet

```text
Deleu et al., Bayesian Structure Learning with Generative Flow Networks,
UAI 2022.
```

https://proceedings.mlr.press/v180/deleu22a.html

Borrow:

```text
sequential construction and posterior sampling of multiple valid DAGs
```

### GFlowNet-EM

```text
Hu et al., GFlowNet-EM for Learning Compositional Latent Variable Models,
ICML 2023.
```

https://proceedings.mlr.press/v202/hu23c.html

Borrow:

```text
posterior sampling for combinatorial latent structures within EM-style
learning
```

## Uncertain Knowledge Graphs

```text
Chen et al., Embedding Uncertain Knowledge Graphs, 2018.
```

https://arxiv.org/abs/1811.10667

Borrow:

```text
explicit confidence on uncertain relations and reasoning about unseen links
```

Boundary:

```text
KG relation confidence does not establish executable precedence; DIRS requires
preconditions, information transfer, and intervention/replay evidence
```

## LLM Workflow And Skill Optimization

### AFlow

https://arxiv.org/abs/2410.10762

Overlap:

```text
MCTS, code-represented workflows, execution feedback, and iterative refinement
```

Difference:

```text
DIRS uses MCTS only on a frozen skill-graph snapshot and reserves persistent
graph edits for an outer loop
```

### SkillGraph

https://arxiv.org/abs/2604.19793

Overlap:

```text
directed reusable execution-transition graph mined from successful agent
trajectories
```

Difference:

```text
DIRS infers latent artifact-compatible traces, maintains graph uncertainty,
and validates typed structural updates
```

### SkillOps

https://arxiv.org/abs/2605.13716

Overlap:

```text
typed skill contracts, graph-structured library, validators, failure modes,
task-time and library-time loops, and maintenance actions
```

Difference:

```text
DIRS focuses on inverse learning from expert artifacts, posterior latent
traces, and fixed-snapshot MCTS sub-DAG selection
```

### Anything2Skill

https://arxiv.org/abs/2606.09316

Overlap:

```text
evidence-grounded procedural skill compilation, persistent SkillBank,
lifecycle and versioned updates
```

Difference:

```text
DIRS explicitly learns uncertain dependency structure and an inverse utility
or preference model, then searches executable sub-DAGs
```

### SkillAdaptor

https://arxiv.org/abs/2606.01311

Overlap:

```text
step-level failure attribution, targeted skill updates, and acceptance checks
```

Difference:

```text
DIRS attributes errors across trace inference, representation, MCTS selection,
execution, evaluation, and data, and governs persistent graph structure
```

