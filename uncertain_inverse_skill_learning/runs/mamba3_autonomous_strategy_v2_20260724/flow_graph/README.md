# Mamba-3 Primitive Flow Graph

Status: `illustrative_contract_schema_not_a_learned_dag`

This folder records two separate objects. They must not be confused:

```text
primitive_skill_dag.json:
  an illustrative contract schema written after inspection; it is not learned
  from the training traces and cannot be used as evidence for DIRS learning

macro_motif_mapping.json:
  an illustrative expansion of S1-S6; also not a learned graph

learned_connected_dag.json:
  the graph actually aggregated from the 19 directed training paths
```

Only `learned_connected_dag.json` may be called learned, and even that is a
count-based baseline over previously extracted proxy traces rather than ground
truth author cognition. No MCTS claim is valid until path-level visit counts,
Q-values, rollouts, backpropagation traces, and baseline comparisons are saved.
