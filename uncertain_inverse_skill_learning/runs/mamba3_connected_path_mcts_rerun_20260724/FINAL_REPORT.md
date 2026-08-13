# Mamba-3 Connected-Path DIRS-MCTS Rerun

Date: `2026-07-24`

Status: `completed_strict_connected_path_cached_rollout_mcts`

## Result

The rerun selected `path_01`, a complete connected path observed in 6 of 19
training traces:

```text
R1_abstract_as_argument
→ G1_problem_gap
→ O1_named_method_or_object
→ M1_architecture_or_mechanism
→ M2_efficiency_or_theory_detail
→ E1_evaluation_setup
→ E2_result_outcome
→ E3_quantitative_anchor
→ I1_interpretation_or_tradeoff
→ S1_bounded_takeaway
→ P1_length_and_placement_prior
```

No arbitrary node addition or macro-name composition was permitted.

## Strict Pipeline

1. Aggregate 19 training proxy traces into one directed acyclic graph.
2. Retain the six complete paths actually observed in training.
3. Exclude two union-graph paths that were connected but never observed.
4. Use one fresh writer, blind to the expert abstract and all prior Mamba-3
   generations, to produce one equal-budget rollout per observed path.
5. Deterministically anonymize and shuffle the six rollouts.
6. Use one fresh evaluator, blind to path identity, frequency, expert text,
   and the private mapping, to assign evidence checks and holistic rewards.
7. Run MCTS on the prefix tree; every action is a legal next node that
   continues at least one observed path.

## Blind Rewards

| Path | Rank | Reward | Training support |
|---|---:|---:|---:|
| path_01 | 1 | 0.965 | 6/19 |
| path_03 | 2 | 0.952 | 3/19 |
| path_02 | 3 | 0.942 | 4/19 |
| path_05 | 4 | 0.935 | 2/19 |
| path_04 | 5 | 0.889 | 3/19 |
| path_06 | 6 | 0.879 | 1/19 |

All six drafts had zero evaluator hard failures.

## MCTS Search

Configuration:

```text
state:
  full selected prefix

action:
  legal next node conditioned on that prefix

prior:
  empirical conditional continuation frequency from training

reward:
  anonymous evaluator holistic preference

simulations:
  192

c_puct:
  2.0
```

Terminal visits:

| Path | Visits |
|---|---:|
| path_01 | 70 |
| path_02 | 39 |
| path_03 | 32 |
| path_04 | 24 |
| path_05 | 19 |
| path_06 | 8 |

The MCTS recommendation matched the exhaustive best path. Every selected
action appeared in its logged valid frontier, and every terminal was one of
the six observed complete paths.

## Baselines

```text
exhaustive best:
  path_01, reward 0.965

frequency-greedy:
  path_01, reward 0.965

uniform random-valid:
  expected reward 0.927
  one-draw probability of selecting the best path 1/6
```

MCTS did not outperform frequency-greedy on this case because the most common
training path was also the blind evaluator's best path. The positive result is
search correctness, not an MCTS advantage.

## Why The Selected Path Won

The evaluator judged its anonymous realization strongest because it:

```text
quantified both quality and latency
explained the roles of the main mechanisms
made the MIMO-quality versus SISO-speed tradeoff explicit
specified training scale and comparison families
retained retrieval, scale, and hardware boundaries
```

## Claim Boundary

This is a real prefix-constrained MCTS over cached LLM rollout rewards, but it
remains a small pilot:

```text
fixed node vocabulary
proxy rather than historical author traces
six observed paths
one rollout per path
one blind LLM evaluator
cached rather than online fresh rollouts at every MCTS visit
one held-out paper
no demonstrated advantage over frequency-greedy
no cross-paper transfer or GFlowNet training
```
