# Prior-Smoothing Diagnostic Report

Date: `2026-07-25`

Status: `completed_post_hoc_diagnostic`

## Result

Softening the empirical-frequency prior changed exploration, but no fixed
mixing value was uniformly reliable. At budget 192:

| η | Epistemic σ | MCTS visit hit | MCTS Q hit | Even-allocation hit |
|---:|---:|---:|---:|---:|
| 0.00 | 0.02 | 55.0% | 90.3% | 88.3% |
| 0.25 | 0.02 | 55.7% | 91.0% | 88.3% |
| 0.50 | 0.02 | 76.0% | 92.7% | 88.3% |
| 1.00 | 0.02 | 33.0% | 86.3% | 88.3% |
| 0.00 | 0.05 | 43.7% | 86.7% | 86.3% |
| 0.25 | 0.05 | 53.7% | 85.3% | 86.3% |
| 0.50 | 0.05 | 64.3% | 84.7% | 86.3% |
| 1.00 | 0.05 | 35.3% | 87.0% | 86.3% |

The intermediate `η=0.5` improved the most-visited rule relative to the raw
frequency prior. Fully uniform local actions did not create a uniform terminal
path prior because different branches contain different numbers and depths of
descendant paths.

## Decision

`η=0.5` is not accepted as a permanent parameter because this was a post-hoc
diagnostic on synthetic Mamba-3-centred utilities. The method should instead
learn or validation-select a context-conditioned prior, with explicit
normalization over descendant path mass.

All four result files passed the connected-prefix trace validator with zero
illegal actions or invalid terminals.
