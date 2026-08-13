# Long-Goal Status

Status as of `2026-07-21T23:30:35Z`:

- Supervisor PID: `658514`
- Supervisor alive: `false`
- Runtime target: `86400` seconds
- Runtime mode: `local_deterministic_no_api`
- Current state: ended after `537` completed iterations
- Elapsed runtime: about `87013` seconds
- Remaining target runtime: about `0` seconds
- Snapshot directories: `537`
- Domain: `LLM Inference / Systems / Token Efficiency`
- Held-out paper: `ICML2026_71057_echo_elastic_speculative_decoding`
- Training examples: `28`
- Blind rule: do not open `holdout_private_after_generation.json`

## First Iteration Result

- Iteration: `1`
- Seed: `20260721`
- Completed loops: `24`
- Converged: `true`
- Converged at loop: `24`
- Final mean replay score: `0.975294`
- Final minimum replay score: `0.958234`
- Target abstract length: `164` words
- Target band: `128` to `198` words
- Minimum loops required: `24`
- MCTS rollouts per example: `5000`
- Stable window: `10`

## Latest Completed Iteration

- Iteration: `537`
- Seed: `20261257`
- Completed loops: `24`
- Converged: `true`
- Converged at loop: `24`
- Final mean replay score: `0.975294`
- Final minimum replay score: `0.958234`
- Minimum loops required: `24`
- MCTS rollouts per example: `5000`
- Stable window: `10`

## Active Iteration

- No active iteration was observed at refresh time.

## Two-Loop Process

Loop 1 updates and audits the learned abstract DAG prior from the 28 training abstracts.

Loop 2 runs connected-subgraph selection and replay evaluation through the MCTS-style harness, then records whether the graph is stable across the required window.

The long supervisor repeats these two loops with fresh seeds and snapshots every pass under `longgoal_iterations/`.

## Public Audit Notes

The cleaned split keeps the ECHO holdout out of `training_trace.json`, with no public `abstract_text` for the holdout. The remaining extraction issues are minor scars in a few training abstracts rather than severe figure, introduction, or arXiv spillover.

## 2026-07-22 Full-Paper Execution-Aware DAG Correction

The abstract-only long goal is not sufficient for the systems/token-efficiency research-gap DAG. A corrected full-paper DIRS simulation was run over 20 papers with introduction, related work, method, experiments, results, limitations, appendix, and artifact evidence.

Corrected run directory:

`/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_fullpaper_train20_section_gap_20260722`

Execution-aware artifacts:

- `DIRS_FULLPAPER_SIMULATION_REPORT.md`
- `dirs_final_dag.json`
- `real_execution_probe.json`
- `gpu_microbenchmark.json`
- `run_real_execution_probe.py`
- `run_fullpaper_dirs_simulation.py`

The revised selected DAG now includes explicit real-experiment nodes:

- `experiments.execution_surface`
- `experiments.gpu_hardware_profile`
- `experiments.api_or_local_backend`
- `experiments.real_benchmark_command`
- `results.runtime_measurement`
- `results.reproduction_status`
- `appendix.artifact_log`

Local execution evidence was collected on an RTX 4090. CUDA was available with 4 visible GPUs, GPU 2 was selected for the synthetic decode/KV attention microbenchmark, and no paid/external API call was invoked.

## 2026-07-22 Strict Author-Style GPU Campaign

The correction was expanded beyond a small probe. The full-paper DAG now includes a strict NeurIPS-style execution loop: exact-rerun feasibility audit, author-style GPU reproduction campaign, measured-gap derivation, proxy-vs-exact reproduction boundary, and raw measurement table.

Campaign artifacts:

- `AUTHOR_STYLE_GPU_REPRODUCTION_CAMPAIGN.md`
- `author_style_gpu_reproduction_campaign.json`

Campaign scale:

- Papers audited for exact rerun readiness: `20`
- GPU measurement rows: `2130`
- Repeated seeds: `30`
- Experiment families: `kv_cache_locality`, `token_merging`, `speculative_decoding_proxy`, `sampling_truncation`, `quantization_compression`, `sparse_kernel_efficiency`
- Inferred measured gaps: `6`

The rerun DAG converged with `35` selected nodes and `44` selected edges. Newly selected strict nodes include `experiments.exact_rerun_feasibility_audit`, `experiments.author_style_reproduction_campaign`, `results.measured_gap_derivation`, `results.proxy_vs_exact_boundary`, and `appendix.raw_measurement_table`.

## 2026-07-22 Loop 2 Author Experiment Simulation

Loop 2 is author-side, not reviewer-side: it simulates forming hypotheses, running or reading measurements, aggregating repeated-seed statistics, making author decisions, revising claims, and writing conclusions.

Loop 2 artifacts:

- `AUTHOR_LOOP2_EXPERIMENT_SIMULATION_REPORT.md`
- `author_loop2_final_dag.json`
- `author_loop2_mcts_trace.jsonl`
- `run_loop2_author_experiment_simulation.py`

Loop 2 converged with `23` selected nodes and `32` selected edges. The selected path is:

`root.author_experiment_loop` -> gap hypotheses -> repo/GPU experiments -> raw measurement read -> repeated-seed statistics -> author decisions -> claim revision -> central conclusion -> paper sections -> appendix artifacts.

Author decisions:

- Reject exact-reproduction wording because `16/20` papers were not exact-rerun-ready locally.
- Accept quality-guard gap because KV/window, token-merge, and sampling controls show quality/probability-mass losses.
- Accept acceptance-limited speculative-decoding gap because effective speedup can collapse when draft acceptance drops.
- Accept hardware-specificity gap because masked sparse and quant-dequant proxies do not automatically produce GPU speedups.
- Revise conclusions to bounded systems claims rather than universal speedup claims.

## 2026-07-22 Loop2 -> Loop1 -> Loop2 20-Paper Author Cycle

The long goal was relaunched as the full author-side cycle requested here:

`Loop 2 per paper` -> `Loop 1 learned prior` -> `final Loop 2 convergence`

Cycle artifacts:

- `LOOP1_LOOP2_20PAPER_AUTHOR_CYCLE_REPORT.md`
- `loop1_loop2_20paper_author_cycle.json`
- `loop2_20paper_author_traces/`
- `run_loop1_loop2_20paper_author_cycle.py`

Convergence result:

- Per-paper Loop 2 converged for `20/20` papers.
- Loop 1 learned `17` core author-DAG nodes and `7` selective domain nodes.
- Final Loop 2 converged in `24` loops with `24` nodes and `34` edges.
- Final trace stabilized under policy `deterministic_full_domain_author_dag_after_loop1`.

The final DAG keeps the whole systems/token-efficiency author protocol explicit:

`root.author_experiment_loop` -> four gap hypotheses -> repo exact-rerun audit plus GPU campaign baselines -> six stress-test families (`kv_cache`, `token_merge`, `speculative_acceptance`, `sampling_mass_entropy`, `quantization_kernel`, `sparse_kernel`) -> raw measurements -> repeated-seed statistics -> author decisions -> claim revision -> section plan -> appendix/raw artifacts.

Final author-cycle decisions:

- Reproducibility-gap decision accepted for `20/20` papers.
- Quality-guard decision accepted for `20/20` papers.
- Speculative acceptance-limit decision accepted for `10/20` papers.
- Hardware-specificity decision accepted for `18/20` papers.
- Final claims must be bounded because exact rerun readiness and measured tradeoff failures dominate the evidence.
