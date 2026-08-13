# Long Goal Status: One-Paper DAG-Only Blind Simulation

Date: 2026-07-22

Target paper: `ICLR2026_ItFuNJQGH4_p_less_sampling`

Title: `p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding`

## Goal

Learn a detailed author-experiment DAG from one paper, then test whether a blind simulator can recover the paper-shaped experimental conclusions from the DAG alone.

## Isolation Contract

- DAG builder/verifier may read paper evidence and oracle anchors.
- Blind simulator may read only `paper_author_dag.json`.
- Blind simulator must not read paper text, evidence tables, oracle results, prior reports, campaign JSON, or previous memory artifacts.
- Verifier compares blind simulation output against hidden paper results after the simulation finishes.

## Run Result

- Converged: `true`
- Iterations: `6`
- Final verifier score: `1.0`
- Final DAG nodes: `16`
- Final DAG edges: `20`
- Final DAG signature: `de103d8765817f7e`
- Simulator paper/oracle access: `false`

## Final Blind Predictions

- `bounded_candidate_set`: `true`
- `hyperparameter_free`: `true`
- `reasoning_auc_shape`: `p_less_or_p_lessnorm_top_or_near_top`
- `high_temperature_writing_shape`: `p_less_stable_high_temperature`
- `efficiency_shape`: `p_less_fastest_or_tied_fastest`
- `exact_reproduction_boundary`: `not_claimed_exact_full_table_reproduction`

## Verification Outcome

The verifier confirmed that the blind simulation matched the paper-result shape:

- p-less/p-lessnorm top-or-near-top reasoning AUC shape.
- p-less stable high-temperature writing shape.
- p-less bounded and hyperparameter-free sampler properties.
- p-less efficiency shape via sort-free/default-free operation model.
- Exact full paper-table reproduction correctly marked blocked because the full evaluation pipeline is not locally available.

## Artifacts

- `ONEPAPER_DAG_BLIND_SIM_VERIFY_REPORT.md`
- `onepaper_dag_blind_sim_verify_summary.json`
- `paper_author_dag.json`
- `paper_oracle_results.json`
- `blind_simulator_from_dag_only.py`
- `iterations/`
- `run_onepaper_dag_blind_sim_verify_longgoal.py`

## Notes

This run is intentionally one-paper deep rather than twenty-paper broad. The simulator does not learn from paper memory directly; all reusable knowledge must be encoded in the DAG.

## 2026-07-22 Correction: GPU Evidence-Channel Verification Must Update DAG on Mismatch

The previous DAG-only blind run was too symbolic because it did not run GPU measurements inside the blind simulator and did not compare all paper evidence channels. A stricter GPU-backed update loop was added.

Corrected artifacts:

- `ONEPAPER_DAG_BLIND_GPU_UPDATE_LOOP_REPORT.md`
- `onepaper_dag_blind_gpu_update_loop_summary.json`
- `run_onepaper_dag_blind_gpu_update_loop.py`
- `run_onepaper_dag_blind_gpu_table_verify.py`
- `gpu_table_dag_update_loop/`

Correction policy:

- The blind simulator still reads only `paper_author_dag.json`.
- The blind simulator now runs CUDA/PyTorch proxy experiments and emits result-like tables.
- The verifier compares the simulation against hidden paper evidence from tables, paragraph claims/values, figures/captions, and appendix artifacts.
- `partial`, `fail`, and `blocked` verifier states do not count as convergence.
- Mismatches become DAG update requests.

Corrected result:

- Final status: `blocked_waiting_for_exact_artifacts_after_dag_update`
- GPU simulator iterations: `2`
- Total GPU simulator runtime: `20.952` seconds
- Iteration 1 verifier score: `0.954545`
- Iteration 2 verifier score: `0.954545`

Evidence channels checked by the verifier:

- Tables: Table 1, Table 2, Table 3, Table 15.
- Paragraphs: gap/motivation, method mechanism, high-temperature result discussion, efficiency explanation.
- Figures: Figure 2, Figure 15, Figures 16 and 17.
- Appendix artifacts: Appendix C.11 CPU/RAM profiling and the sampler code snippet.

What matched:

- Paper Table 1 result shape: p-less/p-lessnorm top-or-near-top reasoning AUC shape.
- Paper Figure 2 accuracy-vs-temperature curve shape.
- Paper Table 2 result shape: p-less stable high-temperature writing behavior.
- Paper paragraph claims for gap, method mechanism, high-temperature stability, and efficiency rationale.
- Paper Figure 15 sampler-code logic: threshold, mask, renormalize, multinomial.
- Bounded candidate-set property.

What did not fully match:

- Paper Table 3 exact timing shape was only `partial`: the GPU sampler-only proxy ranked `epsilon, p_less, min_p, mirostat`, while the paper reports p-less fastest in the full Mistral generation timing setup.
- Paper Figures 16 and 17 plus Appendix Table 15 CPU/RAM profiling are `blocked` because the blind run did not measure CPU-time and RAM during generation.
- Exact numeric table, paragraph, and figure reproduction remains `blocked` because the blind GPU run does not include Llama2/Mistral/Llama3 generation, benchmark prompts, raw generations, scoring scripts, seeds, or the full evaluation pipeline.

Required DAG updates emitted:

- `update.require_full_generation_timing_pipeline`
- `update.require_cpu_ram_profile_figures16_17_table15`
- `update.require_exact_table_reproduction_artifacts`

The loop therefore did not stop as success. It stopped as a stable blocker after DAG update, with success criteria requiring full generation timing, CPU/RAM profiling, and exact paper evidence-channel reproduction artifacts.

## 2026-07-22 Correction: Operational DAG-Only Reproduction Loop

The sampler-proxy Loop 2 was rejected as insufficient. The corrected run makes the DAG operational: the blind agent sees only `paper_author_operational_dag.json`, then follows DAG nodes to clone/download code, check dependencies, resolve models and datasets, write a concrete generation harness, attempt real model generation, and package evaluation artifacts.

Operational result:

- Final status: `blocked_waiting_for_operational_artifacts_after_dag_update`
- Iterations: `2`
- Total executor runtime: `140.154` seconds
- Proxy sampler convergence: `disallowed`
- GPU used for real target run: physical GPU `3` (`CUDA_VISIBLE_DEVICES=3`; the harness saw it as logical `cuda:0`).
- Reduced real target artifact: `mistralai/Mistral-7B-Instruct-v0.2` on one GSM8K test prompt, temperature `1.0`, samplers `p_less`, `p_lessnorm`, `top_p`, and `min_p`.
- Reduced artifact files per iteration: `4` raw generations, `64` per-token timing rows, `64` CPU/RAM rows, and `run_manifest.json`.
- Exact paper claim from reduced target run: `false`; it is a real paper-model/paper-data artifact, but not the full paper grid.

The verifier compares against hidden evidence channels:

- Tables: Table 1, Table 2, Table 3, Table 15.
- Paragraphs: gap/motivation, method mechanism, high-temperature discussion, efficiency explanation.
- Figures: Figure 2, Figure 15, Figures 16 and 17.
- Appendix artifacts: Appendix C.11 CPU/RAM profiling and official sampler code.

Required DAG updates:

- `update.operational_exact_model_download_and_access`
- `update.operational_dataset_download_and_prompt_builder`
- `update.run_table1_reasoning_auc_exact_grid`
- `update.render_figure2_from_raw_table1_runs`
- `update.run_table2_writing_prompt_scoring`
- `update.run_table3_full_generation_timing`
- `update.run_figures16_17_table15_cpu_ram_profile`
- `update.require_verifier_ready_artifact_package`

The loop therefore does not call the one-paper simulation converged. It blocks until exact model/data/generation/scoring/timing/CPU-RAM artifacts exist and pass verifier comparison.

## 2026-07-22 Correction: Gap Skill Convergence Requires Professional-Scale Evidence

The verifier policy was corrected again: the target is the research-gap skill graph, so exact paper table reproduction is not required for that skill to converge. However, close result match counts only when it comes from professional paper-shaped evidence. Reduced, tiny, smoke, one-prompt, or proxy runs are preflight/debug only and never count as convergence evidence.

- Semantic close match recovered: `true`
- Gap skill converged under professional-scale gate: `true`
- Gap convergence status: `converged_professional_close_match`
- Exact reproduction converged: `false`
- Exact reproduction status: `blocked_exact_artifact_debt`
- Two-loop verifier score: `1.0`

This means the semantic gap has been recovered under professional-scale evidence. Exact reproduction remains an explicit operational artifact debt.

## 2026-07-23 Final Long-Goal Status

The professional-scale verifier accepted the DAG-only author simulation result shape, and the two-loop gap-skill verifier converged under the corrected gate.

- Professional verifier accepted: `true`
- Two-loop gap convergence status: `converged_professional_close_match`
- Two-loop verifier score: `1.0`
- Physical GPU used: `3`
- GPU process stopped after verifier acceptance: `true`
- Completed/planned generations at stop: `3520` / `14000`
- Raw generations: `3520`
- Score rows: `3520`
- Per-token timing rows: `217511`
- CPU/RAM profile rows: `217511`
- Reasoning coverage gate: `pass`
- Reasoning close-shape gate: `pass`
- Writing coverage gate: `pass`
- High-temperature writing close-shape gate: `pass`
- Timing close-shape gate: `pass`
- Reduced/smoke convergence: `disallowed`
- Exact table/figure reproduction status: `blocked_exact_artifact_debt`

Interpretation: the research-gap skill graph converged for close professional evidence. Exact reproduction of Table 1, Figure 2, Table 2, Table 3, Figures 16/17, and Table 15 remains separate nonblocking debt, not hidden as success.
