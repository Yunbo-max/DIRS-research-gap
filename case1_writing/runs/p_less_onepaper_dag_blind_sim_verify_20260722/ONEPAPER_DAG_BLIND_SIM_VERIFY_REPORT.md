# One-Paper DAG-Only Blind Simulation and Verification

Date: `2026-07-22T21:21:44Z`
Target: `ICLR2026_ItFuNJQGH4_p_less_sampling`
Title: `p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding`
Venue: `ICLR 2026 Oral`

## Contract

The simulator sees only the DAG file copied into its blind workspace. It does not receive paper text, evidence tables, oracle results, prior reports, campaign JSON, or previous memory artifacts.

Verifier-only files stay outside the blind workspace and are used only after simulation.

## Final Result

- Converged: `true`
- Iterations: `6`
- Final verifier score: `1.0`
- Final DAG nodes: `16`
- Final DAG edges: `20`
- Final DAG signature: `de103d8765817f7e`
- Blind simulation input: `paper_author_dag.json`
- Paper/oracle seen by simulator: `false`

## Iteration Trace

- Iteration `1`: detail `1`, score `0.8`, converged_ready `false`, missing `main_auc_shape, dag_detail_sufficiency`
- Iteration `2`: detail `2`, score `0.8`, converged_ready `false`, missing `main_auc_shape, dag_detail_sufficiency`
- Iteration `3`: detail `3`, score `0.8`, converged_ready `false`, missing `main_auc_shape, dag_detail_sufficiency`
- Iteration `4`: detail `4`, score `0.84`, converged_ready `false`, missing `main_auc_shape`
- Iteration `5`: detail `5`, score `1.0`, converged_ready `true`, missing `none`
- Iteration `6`: detail `5`, score `1.0`, converged_ready `true`, missing `none`

## Final Blind Predictions

- `bounded_candidate_set`: `True`
- `efficiency_shape`: `p_less_fastest_or_tied_fastest`
- `exact_reproduction_boundary`: `not_claimed_exact_full_table_reproduction`
- `high_temperature_writing_shape`: `p_less_stable_high_temperature`
- `hyperparameter_free`: `True`
- `reasoning_auc_shape`: `p_less_or_p_lessnorm_top_or_near_top`

## Final Rankings From DAG-Only Simulation

- `high_temp_quality_proxy`: `p_lessnorm, p_less, min_p, mirostat, top_p, eta, epsilon`
- `operation_cost_proxy`: `p_less, min_p, p_lessnorm, epsilon, eta, top_p, mirostat`
- `quality_proxy`: `p_lessnorm, p_less, min_p, top_p, eta, epsilon, mirostat`
- `speed_proxy`: `epsilon, min_p, p_less, p_lessnorm, top_p, mirostat, eta`

## Verifier Comparison

- `blind_input_contract`: `true` (leakage_hits=[])
- `bounded_candidate_set`: `true` (p_less fallback=0.000)
- `hyperparameter_free`: `true` (p_less and p_lessnorm carried no sampler hyperparameters in the DAG.)
- `main_auc_shape`: `true` (quality ranking=['p_lessnorm', 'p_less', 'min_p', 'top_p'])
- `high_temperature_writing_shape`: `true` (high-temp ranking=['p_lessnorm', 'p_less', 'min_p', 'mirostat'])
- `efficiency_shape`: `true` (operation ranking=['p_less', 'min_p', 'p_lessnorm', 'epsilon'], raw speed ranking=['epsilon', 'min_p', 'p_less', 'p_lessnorm'], p_less_time=0.00074494, fastest=0.00053070)
- `exact_reproduction_boundary`: `true` (blocked_by_missing_full_evaluation_pipeline)
- `dag_detail_sufficiency`: `true` (detail_level=5, node_count=16)

## Artifacts

- Final DAG: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/p_less_onepaper_dag_blind_sim_verify_20260722/paper_author_dag.json`
- Verifier oracle: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/p_less_onepaper_dag_blind_sim_verify_20260722/paper_oracle_results.json`
- Summary JSON: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/p_less_onepaper_dag_blind_sim_verify_20260722/onepaper_dag_blind_sim_verify_summary.json`
- Iterations: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/p_less_onepaper_dag_blind_sim_verify_20260722/iterations`
- Blind simulator source: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/p_less_onepaper_dag_blind_sim_verify_20260722/blind_simulator_from_dag_only.py`