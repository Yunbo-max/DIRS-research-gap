# Battleship Specialized Professional Gate

- Paper id: `ICLR2026_EQhUvWH78U_rational_information_seeking_agents`
- Title: Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People
- Status: `blocked_by_python311_openrouter_full_agent_benchmark_and_guess_who_artifacts`
- Converged: `false`
- Professional ready: `false`
- GPU used: `false` for this paper-specific preflight; core paper execution is API/symbolic, not CUDA.
- DAG signature: `c6430e3580cd9018`
- Repo: `/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/repos/battleship`

## Blockers

- `battleship_python311_runtime_missing`: Current Python is 3.10; pyproject requires ^3.11 and tests fail on enum.StrEnum under Python 3.10.
- `battleship_unit_probe_failed_until_python311_runtime`: Selected tests did not collect/pass; see environment.json.
- `battleship_runtime_dependencies_missing`: Missing runtime imports: lark
- `battleship_openrouter_api_key_missing`: battleship/agents.py constructs an OpenAI client with OpenRouter base_url and requires OPENROUTER_API_KEY.
- `battleship_required_data_artifacts_missing`: Missing data artifacts: readme_named_human_dataset_csv
- `battleship_raw_human_trajectory_files_incomplete`: Raw collaborative human-data directory exists but does not contain the paper-shaped game/round/stage trajectory files expected by README.
- `battleship_full_agent_benchmark_outputs_missing`: Missing operational result directories/files: spotter_benchmark_results, captain_benchmark_results, guess_who_transfer_outputs, openrouter_model_eval_outputs, paper_metric_summary
- `battleship_result_shape_verifier_waiting_for_full_grid`: Verifier cannot compare against paper tables/figures/paragraphs until SpotterQA, CaptainQA, human comparison, OpenRouter model grid, and Guess Who outputs are produced.

## Artifact Paths

- Professional gate: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/battleship/professional_gate_result.json`
- Verifier: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/battleship/battleship_specialized_verifier.json`
- Repo manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/battleship/repo_manifest.json`
- Environment: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/battleship/environment.json`
- Model/data manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/battleship/model_data_manifest.json`
- API runtime manifest: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/battleship/api_runtime_manifest.json`

## Verifier Checks

- `blind_contract`: `pass`
- `repo_path_encoded`: `pass`
- `gpu_requirement_interpretation`: `pass`
- `python311_runtime_gate`: `blocked`
- `openrouter_api_gate`: `blocked`
- `professional_artifact_package`: `blocked`
- `result_shape_comparison_ready`: `blocked`
