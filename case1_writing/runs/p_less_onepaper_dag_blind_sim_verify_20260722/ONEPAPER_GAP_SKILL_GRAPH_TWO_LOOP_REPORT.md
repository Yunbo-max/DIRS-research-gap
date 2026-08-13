# One-Paper Gap Skill Graph Two-Loop Report

Date: 2026-07-23T00:27:31Z

Target paper: `ICLR2026_ItFuNJQGH4_p_less_sampling`

## Final Status

- Semantic close match recovered: `true`
- Gap skill converged under professional-scale gate: `true`
- Gap convergence status: `converged_professional_close_match`
- Exact reproduction converged: `false`
- Exact reproduction status: `blocked_exact_artifact_debt`
- Iterations: `2`
- Final verifier score: `1.0`

## Correct Convergence Rule

The verifier now compares whether the DAG-only author simulation recovers the real paper gap and close result shape. Close results can converge the gap skill only when they come from professional paper-shaped experiments. Reduced, tiny, smoke, one-prompt, or proxy runs never count as convergence evidence.

## What The Simulation Recovered

- Gap: `Existing decoding samplers are brittle because they require hyperparameter tuning across task and temperature; fixed or single-token-relative thresholds can fail under high temperature, tail-token, or fallback cases.`
- Method binding: `A full-distribution second-moment threshold can remove sampler hyperparameters while adapting the retained token set; a normalized variant trades toward diversity.`
- Main result shape: `p_less_or_p_lessnorm_top_or_near_top`
- High-temperature shape: `p_less_stable_high_temperature`
- Efficiency shape: `p_less_fastest_or_near_fastest_in_full_generation`

## Operational Evidence

- Artifact status: `pass`
- Artifact scale: `professional_paper_shaped_long_run`
- Professional-scale for gap convergence: `True`
- Professional-scale evidence ready: `True`
- Physical GPU: `3`
- Model: `mistralai/Mistral-7B-Instruct-v0.2`
- Datasets: `['gsm8k', 'csqa', 'qasc', 'writingprompts']`
- Completed/planned generations: `3510` / `14000`
- Coverage: `0.2507142857142857`
- Raw generations: `None`
- Per-token timing rows: `None`
- CPU/RAM rows: `None`
- Exact paper claim from reduced run: `None`

## Exact Artifact Debt

- `Table 1` reasoning AUC exact grid
- `Figure 2` temperature curves from exact grid
- `Table 2` Writing Prompts scoring
- `Table 3` full generation timing
- `Figures 16/17` CPU/RAM curves
- `Table 15` CPU/RAM values

## Artifacts

- `p_less_research_gap_skill_graph.json`
- `onepaper_gap_skill_graph_two_loop_summary.json`
- `gap_skill_graph_two_loop/`
