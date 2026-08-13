# DIRS Full-Paper Section Gap Simulation Report

Date: `2026-07-22T20:29:37Z`
Domain: `LLM Inference / Systems / Token Efficiency`
Papers: `20`
Simulation: `local_deterministic_dirs_fullpaper_section_gap`
Private holdout read: `false`

## Convergence

- Completed loops: `24`
- Minimum loops: `24`
- Stable window: `10`
- Rollouts per loop: `5000`
- Converged: `true`
- Final score: `1.025497`
- Mean selected node support: `0.995714`
- Mean selected edge support: `0.993182`

## Real Execution Layer

- CUDA available: `true`
- Visible GPU count: `4`
- Selected GPU: `2` `NVIDIA GeForce RTX 4090`
- OpenAI API key present: `true`
- Paid/external API invoked: `false`
- GPU microbenchmark ran: `true`
- Microbenchmark type: `synthetic_decode_kv_attention_microbenchmark`
- Max allocated memory: `88.14 MiB`
- Probe artifact: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_fullpaper_train20_section_gap_20260722/real_execution_probe.json`
- Benchmark artifact: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_fullpaper_train20_section_gap_20260722/gpu_microbenchmark.json`
- Author-style campaign ran: `true`
- Campaign rows: `2130`
- Campaign repeats: `30`
- Campaign runtime: `22.096s`
- Campaign families: `kv_cache_locality, quantization_compression, sampling_truncation, sparse_kernel_efficiency, speculative_decoding_proxy, token_merging`
- Inferred measured gaps: `6`
- Campaign artifact: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_fullpaper_train20_section_gap_20260722/author_style_gpu_reproduction_campaign.json`

Runtime rows:

- context `512` tokens, `float16`, `0.225` ms/decode-token, `4445.27` decode-tokens/s
- context `2048` tokens, `float16`, `0.1334` ms/decode-token, `7496.36` decode-tokens/s
- context `4096` tokens, `float16`, `0.1392` ms/decode-token, `7181.98` decode-tokens/s

## Final Selected DAG Nodes

- `appendix.artifact_log` (appendix): support `20/20`, Store probe JSON, benchmark JSON, logs, commit IDs, and commands used by the run.
- `appendix.raw_measurement_table` (appendix): support `20/20`, Attach machine-readable raw rows and grouped statistics for every GPU experiment family.
- `appendix.reproducibility_or_extra_evidence` (appendix): support `20/20`, Appendix/code artifacts support trust.
- `experiments.ablation_or_control` (experiments): support `18/20`, Ablation or control tests the mechanism.
- `experiments.api_or_local_backend` (experiments): support `20/20`, Record whether results use external API calls, local serving, or no live backend.
- `experiments.author_style_reproduction_campaign` (experiments): support `20/20`, Run GPU stress tests with baselines, controls, repeated seeds, and domain-relevant motifs.
- `experiments.axis_match` (experiments): support `20/20`, Experiment axes match the gap.
- `experiments.baseline_strength` (experiments): support `20/20`, Strong baseline family included.
- `experiments.exact_rerun_feasibility_audit` (experiments): support `20/20`, Audit every paper for exact-rerun readiness, code availability, missing checkpoints, data, and APIs.
- `experiments.execution_surface` (experiments): support `20/20`, Declare local GPU, external API, repo run, or paper-only backend provenance.
- `experiments.gpu_hardware_profile` (experiments): support `20/20`, Record GPU model, CUDA/driver, memory, precision, and backend assumptions.
- `experiments.metric_pairing` (experiments): support `20/20`, Efficiency paired with correctness or quality.
- `experiments.real_benchmark_command` (experiments): support `20/20`, Persist exact runnable command or script, config, seed, batch/context, and artifact path.
- `experiments.scale_or_stress` (experiments): support `20/20`, Stress setting or scale dimension.
- `intro.deployment_pressure` (introduction): support `20/20`, Why the systems constraint matters now.
- `intro.failure_mode` (introduction): support `20/20`, Exact failure under the target constraint.
- `intro.gap_claim` (introduction): support `20/20`, Bounded research gap claim.
- `intro.method_need` (introduction): support `20/20`, Why a new mechanism is needed.
- `intro.prior_family_map` (introduction): support `20/20`, Closest prior family and what it already solves.
- `method.constraint_binding` (method): support `20/20`, Design choices tied to gap constraint.
- `method.mechanism_delta` (method): support `20/20`, Mechanism change from baseline.
- `method.object_definition` (method): support `20/20`, The introduced method object.
- `method.operational_steps` (method): support `20/20`, Executable pipeline or algorithm.
- `related.axis_of_difference` (related_work): support `20/20`, Narrow difference versus prior work.
- `related.closest_baselines` (related_work): support `20/20`, Closest strong baselines.
- `results.exception_scan` (results): support `20/20`, Exceptions, saturation, or weak settings checked.
- `results.measured_gap_derivation` (results): support `20/20`, Derive research gaps from measured tradeoff failures, not only from paper prose.
- `results.mechanism_attribution` (results): support `19/20`, Ablation/theory attributes the gain.
- `results.primary_table_read` (results): support `20/20`, Main result table answers the gap.
- `results.proxy_vs_exact_boundary` (results): support `20/20`, Separate exact reproduction, proxy experiment, code audit, and blocked-paper evidence.
- `results.reproduction_status` (results): support `20/20`, Classify evidence as rerun, microbench-only, code-inspected-only, API-only, or paper-only.
- `results.runtime_measurement` (results): support `20/20`, Report latency, throughput, memory, GPU occupancy, or API cost from the actual run.
- `results.scope_boundary` (limitations): support `20/20`, Limitations bound the claim.
- `results.tradeoff_interpretation` (results): support `20/20`, Tradeoff interpretation.
- `root.fullpaper_gap_argument` (root): support `20/20`, Full paper gap argument unit.

## Final Selected DAG Edges

- `appendix.artifact_log->appendix.reproducibility_or_extra_evidence`: support `20/20`
- `appendix.raw_measurement_table->appendix.artifact_log`: support `20/20`
- `experiments.ablation_or_control->results.primary_table_read`: support `18/20`
- `experiments.api_or_local_backend->experiments.real_benchmark_command`: support `20/20`
- `experiments.author_style_reproduction_campaign->results.measured_gap_derivation`: support `20/20`
- `experiments.author_style_reproduction_campaign->results.runtime_measurement`: support `20/20`
- `experiments.axis_match->experiments.baseline_strength`: support `20/20`
- `experiments.baseline_strength->experiments.metric_pairing`: support `20/20`
- `experiments.exact_rerun_feasibility_audit->experiments.author_style_reproduction_campaign`: support `20/20`
- `experiments.execution_surface->experiments.api_or_local_backend`: support `20/20`
- `experiments.execution_surface->experiments.exact_rerun_feasibility_audit`: support `20/20`
- `experiments.execution_surface->experiments.gpu_hardware_profile`: support `20/20`
- `experiments.gpu_hardware_profile->experiments.real_benchmark_command`: support `20/20`
- `experiments.metric_pairing->experiments.execution_surface`: support `20/20`
- `experiments.metric_pairing->experiments.scale_or_stress`: support `20/20`
- `experiments.real_benchmark_command->experiments.author_style_reproduction_campaign`: support `20/20`
- `experiments.real_benchmark_command->experiments.scale_or_stress`: support `20/20`
- `experiments.real_benchmark_command->results.runtime_measurement`: support `20/20`
- `experiments.scale_or_stress->experiments.ablation_or_control`: support `18/20`
- `intro.deployment_pressure->intro.prior_family_map`: support `20/20`
- `intro.failure_mode->intro.gap_claim`: support `20/20`
- `intro.gap_claim->intro.method_need`: support `20/20`
- `intro.gap_claim->related.closest_baselines`: support `20/20`
- `intro.method_need->method.object_definition`: support `20/20`
- `intro.prior_family_map->intro.failure_mode`: support `20/20`
- `method.constraint_binding->experiments.axis_match`: support `20/20`
- `method.constraint_binding->method.operational_steps`: support `20/20`
- `method.mechanism_delta->method.constraint_binding`: support `20/20`
- `method.object_definition->method.mechanism_delta`: support `20/20`
- `related.closest_baselines->related.axis_of_difference`: support `20/20`
- `results.exception_scan->results.mechanism_attribution`: support `19/20`
- `results.measured_gap_derivation->results.tradeoff_interpretation`: support `20/20`
- `results.mechanism_attribution->results.scope_boundary`: support `19/20`
- `results.primary_table_read->results.tradeoff_interpretation`: support `20/20`
- `results.proxy_vs_exact_boundary->results.exception_scan`: support `20/20`
- `results.reproduction_status->results.exception_scan`: support `20/20`
- `results.reproduction_status->results.proxy_vs_exact_boundary`: support `20/20`
- `results.runtime_measurement->results.primary_table_read`: support `20/20`
- `results.runtime_measurement->results.reproduction_status`: support `20/20`
- `results.scope_boundary->appendix.artifact_log`: support `20/20`
- `results.scope_boundary->appendix.raw_measurement_table`: support `20/20`
- `results.scope_boundary->appendix.reproducibility_or_extra_evidence`: support `20/20`
- `results.tradeoff_interpretation->results.exception_scan`: support `20/20`
- `root.fullpaper_gap_argument->intro.deployment_pressure`: support `20/20`

## Learned Full-Paper Gap Policy

```text
introduction pressure
  -> prior family and near-miss boundary
  -> exact failure mode
  -> bounded gap claim
  -> method mechanism bound to the gap
  -> experiments whose axes match the gap
  -> metric pairs that protect tradeoff claims
  -> execution surface: local GPU / API / paper-only provenance
  -> GPU hardware and API/backend nodes before benchmark claims
  -> exact benchmark command and artifact log
  -> exact-rerun feasibility audit across the full paper set
  -> author-style GPU reproduction campaign with baselines/controls/repeats
  -> measured gap derivation from observed tradeoff failures
  -> proxy-vs-exact reproduction boundary
  -> runtime measurement and reproduction status
  -> scale/stress and ablation evidence
  -> scoped result and limitation boundary
```

## Verifier Reward

Positive signals:
- introduction gap pressure
- closest prior baseline fairness
- method mechanism bound to gap
- experiment axis matches gap
- metric pairing of efficiency and correctness/quality
- execution surface declared as GPU, API, local, or paper-only
- GPU hardware/API backend provenance logged
- exact benchmark command or script persisted
- author-style GPU reproduction campaign with repeated seeds
- exact-rerun feasibility audit across all papers
- measured gap derivation from GPU tradeoff failures
- runtime/memory/API-cost measurement attached to results
- reproduction status stated separately from paper claims
- scale or stress testing
- ablation/control support
- limitation boundary

Negative signals:
- unsupported section node
- unconnected graph
- single-metric tradeoff claim
- GPU speed claim without hardware profile
- API result without external-call or cost provenance
- paper-only result phrased as a local reproduction
- runtime number without command/config/log artifact
- proxy experiment reported as exact paper reproduction
- gap claim derived only from reading, without measurement pressure
- mechanism attribution without ablation/theory
- broad novelty claim without near-miss handling

## Replay Gaps

Selected nodes with less than full support are conditional moves, not universal requirements.
- `experiments.ablation_or_control`: `18/20` support
- `results.mechanism_attribution`: `19/20` support

## Trace Tail

```jsonl
{"edge_count": 44, "loop": 20, "mean_selected_edge_support": 0.993182, "mean_selected_node_support": 0.995714, "node_count": 35, "score": 1.025347, "selected_edges": ["appendix.artifact_log->appendix.reproducibility_or_extra_evidence", "appendix.raw_measurement_table->appendix.artifact_log", "experiments.ablation_or_control->results.primary_table_read", "experiments.api_or_local_backend->experiments.real_benchmark_command", "experiments.author_style_reproduction_campaign->results.measured_gap_derivation", "experiments.author_style_reproduction_campaign->results.runtime_measurement", "experiments.axis_match->experiments.baseline_strength", "experiments.baseline_strength->experiments.metric_pairing", "experiments.exact_rerun_feasibility_audit->experiments.author_style_reproduction_campaign", "experiments.execution_surface->experiments.api_or_local_backend", "experiments.execution_surface->experiments.exact_rerun_feasibility_audit", "experiments.execution_surface->experiments.gpu_hardware_profile", "experiments.gpu_hardware_profile->experiments.real_benchmark_command", "experiments.metric_pairing->experiments.execution_surface", "experiments.metric_pairing->experiments.scale_or_stress", "experiments.real_benchmark_command->experiments.author_style_reproduction_campaign", "experiments.real_benchmark_command->experiments.scale_or_stress", "experiments.real_benchmark_command->results.runtime_measurement", "experiments.scale_or_stress->experiments.ablation_or_control", "intro.deployment_pressure->intro.prior_family_map", "intro.failure_mode->intro.gap_claim", "intro.gap_claim->intro.method_need", "intro.gap_claim->related.closest_baselines", "intro.method_need->method.object_definition", "intro.prior_family_map->intro.failure_mode", "method.constraint_binding->experiments.axis_match", "method.constraint_binding->method.operational_steps", "method.mechanism_delta->method.constraint_binding", "method.object_definition->method.mechanism_delta", "related.closest_baselines->related.axis_of_difference", "results.exception_scan->results.mechanism_attribution", "results.measured_gap_derivation->results.tradeoff_interpretation", "results.mechanism_attribution->results.scope_boundary", "results.primary_table_read->results.tradeoff_interpretation", "results.proxy_vs_exact_boundary->results.exception_scan", "results.reproduction_status->results.exception_scan", "results.reproduction_status->results.proxy_vs_exact_boundary", "results.runtime_measurement->results.primary_table_read", "results.runtime_measurement->results.reproduction_status", "results.scope_boundary->appendix.artifact_log", "results.scope_boundary->appendix.raw_measurement_table", "results.scope_boundary->appendix.reproducibility_or_extra_evidence", "results.tradeoff_interpretation->results.exception_scan", "root.fullpaper_gap_argument->intro.deployment_pressure"], "selected_nodes": ["appendix.artifact_log", "appendix.raw_measurement_table", "appendix.reproducibility_or_extra_evidence", "experiments.ablation_or_control", "experiments.api_or_local_backend", "experiments.author_style_reproduction_campaign", "experiments.axis_match", "experiments.baseline_strength", "experiments.exact_rerun_feasibility_audit", "experiments.execution_surface", "experiments.gpu_hardware_profile", "experiments.metric_pairing", "experiments.real_benchmark_command", "experiments.scale_or_stress", "intro.deployment_pressure", "intro.failure_mode", "intro.gap_claim", "intro.method_need", "intro.prior_family_map", "method.constraint_binding", "method.mechanism_delta", "method.object_definition", "method.operational_steps", "related.axis_of_difference", "related.closest_baselines", "results.exception_scan", "results.measured_gap_derivation", "results.mechanism_attribution", "results.primary_table_read", "results.proxy_vs_exact_boundary", "results.reproduction_status", "results.runtime_measurement", "results.scope_boundary", "results.tradeoff_interpretation", "root.fullpaper_gap_argument"], "signature": "03952e0268262338", "stable_signature_count": 10}
{"edge_count": 44, "loop": 21, "mean_selected_edge_support": 0.993182, "mean_selected_node_support": 0.995714, "node_count": 35, "score": 1.025359, "selected_edges": ["appendix.artifact_log->appendix.reproducibility_or_extra_evidence", "appendix.raw_measurement_table->appendix.artifact_log", "experiments.ablation_or_control->results.primary_table_read", "experiments.api_or_local_backend->experiments.real_benchmark_command", "experiments.author_style_reproduction_campaign->results.measured_gap_derivation", "experiments.author_style_reproduction_campaign->results.runtime_measurement", "experiments.axis_match->experiments.baseline_strength", "experiments.baseline_strength->experiments.metric_pairing", "experiments.exact_rerun_feasibility_audit->experiments.author_style_reproduction_campaign", "experiments.execution_surface->experiments.api_or_local_backend", "experiments.execution_surface->experiments.exact_rerun_feasibility_audit", "experiments.execution_surface->experiments.gpu_hardware_profile", "experiments.gpu_hardware_profile->experiments.real_benchmark_command", "experiments.metric_pairing->experiments.execution_surface", "experiments.metric_pairing->experiments.scale_or_stress", "experiments.real_benchmark_command->experiments.author_style_reproduction_campaign", "experiments.real_benchmark_command->experiments.scale_or_stress", "experiments.real_benchmark_command->results.runtime_measurement", "experiments.scale_or_stress->experiments.ablation_or_control", "intro.deployment_pressure->intro.prior_family_map", "intro.failure_mode->intro.gap_claim", "intro.gap_claim->intro.method_need", "intro.gap_claim->related.closest_baselines", "intro.method_need->method.object_definition", "intro.prior_family_map->intro.failure_mode", "method.constraint_binding->experiments.axis_match", "method.constraint_binding->method.operational_steps", "method.mechanism_delta->method.constraint_binding", "method.object_definition->method.mechanism_delta", "related.closest_baselines->related.axis_of_difference", "results.exception_scan->results.mechanism_attribution", "results.measured_gap_derivation->results.tradeoff_interpretation", "results.mechanism_attribution->results.scope_boundary", "results.primary_table_read->results.tradeoff_interpretation", "results.proxy_vs_exact_boundary->results.exception_scan", "results.reproduction_status->results.exception_scan", "results.reproduction_status->results.proxy_vs_exact_boundary", "results.runtime_measurement->results.primary_table_read", "results.runtime_measurement->results.reproduction_status", "results.scope_boundary->appendix.artifact_log", "results.scope_boundary->appendix.raw_measurement_table", "results.scope_boundary->appendix.reproducibility_or_extra_evidence", "results.tradeoff_interpretation->results.exception_scan", "root.fullpaper_gap_argument->intro.deployment_pressure"], "selected_nodes": ["appendix.artifact_log", "appendix.raw_measurement_table", "appendix.reproducibility_or_extra_evidence", "experiments.ablation_or_control", "experiments.api_or_local_backend", "experiments.author_style_reproduction_campaign", "experiments.axis_match", "experiments.baseline_strength", "experiments.exact_rerun_feasibility_audit", "experiments.execution_surface", "experiments.gpu_hardware_profile", "experiments.metric_pairing", "experiments.real_benchmark_command", "experiments.scale_or_stress", "intro.deployment_pressure", "intro.failure_mode", "intro.gap_claim", "intro.method_need", "intro.prior_family_map", "method.constraint_binding", "method.mechanism_delta", "method.object_definition", "method.operational_steps", "related.axis_of_difference", "related.closest_baselines", "results.exception_scan", "results.measured_gap_derivation", "results.mechanism_attribution", "results.primary_table_read", "results.proxy_vs_exact_boundary", "results.reproduction_status", "results.runtime_measurement", "results.scope_boundary", "results.tradeoff_interpretation", "root.fullpaper_gap_argument"], "signature": "03952e0268262338", "stable_signature_count": 11}
{"edge_count": 44, "loop": 22, "mean_selected_edge_support": 0.993182, "mean_selected_node_support": 0.995714, "node_count": 35, "score": 1.024982, "selected_edges": ["appendix.artifact_log->appendix.reproducibility_or_extra_evidence", "appendix.raw_measurement_table->appendix.artifact_log", "experiments.ablation_or_control->results.primary_table_read", "experiments.api_or_local_backend->experiments.real_benchmark_command", "experiments.author_style_reproduction_campaign->results.measured_gap_derivation", "experiments.author_style_reproduction_campaign->results.runtime_measurement", "experiments.axis_match->experiments.baseline_strength", "experiments.baseline_strength->experiments.metric_pairing", "experiments.exact_rerun_feasibility_audit->experiments.author_style_reproduction_campaign", "experiments.execution_surface->experiments.api_or_local_backend", "experiments.execution_surface->experiments.exact_rerun_feasibility_audit", "experiments.execution_surface->experiments.gpu_hardware_profile", "experiments.gpu_hardware_profile->experiments.real_benchmark_command", "experiments.metric_pairing->experiments.execution_surface", "experiments.metric_pairing->experiments.scale_or_stress", "experiments.real_benchmark_command->experiments.author_style_reproduction_campaign", "experiments.real_benchmark_command->experiments.scale_or_stress", "experiments.real_benchmark_command->results.runtime_measurement", "experiments.scale_or_stress->experiments.ablation_or_control", "intro.deployment_pressure->intro.prior_family_map", "intro.failure_mode->intro.gap_claim", "intro.gap_claim->intro.method_need", "intro.gap_claim->related.closest_baselines", "intro.method_need->method.object_definition", "intro.prior_family_map->intro.failure_mode", "method.constraint_binding->experiments.axis_match", "method.constraint_binding->method.operational_steps", "method.mechanism_delta->method.constraint_binding", "method.object_definition->method.mechanism_delta", "related.closest_baselines->related.axis_of_difference", "results.exception_scan->results.mechanism_attribution", "results.measured_gap_derivation->results.tradeoff_interpretation", "results.mechanism_attribution->results.scope_boundary", "results.primary_table_read->results.tradeoff_interpretation", "results.proxy_vs_exact_boundary->results.exception_scan", "results.reproduction_status->results.exception_scan", "results.reproduction_status->results.proxy_vs_exact_boundary", "results.runtime_measurement->results.primary_table_read", "results.runtime_measurement->results.reproduction_status", "results.scope_boundary->appendix.artifact_log", "results.scope_boundary->appendix.raw_measurement_table", "results.scope_boundary->appendix.reproducibility_or_extra_evidence", "results.tradeoff_interpretation->results.exception_scan", "root.fullpaper_gap_argument->intro.deployment_pressure"], "selected_nodes": ["appendix.artifact_log", "appendix.raw_measurement_table", "appendix.reproducibility_or_extra_evidence", "experiments.ablation_or_control", "experiments.api_or_local_backend", "experiments.author_style_reproduction_campaign", "experiments.axis_match", "experiments.baseline_strength", "experiments.exact_rerun_feasibility_audit", "experiments.execution_surface", "experiments.gpu_hardware_profile", "experiments.metric_pairing", "experiments.real_benchmark_command", "experiments.scale_or_stress", "intro.deployment_pressure", "intro.failure_mode", "intro.gap_claim", "intro.method_need", "intro.prior_family_map", "method.constraint_binding", "method.mechanism_delta", "method.object_definition", "method.operational_steps", "related.axis_of_difference", "related.closest_baselines", "results.exception_scan", "results.measured_gap_derivation", "results.mechanism_attribution", "results.primary_table_read", "results.proxy_vs_exact_boundary", "results.reproduction_status", "results.runtime_measurement", "results.scope_boundary", "results.tradeoff_interpretation", "root.fullpaper_gap_argument"], "signature": "03952e0268262338", "stable_signature_count": 12}
{"edge_count": 44, "loop": 23, "mean_selected_edge_support": 0.993182, "mean_selected_node_support": 0.995714, "node_count": 35, "score": 1.025574, "selected_edges": ["appendix.artifact_log->appendix.reproducibility_or_extra_evidence", "appendix.raw_measurement_table->appendix.artifact_log", "experiments.ablation_or_control->results.primary_table_read", "experiments.api_or_local_backend->experiments.real_benchmark_command", "experiments.author_style_reproduction_campaign->results.measured_gap_derivation", "experiments.author_style_reproduction_campaign->results.runtime_measurement", "experiments.axis_match->experiments.baseline_strength", "experiments.baseline_strength->experiments.metric_pairing", "experiments.exact_rerun_feasibility_audit->experiments.author_style_reproduction_campaign", "experiments.execution_surface->experiments.api_or_local_backend", "experiments.execution_surface->experiments.exact_rerun_feasibility_audit", "experiments.execution_surface->experiments.gpu_hardware_profile", "experiments.gpu_hardware_profile->experiments.real_benchmark_command", "experiments.metric_pairing->experiments.execution_surface", "experiments.metric_pairing->experiments.scale_or_stress", "experiments.real_benchmark_command->experiments.author_style_reproduction_campaign", "experiments.real_benchmark_command->experiments.scale_or_stress", "experiments.real_benchmark_command->results.runtime_measurement", "experiments.scale_or_stress->experiments.ablation_or_control", "intro.deployment_pressure->intro.prior_family_map", "intro.failure_mode->intro.gap_claim", "intro.gap_claim->intro.method_need", "intro.gap_claim->related.closest_baselines", "intro.method_need->method.object_definition", "intro.prior_family_map->intro.failure_mode", "method.constraint_binding->experiments.axis_match", "method.constraint_binding->method.operational_steps", "method.mechanism_delta->method.constraint_binding", "method.object_definition->method.mechanism_delta", "related.closest_baselines->related.axis_of_difference", "results.exception_scan->results.mechanism_attribution", "results.measured_gap_derivation->results.tradeoff_interpretation", "results.mechanism_attribution->results.scope_boundary", "results.primary_table_read->results.tradeoff_interpretation", "results.proxy_vs_exact_boundary->results.exception_scan", "results.reproduction_status->results.exception_scan", "results.reproduction_status->results.proxy_vs_exact_boundary", "results.runtime_measurement->results.primary_table_read", "results.runtime_measurement->results.reproduction_status", "results.scope_boundary->appendix.artifact_log", "results.scope_boundary->appendix.raw_measurement_table", "results.scope_boundary->appendix.reproducibility_or_extra_evidence", "results.tradeoff_interpretation->results.exception_scan", "root.fullpaper_gap_argument->intro.deployment_pressure"], "selected_nodes": ["appendix.artifact_log", "appendix.raw_measurement_table", "appendix.reproducibility_or_extra_evidence", "experiments.ablation_or_control", "experiments.api_or_local_backend", "experiments.author_style_reproduction_campaign", "experiments.axis_match", "experiments.baseline_strength", "experiments.exact_rerun_feasibility_audit", "experiments.execution_surface", "experiments.gpu_hardware_profile", "experiments.metric_pairing", "experiments.real_benchmark_command", "experiments.scale_or_stress", "intro.deployment_pressure", "intro.failure_mode", "intro.gap_claim", "intro.method_need", "intro.prior_family_map", "method.constraint_binding", "method.mechanism_delta", "method.object_definition", "method.operational_steps", "related.axis_of_difference", "related.closest_baselines", "results.exception_scan", "results.measured_gap_derivation", "results.mechanism_attribution", "results.primary_table_read", "results.proxy_vs_exact_boundary", "results.reproduction_status", "results.runtime_measurement", "results.scope_boundary", "results.tradeoff_interpretation", "root.fullpaper_gap_argument"], "signature": "03952e0268262338", "stable_signature_count": 13}
{"edge_count": 44, "loop": 24, "mean_selected_edge_support": 0.993182, "mean_selected_node_support": 0.995714, "node_count": 35, "score": 1.025497, "selected_edges": ["appendix.artifact_log->appendix.reproducibility_or_extra_evidence", "appendix.raw_measurement_table->appendix.artifact_log", "experiments.ablation_or_control->results.primary_table_read", "experiments.api_or_local_backend->experiments.real_benchmark_command", "experiments.author_style_reproduction_campaign->results.measured_gap_derivation", "experiments.author_style_reproduction_campaign->results.runtime_measurement", "experiments.axis_match->experiments.baseline_strength", "experiments.baseline_strength->experiments.metric_pairing", "experiments.exact_rerun_feasibility_audit->experiments.author_style_reproduction_campaign", "experiments.execution_surface->experiments.api_or_local_backend", "experiments.execution_surface->experiments.exact_rerun_feasibility_audit", "experiments.execution_surface->experiments.gpu_hardware_profile", "experiments.gpu_hardware_profile->experiments.real_benchmark_command", "experiments.metric_pairing->experiments.execution_surface", "experiments.metric_pairing->experiments.scale_or_stress", "experiments.real_benchmark_command->experiments.author_style_reproduction_campaign", "experiments.real_benchmark_command->experiments.scale_or_stress", "experiments.real_benchmark_command->results.runtime_measurement", "experiments.scale_or_stress->experiments.ablation_or_control", "intro.deployment_pressure->intro.prior_family_map", "intro.failure_mode->intro.gap_claim", "intro.gap_claim->intro.method_need", "intro.gap_claim->related.closest_baselines", "intro.method_need->method.object_definition", "intro.prior_family_map->intro.failure_mode", "method.constraint_binding->experiments.axis_match", "method.constraint_binding->method.operational_steps", "method.mechanism_delta->method.constraint_binding", "method.object_definition->method.mechanism_delta", "related.closest_baselines->related.axis_of_difference", "results.exception_scan->results.mechanism_attribution", "results.measured_gap_derivation->results.tradeoff_interpretation", "results.mechanism_attribution->results.scope_boundary", "results.primary_table_read->results.tradeoff_interpretation", "results.proxy_vs_exact_boundary->results.exception_scan", "results.reproduction_status->results.exception_scan", "results.reproduction_status->results.proxy_vs_exact_boundary", "results.runtime_measurement->results.primary_table_read", "results.runtime_measurement->results.reproduction_status", "results.scope_boundary->appendix.artifact_log", "results.scope_boundary->appendix.raw_measurement_table", "results.scope_boundary->appendix.reproducibility_or_extra_evidence", "results.tradeoff_interpretation->results.exception_scan", "root.fullpaper_gap_argument->intro.deployment_pressure"], "selected_nodes": ["appendix.artifact_log", "appendix.raw_measurement_table", "appendix.reproducibility_or_extra_evidence", "experiments.ablation_or_control", "experiments.api_or_local_backend", "experiments.author_style_reproduction_campaign", "experiments.axis_match", "experiments.baseline_strength", "experiments.exact_rerun_feasibility_audit", "experiments.execution_surface", "experiments.gpu_hardware_profile", "experiments.metric_pairing", "experiments.real_benchmark_command", "experiments.scale_or_stress", "intro.deployment_pressure", "intro.failure_mode", "intro.gap_claim", "intro.method_need", "intro.prior_family_map", "method.constraint_binding", "method.mechanism_delta", "method.object_definition", "method.operational_steps", "related.axis_of_difference", "related.closest_baselines", "results.exception_scan", "results.measured_gap_derivation", "results.mechanism_attribution", "results.primary_table_read", "results.proxy_vs_exact_boundary", "results.reproduction_status", "results.runtime_measurement", "results.scope_boundary", "results.tradeoff_interpretation", "root.fullpaper_gap_argument"], "signature": "03952e0268262338", "stable_signature_count": 14}
```
