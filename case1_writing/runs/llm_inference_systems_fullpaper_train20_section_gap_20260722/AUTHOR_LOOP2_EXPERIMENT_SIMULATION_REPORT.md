# Loop 2 Author Experiment Simulation

Date: `2026-07-22T20:47:56Z`
Simulation: `loop2_author_experiment_decision_simulation`
Definition: Author forms hypotheses, runs/reads experiments, makes decisions, revises claims, and writes conclusions.
Campaign artifact: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_fullpaper_train20_section_gap_20260722/author_style_gpu_reproduction_campaign.json`
Private holdout read: `false`
Paid/external API invoked: `false`

## What Simulation Means Here

Loop 2 is the author-side scientific decision loop:

```text
gap hypothesis
  -> experiment design / baseline / control
  -> GPU or artifact measurement
  -> repeated-seed statistics
  -> author judgment
  -> claim revision
  -> conclusion and paper sections
```

It is not paper reading, not reviewer-only critique, and not a GPU benchmark by itself.

## Campaign Evidence

- GPU rows read: `2130`
- Repeated seeds: `30`
- Families: `kv_cache_locality, quantization_compression, sampling_truncation, sparse_kernel_efficiency, speculative_decoding_proxy, token_merging`
- Exact-rerun blocked/not-ready: `16/20`
- KV max local-window RMSE: `0.121248`
- Token-merge max RMSE: `0.553301`
- Speculative min effective speedup: `0.000000`
- Sampling min retained mass: `0.029347`
- Sparse controls not faster than dense: `48/270`

## Author Decisions

- `Reject exact-reproduction wording.` 16/20 papers are not exact-rerun-ready under local constraints. Write as: proxy reproduction plus reproducibility boundary.
- `Accept quality-guard gap.` KV local-window RMSE reached 0.121248; token-merge RMSE reached 0.553301; sampling retained mass dropped to 0.029347. Write as: speed claims require paired quality/correctness metrics.
- `Accept acceptance-limited speculative decoding gap.` minimum effective speculative speedup was 0.000000. Write as: draft quality and acceptance distribution are core variables.
- `Accept hardware-specificity gap.` 48/270 sparse controls failed to beat dense; quant-dequant was not faster than fp16 operation in the proxy. Write as: efficiency requires hardware-aware kernels and end-to-end measurement.
- `Revise conclusion to bounded NeurIPS-systems claim.` The measurements support tradeoff and reproducibility gaps, not a universal new algorithmic win. Write as: a strict paper should propose a measured backend-specific policy with artifacts and limits.

## Selected Author DAG Nodes

- `A.appendix_artifacts` support `1.0`: Attach scripts, raw rows, traces, and blocked-rerun audit.
- `C.central_claim` support `1.0`: Conclusion: strict systems gaps must be measured as backend-specific speed/quality/reproducibility tradeoffs.
- `C.paper_sections_to_write` support `1.0`: Write experiments/results/limitations from decisions and artifacts.
- `D.accept_acceptance_limited_gap` support `1.0`: Author decision: speculative speedup is acceptance-limited.
- `D.accept_hardware_specificity_gap` support `1.0`: Author decision: unstructured sparsity/compression needs hardware-aware kernels.
- `D.accept_quality_guard_gap` support `1.0`: Author decision: speed claims need paired quality/correctness guards.
- `D.accept_reproducibility_gap` support `1.0`: Author decision: exact-rerun infrastructure gap survives.
- `D.reject_exact_reproduction_claim` support `1.0`: Author decision: do not claim exact leaderboard reproduction from proxy tests.
- `D.revise_overbroad_speed_claim` support `1.0`: Author decision: soften any universal speedup claim.
- `E.gpu_campaign_baselines` support `1.0`: Run GPU baselines and controls with repeated seeds.
- `E.kv_cache_stress` support `1.0`: Run dense/local/top-k KV-style long-context stress tests.
- `E.quantization_kernel_stress` support `1.0`: Run fp16 versus quant-dequant controls.
- `E.repo_exact_rerun_audit` support `1.0`: Audit 20 papers for repo, checkpoint, dataset, and API readiness.
- `E.sampling_mass_entropy_stress` support `1.0`: Run full softmax, top-k, top-p, entropy-adaptive sampling controls.
- `E.sparse_kernel_stress` support `1.0`: Run dense versus masked-sparse matvec controls.
- `E.speculative_acceptance_stress` support `1.0`: Run draft/target acceptance-limited speculative proxy.
- `E.token_merge_stress` support `1.0`: Run full-attention versus merge proxy controls.
- `H.hardware_specificity_gap` support `1.0`: Hypothesis: speedup requires hardware-aware implementation, not abstract sparsity/compression.
- `H.quality_guard_gap` support `1.0`: Hypothesis: token efficiency needs quality/correctness guards.
- `H.reproducibility_gap` support `1.0`: Hypothesis: exact systems claims are blocked without runnable artifacts.
- `M.raw_measurement_read` support `1.0`: Read raw GPU rows, not just summary prose.
- `M.statistics_over_repeats` support `1.0`: Aggregate repeated-seed means/stds before judging.
- `root.author_experiment_loop` support `1.0`: Loop 2 is the author doing experiments and making scientific decisions.

## Selected Author DAG Edges

- `C.central_claim->C.paper_sections_to_write` support `1.0`
- `C.paper_sections_to_write->A.appendix_artifacts` support `1.0`
- `D.accept_acceptance_limited_gap->D.revise_overbroad_speed_claim` support `1.0`
- `D.accept_hardware_specificity_gap->D.revise_overbroad_speed_claim` support `1.0`
- `D.accept_quality_guard_gap->D.revise_overbroad_speed_claim` support `1.0`
- `D.accept_reproducibility_gap->D.reject_exact_reproduction_claim` support `1.0`
- `D.reject_exact_reproduction_claim->C.central_claim` support `1.0`
- `D.revise_overbroad_speed_claim->C.central_claim` support `1.0`
- `E.gpu_campaign_baselines->E.kv_cache_stress` support `1.0`
- `E.gpu_campaign_baselines->E.quantization_kernel_stress` support `1.0`
- `E.gpu_campaign_baselines->E.sampling_mass_entropy_stress` support `1.0`
- `E.gpu_campaign_baselines->E.sparse_kernel_stress` support `1.0`
- `E.gpu_campaign_baselines->E.speculative_acceptance_stress` support `1.0`
- `E.gpu_campaign_baselines->E.token_merge_stress` support `1.0`
- `E.kv_cache_stress->M.raw_measurement_read` support `1.0`
- `E.quantization_kernel_stress->M.raw_measurement_read` support `1.0`
- `E.repo_exact_rerun_audit->M.raw_measurement_read` support `1.0`
- `E.sampling_mass_entropy_stress->M.raw_measurement_read` support `1.0`
- `E.sparse_kernel_stress->M.raw_measurement_read` support `1.0`
- `E.speculative_acceptance_stress->M.raw_measurement_read` support `1.0`
- `E.token_merge_stress->M.raw_measurement_read` support `1.0`
- `H.hardware_specificity_gap->E.gpu_campaign_baselines` support `1.0`
- `H.quality_guard_gap->E.gpu_campaign_baselines` support `1.0`
- `H.reproducibility_gap->E.repo_exact_rerun_audit` support `1.0`
- `M.raw_measurement_read->M.statistics_over_repeats` support `1.0`
- `M.statistics_over_repeats->D.accept_acceptance_limited_gap` support `1.0`
- `M.statistics_over_repeats->D.accept_hardware_specificity_gap` support `1.0`
- `M.statistics_over_repeats->D.accept_quality_guard_gap` support `1.0`
- `M.statistics_over_repeats->D.accept_reproducibility_gap` support `1.0`
- `root.author_experiment_loop->H.hardware_specificity_gap` support `1.0`
- `root.author_experiment_loop->H.quality_guard_gap` support `1.0`
- `root.author_experiment_loop->H.reproducibility_gap` support `1.0`

## Convergence

- Completed loops: `24`
- Converged: `true`
- Final score: `1.008011`

## Trace Tail

```jsonl
{"edge_count": 32, "loop": 20, "mean_support": 1.0, "node_count": 23, "score": 1.007326, "selected_edges": ["C.central_claim->C.paper_sections_to_write", "C.paper_sections_to_write->A.appendix_artifacts", "D.accept_acceptance_limited_gap->D.revise_overbroad_speed_claim", "D.accept_hardware_specificity_gap->D.revise_overbroad_speed_claim", "D.accept_quality_guard_gap->D.revise_overbroad_speed_claim", "D.accept_reproducibility_gap->D.reject_exact_reproduction_claim", "D.reject_exact_reproduction_claim->C.central_claim", "D.revise_overbroad_speed_claim->C.central_claim", "E.gpu_campaign_baselines->E.kv_cache_stress", "E.gpu_campaign_baselines->E.quantization_kernel_stress", "E.gpu_campaign_baselines->E.sampling_mass_entropy_stress", "E.gpu_campaign_baselines->E.sparse_kernel_stress", "E.gpu_campaign_baselines->E.speculative_acceptance_stress", "E.gpu_campaign_baselines->E.token_merge_stress", "E.kv_cache_stress->M.raw_measurement_read", "E.quantization_kernel_stress->M.raw_measurement_read", "E.repo_exact_rerun_audit->M.raw_measurement_read", "E.sampling_mass_entropy_stress->M.raw_measurement_read", "E.sparse_kernel_stress->M.raw_measurement_read", "E.speculative_acceptance_stress->M.raw_measurement_read", "E.token_merge_stress->M.raw_measurement_read", "H.hardware_specificity_gap->E.gpu_campaign_baselines", "H.quality_guard_gap->E.gpu_campaign_baselines", "H.reproducibility_gap->E.repo_exact_rerun_audit", "M.raw_measurement_read->M.statistics_over_repeats", "M.statistics_over_repeats->D.accept_acceptance_limited_gap", "M.statistics_over_repeats->D.accept_hardware_specificity_gap", "M.statistics_over_repeats->D.accept_quality_guard_gap", "M.statistics_over_repeats->D.accept_reproducibility_gap", "root.author_experiment_loop->H.hardware_specificity_gap", "root.author_experiment_loop->H.quality_guard_gap", "root.author_experiment_loop->H.reproducibility_gap"], "selected_nodes": ["A.appendix_artifacts", "C.central_claim", "C.paper_sections_to_write", "D.accept_acceptance_limited_gap", "D.accept_hardware_specificity_gap", "D.accept_quality_guard_gap", "D.accept_reproducibility_gap", "D.reject_exact_reproduction_claim", "D.revise_overbroad_speed_claim", "E.gpu_campaign_baselines", "E.kv_cache_stress", "E.quantization_kernel_stress", "E.repo_exact_rerun_audit", "E.sampling_mass_entropy_stress", "E.sparse_kernel_stress", "E.speculative_acceptance_stress", "E.token_merge_stress", "H.hardware_specificity_gap", "H.quality_guard_gap", "H.reproducibility_gap", "M.raw_measurement_read", "M.statistics_over_repeats", "root.author_experiment_loop"], "signature": "b091c2ff3b463d01", "stable_signature_count": 18}
{"edge_count": 32, "loop": 21, "mean_support": 1.0, "node_count": 23, "score": 1.008145, "selected_edges": ["C.central_claim->C.paper_sections_to_write", "C.paper_sections_to_write->A.appendix_artifacts", "D.accept_acceptance_limited_gap->D.revise_overbroad_speed_claim", "D.accept_hardware_specificity_gap->D.revise_overbroad_speed_claim", "D.accept_quality_guard_gap->D.revise_overbroad_speed_claim", "D.accept_reproducibility_gap->D.reject_exact_reproduction_claim", "D.reject_exact_reproduction_claim->C.central_claim", "D.revise_overbroad_speed_claim->C.central_claim", "E.gpu_campaign_baselines->E.kv_cache_stress", "E.gpu_campaign_baselines->E.quantization_kernel_stress", "E.gpu_campaign_baselines->E.sampling_mass_entropy_stress", "E.gpu_campaign_baselines->E.sparse_kernel_stress", "E.gpu_campaign_baselines->E.speculative_acceptance_stress", "E.gpu_campaign_baselines->E.token_merge_stress", "E.kv_cache_stress->M.raw_measurement_read", "E.quantization_kernel_stress->M.raw_measurement_read", "E.repo_exact_rerun_audit->M.raw_measurement_read", "E.sampling_mass_entropy_stress->M.raw_measurement_read", "E.sparse_kernel_stress->M.raw_measurement_read", "E.speculative_acceptance_stress->M.raw_measurement_read", "E.token_merge_stress->M.raw_measurement_read", "H.hardware_specificity_gap->E.gpu_campaign_baselines", "H.quality_guard_gap->E.gpu_campaign_baselines", "H.reproducibility_gap->E.repo_exact_rerun_audit", "M.raw_measurement_read->M.statistics_over_repeats", "M.statistics_over_repeats->D.accept_acceptance_limited_gap", "M.statistics_over_repeats->D.accept_hardware_specificity_gap", "M.statistics_over_repeats->D.accept_quality_guard_gap", "M.statistics_over_repeats->D.accept_reproducibility_gap", "root.author_experiment_loop->H.hardware_specificity_gap", "root.author_experiment_loop->H.quality_guard_gap", "root.author_experiment_loop->H.reproducibility_gap"], "selected_nodes": ["A.appendix_artifacts", "C.central_claim", "C.paper_sections_to_write", "D.accept_acceptance_limited_gap", "D.accept_hardware_specificity_gap", "D.accept_quality_guard_gap", "D.accept_reproducibility_gap", "D.reject_exact_reproduction_claim", "D.revise_overbroad_speed_claim", "E.gpu_campaign_baselines", "E.kv_cache_stress", "E.quantization_kernel_stress", "E.repo_exact_rerun_audit", "E.sampling_mass_entropy_stress", "E.sparse_kernel_stress", "E.speculative_acceptance_stress", "E.token_merge_stress", "H.hardware_specificity_gap", "H.quality_guard_gap", "H.reproducibility_gap", "M.raw_measurement_read", "M.statistics_over_repeats", "root.author_experiment_loop"], "signature": "b091c2ff3b463d01", "stable_signature_count": 19}
{"edge_count": 32, "loop": 22, "mean_support": 1.0, "node_count": 23, "score": 1.00955, "selected_edges": ["C.central_claim->C.paper_sections_to_write", "C.paper_sections_to_write->A.appendix_artifacts", "D.accept_acceptance_limited_gap->D.revise_overbroad_speed_claim", "D.accept_hardware_specificity_gap->D.revise_overbroad_speed_claim", "D.accept_quality_guard_gap->D.revise_overbroad_speed_claim", "D.accept_reproducibility_gap->D.reject_exact_reproduction_claim", "D.reject_exact_reproduction_claim->C.central_claim", "D.revise_overbroad_speed_claim->C.central_claim", "E.gpu_campaign_baselines->E.kv_cache_stress", "E.gpu_campaign_baselines->E.quantization_kernel_stress", "E.gpu_campaign_baselines->E.sampling_mass_entropy_stress", "E.gpu_campaign_baselines->E.sparse_kernel_stress", "E.gpu_campaign_baselines->E.speculative_acceptance_stress", "E.gpu_campaign_baselines->E.token_merge_stress", "E.kv_cache_stress->M.raw_measurement_read", "E.quantization_kernel_stress->M.raw_measurement_read", "E.repo_exact_rerun_audit->M.raw_measurement_read", "E.sampling_mass_entropy_stress->M.raw_measurement_read", "E.sparse_kernel_stress->M.raw_measurement_read", "E.speculative_acceptance_stress->M.raw_measurement_read", "E.token_merge_stress->M.raw_measurement_read", "H.hardware_specificity_gap->E.gpu_campaign_baselines", "H.quality_guard_gap->E.gpu_campaign_baselines", "H.reproducibility_gap->E.repo_exact_rerun_audit", "M.raw_measurement_read->M.statistics_over_repeats", "M.statistics_over_repeats->D.accept_acceptance_limited_gap", "M.statistics_over_repeats->D.accept_hardware_specificity_gap", "M.statistics_over_repeats->D.accept_quality_guard_gap", "M.statistics_over_repeats->D.accept_reproducibility_gap", "root.author_experiment_loop->H.hardware_specificity_gap", "root.author_experiment_loop->H.quality_guard_gap", "root.author_experiment_loop->H.reproducibility_gap"], "selected_nodes": ["A.appendix_artifacts", "C.central_claim", "C.paper_sections_to_write", "D.accept_acceptance_limited_gap", "D.accept_hardware_specificity_gap", "D.accept_quality_guard_gap", "D.accept_reproducibility_gap", "D.reject_exact_reproduction_claim", "D.revise_overbroad_speed_claim", "E.gpu_campaign_baselines", "E.kv_cache_stress", "E.quantization_kernel_stress", "E.repo_exact_rerun_audit", "E.sampling_mass_entropy_stress", "E.sparse_kernel_stress", "E.speculative_acceptance_stress", "E.token_merge_stress", "H.hardware_specificity_gap", "H.quality_guard_gap", "H.reproducibility_gap", "M.raw_measurement_read", "M.statistics_over_repeats", "root.author_experiment_loop"], "signature": "b091c2ff3b463d01", "stable_signature_count": 20}
{"edge_count": 32, "loop": 23, "mean_support": 1.0, "node_count": 23, "score": 1.008066, "selected_edges": ["C.central_claim->C.paper_sections_to_write", "C.paper_sections_to_write->A.appendix_artifacts", "D.accept_acceptance_limited_gap->D.revise_overbroad_speed_claim", "D.accept_hardware_specificity_gap->D.revise_overbroad_speed_claim", "D.accept_quality_guard_gap->D.revise_overbroad_speed_claim", "D.accept_reproducibility_gap->D.reject_exact_reproduction_claim", "D.reject_exact_reproduction_claim->C.central_claim", "D.revise_overbroad_speed_claim->C.central_claim", "E.gpu_campaign_baselines->E.kv_cache_stress", "E.gpu_campaign_baselines->E.quantization_kernel_stress", "E.gpu_campaign_baselines->E.sampling_mass_entropy_stress", "E.gpu_campaign_baselines->E.sparse_kernel_stress", "E.gpu_campaign_baselines->E.speculative_acceptance_stress", "E.gpu_campaign_baselines->E.token_merge_stress", "E.kv_cache_stress->M.raw_measurement_read", "E.quantization_kernel_stress->M.raw_measurement_read", "E.repo_exact_rerun_audit->M.raw_measurement_read", "E.sampling_mass_entropy_stress->M.raw_measurement_read", "E.sparse_kernel_stress->M.raw_measurement_read", "E.speculative_acceptance_stress->M.raw_measurement_read", "E.token_merge_stress->M.raw_measurement_read", "H.hardware_specificity_gap->E.gpu_campaign_baselines", "H.quality_guard_gap->E.gpu_campaign_baselines", "H.reproducibility_gap->E.repo_exact_rerun_audit", "M.raw_measurement_read->M.statistics_over_repeats", "M.statistics_over_repeats->D.accept_acceptance_limited_gap", "M.statistics_over_repeats->D.accept_hardware_specificity_gap", "M.statistics_over_repeats->D.accept_quality_guard_gap", "M.statistics_over_repeats->D.accept_reproducibility_gap", "root.author_experiment_loop->H.hardware_specificity_gap", "root.author_experiment_loop->H.quality_guard_gap", "root.author_experiment_loop->H.reproducibility_gap"], "selected_nodes": ["A.appendix_artifacts", "C.central_claim", "C.paper_sections_to_write", "D.accept_acceptance_limited_gap", "D.accept_hardware_specificity_gap", "D.accept_quality_guard_gap", "D.accept_reproducibility_gap", "D.reject_exact_reproduction_claim", "D.revise_overbroad_speed_claim", "E.gpu_campaign_baselines", "E.kv_cache_stress", "E.quantization_kernel_stress", "E.repo_exact_rerun_audit", "E.sampling_mass_entropy_stress", "E.sparse_kernel_stress", "E.speculative_acceptance_stress", "E.token_merge_stress", "H.hardware_specificity_gap", "H.quality_guard_gap", "H.reproducibility_gap", "M.raw_measurement_read", "M.statistics_over_repeats", "root.author_experiment_loop"], "signature": "b091c2ff3b463d01", "stable_signature_count": 21}
{"edge_count": 32, "loop": 24, "mean_support": 1.0, "node_count": 23, "score": 1.008011, "selected_edges": ["C.central_claim->C.paper_sections_to_write", "C.paper_sections_to_write->A.appendix_artifacts", "D.accept_acceptance_limited_gap->D.revise_overbroad_speed_claim", "D.accept_hardware_specificity_gap->D.revise_overbroad_speed_claim", "D.accept_quality_guard_gap->D.revise_overbroad_speed_claim", "D.accept_reproducibility_gap->D.reject_exact_reproduction_claim", "D.reject_exact_reproduction_claim->C.central_claim", "D.revise_overbroad_speed_claim->C.central_claim", "E.gpu_campaign_baselines->E.kv_cache_stress", "E.gpu_campaign_baselines->E.quantization_kernel_stress", "E.gpu_campaign_baselines->E.sampling_mass_entropy_stress", "E.gpu_campaign_baselines->E.sparse_kernel_stress", "E.gpu_campaign_baselines->E.speculative_acceptance_stress", "E.gpu_campaign_baselines->E.token_merge_stress", "E.kv_cache_stress->M.raw_measurement_read", "E.quantization_kernel_stress->M.raw_measurement_read", "E.repo_exact_rerun_audit->M.raw_measurement_read", "E.sampling_mass_entropy_stress->M.raw_measurement_read", "E.sparse_kernel_stress->M.raw_measurement_read", "E.speculative_acceptance_stress->M.raw_measurement_read", "E.token_merge_stress->M.raw_measurement_read", "H.hardware_specificity_gap->E.gpu_campaign_baselines", "H.quality_guard_gap->E.gpu_campaign_baselines", "H.reproducibility_gap->E.repo_exact_rerun_audit", "M.raw_measurement_read->M.statistics_over_repeats", "M.statistics_over_repeats->D.accept_acceptance_limited_gap", "M.statistics_over_repeats->D.accept_hardware_specificity_gap", "M.statistics_over_repeats->D.accept_quality_guard_gap", "M.statistics_over_repeats->D.accept_reproducibility_gap", "root.author_experiment_loop->H.hardware_specificity_gap", "root.author_experiment_loop->H.quality_guard_gap", "root.author_experiment_loop->H.reproducibility_gap"], "selected_nodes": ["A.appendix_artifacts", "C.central_claim", "C.paper_sections_to_write", "D.accept_acceptance_limited_gap", "D.accept_hardware_specificity_gap", "D.accept_quality_guard_gap", "D.accept_reproducibility_gap", "D.reject_exact_reproduction_claim", "D.revise_overbroad_speed_claim", "E.gpu_campaign_baselines", "E.kv_cache_stress", "E.quantization_kernel_stress", "E.repo_exact_rerun_audit", "E.sampling_mass_entropy_stress", "E.sparse_kernel_stress", "E.speculative_acceptance_stress", "E.token_merge_stress", "H.hardware_specificity_gap", "H.quality_guard_gap", "H.reproducibility_gap", "M.raw_measurement_read", "M.statistics_over_repeats", "root.author_experiment_loop"], "signature": "b091c2ff3b463d01", "stable_signature_count": 22}
```
