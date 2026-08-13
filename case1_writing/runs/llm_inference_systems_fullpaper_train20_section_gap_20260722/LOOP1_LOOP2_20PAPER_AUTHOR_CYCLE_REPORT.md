# Loop2 -> Loop1 -> Loop2 20-Paper Author Cycle

Date: `2026-07-22T21:02:36Z`
Domain: `LLM Inference / Systems / Token Efficiency`
Papers: `20`
Simulation definition: `author-side Loop 2 over each paper, Loop 1 prior learning, final author-side Loop 2 convergence`
Private holdout read: `false`
Paid/external API invoked: `false`

## Cycle

```text
Loop 2 per paper: author hypotheses -> experiments -> measurements -> decisions
Loop 1: learn reusable author-decision DAG prior from 20 Loop 2 traces
Loop 2 final: rerun author decision search with the learned prior until stable
```

## Measurement Memory

- GPU rows: `2130`
- Repeated seeds: `30`
- Families: `kv_cache_locality, quantization_compression, sampling_truncation, sparse_kernel_efficiency, speculative_decoding_proxy, token_merging`
- KV max RMSE: `0.121248`
- Token merge max RMSE: `0.553301`
- Speculative min speedup: `0.0`
- Sampling min retained mass: `0.029347`
- Sparse not-faster controls: `48/270`

## Loop 2 Per Paper

- `CVPR2026_103_atoken_unified_visual_tokenizer`: converged `true`, loops `24`, nodes `24`, families `kv_cache_locality, quantization_compression, sampling_truncation, sparse_kernel_efficiency, speculative_decoding_proxy, token_merging`
- `CVPR2026_067_rdvq_differentiable_vq_rate_distortion`: converged `true`, loops `24`, nodes `19`, families `quantization_compression, sparse_kernel_efficiency, token_merging`
- `CVPR2026_065_trellis2_native_compact_structured_latents`: converged `true`, loops `24`, nodes `24`, families `kv_cache_locality, quantization_compression, sampling_truncation, sparse_kernel_efficiency, speculative_decoding_proxy, token_merging`
- `CVPR2026_016_nuwa_class_specific_vit_pruning`: converged `true`, loops `24`, nodes `19`, families `kv_cache_locality, quantization_compression, sparse_kernel_efficiency`
- `CVPR2026_052_seacache_spectral_evolution_cache`: converged `true`, loops `24`, nodes `21`, families `kv_cache_locality, quantization_compression, sampling_truncation, sparse_kernel_efficiency, token_merging`
- `CVPR2026_053_sencache_sensitivity_aware_caching`: converged `true`, loops `24`, nodes `16`, families `kv_cache_locality, token_merging`
- `ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding`: converged `true`, loops `24`, nodes `23`, families `kv_cache_locality, sampling_truncation, sparse_kernel_efficiency, speculative_decoding_proxy, token_merging`
- `ICLR2026_QMItTyQW92_dto_kd_dynamic_tradeoff_distillation`: converged `true`, loops `24`, nodes `18`, families `sparse_kernel_efficiency, token_merging`
- `ICLR2026_H6rDX4w6Al_flashvid_vllm_token_merging`: converged `true`, loops `24`, nodes `20`, families `kv_cache_locality, quantization_compression, sparse_kernel_efficiency, token_merging`
- `ICLR2026_P5B97gZwRb_hyperparameter_trajectory_inference_clot`: converged `true`, loops `24`, nodes `20`, families `kv_cache_locality, quantization_compression, sampling_truncation, sparse_kernel_efficiency`
- `ICLR2026_JEYWpFGzvn_infotok_adaptive_video_tokenizer`: converged `true`, loops `24`, nodes `23`, families `kv_cache_locality, quantization_compression, sampling_truncation, speculative_decoding_proxy, token_merging`
- `ICLR2026_h06l9w1clt_locality_parallel_decoding_ar_image`: converged `true`, loops `24`, nodes `23`, families `kv_cache_locality, quantization_compression, sampling_truncation, speculative_decoding_proxy, token_merging`
- `ICLR2026_o29E01Q6bv_loongrl_long_context_reasoning`: converged `true`, loops `24`, nodes `19`, families `kv_cache_locality, speculative_decoding_proxy, token_merging`
- `ICLR2026_VdLEaGPYWT_sparserl_sparse_cuda_rl`: converged `true`, loops `24`, nodes `19`, families `sampling_truncation, sparse_kernel_efficiency, token_merging`
- `ICLR2026_1J63FJYJKg_mrrope_mixed_radix_rope`: converged `true`, loops `24`, nodes `19`, families `kv_cache_locality, quantization_compression, token_merging`
- `ICLR2026_LaVrNaBNwM_hsd_lossless_speculative_decoding`: converged `true`, loops `24`, nodes `22`, families `quantization_compression, sampling_truncation, speculative_decoding_proxy, token_merging`
- `ICLR2026_ItFuNJQGH4_p_less_sampling`: converged `true`, loops `24`, nodes `22`, families `sampling_truncation, sparse_kernel_efficiency, speculative_decoding_proxy, token_merging`
- `ICLR2026_88ZLp7xYxw_prism_fmri_structured_text`: converged `true`, loops `24`, nodes `21`, families `sampling_truncation, sparse_kernel_efficiency, speculative_decoding_proxy`
- `ICLR2026_EQhUvWH78U_rational_information_seeking_agents`: converged `true`, loops `24`, nodes `22`, families `quantization_compression, sampling_truncation, speculative_decoding_proxy, token_merging`
- `CVPR2026_030_spark_vlm_articulated_reconstruction`: converged `true`, loops `24`, nodes `18`, families `sparse_kernel_efficiency, token_merging`

## Loop 1 Learned Prior

- Core nodes: `17`
- Selective nodes: `7`
- Rare nodes: `0`

Core nodes:

- `A.raw_artifacts` support `1.00`: Attach traces, measurements, and blocked-rerun evidence.
- `C.paper_specific_conclusion` support `1.00`: Write the paper-specific bounded conclusion.
- `C.section_plan` support `1.00`: Route decisions into experiments, results, limitations, and appendix.
- `D.accept_hardware_specificity_gap` support `0.90`: Author accepts hardware-specificity gap.
- `D.accept_quality_guard_gap` support `1.00`: Author accepts quality/correctness guard gap.
- `D.accept_reproducibility_gap` support `1.00`: Author accepts reproducibility gap.
- `D.reject_exact_reproduction_claim` support `1.00`: Author rejects exact-reproduction wording when blocked.
- `D.revise_overbroad_speed_claim` support `1.00`: Author softens universal speedup claims.
- `E.gpu_campaign_baselines` support `1.00`: Use measured GPU baselines and controls.
- `E.repo_exact_rerun_audit` support `1.00`: Audit code, checkpoints, datasets, APIs, and exact-rerun readiness.
- `E.token_merge_stress` support `0.85`: Inspect token-merging stress evidence.
- `H.hardware_specificity_gap` support `0.90`: Hypothesize that speedups depend on hardware-aware implementation.
- `H.quality_guard_gap` support `1.00`: Hypothesize that efficiency must preserve quality/correctness.
- `H.reproducibility_gap` support `1.00`: Hypothesize that exact systems claims require runnable artifacts.
- `M.raw_measurement_read` support `1.00`: Read raw measured rows rather than only paper prose.
- `M.statistics_over_repeats` support `1.00`: Aggregate repeated-seed statistics.
- `root.author_experiment_loop` support `1.00`: Author-side experimental decision process.

Selective nodes:

- `D.accept_acceptance_limited_gap` support `0.50`: Author accepts acceptance-limited speculative gap.
- `E.kv_cache_stress` support `0.60`: Inspect KV/cache long-context stress evidence.
- `E.quantization_kernel_stress` support `0.60`: Inspect quantization/compression kernel evidence.
- `E.sampling_mass_entropy_stress` support `0.60`: Inspect sampling mass/entropy evidence.
- `E.sparse_kernel_stress` support `0.65`: Inspect sparse-kernel evidence.
- `E.speculative_acceptance_stress` support `0.50`: Inspect speculative decoding acceptance evidence.
- `H.acceptance_limited_gap` support `0.50`: Hypothesize that speculative speedups are acceptance-limited.

## Final Loop 2 After Loop 1

- Converged: `true`
- Completed loops: `24`
- Final score: `0.96873`
- Nodes: `24`
- Edges: `34`

Final selected nodes:

- `A.raw_artifacts`
- `C.paper_specific_conclusion`
- `C.section_plan`
- `D.accept_acceptance_limited_gap`
- `D.accept_hardware_specificity_gap`
- `D.accept_quality_guard_gap`
- `D.accept_reproducibility_gap`
- `D.reject_exact_reproduction_claim`
- `D.revise_overbroad_speed_claim`
- `E.gpu_campaign_baselines`
- `E.kv_cache_stress`
- `E.quantization_kernel_stress`
- `E.repo_exact_rerun_audit`
- `E.sampling_mass_entropy_stress`
- `E.sparse_kernel_stress`
- `E.speculative_acceptance_stress`
- `E.token_merge_stress`
- `H.acceptance_limited_gap`
- `H.hardware_specificity_gap`
- `H.quality_guard_gap`
- `H.reproducibility_gap`
- `M.raw_measurement_read`
- `M.statistics_over_repeats`
- `root.author_experiment_loop`

## Author Decisions

- Loop 2 accepted reproducibility-gap decisions for 20/20 papers.
- Loop 2 accepted quality-guard decisions for 20/20 papers.
- Loop 2 accepted speculative acceptance-limit decisions for 10/20 papers.
- Loop 2 accepted hardware-specificity decisions for 18/20 papers.
- Loop 1 therefore treats raw measurements, repeated statistics, claim revision, and appendix artifacts as core author-DAG nodes.
- Final Loop 2 conclusion must stay bounded because exact rerun readiness and measured tradeoff failures dominate the evidence.

## Artifacts

- JSON: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_fullpaper_train20_section_gap_20260722/loop1_loop2_20paper_author_cycle.json`
- Trace directory: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_fullpaper_train20_section_gap_20260722/loop2_20paper_author_traces`
