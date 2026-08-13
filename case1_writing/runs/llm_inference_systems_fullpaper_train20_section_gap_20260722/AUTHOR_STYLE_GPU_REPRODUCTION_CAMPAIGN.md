# Author-Style GPU Reproduction Campaign

Date: `2026-07-22T20:27:30Z`
Domain: `LLM Inference / Systems / Token Efficiency`
Papers audited: `20`
Campaign: `author_style_gpu_reproduction_and_gap_search`
Device: `cuda:1`
Runtime: `22.096s`
Private holdout read: `false`
Paid/external API invoked: `false`

## Device

```json
{
  "chosen": {
    "compute_capability": "8.9",
    "driver_version": "535.183.01",
    "index": 1,
    "memory_total_mib": 24564,
    "memory_used_mib": 21,
    "name": "NVIDIA GeForce RTX 4090"
  },
  "cuda_available": true,
  "gpus": [
    {
      "compute_capability": "8.9",
      "driver_version": "535.183.01",
      "index": 0,
      "memory_total_mib": 24564,
      "memory_used_mib": 19145,
      "name": "NVIDIA GeForce RTX 4090"
    },
    {
      "compute_capability": "8.9",
      "driver_version": "535.183.01",
      "index": 1,
      "memory_total_mib": 24564,
      "memory_used_mib": 21,
      "name": "NVIDIA GeForce RTX 4090"
    },
    {
      "compute_capability": "8.9",
      "driver_version": "535.183.01",
      "index": 2,
      "memory_total_mib": 24564,
      "memory_used_mib": 589,
      "name": "NVIDIA GeForce RTX 4090"
    },
    {
      "compute_capability": "8.9",
      "driver_version": "535.183.01",
      "index": 3,
      "memory_total_mib": 24564,
      "memory_used_mib": 2957,
      "name": "NVIDIA GeForce RTX 4090"
    }
  ]
}
```

## Repo And Exact-Rerun Audit

- `CVPR2026_103_atoken_unified_visual_tokenizer`: repos `2`, syntax-ready `2`, status `blocked`
- `CVPR2026_067_rdvq_differentiable_vq_rate_distortion`: repos `1`, syntax-ready `1`, status `blocked`
- `CVPR2026_065_trellis2_native_compact_structured_latents`: repos `0`, syntax-ready `0`, status `blocked`
- `CVPR2026_016_nuwa_class_specific_vit_pruning`: repos `0`, syntax-ready `0`, status `blocked`
- `CVPR2026_052_seacache_spectral_evolution_cache`: repos `1`, syntax-ready `1`, status `blocked`
- `CVPR2026_053_sencache_sensitivity_aware_caching`: repos `1`, syntax-ready `1`, status `blocked`
- `ICLR2026_g88nt4ieTG_prophet_dlm_early_commit_decoding`: repos `1`, syntax-ready `1`, status `code_ready_needs_model_data`
- `ICLR2026_QMItTyQW92_dto_kd_dynamic_tradeoff_distillation`: repos `0`, syntax-ready `0`, status `paper_only_or_source_missing`
- `ICLR2026_H6rDX4w6Al_flashvid_vllm_token_merging`: repos `1`, syntax-ready `1`, status `code_ready_needs_model_data`
- `ICLR2026_P5B97gZwRb_hyperparameter_trajectory_inference_clot`: repos `0`, syntax-ready `0`, status `blocked`
- `ICLR2026_JEYWpFGzvn_infotok_adaptive_video_tokenizer`: repos `0`, syntax-ready `0`, status `blocked`
- `ICLR2026_h06l9w1clt_locality_parallel_decoding_ar_image`: repos `0`, syntax-ready `0`, status `blocked`
- `ICLR2026_o29E01Q6bv_loongrl_long_context_reasoning`: repos `1`, syntax-ready `1`, status `code_ready_needs_model_data`
- `ICLR2026_VdLEaGPYWT_sparserl_sparse_cuda_rl`: repos `1`, syntax-ready `1`, status `code_ready_needs_model_data`
- `ICLR2026_1J63FJYJKg_mrrope_mixed_radix_rope`: repos `1`, syntax-ready `0`, status `paper_only_or_source_missing`
- `ICLR2026_LaVrNaBNwM_hsd_lossless_speculative_decoding`: repos `0`, syntax-ready `0`, status `blocked`
- `ICLR2026_ItFuNJQGH4_p_less_sampling`: repos `1`, syntax-ready `1`, status `blocked`
- `ICLR2026_88ZLp7xYxw_prism_fmri_structured_text`: repos `0`, syntax-ready `0`, status `blocked`
- `ICLR2026_EQhUvWH78U_rational_information_seeking_agents`: repos `0`, syntax-ready `0`, status `blocked`
- `CVPR2026_030_spark_vlm_articulated_reconstruction`: repos `0`, syntax-ready `0`, status `paper_only_or_source_missing`

## Experiment Families

### kv_cache_locality

Rows: `480`

- n `30`: method=dense_full_kv, context_tokens=16384, ms_mean=0.128457, ms_std=0.001173, items_per_second_mean=7785.375, items_per_second_std=70.151864, quality_error_rmse_vs_dense_mean=0.0, quality_error_rmse_vs_dense_std=0.0
- n `30`: method=dense_full_kv, context_tokens=2048, ms_mean=0.12863, ms_std=0.001184, items_per_second_mean=7774.822667, items_per_second_std=70.241795, quality_error_rmse_vs_dense_mean=0.0, quality_error_rmse_vs_dense_std=0.0
- n `30`: method=dense_full_kv, context_tokens=512, ms_mean=0.128107, ms_std=0.022772, items_per_second_mean=7937.132333, items_per_second_std=736.199747, quality_error_rmse_vs_dense_mean=0.0, quality_error_rmse_vs_dense_std=0.0
- n `30`: method=dense_full_kv, context_tokens=8192, ms_mean=0.128577, ms_std=0.001305, items_per_second_mean=7778.283333, items_per_second_std=77.830799, quality_error_rmse_vs_dense_mean=0.0, quality_error_rmse_vs_dense_std=0.0
- n `30`: method=local_window_1024, context_tokens=16384, ms_mean=0.14406, ms_std=0.001236, items_per_second_mean=6941.584667, items_per_second_std=58.474696, quality_error_rmse_vs_dense_mean=0.049999, quality_error_rmse_vs_dense_std=0.002014
- n `30`: method=local_window_1024, context_tokens=2048, ms_mean=0.144463, ms_std=0.001026, items_per_second_mean=6922.22, items_per_second_std=48.877362, quality_error_rmse_vs_dense_mean=0.036506, quality_error_rmse_vs_dense_std=0.001361
- n `30`: method=local_window_1024, context_tokens=512, ms_mean=0.14076, ms_std=0.000814, items_per_second_mean=7104.309667, items_per_second_std=41.066577, quality_error_rmse_vs_dense_mean=0.0, quality_error_rmse_vs_dense_std=0.0
- n `30`: method=local_window_1024, context_tokens=8192, ms_mean=0.144673, ms_std=0.001585, items_per_second_mean=6912.988333, items_per_second_std=74.518731, quality_error_rmse_vs_dense_mean=0.04844, quality_error_rmse_vs_dense_std=0.001807
- n `30`: method=local_window_256, context_tokens=16384, ms_mean=0.143007, ms_std=0.001117, items_per_second_mean=6993.121, items_per_second_std=53.322228, quality_error_rmse_vs_dense_mean=0.102012, quality_error_rmse_vs_dense_std=0.004729
- n `30`: method=local_window_256, context_tokens=2048, ms_mean=0.14317, ms_std=0.001169, items_per_second_mean=6985.211333, items_per_second_std=56.053114, quality_error_rmse_vs_dense_mean=0.094728, quality_error_rmse_vs_dense_std=0.004796
- n `30`: method=local_window_256, context_tokens=512, ms_mean=0.143757, ms_std=0.001345, items_per_second_mean=6956.716, items_per_second_std=64.257459, quality_error_rmse_vs_dense_mean=0.072497, quality_error_rmse_vs_dense_std=0.002597
- n `30`: method=local_window_256, context_tokens=8192, ms_mean=0.143097, ms_std=0.001161, items_per_second_mean=6988.568, items_per_second_std=55.755458, quality_error_rmse_vs_dense_mean=0.100933, quality_error_rmse_vs_dense_std=0.006191
- ... `4` more grouped rows in JSON

### token_merging

Rows: `630`

- n `30`: method=full_attention_baseline, tokens=1024, kept_tokens=1024, ms_mean=0.13403, ms_std=0.001856, items_per_second_mean=7462.252333, items_per_second_std=102.84336, quality_error_rmse_vs_full_mean=0.0, quality_error_rmse_vs_full_std=0.0
- n `30`: method=full_attention_baseline, tokens=4096, kept_tokens=4096, ms_mean=0.446963, ms_std=0.001072, items_per_second_mean=2237.328333, items_per_second_std=5.337263, quality_error_rmse_vs_full_mean=0.0, quality_error_rmse_vs_full_std=0.0
- n `30`: method=full_attention_baseline, tokens=8192, kept_tokens=8192, ms_mean=2.083067, ms_std=0.001999, items_per_second_mean=480.06, items_per_second_std=0.461844, quality_error_rmse_vs_full_mean=0.0, quality_error_rmse_vs_full_std=0.0
- n `30`: method=norm_topk_merge_proxy, tokens=1024, kept_tokens=256, ms_mean=0.127783, ms_std=0.001923, items_per_second_mean=7827.540667, items_per_second_std=116.586206, quality_error_rmse_vs_full_mean=0.399227, quality_error_rmse_vs_full_std=0.006509
- n `30`: method=norm_topk_merge_proxy, tokens=1024, kept_tokens=512, ms_mean=0.127847, ms_std=0.001794, items_per_second_mean=7823.682, items_per_second_std=108.830286, quality_error_rmse_vs_full_mean=0.248735, quality_error_rmse_vs_full_std=0.005524
- n `30`: method=norm_topk_merge_proxy, tokens=1024, kept_tokens=768, ms_mean=0.127953, ms_std=0.001933, items_per_second_mean=7817.165, items_per_second_std=117.486625, quality_error_rmse_vs_full_mean=0.117903, quality_error_rmse_vs_full_std=0.003213
- n `30`: method=norm_topk_merge_proxy, tokens=4096, kept_tokens=1024, ms_mean=0.13249, ms_std=0.001902, items_per_second_mean=7549.151667, items_per_second_std=106.271968, quality_error_rmse_vs_full_mean=0.213913, quality_error_rmse_vs_full_std=0.002571
- n `30`: method=norm_topk_merge_proxy, tokens=4096, kept_tokens=2048, ms_mean=0.173077, ms_std=0.000946, items_per_second_mean=5777.753, items_per_second_std=31.678768, quality_error_rmse_vs_full_mean=0.13029, quality_error_rmse_vs_full_std=0.001376
- n `30`: method=norm_topk_merge_proxy, tokens=4096, kept_tokens=3072, ms_mean=0.2829, ms_std=0.001203, items_per_second_mean=3534.876333, items_per_second_std=14.993487, quality_error_rmse_vs_full_mean=0.056942, quality_error_rmse_vs_full_std=0.000661
- n `30`: method=norm_topk_merge_proxy, tokens=8192, kept_tokens=2048, ms_mean=0.458467, ms_std=0.001008, items_per_second_mean=2181.166333, items_per_second_std=4.826493, quality_error_rmse_vs_full_mean=0.167643, quality_error_rmse_vs_full_std=0.000715
- n `30`: method=norm_topk_merge_proxy, tokens=8192, kept_tokens=4096, ms_mean=1.000527, ms_std=0.001344, items_per_second_mean=999.463, items_per_second_std=1.341184, quality_error_rmse_vs_full_mean=0.104974, quality_error_rmse_vs_full_std=0.000484
- n `30`: method=norm_topk_merge_proxy, tokens=8192, kept_tokens=6144, ms_mean=1.552063, ms_std=0.001465, items_per_second_mean=644.306667, items_per_second_std=0.608399, quality_error_rmse_vs_full_mean=0.044899, quality_error_rmse_vs_full_std=0.000222
- ... `9` more grouped rows in JSON

### speculative_decoding_proxy

Rows: `120`

- n `30`: draft_noise=0.03, token_accept_rate_mean=0.934115, token_accept_rate_std=0.021029, block_accept_rate_mean=0.752083, block_accept_rate_std=0.081063, effective_speedup_after_acceptance_mean=1.509517, effective_speedup_after_acceptance_std=0.162358
- n `30`: draft_noise=0.08, token_accept_rate_mean=0.815885, token_accept_rate_std=0.036958, block_accept_rate_mean=0.44375, block_accept_rate_std=0.097961, effective_speedup_after_acceptance_mean=0.890647, effective_speedup_after_acceptance_std=0.196411
- n `30`: draft_noise=0.16, token_accept_rate_mean=0.672656, token_accept_rate_std=0.047579, block_accept_rate_mean=0.210417, block_accept_rate_std=0.071231, effective_speedup_after_acceptance_mean=0.42324, effective_speedup_after_acceptance_std=0.143669
- n `30`: draft_noise=0.32, token_accept_rate_mean=0.435677, token_accept_rate_std=0.049603, block_accept_rate_mean=0.035417, block_accept_rate_std=0.026435, effective_speedup_after_acceptance_mean=0.071123, effective_speedup_after_acceptance_std=0.053116

### sampling_truncation

Rows: `180`

- n `30`: method=p_less_entropy_proxy, ms_mean=0.100743, ms_std=0.000708, kept_tokens_mean_mean=550.892233, kept_tokens_mean_std=46.562936, retained_probability_mass_mean=0.124113, retained_probability_mass_std=0.000193, entropy_delta_vs_full_mean=3.735272, entropy_delta_vs_full_std=0.000769
- n `30`: method=softmax_full, ms_mean=0.055363, ms_std=0.001453, kept_tokens_mean_mean=65536, kept_tokens_mean_std=0.0, retained_probability_mass_mean=1.0, retained_probability_mass_std=0.0, entropy_delta_vs_full_mean=0.0, entropy_delta_vs_full_std=0.0
- n `30`: method=top_k_128, ms_mean=0.092037, ms_std=0.001023, kept_tokens_mean_mean=128, kept_tokens_mean_std=0.0, retained_probability_mass_mean=0.029565, retained_probability_mass_std=0.000113, entropy_delta_vs_full_mean=5.788311, entropy_delta_vs_full_std=0.001715
- n `30`: method=top_k_512, ms_mean=0.1, ms_std=0.000898, kept_tokens_mean_mean=512, kept_tokens_mean_std=0.0, retained_probability_mass_mean=0.078062, retained_probability_mass_std=0.000152, entropy_delta_vs_full_mean=4.41696, entropy_delta_vs_full_std=0.001043
- n `30`: method=top_p_0.90, ms_mean=0.745313, ms_std=0.001363, kept_tokens_mean_mean=40033.403133, kept_tokens_mean_std=8.045656, retained_probability_mass_mean=0.899997, retained_probability_mass_std=1e-06, entropy_delta_vs_full_mean=0.300339, entropy_delta_vs_full_std=8.5e-05
- n `30`: method=top_p_0.95, ms_mean=0.745197, ms_std=0.001697, kept_tokens_mean_mean=48528.501033, kept_tokens_mean_std=6.507067, retained_probability_mass_mean=0.949998, retained_probability_mass_std=0.0, entropy_delta_vs_full_mean=0.15985, entropy_delta_vs_full_std=4.3e-05

### quantization_compression

Rows: `360`

- n `30`: method=fp16_baseline, bits=16, elements=1000000, ms_mean=0.052353, ms_std=0.000674, items_per_second_mean=19102813873.034668, items_per_second_std=238314442.312088, rmse_vs_fp16_mean=0.0, rmse_vs_fp16_std=0.0
- n `30`: method=fp16_baseline, bits=16, elements=16000000, ms_mean=0.23196, ms_std=0.000508, items_per_second_mean=68973999785.657, items_per_second_std=148887520.510509, rmse_vs_fp16_mean=0.0, rmse_vs_fp16_std=0.0
- n `30`: method=fp16_baseline, bits=16, elements=4000000, ms_mean=0.052347, ms_std=0.000491, items_per_second_mean=76422702345.79199, items_per_second_std=712655938.111845, rmse_vs_fp16_mean=0.0, rmse_vs_fp16_std=0.0
- n `30`: method=symmetric_quant_dequant, bits=4, elements=1000000, ms_mean=0.08062, ms_std=0.000761, items_per_second_mean=12404565541.952667, items_per_second_std=116144630.40429, rmse_vs_fp16_mean=0.206705, rmse_vs_fp16_std=0.00834
- n `30`: method=symmetric_quant_dequant, bits=4, elements=16000000, ms_mean=0.855503, ms_std=0.000633, items_per_second_mean=18702595625.399666, items_per_second_std=13929001.350459, rmse_vs_fp16_mean=0.226507, rmse_vs_fp16_std=0.009056
- n `30`: method=symmetric_quant_dequant, bits=4, elements=4000000, ms_mean=0.089743, ms_std=0.000546, items_per_second_mean=44573796459.874, items_per_second_std=266550784.780746, rmse_vs_fp16_mean=0.217583, rmse_vs_fp16_std=0.007574
- n `30`: method=symmetric_quant_dequant, bits=6, elements=1000000, ms_mean=0.08089, ms_std=0.000718, items_per_second_mean=12363219042.737667, items_per_second_std=109021358.91456, rmse_vs_fp16_mean=0.046678, rmse_vs_fp16_std=0.001878
- n `30`: method=symmetric_quant_dequant, bits=6, elements=16000000, ms_mean=0.856167, ms_std=0.000481, items_per_second_mean=18687874935.911335, items_per_second_std=10236961.168009, rmse_vs_fp16_mean=0.051149, rmse_vs_fp16_std=0.002044
- n `30`: method=symmetric_quant_dequant, bits=6, elements=4000000, ms_mean=0.089753, ms_std=0.000599, items_per_second_mean=44566950551.00467, items_per_second_std=292654500.972291, rmse_vs_fp16_mean=0.049132, rmse_vs_fp16_std=0.001712
- n `30`: method=symmetric_quant_dequant, bits=8, elements=1000000, ms_mean=0.081087, ms_std=0.000756, items_per_second_mean=12333014163.928667, items_per_second_std=113469985.091566, rmse_vs_fp16_mean=0.011394, rmse_vs_fp16_std=0.000459
- n `30`: method=symmetric_quant_dequant, bits=8, elements=16000000, ms_mean=0.855433, ms_std=0.000494, items_per_second_mean=18704014717.187, items_per_second_std=10813533.443878, rmse_vs_fp16_mean=0.012486, rmse_vs_fp16_std=0.000499
- n `30`: method=symmetric_quant_dequant, bits=8, elements=4000000, ms_mean=0.089683, ms_std=0.000443, items_per_second_mean=44604526669.15967, items_per_second_std=219648948.552478, rmse_vs_fp16_mean=0.011995, rmse_vs_fp16_std=0.000418

### sparse_kernel_efficiency

Rows: `360`

- n `30`: method=dense_matvec, density=1.0, ms_mean=0.047237, ms_std=0.000765, items_per_second_mean=21175.964333, items_per_second_std=341.986939, rmse_vs_dense_mean=0.0, rmse_vs_dense_std=0.0
- n `30`: method=dense_matvec, density=1.0, ms_mean=0.046817, ms_std=0.00068, items_per_second_mean=21367.269, items_per_second_std=309.596016, rmse_vs_dense_mean=0.0, rmse_vs_dense_std=0.0
- n `30`: method=dense_matvec, density=1.0, ms_mean=0.17263, ms_std=0.00103, items_per_second_mean=5792.969667, items_per_second_std=34.208277, rmse_vs_dense_mean=0.0, rmse_vs_dense_std=0.0
- n `30`: method=masked_dense_matvec_control, density=0.125, ms_mean=0.046433, ms_std=0.000638, items_per_second_mean=21541.887333, items_per_second_std=290.086479, rmse_vs_dense_mean=42.540162, rmse_vs_dense_std=0.88858
- n `30`: method=masked_dense_matvec_control, density=0.25, ms_mean=0.046457, ms_std=0.000653, items_per_second_mean=21528.941, items_per_second_std=296.19443, rmse_vs_dense_mean=39.401847, rmse_vs_dense_std=0.95855
- n `30`: method=masked_dense_matvec_control, density=0.5, ms_mean=0.04657, ms_std=0.000696, items_per_second_mean=21474.051667, items_per_second_std=313.238641, rmse_vs_dense_mean=32.112121, rmse_vs_dense_std=0.727559
- n `30`: method=masked_dense_matvec_control, density=0.125, ms_mean=0.046777, ms_std=0.000736, items_per_second_mean=21382.378667, items_per_second_std=340.229665, rmse_vs_dense_mean=59.816909, rmse_vs_dense_std=0.892583
- n `30`: method=masked_dense_matvec_control, density=0.25, ms_mean=0.04684, ms_std=0.000735, items_per_second_mean=21356.963, items_per_second_std=327.851769, rmse_vs_dense_mean=55.244277, rmse_vs_dense_std=0.922727
- n `30`: method=masked_dense_matvec_control, density=0.5, ms_mean=0.04671, ms_std=0.000664, items_per_second_mean=21412.884, items_per_second_std=299.325217, rmse_vs_dense_mean=45.130214, rmse_vs_dense_std=0.644608
- n `30`: method=masked_dense_matvec_control, density=0.125, ms_mean=0.151263, ms_std=0.000988, items_per_second_mean=6611.179, items_per_second_std=43.047728, rmse_vs_dense_mean=85.014283, rmse_vs_dense_std=0.892621
- n `30`: method=masked_dense_matvec_control, density=0.25, ms_mean=0.151593, ms_std=0.000585, items_per_second_mean=6596.658, items_per_second_std=25.698495, rmse_vs_dense_mean=78.663425, rmse_vs_dense_std=0.938193
- n `30`: method=masked_dense_matvec_control, density=0.5, ms_mean=0.15172, ms_std=0.000588, items_per_second_mean=6590.895, items_per_second_std=25.429687, rmse_vs_dense_mean=64.062527, rmse_vs_dense_std=0.765162

## Inferred Research Gaps

- Gap: Exact paper-result reruns are structurally blocked for many papers without checkpoint/data/API provenance.
  Evidence: 16/20 papers are not code-ready exact reruns under local constraints.
  Paper move: A strict systems paper should publish a runnable benchmark harness with fallback tiny fixtures, not only full-scale scripts.
- Gap: KV/cache locality speedups need quality guards, because local windows can diverge from full-context attention.
  Evidence: Max local-window RMSE vs dense was 0.121248 at context 8192.
  Paper move: Report speed together with semantic/correctness deltas under long-context stress.
- Gap: Token merging methods need instance-adaptive retention, not only fixed token budgets.
  Evidence: Max merge RMSE vs full attention was 0.553301 with 256 kept tokens.
  Paper move: Tie merge ratio to failure cases and preserve a task-quality metric next to throughput.
- Gap: Speculative decoding speedup is acceptance-limited, so draft quality is a first-class systems variable.
  Evidence: Lowest effective modeled speedup was 0.0 at draft noise 0.32.
  Paper move: Report acceptance distributions and not just raw draft/target latency.
- Gap: Sampling/token-pruning claims need probability-mass and entropy audits.
  Evidence: Lowest retained mass was 0.029347 for top_k_128.
  Paper move: Frame hyperparameter-free decoding as a constrained mass/entropy problem, not only a recipe.
- Gap: Unstructured sparsity is not automatically a kernel speedup.
  Evidence: 48 masked sparse controls failed to beat dense matvec despite lower density.
  Paper move: Require hardware-aware sparse layout/kernel measurements before claiming sparse efficiency.

## Strict NeurIPS DAG Delta

- gap hypothesis must be paired with a runnable stress test
- experiment section must name backend, command, seed, data fixture, metric, and blocked artifacts
- results must separate exact rerun, proxy rerun, code audit, and paper-only evidence
- gap claims should emerge from measured tradeoff failures, not just related-work language
- appendix must include logs and machine-readable artifacts for every table claim

## Artifacts

- JSON: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_fullpaper_train20_section_gap_20260722/author_style_gpu_reproduction_campaign.json`
- Markdown: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_fullpaper_train20_section_gap_20260722/AUTHOR_STYLE_GPU_REPRODUCTION_CAMPAIGN.md`
