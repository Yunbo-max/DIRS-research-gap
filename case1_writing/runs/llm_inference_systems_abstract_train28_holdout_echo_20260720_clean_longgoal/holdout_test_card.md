# Held-Out Abstract Test Card

Date: `2026-07-20`

Chip: `/tf/notebooks/icml2026_oral_paper_memory_fresh_24h/chips/ICML2026_71057_echo_elastic_speculative_decoding.chip.json`

Paper: `ECHO: Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios`

Domain: `LLM Inference / Systems / Token Efficiency`

## Blind Inputs Allowed

```text
chip facts
domain node/edge support priors
style_profile.json
case1_writing/01_abstract_writing.md
```

## Hidden Until After Generation

```yaml
private_file: holdout_private_after_generation.json
rule: do_not_open_until_after_blind_generation_and_verification
```

## Target Length From Training Priors

```yaml
target_words: 164
target_band: [128, 198]
```

## Test Procedure

```text
1. Read the chip only.
2. Infer paper signature.
3. Select connected abstract sub-DAG using node/edge support and chip compatibility.
4. Generate abstract without reading hidden original.
5. Verify evidence support, order, target length, and scope.
6. Reveal original only for post-generation comparison.
```
