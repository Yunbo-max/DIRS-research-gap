# Held-Out Abstract Test Card

Date: `2026-07-20`

Chip: `/tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/chips/ICLR2026_HwCvaJOiCj_mamba3.chip.json`

Paper: `Mamba-3: Improved Sequence Modeling using State Space Principles`

Domain: `LLM Architecture / Attention / State Space Models`

## Blind Inputs Allowed

```text
chip facts
domain node/edge support priors
style_profile.json
case1_writing/01_abstract_writing.md
```

## Hidden Until After Generation

```yaml
original_abstract_source_path: /tf/notebooks/iclr2026_oral_paper_memory_fresh_248h/text/ICLR2026_HwCvaJOiCj_openreview.txt
original_abstract_sha256: 5d4f7e0577cf0d38693cba49579ba45e1afeb5c7f10e40057f129061266dac0f
original_abstract_word_count: 234
```

## Target Length From Training Priors

```yaml
target_words: 438
target_band: [403, 473]
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
