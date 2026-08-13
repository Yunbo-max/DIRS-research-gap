---
license: mit
language:
- en
pretty_name: DLM-Decoding-Analysis
tags:
- diffusion-language-model
- llada
- decoding-trajectory
- gsm8k
- mmlu
- reasoning
- early-exit
size_categories:
- 10K<n<100K
---

# DLM-Decoding-Analysis

## Diffusion Language Model Knows the Answer Before It Decodes

Pengxiang Li*, Yefan Zhou*, Dilxat Muhtar, Lu Yin, Shilin Yan, Li Shen, Yi Liang, Soroush Vosoughi, Shiwei Liu

**The Fourteenth International Conference on Learning Representations (ICLR 2026)**

**TL;DR:** Diffusion language models often commit to the correct answer
well before they finish decoding. This dataset releases the per-question,
step-by-step decoding trajectories of **LLaDA-8B-Instruct** on **GSM8K**
and **MMLU-STEM** that we use to study this answer-emergence behaviour and
to design early-exit decoding strategies.

* 💻 **Github:** <https://github.com/pixeli99/Prophet>
* 📜 **Paper:** <https://arxiv.org/abs/2508.19982>

---

Decoding trajectories of the **LLaDA-8B-Instruct** diffusion language model
([GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct))
on **GSM8K** and **MMLU-STEM**, recorded step-by-step.

For every test question we save the full sequence-evolution tensor
`x0_history` (the model's denoised prediction at every decoding step), the
positions that get committed at each step (`true_indices_history`), and
metadata about the prompt, generation, and answer. The data lets researchers
study *when and where* the correct answer first emerges during the iterative
masked-prediction decoding of a diffusion LM, and analyse the effect of
remasking policy and answer-position constraints.

---

## Repository contents

The dataset has **8 folders**, organised as a 2 × 2 × 2 grid:

| Benchmark | Remasking policy | Constraint | Folder name |
|---|---|---|---|
| GSM8K | `low_confidence` | yes | `question_histories_low_conf_constraint_index_genlen_step256_blocklen32/` |
| GSM8K | `low_confidence` | no  | `question_histories_low_conf_none_index_genlen_step256_blocklen32/` |
| GSM8K | `random`         | yes | `question_histories_random_constraint_index_genlen_step256_blocklen256/` |
| GSM8K | `random`         | no  | `question_histories_random_none_index_genlen_step256_blocklen256/` |
| MMLU-STEM | `low_confidence` | yes | `question_histories_mmlu_low_confidence_constraint_index_genlen_step128_blocklen128/` |
| MMLU-STEM | `low_confidence` | no  | `question_histories_mmlu_low_confidence_none_index_genlen_step128_blocklen128/` |
| MMLU-STEM | `random`         | yes | `question_histories_mmlu_random_constraint_index_genlen_step128_blocklen128/` |
| MMLU-STEM | `random`         | no  | `question_histories_mmlu_random_none_index_genlen_step128_blocklen128/` |

Each folder contains one PyTorch checkpoint per question:

```
question_{idx:04d}_steps_{steps:03d}.pt
```

* GSM8K folders contain **1,319 files** (full GSM8K test split).
* MMLU-STEM folders contain **3,153 files**, covering the 19 STEM subjects
  defined in the original MMLU paper (`abstract_algebra`, `anatomy`,
  `astronomy`, `college_biology`, `college_chemistry`, `college_computer_science`,
  `college_mathematics`, `college_physics`, `computer_security`, `conceptual_physics`,
  `electrical_engineering`, `elementary_mathematics`, `high_school_biology`,
  `high_school_chemistry`, `high_school_computer_science`, `high_school_mathematics`,
  `high_school_physics`, `high_school_statistics`, `machine_learning`).

---

## Decoding configuration

All trajectories were generated with the LLaDA block-wise iterative
denoising procedure. Common settings:

| Field | GSM8K | MMLU-STEM |
|---|---|---|
| `gen_length` (= total decoding steps) | 256 | 128 |
| `block_length` (`low_confidence` runs) | 32  | 128 |
| `block_length` (`random` runs)         | 256 | 128 |
| `temperature` | 0.0 (deterministic) | 0.0 |
| `cfg_scale`   | 0.0 | 0.0 |

* **`decode_policy` (a.k.a. `remasking`)**:
  * `low_confidence` — at every step the lowest-confidence positions in the
    current block are kept masked and re-predicted next step.
  * `random` — positions to keep masked are chosen uniformly at random.
* **`constraint_policy`**:
  * `constraint` — the answer-introduction tokens are *forced* at fixed
    positions in the generated sequence so the answer always appears in a
    parseable location.
    * GSM8K constraint: `"Answer"` is forced to start at position **220**
      of the 256-token generation (corresponding to `CONSTRAINTS_TEXT = "220:Answer"`).
    * MMLU-STEM constraint: `"The answer is"` is forced at positions
      **120–122** of the 128-token generation (`CONSTRAINTS_TEXT = "120:The, 121:answer, 122:is"`).
  * `none` — no forced tokens; the model decides where to put the answer.

Prompts are CoT-style:

* **GSM8K**: *"Solve the following math problem step by step. The last line
  of your response should be of the form Answer: $ANSWER ..."*
* **MMLU-STEM**: standard 4-choice CoT prompt ending *"The last line of my
  response should be of the form 'The answer is [letter]' ..."*

---

## File format

Each `.pt` file is loaded with `torch.load(...)` and is a Python `dict` with
the following keys:

| Key | Type | Description |
|---|---|---|
| `x0_history` | `list[Tensor]` | One tensor per decoding **block**. Each tensor has shape `[steps_in_block, prompt_token_len + gen_length]` and stores the model's denoised `x0` prediction (token IDs) at every step within that block. Concatenating along `dim=0` yields the full `[total_steps, seq_length]` trajectory. |
| `true_indices_history` | `list[list[Tensor]]` | Outer list = block, inner list = step. Each tensor has shape `[N, 2]`; column `[:, 1]` is the set of positions whose token was *committed* (decoded into the final output) at that step. |
| `correct` | `bool` | Whether `pred_ans == gt_text`. |
| `pred_text` | `str` | Full decoded model output (after the prompt). |
| `pred_ans` | `str` | Extracted predicted answer (e.g. `"72"` for GSM8K, `"A"`/`"B"`/`"C"`/`"D"` for MMLU). |
| `gt_text` | `str` | Ground-truth answer in the same format as `pred_ans`. |
| `pred_token_id` | `list[int]` | Tokeniser IDs of `pred_ans`. |
| `gt_token_id`   | `list[int]` | Tokeniser IDs of `gt_text`. |
| `ans_posidx` | `int` | **Absolute** position of the predicted answer within the full sequence (i.e. `prompt_token_len + offset_in_generated_segment`). |
| `prompt_token_len` | `int` | Length of the tokenised prompt. |
| `gen_ids` | `list[int]` | Generated token IDs (after the prompt; length = `gen_length`). |
| `answer_pos_indices` | `list[int]` | *(MMLU only)* All positions in `gen_ids` where the predicted answer letter appears. |

Tokeniser: the LLaDA-8B-Instruct tokeniser (loaded via
`AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)`).

---

## Quick start

Download the dataset:

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="YefanZhou98/DLM-Decoding-Analysis",
    repo_type="dataset",
)
```

For trajectory-collection scripts, analysis code, and the figure-generation
notebook (`analysis/visualize.ipynb`) used to produce all paper plots from
these files, see the official Prophet repository:
**[github.com/pixeli99/Prophet](https://github.com/pixeli99/Prophet)**.

---

## Statistics

| Folder | # files | gen_length | block_length | total decoding steps |
|---|---:|---:|---:|---:|
| GSM8K (`low_conf`, both constraint settings) | 1,319 | 256 | 32  | 256 |
| GSM8K (`random`,  both constraint settings)  | 1,319 | 256 | 256 | 256 |
| MMLU-STEM (all 4 settings)                   | 3,153 | 128 | 128 | 128 |

Total: **8 × question folders**, ≈ **17,888 trajectory files**.

---

## License

Released under the **MIT License**, matching the upstream LLaDA-8B-Instruct
and MMLU-STEM licences. The underlying GSM8K and MMLU benchmarks retain
their original licences; please cite their original papers if you build on
this data.

## Citation

If you use this dataset, please cite our ICLR 2026 paper:

```bibtex
@inproceedings{li2026diffusion,
  title     = {Diffusion Language Model Knows the Answer Before Decoding},
  author    = {Pengxiang Li and Yefan Zhou and Dilxat Muhtar and Lu Yin and Shilin Yan and Li Shen and Yi Liang and Soroush Vosoughi and Shiwei Liu},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026}
}
```