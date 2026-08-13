# Source Snapshot Manifest

This standalone experience archive was assembled on `2026-08-13` from the
`yunbo/DIRS` tree in the `negative-preference-optimization` workspace.

## Included

- 803 original Markdown documentation and run-report files
- 96 Python source files
- 1 shell script
- 61 YAML graph/configuration files
- 1 CSV configuration table
- 2,764 JSON result, state, verifier, and learned-graph files
- 683 JSONL trace and per-sample result files
- 672 historical execution logs and 1 captured output file
- 4 lightweight timestamp/status marker files
- The conversation note at
  `docs/dirs_research_gap_conversation_2026-08-04.md`

This is 5,086 original DIRS files, plus this manifest, the repository ignore
rules, and the conversation note. Files retain their original relative
locations so links among reports, learned skills, results, and the scripts that
produced them remain understandable.

## Excluded

- Hugging Face caches, downloaded datasets, and model weights
- 17,888 raw `.pt` question-history tensors (8,717,724,604 bytes) from the
  `DLM-Decoding-Analysis` trajectory download
- Python bytecode, cache metadata, and stale PID files
- Two embedded checkouts of the third-party `p-less-sampling` repository
- Git metadata from the source workspace and embedded repositories

The excluded model-cache blobs and raw tensors made the source tree
approximately 23 GB and included individual files far above GitHub's normal
per-file limit. The tensors are intermediate benchmark downloads; their derived
trajectory analysis and all textual results are included. Dependencies can be
downloaded again by the relevant experiment harnesses. Embedded third-party code
should be obtained from its upstream project under its own license.

## Reproduction note

This archive preserves all GitHub-suitable DIRS experience and evidence, but it
is not a self-contained model/data bundle. Historical experiment scripts may
require external repositories, models, datasets, GPUs, or absolute-path updates
described in their adjacent run reports.
