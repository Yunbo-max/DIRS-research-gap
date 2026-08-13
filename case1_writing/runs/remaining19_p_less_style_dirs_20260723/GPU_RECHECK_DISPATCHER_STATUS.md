# GPU Recheck Dispatcher Status

- Updated: `2026-07-24T14:02:55Z`
- Status: `deferred_selected_gpu_has_active_prophet_runner`
- Policy: professional gate rechecks are support only; they cannot converge a paper without full verifier-comparable outputs.
- Selected GPU: `3`
- Pending before dispatch: `0`
- Rechecks executed: `0`
- State: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/gpu_recheck_dispatcher_state.json`
- Report: `/tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/gpu_recheck_dispatcher_report.json`

## GPU Inventory

- GPU `0` NVIDIA GeForce RTX 4090 free=`8395` MiB used=`16169` MiB util=`99`%
- GPU `1` NVIDIA GeForce RTX 4090 free=`8435` MiB used=`16129` MiB util=`99`%
- GPU `2` NVIDIA GeForce RTX 4090 free=`7956` MiB used=`16608` MiB util=`99`%
- GPU `3` NVIDIA GeForce RTX 4090 free=`8415` MiB used=`16149` MiB util=`100`%

## Active Prophet Processes

- `root        2355 97.9  0.2 80346904 1482384 ?    Rsl  13:20 100:14 python /tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/prophet_custom_full_gsm8k_runner.py --gpu 0 --out-dir /tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/table1_threshold_repair_full_gsm8k/table1_threshold_relaxed_6_4_2 --run-label table1_threshold_relaxed_6_4_2 --variants prophet --gen-length 256 --steps 256 --block-length 32 --remasking low_confidence --constraints-text 200:The|201:answer|202:is --answer-start-offset 200 --prompt-profile official_zero_shot --early-threshold 6.0 --mid-threshold 4.0 --late-threshold 2.0`
- `root        2465 97.1  0.2 80349024 1487748 ?    Rsl  13:20  99:17 python /tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/prophet_custom_full_gsm8k_runner.py --gpu 1 --out-dir /tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/table1_threshold_repair_full_gsm8k/table1_threshold_relaxed_5_3_1 --run-label table1_threshold_relaxed_5_3_1 --variants prophet --gen-length 256 --steps 256 --block-length 32 --remasking low_confidence --constraints-text 200:The|201:answer|202:is --answer-start-offset 200 --prompt-profile official_zero_shot --early-threshold 5.0 --mid-threshold 3.0 --late-threshold 1.0`
- `root        2714 97.8  0.2 80347344 1485976 ?    Rsl  13:20 100:01 python /tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/prophet_custom_full_gsm8k_runner.py --gpu 2 --out-dir /tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/table1_threshold_repair_full_gsm8k/table1_threshold_relaxed_4_2_0p5 --run-label table1_threshold_relaxed_4_2_0p5 --variants prophet --gen-length 256 --steps 256 --block-length 32 --remasking low_confidence --constraints-text 200:The|201:answer|202:is --answer-start-offset 200 --prompt-profile official_zero_shot --early-threshold 4.0 --mid-threshold 2.0 --late-threshold 0.5`
- `root        2903 97.9  0.2 80347180 1485896 ?    Rsl  13:20 100:03 python /tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/prophet_custom_full_gsm8k_runner.py --gpu 3 --out-dir /tf/notebooks/yunbo/DIRS/case1_writing/runs/remaining19_p_less_style_dirs_20260723/specialized_runners/prophet/table1_threshold_repair_full_gsm8k/table1_threshold_relaxed_3_1p5_0p5 --run-label table1_threshold_relaxed_3_1p5_0p5 --variants prophet --gen-length 256 --steps 256 --block-length 32 --remasking low_confidence --constraints-text 200:The|201:answer|202:is --answer-start-offset 200 --prompt-profile official_zero_shot --early-threshold 3.0 --mid-threshold 1.5 --late-threshold 0.5`

## Recheck Results

