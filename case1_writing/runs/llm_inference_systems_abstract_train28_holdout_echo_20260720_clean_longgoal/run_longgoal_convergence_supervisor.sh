#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/tf/notebooks/yunbo/DIRS/case1_writing/runs/llm_inference_systems_abstract_train28_holdout_echo_20260720_clean_longgoal"
HARNESS="/tf/notebooks/yunbo/DIRS/case1_writing/scripts/run_abstract_convergence_harness.py"
ITER_DIR="${RUN_DIR}/longgoal_iterations"
mkdir -p "${ITER_DIR}"

DURATION_SECONDS="${DIRS_LONGGOAL_SECONDS:-86400}"
MIN_LOOPS="${DIRS_MIN_LOOPS:-24}"
MAX_LOOPS="${DIRS_MAX_LOOPS:-1000}"
MCTS_ROLLOUTS="${DIRS_MCTS_ROLLOUTS:-5000}"
STABLE_WINDOW="${DIRS_STABLE_WINDOW:-10}"
BASE_SEED="${DIRS_BASE_SEED:-20260720}"
STOP_AFTER_CONVERGED="${DIRS_STOP_AFTER_CONVERGED:-0}"

START_TS="$(date +%s)"
END_TS="$((START_TS + DURATION_SECONDS))"
ITER=1

echo "$$" > "${RUN_DIR}/longgoal_supervisor.pid"
{
  echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run_dir=${RUN_DIR}"
  echo "duration_seconds=${DURATION_SECONDS}"
  echo "min_loops=${MIN_LOOPS}"
  echo "max_loops=${MAX_LOOPS}"
  echo "mcts_rollouts=${MCTS_ROLLOUTS}"
  echo "stable_window=${STABLE_WINDOW}"
  echo "stop_after_converged=${STOP_AFTER_CONVERGED}"
} > "${RUN_DIR}/longgoal_supervisor.status"

while [ "$(date +%s)" -lt "${END_TS}" ]; do
  SEED="$((BASE_SEED + ITER))"
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  PREFIX="iter_$(printf "%04d" "${ITER}")_seed_${SEED}_${STAMP}"
  LOG_PATH="${ITER_DIR}/${PREFIX}.log"
  SNAP_DIR="${ITER_DIR}/${PREFIX}"
  mkdir -p "${SNAP_DIR}"

  {
    echo "iteration=${ITER}"
    echo "seed=${SEED}"
    echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${SNAP_DIR}/iteration.status"

  python "${HARNESS}" \
    --run-dir "${RUN_DIR}" \
    --max-loops "${MAX_LOOPS}" \
    --min-loops "${MIN_LOOPS}" \
    --mcts-rollouts "${MCTS_ROLLOUTS}" \
    --stable-window "${STABLE_WINDOW}" \
    --seed "${SEED}" > "${LOG_PATH}" 2>&1

  cp "${RUN_DIR}/convergence_report.json" "${SNAP_DIR}/convergence_report.json"
  cp "${RUN_DIR}/convergence_trace.jsonl" "${SNAP_DIR}/convergence_trace.jsonl"
  cp "${RUN_DIR}/CONVERGENCE_REPORT.md" "${SNAP_DIR}/CONVERGENCE_REPORT.md"

  CONVERGED="$(python - <<'PY' "${RUN_DIR}/convergence_report.json"
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("converged"))
PY
)"

  {
    echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "converged=${CONVERGED}"
  } >> "${SNAP_DIR}/iteration.status"

  {
    echo "last_iteration=${ITER}"
    echo "last_seed=${SEED}"
    echo "last_finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "last_snapshot=${SNAP_DIR}"
    echo "last_log=${LOG_PATH}"
    echo "last_converged=${CONVERGED}"
  } >> "${RUN_DIR}/longgoal_supervisor.status"

  if [ "${STOP_AFTER_CONVERGED}" = "1" ] && [ "${CONVERGED}" = "True" ]; then
    break
  fi

  ITER="$((ITER + 1))"
done

echo "ended_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RUN_DIR}/longgoal_supervisor.status"
