#!/usr/bin/env bash
# Fixed orchestration for the top_cam 500ep EMIX-only ablation suite.
#
# For every batch runbook (in order) it drives the complete flow, and the
# flow itself is resume-safe — each stage checks its artifacts first:
#
#   1. prepare     check collection/dataset complete; collect + convert only
#                  when the 84x84 LeRobot dataset is missing
#   2. index       emix build (with the batch's emix.build_extra_args filters)
#   3. vis         intersections HTML + crisp PNG per task
#   4. train-eval  train the selected policy families (training-time encoder
#                  debug + emix debug visuals are always on), then evaluate
#   5. summary     write runs/<run_label>/summary/RESULTS.md (auto table plus
#                  a reserved section for agents to fill in manually)
#
# Batches run strictly sequentially so both GPUs stay saturated inside each
# batch (static 5+5 per-GPU task split, one trainer per card).
set -uo pipefail
cd /home/rogerspyke/projects/EMIX/Embodied-Mixup
PY=/home/rogerspyke/miniforge3/envs/emix/bin/python
STAMP=$(date +%Y%m%d-%H%M%S)
LOG=logs/ablation-suite-${STAMP}.log
mkdir -p logs

BATCHES=(
  configs/scripts/run_cen_arm_top_cam_500ep_homefilter.yaml
  configs/scripts/run_cen_arm_top_cam_500ep_timeratio.yaml
  configs/scripts/run_cen_arm_top_cam_500ep_combo.yaml
)

{
  echo "[suite] start $(date '+%F %T') with ${#BATCHES[@]} batches"
  for cfg in "${BATCHES[@]}"; do
    label=$(grep -m1 'run_label:' "$cfg" | awk '{print $2}')
    echo "[suite] ===== batch ${label} (${cfg}) ====="
    if "$PY" scripts/run.py --config "$cfg" >>"$LOG" 2>&1; then
      echo "[suite] batch ${label} finished OK $(date '+%F %T')"
    else
      code=$?
      echo "[suite] batch ${label} FAILED (exit ${code}) $(date '+%F %T') — continuing with next batch"
    fi
  done
  echo "[suite] all batches processed $(date '+%F %T')"
} | tee logs/ablation-suite-status-${STAMP}.log
