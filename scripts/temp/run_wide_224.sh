#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# 3-model pipeline: Baseline | EMIX Linear (50→0) | EMIX Log (50→0)
#
# Each model: train (emix env) → eval (emix-twin env).
# conda run --no-capture-output keeps stdout/stderr live in the terminal.
# ---------------------------------------------------------------------------

RUN_TRAIN="conda run --no-capture-output -n emix python -m embodied_mixup train"
RUN_EVAL="conda run --no-capture-output -n emix-twin python -m embodied_mixup eval"

TRAIN_DIR="configs/train/wide"
EVAL_DIR="configs/eval/wide"

echo "========================================="
echo "1/6  Training: DP Baseline (224×224)"
echo "========================================="
${RUN_TRAIN} --config "${TRAIN_DIR}/place_object_scale_left_wide_dp_baseline.yaml"

echo ""
echo "========================================="
echo "2/6  Evaluation: DP Baseline"
echo "========================================="
${RUN_EVAL} --config "${EVAL_DIR}/place_object_scale_left_dp_baseline.yaml"

echo ""
echo "========================================="
echo "3/6  Training: EMIX Linear 50→0 (224×224)"
echo "========================================="
${RUN_TRAIN} --config "${TRAIN_DIR}/place_object_scale_left_wide_dp_emix_linear.yaml"

echo ""
echo "========================================="
echo "4/6  Evaluation: EMIX Linear 50→0"
echo "========================================="
${RUN_EVAL} --config "${EVAL_DIR}/place_object_scale_left_dp_emix_linear.yaml"

echo ""
echo "========================================="
echo "5/6  Training: EMIX Log 50→0 (224×224)"
echo "========================================="
${RUN_TRAIN} --config "${TRAIN_DIR}/place_object_scale_left_wide_dp_emix_log.yaml"

echo ""
echo "========================================="
echo "6/6  Evaluation: EMIX Log 50→0"
echo "========================================="
${RUN_EVAL} --config "${EVAL_DIR}/place_object_scale_left_dp_emix_log.yaml"

echo ""
echo "All 6 steps completed."
