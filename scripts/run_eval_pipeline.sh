#!/bin/bash
# Full evaluation pipeline: evaluate → metrics → plot_metrics → find_failures
#
# Usage:
#   bash scripts/run_eval_pipeline.sh <run_id> [--flip-x]
#
# Examples:
#   bash scripts/run_eval_pipeline.sh yolo26n_sc_only_all_datasets
#   bash scripts/run_eval_pipeline.sh yolo26n_sc_only_all_datasets --flip-x

set -e

PYTHON=/home/quentinr/.conda/envs/contrast_agnostic/bin/python
ROOT=/home/quentinr/model_cropping_sc_contrast-agnostic_yolo
SCRIPTS=$ROOT/scripts

RUN_ID=${1:?Usage: $0 <run_id> [--flip-x]}
FLIP_X=${2:-}

CHECKPOINT=$ROOT/checkpoints/$RUN_ID/weights/best.pt
PROCESSED=$ROOT/processed/10mm_SI_1mm_axial_3ch_sc_and_canal
SPLITS_DIR=$ROOT/data/datasplits/from_raw
PRED_DIR=$ROOT/predictions/${RUN_ID}${FLIP_X:+_flipx}

echo "=========================================="
echo "RUN_ID   : $RUN_ID"
echo "FLIP_X   : ${FLIP_X:-no}"
echo "PRED_DIR : $PRED_DIR"
echo "=========================================="

# 1. Evaluate
echo ""
echo "[1/4] evaluate.py"
$PYTHON $SCRIPTS/evaluate.py \
    --checkpoint $CHECKPOINT \
    --processed  $PROCESSED \
    --splits-dir $SPLITS_DIR \
    --run-id     ${RUN_ID}${FLIP_X:+_flipx} \
    $FLIP_X

# 2. Metrics
echo ""
echo "[2/4] metrics.py"
$PYTHON $SCRIPTS/metrics.py \
    --inference  $PRED_DIR \
    --processed  $PROCESSED \
    --splits-dir $SPLITS_DIR

# 3. Plot metrics
echo ""
echo "[3/4] plot_metrics.py"
$PYTHON $SCRIPTS/plot_metrics.py \
    --inference  $PRED_DIR \
    --metrics    iou_3d_mm gap_mm_R gap_mm_L gap_mm_P gap_mm_A gap_mm_I gap_mm_S \
    --splits     val test \
    --splits-dir $SPLITS_DIR \
    --conf       0.1

# 4. Failures
echo ""
echo "[4/4] find_failures.py"
$PYTHON $SCRIPTS/find_failures.py \
    --inference  $PRED_DIR \
    --metric     iou_3d_mm \
    --splits     val test \
    --splits-dir $SPLITS_DIR \
    --conf       0.1

echo ""
echo "Done. Results in $PRED_DIR"
