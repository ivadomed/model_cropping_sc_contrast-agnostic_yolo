#!/bin/bash
# Full evaluation pipeline: evaluate → metrics → plot_metrics → find_failures
#
# Usage:
#   bash scripts/run_eval_pipeline.sh --run-id <run_id> [--flip-x] [--start <step>]
#
# Examples:
#   bash scripts/run_eval_pipeline.sh --run-id yolo26n_sc_only_all_datasets
#   bash scripts/run_eval_pipeline.sh --run-id yolo26n_sc_only_all_datasets --flip-x
#   bash scripts/run_eval_pipeline.sh --run-id yolo26n_sc_only_all_datasets --start 2

set -e

PYTHON=/home/quentinr/.conda/envs/contrast_agnostic/bin/python
ROOT=/home/quentinr/model_cropping_sc_contrast-agnostic_yolo
SCRIPTS=$ROOT/scripts

RUN_ID=
FLIP_X=
START=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-id)  RUN_ID=$2; shift 2 ;;
        --flip-x)  FLIP_X=--flip-x;  shift ;;
        --start)   START=$2;  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z $RUN_ID ]] && { echo "Usage: $0 --run-id <run_id> [--flip-x] [--start <step>]"; exit 1; }

if [[ -f $ROOT/checkpoints/$RUN_ID/weights/best.pt ]]; then
    CHECKPOINT=$ROOT/checkpoints/$RUN_ID/weights/best.pt
else
    CHECKPOINT=$ROOT/runs/$RUN_ID/checkpoints/weights/best.pt
fi
PROCESSED=$ROOT/processed/10mm_SI_1mm_axial_3ch_sc_and_canal
SPLITS_DIR=$ROOT/data/datasplits/from_raw
PRED_DIR=$ROOT/predictions/${RUN_ID}${FLIP_X:+_flipx}

echo "=========================================="
echo "RUN_ID   : $RUN_ID"
echo "FLIP_X   : ${FLIP_X:-no}"
echo "PRED_DIR : $PRED_DIR"
echo "=========================================="

# 1. Evaluate
if [[ $START -le 1 ]]; then
echo ""
echo "[1/4] evaluate.py"
$PYTHON $SCRIPTS/evaluate.py \
    --checkpoint $CHECKPOINT \
    --processed  $PROCESSED \
    --splits-dir $SPLITS_DIR \
    --run-id     ${RUN_ID}${FLIP_X:+_flipx} \
    $FLIP_X
fi

# 2. Metrics
if [[ $START -le 2 ]]; then
echo ""
echo "[2/4] metrics.py"
$PYTHON $SCRIPTS/metrics.py \
    --inference  $PRED_DIR \
    --processed  $PROCESSED \
    --splits-dir $SPLITS_DIR
fi

# 3. Plot metrics
if [[ $START -le 3 ]]; then
echo ""
echo "[3/4] plot_metrics.py"
$PYTHON $SCRIPTS/plot_metrics.py \
    --inference  $PRED_DIR \
    --metrics    iou_3d_mm gap_mm_R gap_mm_L gap_mm_P gap_mm_A gap_mm_I gap_mm_S \
    --splits     val test \
    --splits-dir $SPLITS_DIR \
    --conf       0.1
fi

# 4. Failures
if [[ $START -le 4 ]]; then
echo ""
echo "[4/4] find_failures.py"
$PYTHON $SCRIPTS/find_failures.py \
    --inference  $PRED_DIR \
    --metric     iou_3d_mm \
    --splits     val test \
    --splits-dir $SPLITS_DIR \
    --conf       0.1
fi

echo ""
echo "Done. Results in $PRED_DIR"
