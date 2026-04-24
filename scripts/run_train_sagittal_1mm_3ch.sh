#!/bin/bash
# Preprocessing → dataset → training for sagittal 1mm isotropic 3ch pipeline.
#
# Usage:
#   bash scripts/run_train_sagittal_1mm_3ch.sh [run_id]
#
# Examples:
#   bash scripts/run_train_sagittal_1mm_3ch.sh
#   bash scripts/run_train_sagittal_1mm_3ch.sh yolo26n_sagittal_1mm_3ch_v2

set -e

PYTHON=/home/quentinr/.conda/envs/contrast_agnostic/bin/python
ROOT=/home/quentinr/model_cropping_sc_contrast-agnostic_yolo
SCRIPTS=$ROOT/scripts

RUN_ID=${1:-yolo26n_sagittal_1mm_3ch}
PROCESSED=$ROOT/processed/1mm_SI_1mm_axial_3ch_sagittal
DATASET=$ROOT/datasets/1mm_SI_1mm_axial_3ch_sagittal
SPLITS_DIR=$ROOT/data/datasplits/from_raw

echo "=========================================="
echo "RUN_ID    : $RUN_ID"
echo "PROCESSED : $PROCESSED"
echo "DATASET   : $DATASET"
echo "=========================================="

# 1. Preprocessing (skipped if output dir already exists)
echo ""
if [ -d "$PROCESSED" ]; then
    echo "[1/3] preprocess.py — skipped ($PROCESSED already exists)"
else
    echo "[1/3] preprocess.py — sagittal, 1mm isotropic, 3ch"
    $PYTHON $SCRIPTS/preprocess.py \
        --si-res    1.0 \
        --axial-res 1.0 \
        --3ch \
        --plane sagittal
fi

# 2. Build YOLO dataset
echo ""
echo "[2/3] build_dataset.py"
$PYTHON $SCRIPTS/build_dataset.py \
    --processed  $PROCESSED \
    --out        $DATASET \
    --splits-dir $SPLITS_DIR \
    --nc         1 \
    --class-names spine

# 3. Training
echo ""
echo "[3/3] train.py — imgsz=320, batch=512, epochs=100"
cd $ROOT
$PYTHON $SCRIPTS/train.py \
    --run-id      $RUN_ID \
    --dataset-yaml $DATASET/dataset.yaml \
    --imgsz       320 \
    --batch       512 \
    --epochs      100

echo ""
echo "Done. Checkpoint: $ROOT/checkpoints/$RUN_ID/weights/best.pt"
