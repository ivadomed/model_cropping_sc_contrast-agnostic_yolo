#!/bin/bash
# Download all datasets
# Usage: bash download_all_datasets.sh
#
# Prerequisites:
#   - SSH access to data.neuro.polymtl.ca
#   - git-annex installed
#   - conda environment "contrast_agnostic" activated
#   - Praxis datasets must be downloaded manually from spineimage.ca
#
# Re-running is safe: clone is skipped if the directory already exists,
# and git annex get only downloads files not yet present.

set -e

# ---- Configuration ----
DATA_DIR="$HOME/model_cropping_sc_contrast-agnostic_yolo/data/raw"
SCRIPT_DIR="$HOME/model_cropping_sc_contrast-agnostic_yolo/scripts"

# Datasets hosted on data.neuro.polymtl.ca (cloned via SSH)
NEURO_DATASETS=(
    "basel-mp2rage"
    "canproco"
    "dcm-brno"
    "dcm-zurich-lesions-20231115"
    "dcm-zurich-lesions"
    "dcm-zurich"
    "lumbar-epfl"
    "lumbar-vanderbilt"
    "nih-ms-mp2rage"
    "sci-colorado"
    "sci-paris"
    "sci-zurich"
    "sct-testing-large"
    "spider-challenge-2023"
    "whole-spine"
    # "site_006_praxis"
    # "site_007_praxis"
)

# Datasets hosted on GitHub (spine-generic organisation)
GITHUB_DATASETS=(
    "data-multi-subject"
    "data-single-subject"
)

# ---- Setup ----
mkdir -p "$DATA_DIR"

# ---- Clone neuropoly datasets ----
for dataset in "${NEURO_DATASETS[@]}"; do
    echo "=========================================="
    echo "Cloning (neuropoly): $dataset"
    echo "=========================================="
    if [ -d "$DATA_DIR/$dataset" ]; then
        echo "  -> Already exists, skipping clone."
    else
        cd "$SCRIPT_DIR"
        python 01_clone_dataset.py --ofolder "$DATA_DIR" --dataset "$dataset"
    fi
done

# ---- Clone GitHub (spine-generic) datasets ----
for dataset in "${GITHUB_DATASETS[@]}"; do
    echo "=========================================="
    echo "Cloning (GitHub): $dataset"
    echo "=========================================="
    if [ -d "$DATA_DIR/$dataset" ]; then
        echo "  -> Already exists, skipping clone."
    else
        cd "$DATA_DIR"
        git clone "https://github.com/spine-generic/${dataset}"
        cd "$DATA_DIR/$dataset"
        git annex init
    fi
done

# ---- Download NIfTI files via git-annex (parallel) ----
echo ""
echo "=========================================="
echo "Fetching NIfTI files with git annex get (parallel)..."
echo "=========================================="

ALL_DATASETS=("${NEURO_DATASETS[@]}" "${GITHUB_DATASETS[@]}")

pids=()
for dataset in "${ALL_DATASETS[@]}"; do
    if [ -d "$DATA_DIR/$dataset" ]; then
        echo "  -> git annex get: $dataset"
        (cd "$DATA_DIR/$dataset" && git annex get .) &
        pids+=($!)
    fi
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || { echo "ERROR: git annex get failed (pid $pid)"; failed=1; }
done

if [ "$failed" -eq 1 ]; then
    echo "ERROR: one or more git annex get calls failed."
    exit 1
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo ""
echo "REMINDER: Praxis datasets need manual download from spineimage.ca:"
echo "  - site_006_praxis"
echo "  - site_007_praxis"
echo "Place them in: $DATA_DIR"
