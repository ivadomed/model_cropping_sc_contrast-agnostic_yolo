#!/bin/bash
# Download all datasets from the "Monitoring morphometric drift" paper
# Usage: bash download_all_datasets.sh
#
# Prerequisites:
#   - SSH access to data.neuro.polymtl.ca
#   - conda environment "contrast_agnostic" activated
#   - Praxis datasets must be downloaded manually from spineimage.ca

set -e

# ---- Configuration ----
DATA_DIR="$HOME/model_cropping_sc_contrast-agnostic_yolo/data/raw"
SCRIPT_DIR="$HOME/model_cropping_sc_contrast-agnostic_yolo/scripts"

# All datasets from the paper (excluding Praxis which require manual download)
DATASETS=(
    "basel-mp2rage"
    "canproco"
    # "data-multi-subject"
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
    # "site_006_praxis"
    # "site_007_praxis"
)

# ---- Setup ----
mkdir -p "$DATA_DIR"

# ---- Download datasets ----
for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Downloading: $dataset"
    echo "=========================================="
    
    # Skip if already cloned
    if [ -d "$DATA_DIR/$dataset" ]; then
        echo "  -> Already exists, skipping clone."
    else
        cd "$SCRIPT_DIR"
        python 01_clone_dataset.py --ofolder "$DATA_DIR" --dataset "$dataset"
    fi
done

# ---- Download actual NIfTI files via git-annex ----
echo ""
echo "=========================================="
echo "Fetching NIfTI files with git annex get..."
echo "=========================================="

for dataset in "${DATASETS[@]}"; do
    if [ -d "$DATA_DIR/$dataset" ]; then
        echo "  -> git annex get: $dataset"
        cd "$DATA_DIR/$dataset"
        git annex get .
    fi
done

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo ""
echo "REMINDER: Praxis datasets need manual download from spineimage.ca:"
echo "  - site_006_praxis"
echo "  - site_007_praxis"
echo "Place them in: $DATA_DIR"