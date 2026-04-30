#!/bin/bash
# Download all datasets defined in data/datasets.yaml.
# Usage: bash scripts/download_all_datasets.sh
#
# Prerequisites:
#   - SSH access to data.neuro.polymtl.ca and spineimage.ca
#   - git-annex installed
#   - conda environment "contrast_agnostic" activated

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # run from project root

DATA_DIR="data/raw"

mkdir -p "$DATA_DIR"

# Extract all dataset names from the registry
ALL_DATASETS=$(python - <<'EOF'
import yaml
with open("data/datasets.yaml") as f:
    reg = yaml.safe_load(f)
for d in reg["datasets"]:
    print(d["name"])
EOF
)

# ---- Clone each dataset ----
for dataset in $ALL_DATASETS; do
    echo "=========================================="
    echo "Cloning: $dataset"
    echo "=========================================="
    if [ -d "$DATA_DIR/$dataset" ]; then
        echo "  -> Already exists, skipping clone."
    else
        python scripts/01_clone_dataset.py --ofolder "$DATA_DIR" --dataset "$dataset"
    fi
done

# ---- Download NIfTI files via git-annex (parallel) ----
echo ""
echo "=========================================="
echo "Fetching NIfTI files with git annex get (parallel)..."
echo "=========================================="

pids=()
for dataset in $ALL_DATASETS; do
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
