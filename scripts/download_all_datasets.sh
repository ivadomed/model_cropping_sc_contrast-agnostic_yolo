#!/bin/bash
# Download all datasets defined in data/datasets.yaml.
# Usage: bash scripts/download_all_datasets.sh
#
# Prerequisites:
#   - SSH key access to data.neuro.polymtl.ca and spineimage.ca
#   - git-annex installed
#   - conda environment "contrast_agnostic" activated

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # run from project root

DATA_DIR="data/raw"
LOG="$DATA_DIR/git_branch_commit.log"

mkdir -p "$DATA_DIR"

# Parse datasets.yaml once → lines of "name|url_ssh|commit"
DATASETS=$(python - <<'EOF'
import yaml
with open("data/datasets.yaml") as f:
    for d in yaml.safe_load(f)["datasets"]:
        print(f"{d['name']}|{d['url_ssh']}|{d.get('commit') or ''}")
EOF
)

# ---- Clone each dataset ----
for entry in $DATASETS; do
    name="${entry%%|*}"; rest="${entry#*|}"
    url="${rest%%|*}";   commit="${rest#*|}"

    echo "=========================================="
    echo "Cloning: $name"
    echo "=========================================="

    if [ -d "$DATA_DIR/$name" ]; then
        echo "  -> Already exists, skipping clone."
        continue
    fi

    git clone "$url" "$DATA_DIR/$name"
    git -C "$DATA_DIR/$name" annex dead here
    [ -n "$commit" ] && git -C "$DATA_DIR/$name" checkout -q "$commit"

    branch=$(git -C "$DATA_DIR/$name" rev-parse --abbrev-ref HEAD)
    actual=$(git -C "$DATA_DIR/$name" rev-parse HEAD)
    echo "$name: git-$branch-$actual" >> "$LOG"
done

# ---- Download NIfTI files via git-annex (parallel) ----
echo ""
echo "=========================================="
echo "Fetching NIfTI files with git annex get (parallel)..."
echo "=========================================="

pids=()
dataset_names=()
for entry in $DATASETS; do
    name="${entry%%|*}"
    if [ -d "$DATA_DIR/$name" ]; then
        echo "  -> git annex get: $name"
        (cd "$DATA_DIR/$name" && git annex get .) &
        pids+=($!)
        dataset_names+=("$name")
    fi
done

failed_datasets=()
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || failed_datasets+=("${dataset_names[$i]}")
done

echo ""
echo "=========================================="
if (( ${#failed_datasets[@]} > 0 )); then
    echo "WARNING: git annex get incomplete for:"
    printf "  - %s\n" "${failed_datasets[@]}"
    echo "  Affected subjects will be skipped by preprocess.py."
else
    echo "Done!"
fi
echo "=========================================="
