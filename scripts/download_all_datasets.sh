#!/bin/bash
# Download all datasets defined in configs/datasets.yaml.
# Usage: bash scripts/download_all_datasets.sh
#
# Prerequisites:
#   - git-annex installed
#   - conda environment "contrast_agnostic" activated
#   - SSH key access recommended (falls back to HTTPS if SSH unavailable)

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # run from project root

DATA_DIR="data/raw"
LOG="$DATA_DIR/git_branch_commit.log"

mkdir -p "$DATA_DIR"

# Parse datasets.yaml once → lines of "name|url_ssh|url_https|commit"
DATASETS=$(python - <<'EOF'
import yaml
with open("configs/datasets.yaml") as f:
    for d in yaml.safe_load(f)["datasets"]:
        print(f"{d['name']}|{d.get('url_ssh') or ''}|{d.get('url_https') or ''}|{d.get('commit') or ''}")
EOF
)

# ---- Clone each dataset ----
while IFS='|' read -r name url_ssh url_https commit; do
    echo "=========================================="
    echo "Cloning: $name"
    echo "=========================================="

    if [ -d "$DATA_DIR/$name" ]; then
        echo "  -> Already exists, skipping clone."
        continue
    fi

    cloned=0
    if [ -n "$url_ssh" ] && git clone "$url_ssh" "$DATA_DIR/$name" 2>/dev/null; then
        echo "  -> Cloned via SSH."
        cloned=1
    elif [ -n "$url_https" ] && git clone "$url_https" "$DATA_DIR/$name"; then
        echo "  -> SSH unavailable, cloned via HTTPS."
        cloned=1
    fi

    if [ "$cloned" -eq 0 ]; then
        echo "  ERROR: failed to clone $name, skipping."
        continue
    fi

    git -C "$DATA_DIR/$name" annex dead here
    [ -n "$commit" ] && git -C "$DATA_DIR/$name" checkout -q "$commit"

    branch=$(git -C "$DATA_DIR/$name" rev-parse --abbrev-ref HEAD)
    actual=$(git -C "$DATA_DIR/$name" rev-parse HEAD)
    echo "$name: git-$branch-$actual" >> "$LOG"
done <<< "$DATASETS"

# ---- Download NIfTI files via git-annex (parallel) ----
echo ""
echo "=========================================="
echo "Fetching NIfTI files with git annex get (parallel)..."
echo "=========================================="

pids=()
dataset_names=()
while IFS='|' read -r name url_ssh url_https commit; do
    if [ -d "$DATA_DIR/$name" ]; then
        echo "  -> git annex get: $name"
        (cd "$DATA_DIR/$name" && git annex get .) &
        pids+=($!)
        dataset_names+=("$name")
    fi
done <<< "$DATASETS"

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
