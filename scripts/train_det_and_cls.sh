#!/bin/bash
# Train the detector and the classifier in one command (the two runs a release needs).
# Mode is forced per-run via --mode, independently of configs/training.yaml.
#
# Usage: bash scripts/train_all.sh [run_root] [--no-wandb]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_ROOT="runs/$(date +%Y%m%d_%H%M%S)"
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-wandb) EXTRA_ARGS+=(--no-wandb) ;;
        *)          RUN_ROOT="$arg" ;;
    esac
done
DET_RUN="${RUN_ROOT}_det"
CLS_RUN="${RUN_ROOT}_cls"

python scripts/run_pipeline.py --run-dir "$DET_RUN" --mode detection      "${EXTRA_ARGS[@]}"
python scripts/run_pipeline.py --run-dir "$CLS_RUN"  --mode classification "${EXTRA_ARGS[@]}"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Both runs complete."
echo "  Detector run   : ${DET_RUN}"
echo "  Classifier run : ${CLS_RUN}"
echo ""
echo "  Next — export a release bundle (tags this repo model-v<MODEL_VERSION>):"
echo "    python scripts/export_model.py --det-run-dir ${DET_RUN} --cls-run-dir ${CLS_RUN} --version <MODEL_VERSION>"
echo "  Then, in the sc-crop repo, publish it:"
echo "    bash scripts/publish_release.sh --export-dir release_export/ --package-version <PACKAGE_VERSION>"
echo "══════════════════════════════════════════════════════════════"
