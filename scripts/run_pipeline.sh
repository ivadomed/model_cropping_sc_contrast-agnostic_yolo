#!/bin/bash
# Prerequisite: conda activate contrast_agnostic
# Usage: bash scripts/run_pipeline.sh [--start N] [--end N] [--no-wandb]

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python scripts/run_pipeline.py "$@"
