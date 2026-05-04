#!/bin/bash
# =============================================================================
# Spine detection — full training pipeline (axial or sagittal)
#
# Steps:
#   1 = download datasets
#   2 = make splits      (make_splits: true in configs/pipeline.yaml)
#   3 = preprocess
#   4 = build YOLO dataset
#   5 = train
#   6 = evaluate  (inference on all patients)
#   7 = metrics
#   8 = plot metrics
#   9 = find failures
#
# Configuration:
#   Edit configs/pipeline.yaml, configs/preprocess.yaml,
#        configs/dataset.yaml,  configs/training.yaml
#   then: bash scripts/run_pipeline.sh
#
# Prerequisites:
#   conda activate contrast_agnostic
#   set_slot <N>
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # run from project root

# ─── Read pipeline config ─────────────────────────────────────────────────────

eval "$(python - <<'EOF'
import yaml, sys

cfg = yaml.safe_load(open("configs/pipeline.yaml"))
pre = yaml.safe_load(open("configs/preprocess.yaml"))

plane = cfg.get("plane", "axial")
if plane not in ("axial", "sagittal"):
    sys.exit(f"ERROR: plane must be axial or sagittal, got {plane!r}")

plane_cfg = pre.get(plane, {})
si_res      = plane_cfg.get("si_res", 10.0)
inplane_res = plane_cfg.get("inplane_res", 1.0)
sc_pad      = plane_cfg.get("sc_pad", "")

# Auto-generated paths
si_int      = str(si_res).rstrip("0").rstrip(".")
ip_int      = str(inplane_res).rstrip("0").rstrip(".")

if plane == "axial":
    processed = f"processed/pipeline_{si_int}mm_SI_{ip_int}mm_axial_3ch"
else:
    processed = f"processed/pipeline_{si_int}mm_SI_{ip_int}mm_axial_{ip_int}mm_RL_3ch_sagittal_sc{sc_pad}mm"

seed        = cfg.get("seed", 50)
splits_dir  = f"data/datasplits_seed{seed}"
start_step  = cfg.get("start_step", 1)
end_step    = cfg.get("end_step", 9)
make_splits = str(cfg.get("make_splits", True)).lower()

override_processed = cfg.get("override_processed_dir", "") or ""
override_dataset   = cfg.get("override_dataset_dir",   "") or ""
override_run_id    = cfg.get("override_run_id",         "") or ""

from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

if plane == "axial":
    dataset_dir = f"datasets/pipeline_{si_int}mm_SI_{ip_int}mm_axial_3ch_{ts}"
    run_id      = f"pipeline_yolo26n_axial_{ts}"
else:
    training_cfg = yaml.safe_load(open("configs/training.yaml"))
    epochs = training_cfg.get("epochs", 200)
    dataset_dir = f"datasets/pipeline_{si_int}mm_SI_{ip_int}mm_axial_{ip_int}mm_RL_3ch_sagittal_sc{sc_pad}mm_{ts}"
    run_id      = f"pipeline_yolo26n_sagittal_sc{sc_pad}mm_{epochs}ep_{ts}"

if override_processed: processed   = override_processed
if override_dataset:   dataset_dir = override_dataset
if override_run_id:    run_id      = override_run_id

print(f"PLANE={plane}")
print(f"SEED={seed}")
print(f"START_STEP={start_step}")
print(f"END_STEP={end_step}")
print(f"MAKE_SPLITS={make_splits}")
print(f"SPLITS_DIR={splits_dir}")
print(f"PROCESSED_DIR={processed}")
print(f"DATASET_DIR={dataset_dir}")
print(f"RUN_ID={run_id}")
print(f"TS={ts}")
EOF
)"

export SEED

CHECKPOINT="checkpoints/${RUN_ID}/weights/best.pt"
PREDICTIONS_DIR="predictions/${RUN_ID}"

# ─── Expected datasets — derived from data/datasets.yaml ─────────────────────

mapfile -t EXPECTED_DATASETS < <(python -c "
import yaml
with open('data/datasets.yaml') as f:
    print('\n'.join(d['name'] for d in yaml.safe_load(f)['datasets']))
")

# ─── Helpers ──────────────────────────────────────────────────────────────────

step() {
    local n=$1 label=$2
    (( n >= START_STEP && n <= END_STEP )) || return 1
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    printf "  STEP %d/%d : %s\n" "$n" "$END_STEP" "$label"
    echo "══════════════════════════════════════════════════════════════"
}

warn_missing_datasets() {
    local missing=()
    for ds in "${EXPECTED_DATASETS[@]}"; do
        [[ -d "data/raw/${ds}" ]] || missing+=("$ds")
    done
    if (( ${#missing[@]} > 0 )); then
        echo "WARNING: the following datasets are missing from data/raw/ and will be skipped:"
        printf "  - %s\n" "${missing[@]}"
        echo ""
    fi
}

# ─── Validation & summary ─────────────────────────────────────────────────────

if ! (( START_STEP >= 1 && END_STEP <= 9 && START_STEP <= END_STEP )); then
    echo "ERROR: START_STEP=${START_STEP} END_STEP=${END_STEP} — must be in [1,9] with START ≤ END" >&2
    exit 1
fi

echo "══════════════════════════════════════════════════════════════"
echo "  Spine detection pipeline"
echo "  Timestamp   : ${TS}"
echo "  Plane       : ${PLANE}"
echo "  Steps       : ${START_STEP} → ${END_STEP}"
echo "  Run ID      : ${RUN_ID}"
echo "  Processed   : ${PROCESSED_DIR}"
echo "  Dataset     : ${DATASET_DIR}"
echo "  Checkpoint  : ${CHECKPOINT}"
echo "  Predictions : ${PREDICTIONS_DIR}"
echo "══════════════════════════════════════════════════════════════"

# ─── Config snapshot ─────────────────────────────────────────────────────────
# Saved early so it's present even if the pipeline fails mid-way.

RUN_DIR="runs/${TS}"
mkdir -p "${RUN_DIR}"
cp configs/pipeline.yaml configs/preprocess.yaml configs/dataset.yaml configs/training.yaml "${RUN_DIR}/"
echo "Config snapshot saved to ${RUN_DIR}/"

# ─── Step 1 : Download datasets ───────────────────────────────────────────────

if step 1 "Download datasets"; then
    bash scripts/download_all_datasets.sh
fi

# ─── Step 2 : Make splits ─────────────────────────────────────────────────────

if step 2 "Make splits"; then
    if [[ "$MAKE_SPLITS" == "true" ]]; then
        python scripts/make_splits.py \
            --raw data/raw \
            --out "${SPLITS_DIR}" \
            --seed "${SEED}"
    else
        echo "make_splits=false — using committed splits in ${SPLITS_DIR}"
    fi
fi

# ─── Step 3 : Preprocess ──────────────────────────────────────────────────────

if step 3 "Preprocess — ${PLANE}"; then
    warn_missing_datasets
    python scripts/preprocess.py \
        --config configs/preprocess.yaml \
        --plane  "${PLANE}" \
        --out    "${PROCESSED_DIR}"
fi

# ─── Step 4 : Build YOLO dataset ──────────────────────────────────────────────

if step 4 "Build YOLO dataset"; then
    python scripts/build_dataset.py \
        --config    configs/dataset.yaml \
        --processed "${PROCESSED_DIR}" \
        --out       "${DATASET_DIR}"

    # Copy the configs snapshot into the dataset dir for W&B traceability
    cp "${RUN_DIR}"/*.yaml "${DATASET_DIR}/"
fi

# ─── Step 5 : Train ───────────────────────────────────────────────────────────

if step 5 "Train"; then
    python scripts/train.py \
        --config       configs/training.yaml \
        --dataset-yaml "${DATASET_DIR}/dataset.yaml" \
        --run-id       "${RUN_ID}"
fi

# ─── Step 6 : Evaluate ────────────────────────────────────────────────────────

if step 6 "Evaluate — inference on all patients"; then
    python scripts/evaluate.py \
        --checkpoint "${CHECKPOINT}" \
        --processed  "${PROCESSED_DIR}" \
        --out        predictions || true
fi

# ─── Step 7 : Metrics ─────────────────────────────────────────────────────────

if step 7 "Compute metrics"; then
    python scripts/metrics.py \
        --inference  "${PREDICTIONS_DIR}" \
        --splits-dir "${SPLITS_DIR}"
fi

# ─── Step 8 : Plot metrics ────────────────────────────────────────────────────

if step 8 "Plot metrics"; then
    python scripts/plot_metrics.py \
        --inference  "${PREDICTIONS_DIR}" \
        --splits-dir "${SPLITS_DIR}"
fi

# ─── Step 9 : Find failures ───────────────────────────────────────────────────

if step 9 "Find failures"; then
    python scripts/find_failures.py \
        --inference  "${PREDICTIONS_DIR}" \
        --splits-dir "${SPLITS_DIR}"
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Pipeline complete — results in ${PREDICTIONS_DIR}/"
echo "  Config snapshot  — ${RUN_DIR}/"
echo "══════════════════════════════════════════════════════════════"
