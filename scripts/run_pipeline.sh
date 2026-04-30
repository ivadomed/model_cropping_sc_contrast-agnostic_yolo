#!/bin/bash
# =============================================================================
# Spine detection — full training pipeline (axial or sagittal)
#
# Steps:
#   1 = download datasets
#   2 = make splits      (skipped by default: MAKE_SPLITS=false)
#   3 = preprocess
#   4 = build YOLO dataset
#   5 = train
#   6 = evaluate  (inference on all patients)
#   7 = metrics
#   8 = plot metrics
#   9 = find failures
#
# Usage:
#   Edit PLANE, START_STEP, END_STEP at the top of this script, then:
#   bash scripts/run_pipeline.sh
#
# Prerequisites (must be done before running this script):
#   conda activate contrast_agnostic
#   set_slot <N>
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # run from project root

# ─── Pipeline control ─────────────────────────────────────────────────────────

PLANE=sagittal       # axial | sagittal
START_STEP=6      # 1–9
END_STEP=9        # 1–9
MAKE_SPLITS=false # true = regenerate splits from data/raw (step 2)
export SEED=50    # global random seed — propagated to all Python scripts via env

# ─── Parameters — AXIAL ───────────────────────────────────────────────────────

AXIAL_SI_RES=10.0        # SI resampling resolution (mm)
AXIAL_INPLANE_RES=1.0    # in-plane (RL, AP) resampling resolution (mm)

# ─── Parameters — SAGITTAL ────────────────────────────────────────────────────

SAG_SI_RES=1.0           # SI resampling resolution (mm)
SAG_INPLANE_RES=1.0      # in-plane (RL, AP) resampling resolution (mm)
SAG_SC_PAD=10            # slices kept = SC extent ± this value (mm)
SAG_SC_RATIO=3           # SC/non-SC slice balance ratio in the YOLO dataset

# ─── Parameters — DATASET OVERSAMPLING ───────────────────────────────────────
# Per-dataset slice multipliers applied during build_dataset (step 4).
# Format: "dataset_name:factor"  — omit a dataset to use factor 1 (no change).
# Example: lumbar datasets oversampled ×5 because they are underrepresented.

DATASET_FACTORS=(
    lumbar-epfl:7
    lumbar-vanderbilt:7
    spider-challenge-2023:10
    sct-testing-large:2
    sci-zurich:2
    canproco:2
)

# ─── Parameters — TRAINING (shared) ──────────────────────────────────────────

WITH_CANAL=false         # true = also extract spinal canal (class 1) during preprocessing

MODEL="yolo26n.pt"
EPOCHS=200
IMGSZ=320
FL_GAMMA=2               # focal loss gamma (0 = standard BCE)
WORKERS=4

# ─── Path overrides (optional) ───────────────────────────────────────────────
# Leave empty to use auto-generated paths.
# Use when resuming from an existing directory created outside this pipeline.
# NOTE: overrides only affect steps 4+ (build, train, eval, metrics, failures).
#       Step 3 (preprocess) always writes to the auto-generated PROCESSED_DIR.

OVERRIDE_PROCESSED_DIR="/home/quentinr/model_cropping_sc_contrast-agnostic_yolo/processed/1mm_SI_1mm_axial_1mm_RL_3ch_sagittal_sc10mm"   # e.g. processed/10mm_SI_1mm_axial_3ch
OVERRIDE_DATASET_DIR=""     # e.g. datasets/10mm_SI_1mm_axial_3ch
OVERRIDE_RUN_ID="yolo26n_sagittal_sc10mm_focal23"          # e.g. yolo26n_axial_v2

# ─── Derived values (do not edit) ─────────────────────────────────────────────

SPLITS_DIR="data/datasplits_seed${SEED}"

if [[ "$PLANE" == "axial" ]]; then

    PROCESSED_DIR="processed/pipeline_${AXIAL_SI_RES%.*}mm_SI_${AXIAL_INPLANE_RES%.*}mm_axial_3ch"
    DATASET_DIR="datasets/pipeline_${AXIAL_SI_RES%.*}mm_SI_${AXIAL_INPLANE_RES%.*}mm_axial_3ch"
    RUN_ID="pipeline_run_yolo26n_axial_${EPOCHS}ep"

    PREPROCESS_ARGS=(
        --si-res    "${AXIAL_SI_RES}"
        --axial-res "${AXIAL_INPLANE_RES}"
        --3ch
        --out       "${PROCESSED_DIR}"
    )
    BUILD_DATASET_EXTRA_ARGS=()

elif [[ "$PLANE" == "sagittal" ]]; then

    PROCESSED_DIR="processed/pipeline_${SAG_SI_RES%.*}mm_SI_${SAG_INPLANE_RES%.*}mm_axial_${SAG_INPLANE_RES%.*}mm_RL_3ch_sagittal_sc${SAG_SC_PAD}mm"
    DATASET_DIR="datasets/pipeline_${SAG_SI_RES%.*}mm_SI_${SAG_INPLANE_RES%.*}mm_axial_${SAG_INPLANE_RES%.*}mm_RL_3ch_sagittal_sc${SAG_SC_PAD}mm"
    RUN_ID="pipeline_run_yolo26n_sagittal_sc${SAG_SC_PAD}mm_${EPOCHS}ep"

    PREPROCESS_ARGS=(
        --si-res    "${SAG_SI_RES}"
        --axial-res "${SAG_INPLANE_RES}"
        --rl-res    "${SAG_INPLANE_RES}"
        --3ch
        --plane     sagittal
        --sc-pad    "${SAG_SC_PAD}"
        --out       "${PROCESSED_DIR}"
    )
    BUILD_DATASET_EXTRA_ARGS=(--sc-ratio "${SAG_SC_RATIO}")

else
    echo "ERROR: PLANE must be 'axial' or 'sagittal', got '${PLANE}'" >&2
    exit 1
fi

[[ "$WITH_CANAL" == "true" ]] && PREPROCESS_ARGS+=(--with-canal)

# Apply path overrides if set
[[ -n "$OVERRIDE_PROCESSED_DIR" ]] && PROCESSED_DIR="$OVERRIDE_PROCESSED_DIR"
[[ -n "$OVERRIDE_DATASET_DIR"   ]] && DATASET_DIR="$OVERRIDE_DATASET_DIR"
[[ -n "$OVERRIDE_RUN_ID"        ]] && RUN_ID="$OVERRIDE_RUN_ID"

# Assemble build_dataset args after overrides so paths are always correct
BUILD_DATASET_ARGS=(
    --processed       "${PROCESSED_DIR}"
    --out             "${DATASET_DIR}"
    --splits-dir      "${SPLITS_DIR}"
    --dataset-factors "${DATASET_FACTORS[@]}"
    "${BUILD_DATASET_EXTRA_ARGS[@]}"
)

CHECKPOINT="checkpoints/${RUN_ID}/weights/best.pt"
PREDICTIONS_DIR="predictions/${RUN_ID}"

# ─── Expected datasets — warn if any are missing ──────────────────────────────

EXPECTED_DATASETS=(
    data-multi-subject
    basel-mp2rage
    dcm-zurich
    lumbar-vanderbilt
    nih-ms-mp2rage
    canproco
    sci-colorado
    sci-paris
    sci-zurich
    sct-testing-large
    lumbar-epfl
    dcm-brno
    dcm-zurich-lesions
    dcm-zurich-lesions-20231115
    spider-challenge-2023
    whole-spine
    site_006_praxis      # manual download from spineimage.ca
    site_007_praxis      # manual download from spineimage.ca
)

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
echo "  Plane       : ${PLANE}"
echo "  Steps       : ${START_STEP} → ${END_STEP}"
echo "  Run ID      : ${RUN_ID}"
echo "  Processed   : ${PROCESSED_DIR}"
echo "  Dataset     : ${DATASET_DIR}"
echo "  Checkpoint  : ${CHECKPOINT}"
echo "  Predictions : ${PREDICTIONS_DIR}"
echo "══════════════════════════════════════════════════════════════"

# ─── Step 1 : Download datasets ───────────────────────────────────────────────

if step 1 "Download datasets"; then
    bash scripts/download_all_datasets.sh
fi

# ─── Step 2 : Make splits ─────────────────────────────────────────────────────
# Disabled by default — committed splits in data/datasplits_seed<N>/ are the reference.
# Enable only to regenerate from scratch (changes results, commit afterwards).

if step 2 "Make splits"; then
    if [[ "$MAKE_SPLITS" == "true" ]]; then
        python scripts/make_splits.py \
            --raw data/raw \
            --out "${SPLITS_DIR}" \
            --seed "${SEED}"
    else
        echo "MAKE_SPLITS=false — using committed splits in ${SPLITS_DIR}"
    fi
fi

# ─── Step 3 : Preprocess ──────────────────────────────────────────────────────

if step 3 "Preprocess — ${PLANE}"; then
    warn_missing_datasets
    python scripts/preprocess.py "${PREPROCESS_ARGS[@]}"
fi

# ─── Step 4 : Build YOLO dataset ──────────────────────────────────────────────

if step 4 "Build YOLO dataset"; then
    python scripts/build_dataset.py "${BUILD_DATASET_ARGS[@]}"
fi

# Write pipeline_config.yaml into the dataset dir for W&B traceability
mkdir -p "${DATASET_DIR}"
python - <<EOF
import yaml
factors = {}
for entry in "${DATASET_FACTORS[*]}".split():
    k, v = entry.split(":")
    factors[k] = float(v)
config = {
    "plane":          "${PLANE}",
    "seed":           ${SEED},
    "model":          "${MODEL}",
    "epochs":         ${EPOCHS},
    "imgsz":          ${IMGSZ},
    "fl_gamma":       ${FL_GAMMA},
    "workers":        ${WORKERS},
    "with_canal":     "${WITH_CANAL}" == "true",
    "dataset_factors": factors,
    "processed_dir":  "${PROCESSED_DIR}",
    "dataset_dir":    "${DATASET_DIR}",
    "run_id":         "${RUN_ID}",
}
if "${PLANE}" == "axial":
    config["si_res_mm"]     = ${AXIAL_SI_RES}
    config["inplane_res_mm"] = ${AXIAL_INPLANE_RES}
else:
    config["si_res_mm"]     = ${SAG_SI_RES}
    config["inplane_res_mm"] = ${SAG_INPLANE_RES}
    config["sc_pad_mm"]     = ${SAG_SC_PAD}
    config["sc_ratio"]      = ${SAG_SC_RATIO}
with open("${DATASET_DIR}/pipeline_config.yaml", "w") as f:
    yaml.dump(config, f, sort_keys=False, default_flow_style=False)
print("Pipeline config written to ${DATASET_DIR}/pipeline_config.yaml")
EOF

# ─── Step 5 : Train ───────────────────────────────────────────────────────────

if step 5 "Train — ${MODEL} × ${EPOCHS} epochs (fl-gamma=${FL_GAMMA})"; then
    python scripts/train.py \
        --dataset-yaml "${DATASET_DIR}/dataset.yaml" \
        --run-id       "${RUN_ID}" \
        --model        "${MODEL}" \
        --epochs       "${EPOCHS}" \
        --imgsz        "${IMGSZ}" \
        --workers      "${WORKERS}" \
        --fl-gamma     "${FL_GAMMA}"
fi

# ─── Step 6 : Evaluate ────────────────────────────────────────────────────────
# Writes predictions/<RUN_ID>/<dataset>/<patient>/txt|png|volume + gt/ symlink.

if step 6 "Evaluate — inference on all patients"; then
    python scripts/evaluate.py \
        --checkpoint "${CHECKPOINT}" \
        --processed  "${PROCESSED_DIR}" \
        --out        predictions
fi

# ─── Step 7 : Metrics ─────────────────────────────────────────────────────────
# Uses gt/ symlinks written by step 6 — no --processed needed.

if step 7 "Compute metrics"; then
    python scripts/metrics.py \
        --inference "${PREDICTIONS_DIR}"
fi

# ─── Step 8 : Plot metrics ────────────────────────────────────────────────────

if step 8 "Plot metrics"; then
    python scripts/plot_metrics.py \
        --inference "${PREDICTIONS_DIR}"
fi

# ─── Step 9 : Find failures ───────────────────────────────────────────────────

if step 9 "Find failures"; then
    python scripts/find_failures.py \
        --inference "${PREDICTIONS_DIR}"
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Pipeline complete — results in ${PREDICTIONS_DIR}/"
echo "══════════════════════════════════════════════════════════════"
