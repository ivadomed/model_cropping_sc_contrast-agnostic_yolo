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

PLANE=axial       # axial | sagittal
START_STEP=1      # 1–9
END_STEP=9        # 1–9
MAKE_SPLITS=true # true = regenerate splits from data/raw (step 2)
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
    # lumbar-epfl:7
    # lumbar-vanderbilt:7
    # spider-challenge-2023:3
    # whole-spine:3
    # sct-testing-large:1
    # sci-zurich:1
    # canproco:1
)

# Oversample ×2 slices with no SC or within this distance of the SC boundary (mm).
# Set to "" to disable.
BORDER_OVERSAMPLE_MM=""

# ─── Parameters — TRAINING (shared) ──────────────────────────────────────────

WITH_CANAL=false         # true = also extract spinal canal (class 1) during preprocessing

MODEL="yolo26n.pt"
EPOCHS=200
IMGSZ=320
FL_GAMMA=2               # focal loss gamma (0 = standardq BCE)
WORKERS=2
EXTRA_AUGMENT=false       # true = extra MRI albumentations (bias field, invert, affine zoom…)
WANDB_ENTITY="quentin-revillon-neuropoly" # W&B entity: "" = compte par défaut, "quentin-revillon-neuropoly" = compte perso, "neuropoly" = lab

# ─── Path overrides (optional) ───────────────────────────────────────────────
# Leave empty to use auto-generated paths.
# Use when resuming from an existing directory created outside this pipeline.
# NOTE: overrides only affect steps 4+ (build, train, eval, metrics, failures).
#       Step 3 (preprocess) always writes to the auto-generated PROCESSED_DIR.

OVERRIDE_PROCESSED_DIR=""   # e.g. processed/10mm_SI_1mm_axial_3ch
OVERRIDE_DATASET_DIR=""     # e.g. datasets/10mm_SI_1mm_axial_3ch
OVERRIDE_RUN_ID=""          # e.g. yolo26n_axial_v2

# ─── Derived values (do not edit) ─────────────────────────────────────────────

# Timestamp generated once at pipeline start — shared by DATASET_DIR and RUN_ID.
# Override via OVERRIDE_DATASET_DIR / OVERRIDE_RUN_ID to resume an existing run.
TS=$(date +%Y%m%d_%H%M%S)

SPLITS_DIR="data/datasplits_seed${SEED}"

if [[ "$PLANE" == "axial" ]]; then

    PROCESSED_DIR="processed/pipeline_${AXIAL_SI_RES%.*}mm_SI_${AXIAL_INPLANE_RES%.*}mm_axial_3ch"
    DATASET_DIR="datasets/pipeline_${AXIAL_SI_RES%.*}mm_SI_${AXIAL_INPLANE_RES%.*}mm_axial_3ch_${TS}"
    RUN_ID="pipeline_yolo26n_axial_${TS}"

    PREPROCESS_ARGS=(
        --si-res    "${AXIAL_SI_RES}"
        --axial-res "${AXIAL_INPLANE_RES}"
        --3ch
        --out       "${PROCESSED_DIR}"
    )
    BUILD_DATASET_EXTRA_ARGS=()

elif [[ "$PLANE" == "sagittal" ]]; then

    PROCESSED_DIR="processed/pipeline_${SAG_SI_RES%.*}mm_SI_${SAG_INPLANE_RES%.*}mm_axial_${SAG_INPLANE_RES%.*}mm_RL_3ch_sagittal_sc${SAG_SC_PAD}mm"
    DATASET_DIR="datasets/pipeline_${SAG_SI_RES%.*}mm_SI_${SAG_INPLANE_RES%.*}mm_axial_${SAG_INPLANE_RES%.*}mm_RL_3ch_sagittal_sc${SAG_SC_PAD}mm_${TS}"
    RUN_ID="pipeline_yolo26n_sagittal_sc${SAG_SC_PAD}mm_${EPOCHS}ep_${TS}"

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
[[ -n "$BORDER_OVERSAMPLE_MM" ]] && BUILD_DATASET_ARGS+=(--border-oversample "${BORDER_OVERSAMPLE_MM}")

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

    # Write pipeline_config.yaml for W&B traceability (read by train.py at step 5)
    export PIPELINE_PLANE="$PLANE"
    export PIPELINE_DATASET_DIR="$DATASET_DIR"
    export PIPELINE_DATASET_FACTORS="${DATASET_FACTORS[*]}"
    export PIPELINE_BORDER_MM="$BORDER_OVERSAMPLE_MM"
    export PIPELINE_MODEL="$MODEL"
    export PIPELINE_EPOCHS="$EPOCHS"
    export PIPELINE_IMGSZ="$IMGSZ"
    export PIPELINE_FL_GAMMA="$FL_GAMMA"
    export PIPELINE_WORKERS="$WORKERS"
    export PIPELINE_WITH_CANAL="$WITH_CANAL"
    export PIPELINE_PROCESSED_DIR="$PROCESSED_DIR"
    export PIPELINE_RUN_ID="$RUN_ID"
    if [[ "$PLANE" == "axial" ]]; then
        export PIPELINE_SI_RES="$AXIAL_SI_RES"
        export PIPELINE_INPLANE_RES="$AXIAL_INPLANE_RES"
        export PIPELINE_SC_PAD=""
        export PIPELINE_SC_RATIO=""
    else
        export PIPELINE_SI_RES="$SAG_SI_RES"
        export PIPELINE_INPLANE_RES="$SAG_INPLANE_RES"
        export PIPELINE_SC_PAD="$SAG_SC_PAD"
        export PIPELINE_SC_RATIO="$SAG_SC_RATIO"
    fi
    python - <<'PYEOF'
import os, yaml
plane        = os.environ["PIPELINE_PLANE"]
dataset_dir  = os.environ["PIPELINE_DATASET_DIR"]
factors      = {}
for entry in os.environ.get("PIPELINE_DATASET_FACTORS", "").split():
    k, v = entry.split(":")
    factors[k] = float(v)
border_mm_str = os.environ.get("PIPELINE_BORDER_MM", "")
config = {
    "plane":               plane,
    "seed":                int(os.environ["SEED"]),
    "model":               os.environ["PIPELINE_MODEL"],
    "epochs":              int(os.environ["PIPELINE_EPOCHS"]),
    "imgsz":               int(os.environ["PIPELINE_IMGSZ"]),
    "fl_gamma":            float(os.environ["PIPELINE_FL_GAMMA"]),
    "workers":             int(os.environ["PIPELINE_WORKERS"]),
    "with_canal":          os.environ["PIPELINE_WITH_CANAL"] == "true",
    "dataset_factors":     factors,
    "border_oversample_mm": float(border_mm_str) if border_mm_str else None,
    "processed_dir":       os.environ["PIPELINE_PROCESSED_DIR"],
    "dataset_dir":         dataset_dir,
    "run_id":              os.environ["PIPELINE_RUN_ID"],
}
if plane == "axial":
    config["si_res_mm"]      = float(os.environ["PIPELINE_SI_RES"])
    config["inplane_res_mm"] = float(os.environ["PIPELINE_INPLANE_RES"])
else:
    config["si_res_mm"]      = float(os.environ["PIPELINE_SI_RES"])
    config["inplane_res_mm"] = float(os.environ["PIPELINE_INPLANE_RES"])
    config["sc_pad_mm"]      = float(os.environ.get("PIPELINE_SC_PAD", 0))
    config["sc_ratio"]       = int(os.environ.get("PIPELINE_SC_RATIO", 0))
with open(f"{dataset_dir}/pipeline_config.yaml", "w") as f:
    yaml.dump(config, f, sort_keys=False, default_flow_style=False)
print(f"Pipeline config written to {dataset_dir}/pipeline_config.yaml")
PYEOF
fi

# ─── Step 5 : Train ───────────────────────────────────────────────────────────

if step 5 "Train — ${MODEL} × ${EPOCHS} epochs (fl-gamma=${FL_GAMMA})"; then
    python scripts/train.py \
        --dataset-yaml "${DATASET_DIR}/dataset.yaml" \
        --run-id       "${RUN_ID}" \
        --model        "${MODEL}" \
        --epochs       "${EPOCHS}" \
        --imgsz        "${IMGSZ}" \
        --workers      "${WORKERS}" \
        --fl-gamma     "${FL_GAMMA}" \
        ${WANDB_ENTITY:+--wandb-entity "${WANDB_ENTITY}"} \
        $([[ "$EXTRA_AUGMENT" == "false" ]] && echo "--no-extra-augment")
fi

# ─── Step 6 : Evaluate ────────────────────────────────────────────────────────
# Writes predictions/<RUN_ID>/<dataset>/<patient>/txt|png|volume + gt/ symlink.

if step 6 "Evaluate — inference on all patients"; then
    python scripts/evaluate.py \
        --checkpoint "${CHECKPOINT}" \
        --processed  "${PROCESSED_DIR}" \
        --out        predictions || true
fi

# ─── Step 7 : Metrics ─────────────────────────────────────────────────────────
# Uses gt/ symlinks written by step 6 — no --processed needed.

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
echo "══════════════════════════════════════════════════════════════"
