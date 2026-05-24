#!/bin/bash
# Download TotalSegmentator v201 from Zenodo and build a BIDS-compatible symlink tree.
#
# Source : https://zenodo.org/records/10047292  (22 Go, 1228 sujets CT)
# Output :
#   data/raw/Totalsegmentator_dataset_v201/    ← ZIP extrait (fichiers réels)
#   data/raw/totalsegmentator/                 ← arbre BIDS avec symlinks
#     sub-s0001/anat/sub-s0001_ct.nii.gz       → .../s0001/ct.nii.gz
#     derivatives/labels/sub-s0001/anat/
#       sub-s0001_ct_label-SC_seg.nii.gz       → .../s0001/segmentations/spinal_cord.nii.gz
#
# Usage:
#   bash scripts/download_totalsegmentator.sh
#   bash scripts/download_totalsegmentator.sh --dest /chemin/alternatif

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # run depuis la racine du projet

# ── Paramètres ────────────────────────────────────────────────────────────────
DATA_DIR="${1:-data/raw}"
ZENODO_URL="https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip?download=1"
ZIP_FILE="$DATA_DIR/Totalsegmentator_dataset_v201.zip"
EXTRACT_DIR="$DATA_DIR/Totalsegmentator_dataset_v201"
BIDS_DIR="$DATA_DIR/totalsegmentator"
LOG="$DATA_DIR/git_branch_commit.log"

mkdir -p "$DATA_DIR"

# ── 1. Téléchargement ─────────────────────────────────────────────────────────
if [ -d "$EXTRACT_DIR" ]; then
    echo "-> Répertoire $EXTRACT_DIR déjà présent, téléchargement sauté."
elif [ -f "$ZIP_FILE" ]; then
    echo "-> ZIP déjà présent, extraction sautée (lancer l'étape 2 directement)."
else
    echo "=========================================="
    echo "Téléchargement TotalSegmentator v201 (~22 Go)"
    echo "=========================================="
    wget -c "$ZENODO_URL" -O "$ZIP_FILE"
fi

# ── 2. Extraction ─────────────────────────────────────────────────────────────
if [ ! -d "$EXTRACT_DIR" ] && [ -f "$ZIP_FILE" ]; then
    echo "=========================================="
    echo "Extraction du ZIP..."
    echo "=========================================="
    unzip -q "$ZIP_FILE" -d "$DATA_DIR"
    echo "-> Extraction terminée : $EXTRACT_DIR"
fi

[ ! -d "$EXTRACT_DIR" ] && { echo "ERROR: $EXTRACT_DIR introuvable après extraction."; exit 1; }

# ── 3. Structure BIDS (symlinks) ──────────────────────────────────────────────
if [ -d "$BIDS_DIR" ]; then
    echo "-> Structure BIDS $BIDS_DIR déjà présente, sautée."
else
    echo "=========================================="
    echo "Création de la structure BIDS (symlinks)..."
    echo "=========================================="
    python - "$EXTRACT_DIR" "$BIDS_DIR" <<'PYEOF'
import sys
from pathlib import Path

source_dir = Path(sys.argv[1]).resolve()
bids_dir   = Path(sys.argv[2]).resolve()

n_ok = n_skip = 0
for subj_dir in sorted(source_dir.iterdir()):
    if not subj_dir.is_dir():
        continue
    ct      = subj_dir / "ct.nii.gz"
    sc_mask = subj_dir / "segmentations" / "spinal_cord.nii.gz"
    if not ct.exists() or not sc_mask.exists():
        n_skip += 1
        continue

    sub_id   = f"sub-{subj_dir.name}"   # s0001 → sub-s0001

    # Image
    anat_dir = bids_dir / sub_id / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    img_link = anat_dir / f"{sub_id}_ct.nii.gz"
    if not img_link.exists():
        img_link.symlink_to(ct)

    # Masque SC
    label_dir = bids_dir / "derivatives" / "labels" / sub_id / "anat"
    label_dir.mkdir(parents=True, exist_ok=True)
    mask_link = label_dir / f"{sub_id}_ct_label-SC_seg.nii.gz"
    if not mask_link.exists():
        mask_link.symlink_to(sc_mask)

    n_ok += 1

print(f"   {n_ok} sujets traités, {n_skip} sautés (ct.nii.gz ou spinal_cord.nii.gz manquant).")
PYEOF
    echo "-> Structure BIDS créée : $BIDS_DIR"
fi

# ── 4. Log ────────────────────────────────────────────────────────────────────
if ! grep -q "totalsegmentator" "$LOG" 2>/dev/null; then
    echo "totalsegmentator: zenodo-10047292-v201" >> "$LOG"
fi

echo ""
echo "=========================================="
echo "TotalSegmentator prêt dans $BIDS_DIR"
echo "=========================================="
