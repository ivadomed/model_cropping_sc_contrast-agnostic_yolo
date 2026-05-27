#!/bin/bash
# ============================================================
# sc-crop model release — full pipeline
#
# Modifie UNIQUEMENT les variables ci-dessous entre deux releases.
# Le reste est automatique.
#
# Usage:
#   bash scripts/release.sh          # export + release complet
#   bash scripts/release.sh --export-only   # export seul (sans push/release)
#   bash scripts/release.sh --release-only  # release seul (si export déjà fait)
# ============================================================
set -euo pipefail

# ── VARIABLES À MODIFIER ENTRE DEUX RELEASES ────────────────
DET_RUN="runs/20260524_224406"        # run du détecteur
CLS_RUN="runs/20260525_150625"        # run du classifieur
MODEL_VERSION="0.0.6"                 # version du modèle  → tag vMODEL_VERSION sur sc-crop
PACKAGE_VERSION="0.1.5"              # version du package → tag vPACKAGE_VERSION sur sc-crop
DET_CHECKPOINT="best.pt"             # poids détecteur  : best.pt | last.pt
CLS_CHECKPOINT="best.pt"             # poids classifieur: best.pt | loss_best.pt | last.pt
# ─────────────────────────────────────────────────────────────

TRAINING_REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCCROP_REPO="/home/quentinr/sc-crop"
PYTHON="/home/quentinr/.conda/envs/contrast_agnostic/bin/python"
OUT_DIR="${TRAINING_REPO}/release_export"

EXPORT_ONLY=false
RELEASE_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --export-only)  EXPORT_ONLY=true ;;
        --release-only) RELEASE_ONLY=true ;;
    esac
done

# ── Couleurs ─────────────────────────────────────────────────
GRN="\033[32m"; YLW="\033[33m"; BLD="\033[1m"; RST="\033[0m"
step() { echo -e "\n${BLD}${GRN}▶ $*${RST}"; }
info() { echo -e "  ${YLW}$*${RST}"; }


# ════════════════════════════════════════════════════════════
# PHASE 1 — EXPORT (ONNX + SHA256)
# ════════════════════════════════════════════════════════════
if [ "$RELEASE_ONLY" = false ]; then
    step "Phase 1 — Export des modèles (ONNX + SHA256)"
    info "Détecteur  : ${DET_RUN}"
    info "Classifieur: ${CLS_RUN}"
    info "Sortie     : ${OUT_DIR}"

    "$PYTHON" "${TRAINING_REPO}/scripts/export_model.py" \
        --run-dir        "${TRAINING_REPO}/${DET_RUN}" \
        --cls-run-dir    "${TRAINING_REPO}/${CLS_RUN}" \
        --version        "${MODEL_VERSION}" \
        --out-dir        "${OUT_DIR}" \
        --det-checkpoint "${DET_CHECKPOINT}" \
        --cls-checkpoint "${CLS_CHECKPOINT}"

    echo ""
    info "Fichiers produits :"
    ls -lh "${OUT_DIR}"
fi

[ "$EXPORT_ONLY" = true ] && { echo -e "\n${GRN}Export terminé. Lance sans --export-only pour continuer.${RST}"; exit 0; }


# ════════════════════════════════════════════════════════════
# PHASE 2 — LECTURE DU SHA256
# ════════════════════════════════════════════════════════════
step "Phase 2 — Lecture des SHA256"

SHA256_YAML="${OUT_DIR}/sha256.yaml"
[ -f "$SHA256_YAML" ] || { echo "ERREUR : ${SHA256_YAML} introuvable. Lance d'abord l'export."; exit 1; }

SHA_MODEL_ONNX=$(  "$PYTHON" -c "import yaml; d=yaml.safe_load(open('${SHA256_YAML}')); print(d['assets']['model.onnx'])")
SHA_MODEL_PT=$(    "$PYTHON" -c "import yaml; d=yaml.safe_load(open('${SHA256_YAML}')); print(d['assets']['model.pt'])")
SHA_CLS_ONNX=$(    "$PYTHON" -c "import yaml; d=yaml.safe_load(open('${SHA256_YAML}')); print(d['assets']['cls_model.onnx'])")
SHA_CLS_PT=$(      "$PYTHON" -c "import yaml; d=yaml.safe_load(open('${SHA256_YAML}')); print(d['assets']['cls_model.pt'])")
EXPORT_GIT_HASH=$( "$PYTHON" -c "import yaml; d=yaml.safe_load(open('${SHA256_YAML}')); print(d['export_git_hash'])")

info "model.onnx     : ${SHA_MODEL_ONNX}"
info "model.pt       : ${SHA_MODEL_PT}"
info "cls_model.onnx : ${SHA_CLS_ONNX}"
info "cls_model.pt   : ${SHA_CLS_PT}"


# ════════════════════════════════════════════════════════════
# PHASE 3 — RELEASE GITHUB SUR sc-crop
# ════════════════════════════════════════════════════════════
step "Phase 3 — Création de la release GitHub (modèle v${MODEL_VERSION})"

gh release create "v${MODEL_VERSION}" \
    "${OUT_DIR}/model.pt"       \
    "${OUT_DIR}/model.onnx"     \
    "${OUT_DIR}/cls_model.pt"   \
    "${OUT_DIR}/cls_model.onnx" \
    --repo ivadomed/sc-crop \
    --title "sc-crop model v${MODEL_VERSION}" \
    --notes "det=${DET_RUN##*/}  cls=${CLS_RUN##*/}  export_commit=${EXPORT_GIT_HASH:0:12}"

info "Release v${MODEL_VERSION} créée sur ivadomed/sc-crop"


# ════════════════════════════════════════════════════════════
# PHASE 4 — TAG DU REPO DE TRAINING
# ════════════════════════════════════════════════════════════
step "Phase 4 — Tag du repo de training (model-v${MODEL_VERSION})"

git -C "${TRAINING_REPO}" tag "model-v${MODEL_VERSION}"
git -C "${TRAINING_REPO}" push origin "model-v${MODEL_VERSION}"

info "Tag model-v${MODEL_VERSION} créé sur $(git -C ${TRAINING_REPO} remote get-url origin)"


# ════════════════════════════════════════════════════════════
# PHASE 5 — MISE À JOUR DE sc-crop/download.py
# ════════════════════════════════════════════════════════════
step "Phase 5 — Mise à jour de sc_crop/download.py"

"$PYTHON" - <<PYEOF
import re, textwrap
from pathlib import Path

path = Path("${SCCROP_REPO}/sc_crop/download.py")
src  = path.read_text()

new_assets = textwrap.dedent("""
    _MODEL_TAG = "v${MODEL_VERSION}"
    _BASE_URL = f"https://github.com/ivadomed/sc-crop/releases/download/{_MODEL_TAG}"

    _ASSETS = {
        "model.onnx": {
            "url": f"{_BASE_URL}/model.onnx",
            "sha256": "${SHA_MODEL_ONNX}",
        },
        "cls_model.onnx": {
            "url": f"{_BASE_URL}/cls_model.onnx",
            "sha256": "${SHA_CLS_ONNX}",
        },
        "model.pt": {
            "url": f"{_BASE_URL}/model.pt",
            "sha256": "${SHA_MODEL_PT}",
        },
        "cls_model.pt": {
            "url": f"{_BASE_URL}/cls_model.pt",
            "sha256": "${SHA_CLS_PT}",
        },
    }
""").strip()

# Replace from _MODEL_TAG through end of _ASSETS block
src = re.sub(
    r'_MODEL_TAG\s*=.*?(?=\n_CACHE_DIR)',
    new_assets + "\n",
    src,
    flags=re.DOTALL,
)
path.write_text(src)
print("  download.py mis à jour")
PYEOF


# ════════════════════════════════════════════════════════════
# PHASE 6 — MISE À JOUR DE VERSIONS.md
# ════════════════════════════════════════════════════════════
step "Phase 6 — Mise à jour de VERSIONS.md"

VERSIONS_MD="${SCCROP_REPO}/VERSIONS.md"
NEW_ROW="| v${PACKAGE_VERSION} | v${MODEL_VERSION} | det=${DET_RUN##*/}  cls=${CLS_RUN##*/} | \`${EXPORT_GIT_HASH:0:12}\` |"

# Insert new row just after the header row (second | line)
"$PYTHON" - <<PYEOF
from pathlib import Path

path = Path("${VERSIONS_MD}")
lines = path.read_text().splitlines()

# Find the separator row (|---|---| line) and insert after it
insert_after = next(i for i, l in enumerate(lines) if l.startswith("|---"))
lines.insert(insert_after + 1, "${NEW_ROW}")
path.write_text("\n".join(lines) + "\n")
print("  VERSIONS.md mis à jour")
PYEOF


# ════════════════════════════════════════════════════════════
# PHASE 7 — BUMP VERSION DU PACKAGE sc-crop
# ════════════════════════════════════════════════════════════
step "Phase 7 — Bump version package → v${PACKAGE_VERSION}"

sed -i "s/^version\s*=\s*\".*\"/version     = \"${PACKAGE_VERSION}\"/" "${SCCROP_REPO}/pyproject.toml"
sed -i "s/__version__\s*=\s*\".*\"/__version__       = \"${PACKAGE_VERSION}\"/" "${SCCROP_REPO}/sc_crop/__init__.py"

info "pyproject.toml  : version = \"${PACKAGE_VERSION}\""
info "__init__.py     : __version__ = \"${PACKAGE_VERSION}\""


# ════════════════════════════════════════════════════════════
# PHASE 8 — COMMIT + TAG + PUSH sc-crop
# ════════════════════════════════════════════════════════════
step "Phase 8 — Commit + tag v${PACKAGE_VERSION} + push sc-crop"

git -C "${SCCROP_REPO}" add \
    sc_crop/download.py \
    sc_crop/__init__.py \
    pyproject.toml \
    VERSIONS.md

git -C "${SCCROP_REPO}" commit \
    -m "release: v${PACKAGE_VERSION} — model v${MODEL_VERSION} (det=${DET_RUN##*/} cls=${CLS_RUN##*/})"

git -C "${SCCROP_REPO}" tag "v${PACKAGE_VERSION}"
git -C "${SCCROP_REPO}" push
git -C "${SCCROP_REPO}" push --tags

info "Tag v${PACKAGE_VERSION} poussé sur $(git -C ${SCCROP_REPO} remote get-url origin)"


# ════════════════════════════════════════════════════════════
# RÉSUMÉ
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${BLD}${GRN}════ Release terminée ════${RST}"
echo -e "  Modèle     : ${GRN}v${MODEL_VERSION}${RST}  → https://github.com/ivadomed/sc-crop/releases/tag/v${MODEL_VERSION}"
echo -e "  Package    : ${GRN}v${PACKAGE_VERSION}${RST} → pip install git+https://github.com/ivadomed/sc-crop.git@v${PACKAGE_VERSION}"
echo -e "  Training   : tag model-v${MODEL_VERSION} sur $(git -C ${TRAINING_REPO} remote get-url origin)"
echo -e "  VERSIONS.md: entrée ajoutée"
