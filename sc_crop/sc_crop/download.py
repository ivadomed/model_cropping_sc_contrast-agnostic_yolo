"""
Model download for sc_crop (standalone).

Pattern identique à SCT :
  SCT      : modèle dans $SCT_DIR/data/sc_crop_models/  via sct_dir_local_path()
  Standalone: modèle dans ~/.sc_crop/sc_crop_models/    via ensure_model()

Lors de l'intégration SCT, la seule ligne qui change dans core.py est :
  - from sc_crop.download import ensure_model; model_path = ensure_model()
  + from spinalcordtoolbox.utils.sys import sct_dir_local_path
  + model_path = Path(sct_dir_local_path('data', 'sc_crop_models', 'model.onnx'))

Usage:
    sc-crop download
    from sc_crop.download import ensure_model; model_path = ensure_model()
"""

import urllib.request
import zipfile
from pathlib import Path

# Mis à jour à chaque release.
_RELEASE_URL = (
    "https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo"
    "/releases/download/v0.2.0/sc_crop_models_v0.2.0.zip"
)

_DATA_DIR = Path.home() / ".sc_crop" / "sc_crop_models"


def ensure_model() -> Path:
    """Retourne le chemin vers model.onnx, télécharge depuis la release si absent."""
    model_path = _DATA_DIR / "model.onnx"
    if model_path.exists():
        return model_path
    download()
    return model_path


def download() -> None:
    """Télécharge et décompresse le zip de release dans ~/.sc_crop/sc_crop_models/."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = _DATA_DIR / "_download.zip"

    print(f"Downloading sc_crop model from release …")
    urllib.request.urlretrieve(_RELEASE_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(_DATA_DIR)
    zip_path.unlink()

    if not (_DATA_DIR / "model.onnx").exists():
        raise RuntimeError(
            f"model.onnx absent de {_DATA_DIR} après téléchargement — "
            "vérifier la structure du zip de release."
        )
    print(f"Model saved to {_DATA_DIR}")
