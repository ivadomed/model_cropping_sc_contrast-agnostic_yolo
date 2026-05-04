# sc_crop — architecture et workflow

## Pattern de distribution (analogue à SCT)

| | SCT | sc_crop standalone |
|---|---|---|
| Modèle stocké dans | `$SCT_DIR/data/sc_crop_models/` | `~/.sc_crop/sc_crop_models/` |
| Résolution du chemin | `sct_dir_local_path('data', 'sc_crop_models', 'model.onnx')` | `ensure_model()` dans `sc_crop/download.py` |
| Téléchargement explicite | `sct_download_data -d sc_crop_models` | `sc-crop download` |
| Téléchargement automatique | non | oui, au premier `sc-crop image.nii.gz` |
| Modèle absent au runtime | `FileNotFoundError` | auto-download puis inférence |

---

## Workflow développeur (entraînement → release)

```bash
# 1. Lancer un run complet
python scripts/run_pipeline.py

# 2. Exporter le meilleur checkpoint en zip de release
conda activate contrast_agnostic
python scripts/export_model.py --run-dir runs/20260504_134652 --version 0.2.0
# → sc_crop_models_v0.2.0.zip  contient model.onnx + config.yaml

# 3. Attacher le zip à la GitHub release v0.2.0

# 4. Mettre à jour _RELEASE_URL dans sc_crop/sc_crop/download.py
#    "…/releases/download/v0.2.0/sc_crop_models_v0.2.0.zip"
```

---

## Workflow utilisateur final (sans SCT)

```bash
pip install -e /path/to/repo/sc_crop
sc-crop download          # télécharge dans ~/.sc_crop/sc_crop_models/
sc-crop t2.nii.gz         # produit t2_crop.nii.gz au même endroit
```

---

## Migration SCT (PR future)

Dans `/home/quentinr/spinalcordtoolbox/spinalcordtoolbox/sc_crop/core.py`,
remplacer les deux lignes de résolution du modèle :

```python
# MAINTENANT — standalone
from sc_crop.download import ensure_model
model_path = ensure_model()          # → ~/.sc_crop/sc_crop_models/model.onnx
config     = load_config(model_path.parent)

# APRÈS MIGRATION — SCT natif
from spinalcordtoolbox.utils.sys import sct_dir_local_path
model_path = Path(sct_dir_local_path('data', 'sc_crop_models', 'model.onnx'))
config     = load_config(model_path.parent)
```

Ajouter l'entrée dans `/home/quentinr/spinalcordtoolbox/spinalcordtoolbox/download.py` :

```python
"sc_crop_models": {
    "mirrors": [
        "https://github.com/neuropoly/model_cropping_sc_contrast-agnostic_yolo"
        "/releases/download/v0.2.0/sc_crop_models_v0.2.0.zip"
    ],
    "default_location": os.path.join(__sct_dir__, "data", "sc_crop_models"),
    "download_type": "Models",
    "default": False,
},
```

`sct_crop_sc` est déjà enregistré dans `setup.py` (ligne 76). Rien d'autre à faire.

---

## Structure des fichiers

```
repo/
├── scripts/
│   └── export_model.py          ← dev : produit le zip de release
└── sc_crop/
    ├── pyproject.toml
    └── sc_crop/
        ├── crop.py              ← moteur d'inférence (zéro dépendance SCT)
        ├── download.py          ← ensure_model(), download() → ~/.sc_crop/
        ├── cli.py               ← sc-crop image.nii.gz | sc-crop download
        └── models/
            └── config.yaml      ← versionné (template), model.onnx gitignored

spinalcordtoolbox/spinalcordtoolbox/
└── sc_crop/
    ├── core.py                  ← crop_sc() : wrapping aujourd'hui, natif après PR
    ├── download.py              ← supprimé lors de la PR SCT
    └── INTEGRATION.md           ← détails de l'intégration SCT
```
