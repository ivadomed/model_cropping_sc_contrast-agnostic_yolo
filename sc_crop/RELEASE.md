# sc_crop — release et utilisation

## Prérequis serveur

### Installer gh (GitHub CLI) sans sudo

```bash
conda install -c conda-forge gh
```

### Authentifier gh

1. Sur `https://github.com/settings/tokens/new` :
   - **Token name** : `tassan-server`
   - **Expiration** : `No expiration`
   - **Scopes** : cocher uniquement `repo`
   - Cliquer **Generate token** et copier la chaîne (`ghp_XXXX...`)
   - ⚠️ Ne jamais partager ce token — le révoquer immédiatement sur `https://github.com/settings/tokens` s'il est compromis

2. Sur le serveur :

```bash
gh auth login --with-token <<< "ghp_XXXXXXXXXXXX"
gh auth status   # vérifier
```

---

## 1. Serveur — exporter le modèle

```bash
conda activate contrast_agnostic
cd /home/quentinr/model_cropping_sc_contrast-agnostic_yolo
python scripts/export_model.py --run-dir runs/20260504_134652 --version 0.1.0
# → sc_crop_models_v0.1.0.zip dans le dossier courant
```

---

## 2. Serveur — mettre à jour l'URL de release

Dans `sc_crop/sc_crop/download.py`, mettre à jour `_RELEASE_URL` :

```python
_RELEASE_URL = (
    "https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo"
    "/releases/download/v0.1.0/sc_crop_models_v0.1.0.zip"
)
```

Puis committer et pusher :

```bash
git add sc_crop/sc_crop/download.py
git commit -m "sc_crop: set release URL v0.1.0"
git push
```

---

## 3. Serveur — créer la release GitHub et uploader le zip

```bash
gh release create v0.1.0 sc_crop_models_v0.1.0.zip \
  --title "sc_crop v0.1.0" \
  --notes "First model release — axial, 3ch, si_res=10mm, inplane_res=1mm"
```

---

## 4. PC local — cloner le repo et installer le package

```bash
git clone git@github.com:ivadomed/model_cropping_sc_contrast-agnostic_yolo.git
cd model_cropping_sc_contrast-agnostic_yolo
pip install -e sc_crop/
```

---

## 5. PC local — télécharger le modèle

```bash
sc-crop download
# → télécharge dans ~/.sc_crop/sc_crop_models/
```

---

## 6. PC local — lancer l'inférence

```bash
sc-crop t2.nii.gz
# → t2_crop.nii.gz au même endroit, même orientation/résolution/espace
```

Options disponibles :

```bash
sc-crop t2.nii.gz -o output_crop.nii.gz   # chemin de sortie explicite
sc-crop t2.nii.gz --padding 20            # marge autour de la bbox (mm, défaut 10)
sc-crop t2.nii.gz --conf 0.05             # seuil de confiance (défaut 0.1)
sc-crop t2.nii.gz --device cpu            # forcer CPU
```

---

## Mettre à jour le modèle (nouvelle release)

Reprendre depuis l'étape 1 avec un nouveau `--version`, mettre à jour `_RELEASE_URL`,
et les utilisateurs relancent `sc-crop download` pour obtenir le nouveau modèle.
