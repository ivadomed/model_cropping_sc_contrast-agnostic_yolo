ARCHITECTURE DU PROJET — RÈGLES POUR CLAUDE CODE
=================================================

STYLE DE CODE
- Pas de gestion d'exceptions, pas de try/catch — le code plante sur inputs invalides
- Une seule responsabilité par fonction
- Code court, glue coding sur les primitives existantes
- SCT (spinalcordtoolbox) en priorité, nibabel en fallback

STRUCTURE GÉNÉRALE
- data/raw/ est en lecture seule, jamais écrit par du code
- processed/ contient uniquement les slices 2D extraites des volumes, pas de volumes
- predictions/ et reconstructions/ sont organisés par wandb run id
- sandbox/ est un espace de test, pas de sous-dossier run id, écrasé à chaque run
- gitignored : predictions/, reconstructions/, sandbox/, datasets/, checkpoints/

RESAMPLING
- images     → rpi + resampling spline ordre 3 via SCT
- masques GT → rpi + resampling linéaire ordre 1 via SCT
- résolution cible : médiane de la résolution sur le dataset

CONVENTIONS DE NOMMAGE
- les masques sont identifiés par le suffixe _mask.nii.gz
- les slices sont nommées slice_NNN.png / slice_NNN.txt (NNN = index zero-padded)
- les dossiers de run sont nommés par le wandb run id (ex: a3f8x2k1)

DATA/PROCESSED — hiérarchie exacte
  processed/
  └── <site>/
      └── <patient>/
          ├── png/                ← slices 2D extraites du volume resamplé
          │   ├── slice_000.png
          │   └── slice_001.png
          ├── txt/                ← labels YOLO ground truth par slice
          │   ├── slice_000.txt   format : class x_center y_center width height (normalisé)
          │   └── slice_001.txt
          └── volume/
              └── bbox_3d.txt     ← bbox 3D ground truth reconstruite depuis txt/

PREDICTIONS — hiérarchie exacte
  predictions/
  └── <run_id>/                   ← wandb run id
      └── <site>/
          └── <patient>/
              ├── slices/
              │   ├── png/        ← slices visualisées avec bbox prédites superposées
              │   │   ├── slice_000.png
              │   │   └── slice_001.png
              │   └── txt/        ← prédictions YOLO brutes par slice
              │       ├── slice_000.txt
              │       └── slice_001.txt
              └── volume/
                  └── bbox_3d.txt ← bbox 3D reconstruite depuis slices/txt/

RECONSTRUCTIONS — hiérarchie exacte
  reconstructions/
  └── <run_id>/                       ← wandb run id
      └── <site>/
          └── <patient>/
              ├── original.nii.gz         ← symlink -> data/raw/<site>/<patient>/
              ├── resampled.nii.gz        ← généré à la volée depuis raw/ :
              │                              rpi + resampling spline ordre 3 via SCT
              ├── mask_resampled.nii.gz   ← généré à la volée depuis raw/ :
              │                              rpi + resampling linéaire ordre 1 via SCT
              ├── pred_slices_stacked.nii.gz  ← volume binaire : empilement des bbox
              │                                  2D prédites (depuis predictions/<run_id>/)
              └── gt_slices_stacked.nii.gz    ← volume binaire : empilement des labels
                                                 YOLO GT (depuis processed/)
  - tous les volumes partagent le même espace/résolution que resampled.nii.gz
  - resampled.nii.gz et mask_resampled.nii.gz sont toujours générés depuis raw/,
    jamais copiés depuis processed/

SANDBOX — hiérarchie exacte
  sandbox/
  └── <patient>/                  ← pas de run id, usage test uniquement
      ├── png/
      ├── txt/
      ├── volume/
      │   └── bbox_3d.txt
      ├── slices/
      │   ├── png/
      │   └── txt/
      ├── volume_pred/
      │   └── bbox_3d.txt
      ├── resampled.nii.gz
      ├── mask_resampled.nii.gz
      ├── pred_slices_stacked.nii.gz
      └── gt_slices_stacked.nii.gz

DATASETS — généré par build_dataset.py, jamais versionné
  datasets/
  ├── dataset.yaml                ← config YOLO (classes, chemins)
  ├── images/
  │   ├── train/                  ← symlinks vers processed/<site>/<patient>/png/
  │   ├── val/
  │   └── test/
  └── labels/
      ├── train/                  ← symlinks vers processed/<site>/<patient>/txt/
      ├── val/
      └── test/

SPLITS — un yaml par site + un global
  data/splits/
  ├── site_a.yaml
  ├── site_b.yaml
  └── global.yaml

  format d'un split yaml :
    train: [site_a/patient_01, site_a/patient_02, ...]
    val:   [site_a/patient_03, ...]
    test:  [site_a/patient_04, ...]
  les chemins sont relatifs depuis data/processed/

SCRIPTS — un script, une responsabilité
  scripts/
  ├── preprocess.py     ← data/raw/ → data/processed/ (png/ + txt/ + volume/bbox_3d.txt)
  ├── build_dataset.py  ← data/processed/ + data/splits/ → datasets/ (symlinks)
  ├── train.py          ← datasets/ → checkpoints/<run_id>/
  ├── infer.py          ← datasets/ + checkpoints/ → predictions/<run_id>/
  ├── reconstruct.py    ← predictions/<run_id>/ + data/raw/ → reconstructions/<run_id>/
  ├── evaluate.py       ← predictions/<run_id>/ vs processed/ → métriques wandb
  └── run_pipeline.py   ← image + masque → sandbox/ (pipeline complète pour test)