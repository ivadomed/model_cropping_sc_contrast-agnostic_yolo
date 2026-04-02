ARCHITECTURE DU PROJET — RÈGLES POUR CLAUDE CODE
=================================================

STYLE DE CODE
- Pas de gestion d'exceptions, pas de try/catch — le code plante sur inputs invalides
- Une seule responsabilité par fonction
- Code court : favoriser les primitives et librairies existantes (glue coding)
- Fichiers courts : un script = une responsabilité, pas d'abstractions prématurées
- nibabel pour NIfTI (SCT non requis)
- Pas de résolution cible codée en dur — le code plante si les données ne sont pas au format attendu
- Toujours mettre à jour le fichier CLAUDE.md du projet pour refléter exactement l'état du projet

PRINCIPE DE SIMPLIFICATION
- Pas de helpers inutiles : si une fonction n'est utilisée qu'une fois, l'inliner
- Pas de fallback : dict explicite par dataset plutôt que logique de découverte générique
- Pas de classes : fonctions + dicts suffisent pour ce projet
- Les stats comparables entre datasets nécessitent une orientation commune (LAS) ;
  la reorientation est virtuelle (permutation d'axes via ornt_transform, aucun voxel chargé)

ENVIRONNEMENT
- conda environment : contrast_agnostic
- toujours activer avant d'exécuter du code : conda activate contrast_agnostic
- ultralytics 8.4.23 (YOLO26 = modèle le plus récent, défaut : yolo26n.pt)
- albumentations v2 installé (API : std_range=, scale_range= au lieu de var_limit=, scale_min/max=)
- wandb installé, compte : quentin-revillon (neuropoly), project : spine_detection

ÉTAT DES DONNÉES
- processed_10mm_SI/ : COMPLET — tous datasets préprocessés à 10mm SI
- processed_1mm_SI/  : EN COURS — preprocessing lancé, pas encore terminé
- datasets/          : dataset YOLO 10mm construit depuis processed_10mm_SI/
- datasets_1mm_SI/   : à construire une fois processed_1mm_SI/ terminé

ENTRAÎNEMENTS RÉALISÉS
- yolo26_10mm_SI : premier run complet, 10mm SI, yolo26n, epochs 100, imgsz 640
  → sauvegardé dans runs/detect/checkpoints/yolo26_10mm_SI/weights/{best,last}.pt
  → problème observé : val/cls_loss diverge (~epoch 6), mAP50/mAP50-95 s'effondre
    cause suspectée : déséquilibre train/val sur slices vides (extrémités, cerveau)
    avec 10mm peu de slices par volume → sensible à la composition du split
  → best.pt correspond à l'epoch ~4-5 (peak mAP50-95 ~0.5)
- DÉCISION : train.py corrigé pour sauvegarder dans checkpoints/<run_id>/ (path absolu)
  les runs précédents sont dans runs/detect/checkpoints/

STRUCTURE GÉNÉRALE
- data/raw/ est en lecture seule, jamais écrit par du code
- processed_{res}mm_SI/ contient uniquement les slices 2D extraites des volumes, pas de volumes
- predictions/ et reconstructions/ sont organisés par run id
- sandbox/ est un espace de test, pas de sous-dossier run id, écrasé à chaque run
- gitignored : processed*/, predictions/, reconstructions/, sandbox/, datasets*/, checkpoints/, runs/,
               dataset_stats.csv, metrics_*.csv

PRÉPROCESSING
- réorientation LAS avant extraction des slices
- resampling de l'axe SI uniquement (axe 2 en LAS) via --si-res (ex: 1.0, 10.0)
  order=3 pour les images, order=0 pour les masques binaires (scipy.ndimage.zoom)
- export PNG : slices natives normalisées uint8, pas de resize ni de padding in-plane
- resize in-plane délégué à YOLO via le paramètre imgsz (training et inférence)
- dossier de sortie nommé automatiquement processed_{res}mm_SI (ex: processed_1mm_SI)
- Z = min(img, mask) sur l'axe SI : le zoom scipy arrondit indépendamment → peut différer d'1 voxel
- meta.yaml par patient : raw_image, raw_mask, shape_las [H,W,Z], si_res_mm

DÉCOUVERTE DES MASQUES — table explicite par dataset dans preprocess.py
  DATASET_MASK_SUFFIX = {
    "data-multi-subject":           "_label-SC_seg.nii.gz",
    "basel-mp2rage":                "_label-SC_seg.nii.gz",
    "dcm-zurich":                   "_label-SC_seg.nii.gz",
    "lumbar-vanderbilt":            "_label-SC_seg.nii.gz",
    "nih-ms-mp2rage":               "_label-SC_seg.nii.gz",
    "canproco":                     "_seg-manual.nii.gz",
    "sci-colorado":                 "_seg-manual.nii.gz",
    "sci-paris":                    "_seg-manual.nii.gz",
    "sci-zurich":                   "_seg-manual.nii.gz",
    "sct-testing-large":            "_seg-manual.nii.gz",
    "lumbar-epfl":                  "_seg-manual.nii.gz",
    "dcm-brno":                     "_seg.nii.gz",
    "dcm-zurich-lesions":           "_label-SC_mask-manual.nii.gz",
    "dcm-zurich-lesions-20231115":  "_label-SC_mask-manual.nii.gz",
  }
  - plante sur dataset inconnu (pas de fallback)
  - cherche toujours dans derivatives/labels/<sub>/[ses-*/]anat/
  - DWI exclu (pas de sous-dossier dwi/)

DATA SOURCES — structure par source dans data/raw/
  data-multi-subject/
    <subject>/anat/                       ← volumes (contraste variable)
    derivatives/
      labels/<subject>/
        anat/  ← *_label-SC_seg.nii.gz — utilisé
        dwi/   ← ignoré
      labels_softseg/                     ← ignoré

DATA/PROCESSED — hiérarchie exacte
  processed_{res}mm_SI/
  └── <dataset>/
      └── <subject>[_<contrast>]/
          ├── png/                ← slices 2D natives normalisées uint8
          │   └── slice_NNN.png
          ├── txt/                ← labels YOLO GT par slice
          │   └── slice_NNN.txt   format : class cx cy w h (normalisé [0,1])
          ├── volume/
          │   └── bbox_3d.txt     ← bbox 3D GT : row1 row2 col1 col2 z1 z2 (voxels)
          └── meta.yaml           ← raw_image, raw_mask, shape_las, si_res_mm

PREDICTIONS — hiérarchie exacte
  predictions/
  └── <run_id>/
      └── <dataset>/
          └── <patient>/
              ├── slices/
              │   ├── png/        ← slices visualisées avec bbox prédites superposées
              │   └── txt/        ← prédictions YOLO brutes par slice
              └── volume/
                  └── bbox_3d.txt ← bbox 3D reconstruite depuis slices/txt/

RECONSTRUCTIONS — hiérarchie exacte
  reconstructions/
  └── <run_id>/
      └── <dataset>/
          └── <patient>/
              ├── original.nii.gz             ← symlink -> data/raw/...
              ├── mask_original.nii.gz        ← symlink -> data/raw/...*_mask.nii.gz
              ├── pred_slices_stacked.nii.gz  ← volume binaire reconstruit depuis bbox prédites
              └── gt_slices_stacked.nii.gz    ← volume binaire reconstruit depuis labels GT

SANDBOX — hiérarchie exacte (usage test uniquement, écrasé à chaque run)
  sandbox/
  └── <patient>/
      ├── png/ txt/ volume/bbox_3d.txt          ← préprocessing GT
      ├── slices/png/ slices/txt/ volume_pred/  ← inférence
      ├── original.nii.gz mask_original.nii.gz  ← symlinks
      ├── pred_slices_stacked.nii.gz
      └── gt_slices_stacked.nii.gz

DATASETS — généré par build_dataset.py, jamais versionné
  datasets[_<suffix>]/
  ├── dataset.yaml                ← config YOLO (path absolu, classes, splits)
  ├── images/train/ val/ test/    ← symlinks plats vers processed/.../png/
  └── labels/train/ val/ test/    ← symlinks plats vers processed/.../txt/
  nommage symlink : <dataset>_<subject>[_<contrast>]_slice_NNN.png

SPLITS — un yaml par dataset (data/datasplits/)
  data/datasplits/datasplit_<dataset>_seed50.yaml
  format : train/val/test: [sub-xxx, ...]  (noms de sujets BIDS)
  build_dataset.py mappe sub-xxx → tous les dossiers sub-xxx_* dans processed/

TRAIN — décisions
- imgsz=320 (retour au défaut de l'ancien modèle qui convergeait — feature maps plus petites,
  moins d'ancres négatives, bbox SC proportionnellement plus large → convergence plus stable)
- augmentations MRI via albumentations : GaussNoise, GaussianBlur, Downscale, RandomGamma
  injectées via callback on_train_start (inject_mri_augmentations)
- hsv_v=0.15, degrees=15, scale=0.2, translate=0.1, fliplr=0.5, flipud=0.5
- mosaic=0 (images médicales, pas de mosaïque)
- critère best.pt : mAP50-95 sur validation (fitness = 0.1*mAP50 + 0.9*mAP50-95)
- wandb.init() avant model.train() pour que ultralytics utilise le run existant
- project= passé en chemin absolu pour éviter runs/detect/ prefix ultralytics

SCRIPTS — un script, une responsabilité
  scripts/
  ├── explore_stats.py  ← data/raw/ → dataset_stats.csv
  │                       reorientation virtuelle LAS, rapporte shape/résolution/FOV mm
  ├── preprocess.py     ← data/raw/ → processed_{res}mm_SI/
  │                       --si-res obligatoire, réoriente LAS, rééchantillonne axe SI,
  │                       export PNG uint8 + txt YOLO + volume/bbox_3d.txt + meta.yaml
  ├── build_dataset.py  ← processed/ + data/datasplits/*.yaml → datasets/
  │                       --processed processed_10mm_SI --out datasets_10mm_SI
  ├── train.py          ← datasets/ → checkpoints/<run_id>/weights/{best,last}.pt
  │                       --model yolo26n.pt --epochs 100 --imgsz 640
  ├── infer.py          ← processed/ + checkpoint → predictions/<run_id>/
  ├── reconstruct.py    ← predictions/<run_id>/ + data/raw/ → reconstructions/<run_id>/
  ├── evaluate.py       ← processed/ + checkpoint → metrics_<run_id>.csv
  │                       inference directe conf=0.001 (courbe PR complète)
  │                       métriques 2D : IoU, Dice, recall50, precision50, f1_50, AP50, AP50:95
  │                       décomposées : global / split / dataset / dataset×contrast / dataset×contrast×split
  └── run_pipeline.py   ← image + masque + --si-res → sandbox/ (test end-to-end)
