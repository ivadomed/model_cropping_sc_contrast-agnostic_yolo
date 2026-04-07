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
- meta.yaml par patient : raw_image, raw_mask, shape_las [H,W,Z], si_res_mm, rl_res_mm, ap_res_mm
  rl_res_mm/ap_res_mm = résolution native dans le plan axial (LAS axes 0 et 1), non modifiée par resampling
  pour patcher des meta.yaml existants sans re-préprocesser : preprocess.py --update-meta --out <dir>

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
          ├── txt/                ← labels YOLO GT par slice (toujours présent, vide si pas de SC)
          │   └── slice_NNN.txt   format : "0 cx cy w h" normalisé [0,1], ou vide
          ├── volume/
          │   └── bbox_3d.txt     ← bbox 3D GT : row1 row2 col1 col2 z1 z2 (voxels)
          └── meta.yaml           ← raw_image, raw_mask, shape_las [H,W,Z],
                                     si_res_mm, rl_res_mm, ap_res_mm

PREDICTIONS — deux hiérarchies selon le script d'origine

  evaluate.py → structure miroir de processed/ (utilisée par metrics.py et find_failures.py)
  predictions/
  └── <run_id>/
      └── <dataset>/
          └── <patient>/
              ├── png/            ← overlay GT (vert) + pred (rouge) par slice
              │   └── slice_NNN.png
              ├── txt/            ← prédiction par slice (vide si pas de détection)
              │   └── slice_NNN.txt  format : "0 cx cy w h conf" (6 champs, conf ajouté)
              └── volume/
                  └── bbox_3d.txt ← bbox 3D reconstruite depuis txt/

  infer.py → structure avec sous-dossier slices/ (utilisée par reconstruct.py)
  predictions/
  └── <run_id>/
      └── <dataset>/
          └── <patient>/
              ├── slices/
              │   ├── png/        ← slices avec bbox pred superposée
              │   └── txt/        ← prédictions YOLO brutes (5 champs, sans conf)
              └── volume/
                  └── bbox_3d.txt

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
  metrics.py/find_failures.py : sujets absents du split → marqués "unknown" dans le CSV

  CAS CONNUS DE SUJETS "UNKNOWN" :
  - nih-ms-mp2rage      : aucun fichier datasplit → tous les sujets sont unknown par construction
  - dcm-zurich          : le split contient des IDs type "sub-260155" mais processed/ contient
                          "sub-001", "sub-002" (renommage lors du téléchargement) → zéro match
  - sct-testing-large   : le split couvre un sous-ensemble de sites (amuVirginie, karoTobiasMS…)
                          processed/ contient aussi d'autres sites (amuAMU15, amuPAM50…) non splittés

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
  │                       meta.yaml inclut : shape_las, si_res_mm, rl_res_mm, ap_res_mm
  │                       --update-meta --out <dir> : patche les meta.yaml existants sans re-préprocesser
  ├── build_dataset.py  ← processed/ + data/datasplits/*.yaml → datasets/
  │                       --processed processed_10mm_SI --out datasets_10mm_SI
  ├── train.py          ← datasets/ → checkpoints/<run_id>/weights/{best,last}.pt
  │                       --model yolo26n.pt --epochs 100 --imgsz 640
  ├── infer.py          ← processed/ + checkpoint → predictions/<run_id>/ (structure slices/)
  │                       filtré par split yaml + partition, txt sans conf (5 champs)
  ├── reconstruct.py    ← predictions/<run_id>/ + data/raw/ → reconstructions/<run_id>/
  ├── evaluate.py       ← processed/ + checkpoint → predictions/<run_id>/
  │                       seuil unique CONF_THRESH=0.25 (défaut, injectable via --conf)
  │                       par patient : txt (bbox + conf), png (GT vert + pred rouge), volume/bbox_3d.txt
  │                       format txt préd : "0 cx cy w h conf" (champ conf en plus du format YOLO standard)
  ├── metrics.py        ← --inference predictions/<run_id>/ --processed processed/
  │                       → predictions/<run_id>/<dataset>/<patient>/metrics/slices.csv  (une ligne par slice)
  │                       → predictions/<run_id>/metrics.csv  (agrégé cross-dataset/contrast/split)
  │                       → predictions/<run_id>/report.html  (tableau IoU par dataset et par dataset×contraste)
  │                       colonnes slices.csv : slice_idx, has_gt, has_pred, pred_conf,
  │                         iou (vs GT même slice, 0 si absent), iou_nearest_gt (vs GT voisin si pas de GT, 0 si pas de pred),
  │                         z_dist_to_ref_gt, ref_gt_slice, is_fp, is_fn
  │                       seuil CONF_THRESH=0.5 (injectable via --conf) pour precision/recall/f1
  ├── find_failures.py  ← --inference predictions/<run_id>/ --processed processed/ → predictions/<run_id>/failures.csv
  │                       pour chaque slice prédite : IoU vs GT de la slice la plus proche ayant un GT
  │                       (même slice si GT présent, sinon voisin en z le plus proche)
  │                       trié par iou_nearest_gt croissant — pires cas en premier
  ├── predict_volume.py ← image.nii.gz + checkpoint → bbox_pred.nii.gz (overlay FSLeyes)
  │                       réoriente LAS, infère à --si-res (défaut 10.0mm), reprojette sur résolution native
  │                       zoom_factor = orig_si_mm / si_res → round(z_orig * zoom_factor) = z_inf
  │                       même dimensions et affine que l'entrée LAS → superposable dans FSLeyes
  └── run_pipeline.py   ← image + masque + --si-res → sandbox/ (test end-to-end)
