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
- processed/10mm_SI/            : COMPLET — tous datasets, 10mm SI, résolution native axiale, PNG grayscale
- processed/10mm_SI_1mm_axial/  : COMPLET — 10mm SI + 1mm isotropique axial (nibabel order=1), PNG grayscale
- processed/10mm_SI_1mm_axial_3ch/ : COMPLET — même resampling + PNG pseudo-RGB 3ch (R=prev, G=cur, B=next)
- datasets/10mm_SI/             : dataset YOLO construit depuis processed/10mm_SI/
- datasets/10mm_SI_1mm_axial/   : dataset YOLO construit depuis processed/10mm_SI_1mm_axial/
- datasets/10mm_SI_1mm_axial_3ch/ : dataset YOLO construit depuis processed/10mm_SI_1mm_axial_3ch/

ENTRAÎNEMENTS RÉALISÉS
- yolo26_10mm_SI : premier run complet, 10mm SI, yolo26n, epochs 100, imgsz 640
  → sauvegardé dans runs/detect/checkpoints/yolo26_10mm_SI/weights/{best,last}.pt
  → problème observé : val/cls_loss diverge (~epoch 6), mAP50/mAP50-95 s'effondre
    cause suspectée : déséquilibre train/val sur slices vides (extrémités, cerveau)
    avec 10mm peu de slices par volume → sensible à la composition du split
  → best.pt correspond à l'epoch ~4-5 (peak mAP50-95 ~0.5)
- DÉCISION : train.py corrigé pour sauvegarder dans checkpoints/<run_id>/ (path absolu)
  les runs précédents sont dans runs/detect/checkpoints/
- yolo26_10mm_aug_320_tassan : run principal de référence, 10mm SI résolution native axiale
  → checkpoints/yolo26_10mm_aug_320_tassan/weights/best.pt
  → best.pt = fitness = 0.1·mAP50 + 0.9·mAP50-95 sur val ; patience=20

ÉVALUATIONS EN COURS / RÉALISÉES
- yolo26_10mm_aug_320_tassan sur processed/10mm_SI          → predictions/yolo26_10mm_aug_320_tassan/
- yolo26_10mm_aug_320_tassan sur processed/10mm_SI_1mm_axial → predictions/yolo26_10mm_aug_320_tassan_1mm_axial/
- yolo26_1mm_axial sur processed/10mm_SI_1mm_axial          → predictions/yolo26_1mm_axial/

STRUCTURE GÉNÉRALE
- data/raw/ est en lecture seule, jamais écrit par du code
- processed/<variant>/ contient uniquement les slices 2D extraites des volumes, pas de volumes
- datasets/<variant>/ contient les symlinks plats vers processed/ pour YOLO
- predictions/ et reconstructions/ sont organisés par run id
- sandbox/ est un espace de test, pas de sous-dossier run id, écrasé à chaque run
- gitignored : processed/, predictions/, reconstructions/, sandbox/, datasets/, checkpoints/, runs/,
               dataset_stats.csv, metrics_*.csv

PRÉPROCESSING
- réorientation LAS avant extraction des slices
- resampling de l'axe SI uniquement (axe 2 en LAS) via --si-res (ex: 1.0, 10.0)
  order=3 pour les images, order=0 pour les masques binaires (scipy.ndimage.zoom)
- export PNG : slices natives normalisées uint8, pas de resize ni de padding in-plane
- resize in-plane délégué à YOLO via le paramètre imgsz (training et inférence)
- dossier de sortie nommé automatiquement processed/{res}mm_SI (ex: processed/10mm_SI)
- Z = min(img, mask) sur l'axe SI : le zoom scipy arrondit indépendamment → peut différer d'1 voxel
- meta.yaml par patient : raw_image, raw_mask, shape_las [H,W,Z], si_res_mm, rl_res_mm, ap_res_mm
  rl_res_mm/ap_res_mm = résolution native dans le plan axial (LAS axes 0 et 1), non modifiée par resampling
  pour patcher des meta.yaml existants sans re-préprocesser : preprocess.py --update-meta --out <dir>

DÉCOUVERTE DES MASQUES — tables explicites par dataset dans preprocess.py
  DATASET_MASK_SUFFIX (SC, class 0) = {
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
    "spider-challenge-2023":        "_label-SC_seg.nii.gz",
    "whole-spine":                  "_label-SC_seg.nii.gz",
  }
  DATASET_CANAL_SUFFIX (canal rachidien, class 1) = {
    "data-multi-subject":    "_label-canal_seg.nii.gz",
    "spider-challenge-2023": "_label-canal_seg.nii.gz",
    "whole-spine":           "_label-canal_seg.nii.gz",
  }
  - plante sur dataset inconnu (pas de fallback)
  - cherche toujours dans derivatives/labels/<sub>/[ses-*/]anat/
  - DWI exclu (pas de sous-dossier dwi/)
  - --with-canal : active l'extraction canal ; si canal mask absent pour un patient, seul SC est écrit

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
          │   └── slice_NNN.txt   format sans canal : "0 cx cy w h" normalisé [0,1], ou vide
          │                       format avec canal : jusqu'à 2 lignes — "0 cx cy w h" (SC) + "1 cx cy w h" (canal)
          ├── volume/
          │   ├── bbox_3d.txt        ← bbox 3D GT SC  : row1 row2 col1 col2 z1 z2 (voxels)
          │   └── bbox_3d_canal.txt  ← bbox 3D GT canal (seulement si --with-canal, même format)
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

DATA/DATASETS REGISTRY — data/datasets.yaml
  - Registre de tous les datasets : name, url_ssh, url_https, host (neuro|github|spineimage), commit (pinned pour reproductibilité)
  - Commits pinned = état exact utilisé pour les entraînements actuels
  - datasets spineimage : clonés via SSH Gitea (spineimage.ca) — nécessite une clé SSH sur le dépôt
  - 01_clone_dataset.py et download_all_datasets.sh lisent exclusivement ce fichier

SCRIPTS — un script, une responsabilité
  scripts/
  ├── 01_clone_dataset.py ← clone un dataset depuis datasets.yaml (--ofolder --dataset)
  │                         lit data/datasets.yaml pour url/host/commit
  │                         tous les datasets : git clone + git annex dead here (client-only)
  │                         si commit pinné : git checkout <commit> après clone
  │                         ajoute commit dans data/raw/git_branch_commit.log
  ├── download_all_datasets.sh ← clone tous les datasets non-spineimage + git annex get parallèle
  │                              lit la liste de datasets depuis data/datasets.yaml via Python
  ├── explore_stats.py  ← data/raw/ → dataset_stats.csv
  │                       reorientation virtuelle LAS, rapporte shape/résolution/FOV mm
  ├── preprocess.py     ← data/raw/ → processed_{res}mm_SI[_{axial}mm_axial][_3ch]/
  │                       --si-res obligatoire, réoriente LAS, rééchantillonne via nibabel.processing (order=1)
  │                       --axial-res : rééchantillonnage isotropique du plan axial (RL, AP) en même temps que SI
  │                       --3ch : export PNG pseudo-RGB (R=slice-1, G=slice courante, B=slice+1), bords = noir
  │                       export PNG + txt YOLO + volume/bbox_3d.txt + meta.yaml
  │                       meta.yaml inclut : shape_las, si_res_mm, rl_res_mm, ap_res_mm
  │                                          axial_res_mm (si --axial-res), channels=3 (si --3ch)
  │                       séquentiel (pas de multiprocessing — nibabel/scipy single-threadé, oversubscription)
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
  │                       → predictions/<run_id>/patients.csv  (une ligne par volume 3D : fp_rate, fn_rate, iou_mean, score_fail)
  │                       → predictions/<run_id>/metrics.csv  (agrégé cross-dataset/contrast/split)
  │                       → predictions/<run_id>/report.html  (tableau IoU par dataset et par dataset×contraste)
  │                       colonnes slices.csv : slice_idx, has_gt, has_pred, pred_conf,
  │                         iou (vs GT même slice, 0 si absent), iou_nearest_gt (vs GT voisin si pas de GT, 0 si pas de pred),
  │                         z_dist_to_ref_gt, ref_gt_slice, is_fp, is_fn
  │                       seuil CONF_THRESH=0.5 (injectable via --conf) pour precision/recall/f1
  │                       SOURCE DE VÉRITÉ : slices.csv et patients.csv sont la base de tous les scripts aval
  ├── find_failures.py  ← --inference predictions/<run_id>/  (requiert patients.csv de metrics.py)
  │                       → predictions/<run_id>/<dataset>/failures/failures.csv  (top-K volumes par score_fail)
  │                       → predictions/<run_id>/<dataset>/failures/NNN_<stem> → ../<stem>  (symlinks)
  │                       score_fail = (fp_rate + fn_rate) / 2 ; --top-k (défaut 20) ; --split optionnel
  ├── border_metrics.py ← --inference predictions/<run_id>/  (requiert slices.csv de metrics.py)
  │                       analyse les deux extrémités de la moelle indépendamment :
  │                         SUPERIOR (jonction moelle/cerveau) : boundary_z = max GT slice
  │                           niveau 0 = dernière slice avec GT ; -k = dans la moelle ; +k = au-dessus (FP zone)
  │                         INFERIOR (début des lombaires) : boundary_z = min GT slice
  │                           niveau 0 = première slice avec GT ; -k = en dessous (FP zone) ; +k = dans la moelle
  │                       → border_metrics_{superior,inferior}.csv  (une ligne par slice par niveau)
  │                       → border_iou_{superior,inferior}.png      (violin IoU, GT slices uniquement)
  │                       → border_fp_fn_{superior,inferior}.png    (barres FP/FN, niveaux -N à +N)
  │                       n_total affiché dans les labels de l'axe x des barres (dénominateur réel)
  │                       FP = prédiction présente avec IoU < seuil ; FN = GT présent sans aucune prédiction
  │                       splits traités : test + unknown ; paramètres : --n (défaut 5), --conf (0.5), --iou-thresh (0.5)
  │                       --datasets filtre sur un sous-ensemble ; par défaut auto-détection via slices.csv existants
  │                       PAS de --processed : lit slices.csv depuis predictions/<run_id>/<dataset>/<patient>/metrics/
  ├── predict_volume.py ← image.nii.gz + checkpoint → bbox_pred.nii.gz (overlay FSLeyes)
  │                       réoriente LAS, infère à --si-res (défaut 10.0mm), reprojette sur résolution native
  │                       zoom_factor = orig_si_mm / si_res → round(z_orig * zoom_factor) = z_inf
  │                       même dimensions et affine que l'entrée LAS → superposable dans FSLeyes
  └── run_pipeline.py   ← image + masque + --si-res → sandbox/ (test end-to-end)
