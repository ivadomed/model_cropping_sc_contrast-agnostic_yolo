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
- conda environment : sc_crop_training
- toujours activer avant d'exécuter du code : conda activate sc_crop_training
- ultralytics 8.4.23 (YOLO26 = modèle le plus récent, défaut : yolo26n.pt)
- albumentations v2 installé (API : std_range=, scale_range= au lieu de var_limit=, scale_min/max=)
- wandb installé, compte : quentin-revillon (neuropoly), project : spine_detection
- sur romane.neuro.polymtl.ca (4× RTX A6000, driver 535.288.01, CUDA max 12.2) : env recréé en
  python=3.12 + torch==2.5.1+cu121 / torchvision==0.20.1+cu121 (torch==2.8.0 de requirements.txt
  n'a que des builds cu126/cu128/cu129, tous incompatibles avec ce driver — voir README
  "Older GPU driver"). Sans ce fix, torch.cuda.is_available() retourne False silencieusement
  alors que device_count() voit les 4 GPU.

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
- resampling de tous les axes via --si-res / --axial-res / --rl-res (nibabel resample_to_output, order=1 img, order=0 mask)
- export PNG : slices natives normalisées uint8, pas de resize ni de padding in-plane
- resize in-plane délégué à YOLO via le paramètre imgsz (training et inférence)
- dossier de sortie nommé automatiquement depuis les résolutions et options choisies
- Z = min(img, mask) sur l'axe d'itération : le resampling peut différer d'1 voxel entre img et mask

CONVENTION D'ORIENTATION DES SLICES AXIALES (après LAS)
  slice = data[:, :, z].T[::-1, ::-1]  →  shape (AP_dim, RL_dim)
  - rows = AP  : row 0 = Anterior (voxel antérieur le plus extrême)
  - cols = RL  : col 0 = Left     (voxel gauche le plus extrême)
  - z    = SI  : z=0  = Superior  (slice_000 = tranche la plus supérieure)
  - H = AP_dim = shape_las[1],  W = RL_dim = shape_las[0],  Z = SI_dim = shape_las[2]
  - YOLO cx = RL position (normalized by W),  cy = AP position (normalized by H)
  3ch axial : R=slice_Superior, G=current, B=slice_Inferior (voisins dans l'ordre de sortie)

CONVENTION D'ORIENTATION DES SLICES SAGITTALES (après LAS)
  slice = data[r, :, ::-1].T  →  shape (SI_dim, AP_dim)  — Superior en haut
  - rows = SI : row 0 = Superior
  - cols = AP : col 0 = Posterior
  - z    = RL : z=0  = premier RL voxel (Right)
  - H = SI_dim = shape_las[2],  W = AP_dim = shape_las[1],  Z = RL_dim = shape_las[0]

plane_res(meta) → (row_res, col_res, z_res) :
  axial   : (ap_res, rl_res, si_res)
  sagittal: (si_res, ap_res, rl_res)

- meta.yaml par patient : raw_image, raw_mask, shape_las [RL_dim, AP_dim, SI_dim], si_res_mm, rl_res_mm, ap_res_mm
  rl_res_mm/ap_res_mm = résolution dans le plan axial après resampling
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
  - cherche dans derivatives/<labels_dir>/<sub>/[ses-*/]{anat,func,dwi}/
  - img_glob (datasets.yaml) : si défini, l'image est trouvée par glob dans le dossier du sujet
    plutôt que par dérivation depuis le nom du masque (utilisé pour ds005143 : "*_bold.nii.gz")
  - --with-canal : active l'extraction canal ; si canal mask absent pour un patient, seul SC est écrit

DATA SOURCES — structure par source dans data/raw/
  data-multi-subject/
    <subject>/anat/                       ← volumes (contraste variable)
    derivatives/
      labels/<subject>/
        anat/  ← *_label-SC_seg.nii.gz — utilisé
        dwi/   ← *_rec-average_dwi_label-SC_seg.nii.gz — utilisé
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

  evaluate.py → structure dans predictions/<run_id>/predictions/ (utilisée par metrics.py et find_failures.py)
  predictions/
  └── <run_id>/
      ├── predictions/
      │   └── <dataset>/
      │       └── <patient>/
      │           ├── png/            ← overlay GT (vert) + pred (rouge) par slice
      │           │   └── slice_NNN.png
      │           ├── txt/            ← prédiction par slice (vide si pas de détection)
      │           │   └── slice_NNN.txt  format : "0 cx cy w h conf" (6 champs, conf ajouté)
      │           ├── volume/
      │           │   └── bbox_3d.txt ← bbox 3D reconstruite depuis txt/
      │           ├── gt/             ← symlink → processed/<dataset>/<patient>/
      │           ├── meta.yaml
      │           └── metrics/
      │               └── patient.csv
      ├── metrics/
      │   ├── per_split/
      │   │   └── <split>/
      │   │       └── <metric>/
      │   │           └── conf0.1/
      │   │               ├── <metric>_conf0.1.png   ← violin plot (plot_metrics.py)
      │   │               └── failures/
      │   │                   └── <dataset>/         ← find_failures.py
      │   │                       ├── ranking.csv
      │   │                       └── 001_<stem>/
      │   │                           ├── data → symlink
      │   │                           └── overview.png
      │   └── globals/
      │       └── <metric>/
      │           └── conf0.1/
      │               └── <metric>_globals_conf0.1.png  ← tous splits, couleurs différentes + max dashed lines
      └── patients.csv

DATASETS — généré par build_dataset.py, jamais versionné
  datasets[_<suffix>]/
  ├── dataset.yaml                ← config YOLO (path absolu, classes, splits)
  ├── images/train/ val/ test/    ← symlinks plats vers processed/.../png/
  └── labels/train/ val/ test/    ← symlinks plats vers processed/.../txt/
  nommage symlink : <dataset>_<subject>[_<contrast>]_slice_NNN.png

SPLITS — un yaml par dataset, régénéré à chaque run dans <run-dir>/datasplits/
  <run-dir>/datasplits/datasplit_<dataset>_seed50.yaml
  data/datasplits_seed50/ contient la référence trackée (seed50, générée une fois, committée)
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
  + custom : SCCenteredCrop (p=0.3), BiasField (p=0.2), RandomInvert (p=0.25), Contrast (p=0.15)
  injectées via callback on_train_start (inject_mri_augmentations)
- hsv_v=0.15, degrees=15, scale=0.2, translate=0.1, fliplr=0.5, flipud=0.5
- mosaic=0 (images médicales, pas de mosaïque)
- critère best.pt : mAP50-95 sur validation (fitness = 0.1*mAP50 + 0.9*mAP50-95)
- wandb.init() avant model.train() pour que ultralytics utilise le run existant
- project= passé en chemin absolu pour éviter runs/detect/ prefix ultralytics

DATA/DATASETS REGISTRY — configs/datasets.yaml
  - Registre de tous les datasets : name, url_ssh, url_https, host (neuro|github|spineimage), commit (pinned pour reproductibilité)
  - Commits pinned = état exact utilisé pour les entraînements actuels
  - datasets spineimage : clonés via SSH Gitea (spineimage.ca) — nécessite une clé SSH sur le dépôt
  - download_all_datasets.sh lit exclusivement ce fichier

SCRIPTS — un script, une responsabilité
  scripts/
  ├── download_all_datasets.sh ← clone tous les datasets + git annex get parallèle
  │                              parse configs/datasets.yaml via Python one-liner (name|url_ssh|commit)
  │                              git clone + git annex dead here + git checkout <commit> si pinné
  │                              loggue les commits dans data/raw/git_branch_commit.log
  ├── preprocess.py     ← data/raw/ → processed_{res}mm_SI[_{axial}mm_axial][_3ch]/
  │                       --si-res obligatoire, réoriente LAS, rééchantillonne via nibabel.processing (order=1)
  │                       --axial-res : rééchantillonnage isotropique du plan axial (RL, AP) en même temps que SI
  │                       --3ch : export PNG pseudo-RGB (R=slice-1, G=slice courante, B=slice+1), bords = noir
  │                       export PNG + txt YOLO + volume/bbox_3d.txt + meta.yaml
  │                       meta.yaml inclut : shape_las, si_res_mm, rl_res_mm, ap_res_mm
  │                                          axial_res_mm (si --axial-res), channels=3 (si --3ch)
  │                       séquentiel (pas de multiprocessing — nibabel/scipy single-threadé, oversubscription)
  │                       écrit processed/<variant>/skipped.log (TSV: dataset/subject/reason) si des sujets
  │                       sont sautés : missing_nifti (git annex non téléchargé) ou no_sc_voxels (masque vide)
  │                       --update-meta --out <dir> : patche les meta.yaml existants sans re-préprocesser
  ├── build_dataset.py  ← processed/ + <run-dir>/datasplits/*.yaml → datasets/
  │                       --processed processed_10mm_SI --out datasets_10mm_SI
  ├── train.py          ← datasets/ → checkpoints[_cls]/<run_id>/weights/{best,last}.pt
  │                       --mode detection|classification --dataset <yaml|dir> --run-dir <dir>
  │                       dispatche sur mode, lit configs/training.yaml (sections detection/classification)
  │                       sauvegarde resolved_config.yaml (toutes valeurs + git hash) dans run-dir
  ├── evaluate.py       ← processed/ + checkpoint → predictions/<run_id>/predictions/
  │                       seuil unique CONF_THRESH=0.1 (défaut, injectable via --conf)
  │                       par patient : txt (bbox + conf), png (GT vert + pred rouge), volume/bbox_3d.txt
  │                       format txt préd : "0 cx cy w h conf" (champ conf en plus du format YOLO standard)
  ├── metrics.py        ← --inference predictions/<run_id>/ --processed processed/
  │                       → predictions/<run_id>/predictions/<dataset>/<patient>/metrics/patient.csv
  │                       → predictions/<run_id>/patients.csv
  │                       colonnes patient.csv (une ligne par conf_thresh) : conf_thresh, iou_3d_mm,
  │                         gap_mm_R/L/P/A/I/S (padding mm signé par face, LAS)
  │                       reconstruit le bbox 3D pred depuis txt/ (union des slices au-dessus du seuil)
  │                         et le compare au bbox 3D GT (processed/.../volume/bbox_3d.txt)
  │                       SOURCE DE VÉRITÉ : patients.csv + patient.csv sont la base de tous les scripts aval
  │                       --metrics restreint les colonnes calculées/patchées (défaut : les 7 ci-dessus)
  ├── find_failures.py  ← --inference predictions/<run_id>/  (requiert patients.csv de metrics.py)
  │                       → predictions/<run_id>/metrics/per_split/<split>/<metric>/<conf>/failures/<dataset>/
  │                       classement indépendant par métrique (iou_3d_mm croissant, gap_mm_* décroissant,
  │                         gap_mm_*_neg croissant) ; --top-k (défaut 10) ; --split optionnel
  ├── run_pipeline.py   ← orchestrateur des 9 étapes (download→preprocess→splits→build→train→
  │                       eval→metrics→plot→failures), lit configs/*.yaml, --mode override le mode
  ├── train_det_and_cls.sh      ← lance run_pipeline.py en mode detection PUIS classification
  │                       (deux run-dirs distincts <root>_det/ et <root>_cls/), imprime la
  │                       commande export_model.py à lancer ensuite
  └── export_model.py ← exporte + tag ce repo ; la publication (sc-crop, PyPI) se fait
                          depuis l'autre repo via scripts/publish_release.sh — voir MIGRATION.md
