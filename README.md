# Spinal Cord Detection with YOLO

2D bounding-box detection of the spinal cord on axial MRI slices using YOLOv26n.  
The model is trained on 15 open-access MRI datasets covering cervical and lumbar SC, multiple contrasts and pathologies.

## Pipeline overview

```
data/raw/  →  processed/  →  datasets/  →  checkpoints/  →  predictions/  →  metrics + plots + failures
```

---

## Installation

```bash
git clone https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo
cd model_cropping_sc_contrast-agnostic_yolo
conda create -n contrast_agnostic python=3.11
conda activate contrast_agnostic
pip install -r requirements.txt
```

---

## Step 1 — Download datasets

Requires SSH access to `data.neuro.polymtl.ca` and git-annex.

```bash
bash scripts/download_all_datasets.sh
```

Datasets are downloaded to `data/raw/`. The 15 datasets are:
`basel-mp2rage`, `canproco`, `data-multi-subject`, `dcm-brno`, `dcm-zurich`, `dcm-zurich-lesions`, `dcm-zurich-lesions-20231115`, `lumbar-epfl`, `lumbar-vanderbilt`, `nih-ms-mp2rage`, `sci-colorado`, `sci-paris`, `sci-zurich`, `sct-testing-large`.

---

## Step 2 — Generate train/val/test splits

Reads actual subject folders from `data/raw/` and splits 50/20/30 per dataset with a fixed seed.

```bash
python scripts/make_splits.py
```

Output: `data/datasplits/from_raw/datasplit_<dataset>_seed50.yaml` (one file per dataset).

---

## Step 3 — Preprocess

Reorients volumes to LAS, resamples SI axis to 10 mm and axial plane to 1 mm isotropic, exports axial PNG slices and YOLO bounding-box labels.

```bash
python scripts/preprocess.py --si-res 10.0 --axial-res 1.0
```

Output: `processed/10mm_SI_1mm_axial/<dataset>/<patient>/png/` and `txt/`.

For pseudo-RGB (2.5D) input (R=prev slice, G=current, B=next):

```bash
python scripts/preprocess.py --si-res 10.0 --axial-res 1.0 --3ch
```

---

## Step 4 — Build YOLO dataset

Creates flat symlinks into `datasets/` directories ready for YOLO training.

```bash
python scripts/build_dataset.py \
    --processed processed/10mm_SI_1mm_axial \
    --out datasets/10mm_SI_1mm_axial
```

---

## Step 5 — Train

```bash
python scripts/train.py \
    --dataset-yaml datasets/10mm_SI_1mm_axial/dataset.yaml \
    --run-id yolo26_1mm_axial
```

Checkpoint saved to `checkpoints/yolo26_1mm_axial/weights/best.pt`.  
Training is logged to Weights & Biases (project `spine_detection`).

---

## Step 6 — Run inference

Runs the model on all processed volumes and saves predicted bounding boxes and overlay images.

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/yolo26_1mm_axial/weights/best.pt \
    --processed processed/10mm_SI_1mm_axial
```

Output: `predictions/yolo26_1mm_axial/<dataset>/<patient>/png|txt|volume/`.

---

## Step 7 — Compute metrics

Computes per-slice and per-patient metrics at all confidence thresholds (0 → 1).

```bash
python scripts/metrics.py \
    --inference predictions/yolo26_1mm_axial \
    --processed processed/10mm_SI_1mm_axial
```

Key outputs:
- `predictions/yolo26_1mm_axial/<dataset>/<patient>/metrics/patient.csv` — metrics per confidence threshold
- `predictions/yolo26_1mm_axial/patients.csv` — patient index
- `predictions/yolo26_1mm_axial/report.html` — IoU summary per dataset

---

## Step 8 — Plot metrics

Violin plots of `iou_gt_mean` and `iou_all_mean` per dataset, for each confidence threshold.

```bash
python scripts/plot_metrics.py \
    --inference predictions/yolo26_1mm_axial \
    --conf-sweep
```

Generates one plot per metric per threshold in `predictions/yolo26_1mm_axial/plots/<split>/<metric>/`.

| Metric | Definition |
|---|---|
| `iou_gt_mean` | Mean IoU on SC slices (missed SC slices contribute 0) |
| `iou_all_mean` | Mean IoU on all slices (false detections also contribute 0) |

---

## Step 9 — Inspect failures

Ranks the worst-performing volumes per dataset and generates overview images.

```bash
python scripts/find_failures.py \
    --inference predictions/yolo26_1mm_axial
```

Output: `predictions/yolo26_1mm_axial/<dataset>/failures/<split>/<metric>/`
- `ranking.csv` — top-10 worst volumes
- `001_<patient>/overview.png` — all predicted slices tiled in a grid
- `001_<patient>/data` — symlink to the full patient directory (PNG overlays, predicted bboxes)

---

## Repository structure

```
data/
  raw/                      ← BIDS datasets (read-only)
  datasplits/from_raw/      ← train/val/test split YAMLs
processed/                  ← preprocessed PNG slices + YOLO labels
datasets/                   ← flat symlinks for YOLO training
checkpoints/                ← trained model weights
predictions/                ← inference outputs, metrics, plots, failures
scripts/                    ← all pipeline scripts
```

`processed/`, `datasets/`, `checkpoints/`, `predictions/` are gitignored.
