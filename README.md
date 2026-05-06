# Spinal cord detection

## sc_crop — use the model without training

`sc_crop` is a standalone Python package that crops any NIfTI volume around the spinal cord. It uses the pre-trained YOLO model from this repository and requires no knowledge of the training pipeline.

### Install

**Option A — venv**

```bash
mkdir ~/sc_crop && cd ~/sc_crop
python3.11 -m venv venv
git clone https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo.git
source venv/bin/activate
pip install -e model_cropping_sc_contrast-agnostic_yolo/sc_crop/
```

**Option B — conda**

```bash
mkdir ~/sc_crop && cd ~/sc_crop
conda create -p venv python=3.11
git clone https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo.git
conda activate ./venv
pip install -e model_cropping_sc_contrast-agnostic_yolo/sc_crop/
```

The environment must be activated before each use:

```bash
source ~/sc_crop/venv/bin/activate   # venv
sc-crop t2.nii.gz
```

```bash
conda activate ~/sc_crop/venv        # conda
sc-crop t2.nii.gz
```

**Optional — use `sc-crop` without activating the environment each time:**

```bash
mkdir -p ~/.local/bin
ln -s ~/sc_crop/venv/bin/sc-crop ~/.local/bin/sc-crop
```

Make sure `~/.local/bin` is in your `PATH` (add to `~/.bashrc` or `~/.zshrc` if needed):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

After this, `sc-crop` works directly from any terminal without activation.

### Download the model

```bash
sc-crop download
# downloads to ~/.sc_crop/sc_crop_models/
```

### Crop a volume

```bash
sc-crop t2.nii.gz
# → t2_crop.nii.gz next to the input, same orientation / resolution / world space

sc-crop t2.nii.gz -o output.nii.gz     # explicit output path
sc-crop t2.nii.gz --padding-rl 10      # Right-Left padding in mm (default 10)
sc-crop t2.nii.gz --padding-ap 15      # Anterior-Posterior padding in mm (default 15)
sc-crop t2.nii.gz --padding-si 20      # Superior-Inferior padding in mm (default 20)
sc-crop t2.nii.gz --conf 0.1           # confidence threshold (default 0.1)
sc-crop t2.nii.gz --debug              # also saves t2_debug.png (per-slice panel)
```

### Python API

```python
from sc_crop.crop import run
out = run("t2.nii.gz")
out = run("t2.nii.gz", padding_rl_mm=10, padding_ap_mm=15, padding_si_mm=20, conf=0.1, debug=True)
```

### Requirements

Python ≥ 3.8, automatically installed by pip: `ultralytics`, `nibabel`, `numpy`, `pillow`, `pyyaml`.

---

## Project description

### The goal
Finding the minimal 3d bounding box enclosing the whole spinal cord on any MRI volume.

### The method
- Detecting the spinal cord on 2.5D axial and sagital slices
- Combining the detections to recreate a 3d bounding box

### The detector
We use YOLO26n released by ultralytics

### The datasets used
We use 18 MRI datasets covering cervical and lumbar SC, multiple contrasts and pathologies

From data.neuro.polymtl.ca:
1. [basel-mp2rage](https://data.neuro.polymtl.ca/datasets/basel-mp2rage.git), 
2. [canproco](https://data.neuro.polymtl.ca/datasets/canproco.git), 
3. [data-multi-subject](https://data.neuro.polymtl.ca/datasets/data-multi-subject.git), 
4. [dcm-brno](https://data.neuro.polymtl.ca/datasets/dcm-brno.git), 
5. [dcm-zurich](https://data.neuro.polymtl.ca/datasets/dcm-zurich.git), 
6. [dcm-zurich-lesions](https://data.neuro.polymtl.ca/datasets/dcm-zurich-lesions.git),
7. [dcm-zurich-lesions-20231115](https://data.neuro.polymtl.ca/datasets/dcm-zurich-lesions-20231115.git), 
8. [lumbar-epfl](https://data.neuro.polymtl.ca/datasets/lumbar-epfl.git),
9. [lumbar-vanderbilt](https://data.neuro.polymtl.ca/datasets/lumbar-vanderbilt.git), 
10. [nih-ms-mp2rage](https://data.neuro.polymtl.ca/datasets/nih-ms-mp2rage.git), 
11. [sci-colorado](https://data.neuro.polymtl.ca/datasets/sci-colorado.git),
12. [sci-paris](https://data.neuro.polymtl.ca/datasets/sci-paris.git), 
13. [sci-zurich](https://data.neuro.polymtl.ca/datasets/sci-zurich.git), 
14. [sct-testing-large](https://data.neuro.polymtl.ca/datasets/sct-testing-large.git)
15. [spider-challenge-2023](https://data.neuro.polymtl.ca/datasets/spider-challenge-2023.git)
16. [whole-spine](https://data.neuro.polymtl.ca/datasets/whole-spine.git)

From spineimage.ca:
17. [site_006](https://spineimage.ca/MON/site_006), 
18. [site_007](https://spineimage.ca/VGH/site_007)

## Training the model

###  Install the repository

```bash
git clone https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo
cd model_cropping_sc_contrast-agnostic_yolo
conda create -n contrast_agnostic python=3.11
conda activate contrast_agnostic
pip install -r requirements.txt
```
### Install or check that you have git annex installed
```bash
sudo apt install git-annex
```

### Add you public ssh key to the datasets repositories
Add your public ssh key to [data.neuro.polymtl.ca](https://data.neuro.polymtl.ca/user/settings/keys) and to [spineimage.ca](https://spineimage.ca/user/settings/keys)

---
### Automatic pipeline

The whole pipeline from downloading datasets to training, evaluating and exporting the model can be run by: 
```
bash scripts/run_pipeline.sh
```
Many global parameters allow to fine-tunne the pipeline to compare different approaches.

Otherwise you can start the pipeline from any step (detailed below), everything is specified in the script.

The following variables are editable at the top of `scripts/run_pipeline.sh`:

| Variable | Description |
|---|---|
| `PLANE` | `axial` or `sagittal` |
| `START_STEP` / `END_STEP` | Run only a subset of steps (1–9) |
| `MAKE_SPLITS` | `true` to regenerate train/val/test splits from scratch |
| `SEED` | Global random seed propagated to all scripts (default `50`) |
| `AXIAL_SI_RES` / `AXIAL_INPLANE_RES` | Resampling resolutions for axial plane (mm) |
| `SAG_SI_RES` / `SAG_INPLANE_RES` | Resampling resolutions for sagittal plane (mm) |
| `SAG_SC_PAD` | Sagittal: slices kept = SC extent ± this value (mm) |
| `SAG_SC_RATIO` | Sagittal: SC/non-SC slice balance ratio in the YOLO dataset |
| `DATASET_FACTORS` | Per-dataset oversampling multipliers applied at step 4 (train split only) |
| `WITH_CANAL` | `true` to also extract the spinal canal as a second detection class |
| `MODEL` | YOLO model variant (e.g. `yolo26n.pt`, `yolo26s.pt`) |
| `EPOCHS` | Number of training epochs |
| `IMGSZ` | Input image size for training |
| `FL_GAMMA` | Focal loss gamma (`0` = standard BCE) |
| `WORKERS` | Number of dataloader workers |
| `OVERRIDE_PROCESSED_DIR` / `OVERRIDE_DATASET_DIR` / `OVERRIDE_RUN_ID` | Point steps 4–9 to an existing directory (useful to resume or re-evaluate without re-preprocessing) |

Output directories are prefixed with `pipeline_` (processed, datasets) and `pipeline_run_` (checkpoints, predictions) to avoid collisions with manually created runs.

---

#### Step 1 — Download datasets

Requires SSH access to `data.neuro.polymtl.ca`, `https://spineimage.ca/` and `https://github.com/spine-generic` (open source) and to have git-annex installed.


```bash
bash scripts/download_all_datasets.sh
```

Datasets are downloaded to `data/raw/`. The 18 datasets are:
`basel-mp2rage`, `canproco`, `data-multi-subject`, `dcm-brno`, `dcm-zurich`, `dcm-zurich-lesions`, `dcm-zurich-lesions-20231115`, `lumbar-epfl`, `lumbar-vanderbilt`, `nih-ms-mp2rage`, `sci-colorado`, `sci-paris`, `sci-zurich`, `sct-testing-large`, `spider-challenge-2023`, `whole-spine`, `site_006_praxis`, `site_007_praxis`.

---

#### Step 2 — Generate train/val/test splits

Reads actual subject folders from `data/raw/` and splits 50/20/30 per dataset with a fixed seed.

```bash
python scripts/make_splits.py
```

Output: `data/datasplits/from_raw/datasplit_<dataset>_seed<SEED>.yaml` (one file per dataset).

---

#### Step 3 — Preprocess

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

#### Step 4 — Build YOLO dataset

Creates flat symlinks into `datasets/` directories ready for YOLO training.

```bash
python scripts/build_dataset.py \
    --processed processed/10mm_SI_1mm_axial \
    --out datasets/10mm_SI_1mm_axial
```

---

#### Step 5 — Train

```bash
python scripts/train.py \
    --dataset-yaml datasets/10mm_SI_1mm_axial/dataset.yaml \
    --run-id yolo26_1mm_axial
```

Checkpoint saved to `checkpoints/yolo26_1mm_axial/weights/best.pt`.  
Training is logged to Weights & Biases (project `spine_detection`).

---

#### Step 6 — Run inference

Runs the model on all processed volumes and saves predicted bounding boxes and overlay images.

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/yolo26_1mm_axial/weights/best.pt \
    --processed processed/10mm_SI_1mm_axial
```

Output: `predictions/yolo26_1mm_axial/<dataset>/<patient>/png|txt|volume/`.

---

#### Step 7 — Compute metrics

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

#### Step 8 — Plot metrics

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

#### Step 9 — Inspect failures

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

#### Repository structure

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
