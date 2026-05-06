# Contrast-agnostic spinal cord detection and cropping

Detects the spinal cord on any MRI volume and outputs a tight 3D bounding box (a text file countaining the coordinates of the bounding box by default). Works across contrasts (T1, T2, MP2RAGE, DWI…), field strengths, and pathologies. Based on a YOLO26n model trained on multiple datasets covering cervical and lumbar spine.

<img width="1713" height="727" alt="image" src="https://github.com/user-attachments/assets/d8958227-06b6-4430-9378-4a6f91e9741d" />

---

## sc_crop — crop an MRI volume around the spinal cord

`sc_crop` is a standalone Python package. It uses the pre-trained model from this repository and requires no knowledge of the training pipeline.

### Install

**Option A — conda (recommended)**

```bash
mkdir sc_crop && cd sc_crop
conda create -p venv python=3.12
git clone https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo.git
conda activate ./venv
pip install model_cropping_sc_contrast-agnostic_yolo/sc_crop/
```

**Option B — venv**

```bash
mkdir sc_crop && cd sc_crop
python3.12 -m venv venv
git clone https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo.git
source venv/bin/activate
pip install model_cropping_sc_contrast-agnostic_yolo/sc_crop/
```

**_Optional_ — Run these commands to use `sc_crop` without having to activate the virtual environment each time:**

```bash
mkdir -p ~/.local/bin
ln -s sc_crop/venv/bin/sc_crop ~/.local/bin/sc_crop
```

Make sure `~/.local/bin` is in your `PATH` (add to `~/.bashrc` or `~/.zshrc` if needed):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

After this, `sc_crop` works directly from any terminal without environment activation.
___

### Usage

#### Environment activation (Skip if the environment is already activated or if _Optional_ section was followed)

```bash
source sc_crop/venv/bin/activate   # venv
```

```bash
conda activate sc_crop/venv        # conda
```

#### Download the model (first use only):

```bash
sc_crop download
```

#### Crop a volume around the spinal cord

```bash
sc_crop -i t2.nii.gz
```

Outputs `t2_bbox.txt` next to the input with the inclusive voxel bounding box in native image space, compatible with SCT's `sct_crop_image`.

##### Optional parameters

| Parameter | Description | Default |
|---|---|---|
| `-o OUTPUT` | Output path (bbox txt, or crop volume if `--crop`) | `<stem>_bbox.txt` |
| `--crop` | Also save the cropped volume | off |
| `--las` | Output cropped volume in LAS orientation (requires `--crop`) | off |
| `--no-translate` | Do not update affine (by default affine is updated for correct FSLeyes overlay) | off |
| `--padding-rl MM` | Right-Left padding in mm | 10 |
| `--padding-ap MM` | Anterior-Posterior padding in mm | 15 |
| `--padding-si MM` | Superior-Inferior padding in mm | 20 |
| `--conf FLOAT` | Detection confidence threshold | from config |
| `--debug` | Save `<stem>_debug.png` (per-slice panel with bbox) | off |
| `--time` | Print elapsed time per pipeline step | off |


### Python API

```python
from sc_crop.crop import run

result = run("t2.nii.gz")                             # bbox txt only
result = run("t2.nii.gz", crop=True)                  # + cropped volume (native)
result = run("t2.nii.gz", crop=True, las=True)        # + cropped volume (LAS)
result = run("t2.nii.gz", crop=True, translate=False) # affine NOT updated

# result keys: bbox_file, xmin, xmax, ymin, ymax, zmin, zmax, original_axcodes
# + output (if crop=True)
```

### Requirements (automatically downloaded)

Python ≥ 3.8. Pinned versions installed automatically by pip:
`nibabel==5.3.3`, `numpy==2.0.2`, `pillow==11.3.0`, `pyyaml==6.0.2`, `ultralytics==8.4.33`.

---

## Training the model

### Method

- Spinal cord detected on 2.5D axial and sagittal slices using YOLO26n
- Detections aggregated across slices to reconstruct a 3D bounding box

### Datasets

18 MRI datasets covering cervical and lumbar spine, multiple contrasts and pathologies.

From data.neuro.polymtl.ca:
1. [basel-mp2rage](https://data.neuro.polymtl.ca/datasets/basel-mp2rage.git)
2. [canproco](https://data.neuro.polymtl.ca/datasets/canproco.git)
3. [data-multi-subject](https://data.neuro.polymtl.ca/datasets/data-multi-subject.git)
4. [dcm-brno](https://data.neuro.polymtl.ca/datasets/dcm-brno.git)
5. [dcm-zurich](https://data.neuro.polymtl.ca/datasets/dcm-zurich.git)
6. [dcm-zurich-lesions](https://data.neuro.polymtl.ca/datasets/dcm-zurich-lesions.git)
7. [dcm-zurich-lesions-20231115](https://data.neuro.polymtl.ca/datasets/dcm-zurich-lesions-20231115.git)
8. [lumbar-epfl](https://data.neuro.polymtl.ca/datasets/lumbar-epfl.git)
9. [lumbar-vanderbilt](https://data.neuro.polymtl.ca/datasets/lumbar-vanderbilt.git)
10. [nih-ms-mp2rage](https://data.neuro.polymtl.ca/datasets/nih-ms-mp2rage.git)
11. [sci-colorado](https://data.neuro.polymtl.ca/datasets/sci-colorado.git)
12. [sci-paris](https://data.neuro.polymtl.ca/datasets/sci-paris.git)
13. [sci-zurich](https://data.neuro.polymtl.ca/datasets/sci-zurich.git)
14. [sct-testing-large](https://data.neuro.polymtl.ca/datasets/sct-testing-large.git)
15. [spider-challenge-2023](https://data.neuro.polymtl.ca/datasets/spider-challenge-2023.git)
16. [whole-spine](https://data.neuro.polymtl.ca/datasets/whole-spine.git)

From spineimage.ca:
17. [site_006](https://spineimage.ca/MON/site_006)
18. [site_007](https://spineimage.ca/VGH/site_007)

### Install

```bash
git clone https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo
cd model_cropping_sc_contrast-agnostic_yolo
conda create -n contrast_agnostic python=3.11
conda activate contrast_agnostic
pip install -r requirements.txt
```

```bash
sudo apt install git-annex
```

Add your public SSH key to [data.neuro.polymtl.ca](https://data.neuro.polymtl.ca/user/settings/keys) and to [spineimage.ca](https://spineimage.ca/user/settings/keys).

### Automatic pipeline

The full pipeline (download → train → evaluate) can be run with:

```bash
bash scripts/run_pipeline.sh
```

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
| `OVERRIDE_PROCESSED_DIR` / `OVERRIDE_DATASET_DIR` / `OVERRIDE_RUN_ID` | Point steps 4–9 to an existing directory |

Output directories are prefixed with `pipeline_` (processed, datasets) and `pipeline_run_` (checkpoints, predictions).

### Step-by-step pipeline

#### Step 1 — Download datasets

```bash
bash scripts/download_all_datasets.sh
```

Datasets are downloaded to `data/raw/`.

---

#### Step 2 — Generate train/val/test splits

```bash
python scripts/make_splits.py
```

Output: `data/datasplits/from_raw/datasplit_<dataset>_seed<SEED>.yaml` (one file per dataset).

---

#### Step 3 — Preprocess

```bash
python scripts/preprocess.py --si-res 10.0 --axial-res 1.0
```

For pseudo-RGB (2.5D) input (R=prev slice, G=current, B=next):

```bash
python scripts/preprocess.py --si-res 10.0 --axial-res 1.0 --3ch
```

Output: `processed/10mm_SI_1mm_axial/<dataset>/<patient>/png/` and `txt/`.

---

#### Step 4 — Build YOLO dataset

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

Checkpoint saved to `checkpoints/yolo26_1mm_axial/weights/best.pt`. Training logged to Weights & Biases (project `spine_detection`).

---

#### Step 6 — Run inference

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/yolo26_1mm_axial/weights/best.pt \
    --processed processed/10mm_SI_1mm_axial
```

Output: `predictions/yolo26_1mm_axial/<dataset>/<patient>/png|txt|volume/`.

---

#### Step 7 — Compute metrics

```bash
python scripts/metrics.py \
    --inference predictions/yolo26_1mm_axial \
    --processed processed/10mm_SI_1mm_axial
```

---

#### Step 8 — Plot metrics

```bash
python scripts/plot_metrics.py \
    --inference predictions/yolo26_1mm_axial \
    --conf-sweep
```

| Metric | Definition |
|---|---|
| `iou_gt_mean` | Mean IoU on SC slices (missed SC slices contribute 0) |
| `iou_all_mean` | Mean IoU on all slices (false detections also contribute 0) |

---

#### Step 9 — Inspect failures

```bash
python scripts/find_failures.py \
    --inference predictions/yolo26_1mm_axial
```

---

### Repository structure

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
