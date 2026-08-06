# Spinal cord detection model — training

Trains the YOLO26n detector + classifier that find the spinal cord on **axial** MRI slices, contrast-agnostic (T1, T2, MP2RAGE, DWI…), across field strengths and pathologies, cervical and lumbar. Per-slice detections are aggregated into a 3D bounding box.

<img width="1713" height="727" alt="image" src="https://github.com/user-attachments/assets/d8958227-06b6-4430-9378-4a6f91e9741d" />

*(example crop produced by `sc-crop`, the inference package built from a model trained here)*

This repository only trains the model. **It does not run inference.** To crop a volume with an already-trained model, use [`sc-crop`](https://github.com/ivadomed/sc-crop) — a separate, standalone Python package/repository

<img width="8034" height="1227" alt="Dataset Export and Release-2026-08-06-201714" src="https://github.com/user-attachments/assets/45ab2443-0b68-409b-9e7d-3a593ff2b918" />


---

### Method

- Spinal cord detected on 2.5D **axial** slices using YOLO26n
- A YOLO26n image classifier gives a second opinion to say wheter there is spinal cord or not in the image (used to avoid detections of the spinal cord in the brain)
- Detections aggregated across slices to reconstruct a 3D bounding box



### Install

```bash
git clone https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo
cd model_cropping_sc_contrast-agnostic_yolo
conda create -n sc_crop_training python=3.13 -y
conda activate sc_crop_training
pip install -r requirements.txt
```

> **Blackwell GPU (RTX PRO 6000, RTX 5090, sm_120+):** `requirements.txt` pins `torch==2.8.0` which requires CUDA 12.8 wheels not on PyPI. Install PyTorch first:
> ```bash
> pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
> pip install -r requirements.txt
> ```

> **Older GPU driver (`nvidia-smi` reports CUDA Version < 12.6, e.g. driver 535.x on Ampere-class GPUs like RTX A6000):** the only CUDA builds published for `torch==2.8.0` are `cu126`/`cu128`/`cu129`, all requiring driver ≥ 12.6 — none work here. `torch.cuda.is_available()` will silently return `False` (while `device_count()` still reports your GPUs) instead of a clear install error. The newest CUDA build compatible with driver ≤ 12.2 is `cu121`, which has no Python 3.13 wheel for `torchvision` — use **Python 3.12** for this env instead, and override torch/torchvision *after* `requirements.txt` so the `torch==2.8.0` pin doesn't clobber it:
> ```bash
> conda create -n sc_crop_training python=3.12 -y
> conda activate sc_crop_training
> pip install -r requirements.txt
> pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
> ```

```bash
sudo apt install git-annex
```

Add your public SSH key to [data.neuro.polymtl.ca](https://data.neuro.polymtl.ca/user/settings/keys) and to [spineimage.ca](https://spineimage.ca/user/settings/keys).



### Train (the one command) that dowloads datasets, train a detector and classifier

Data dowloading and preprocessing are skipped if already done.

```bash
bash scripts/train_all.sh              # add --no-wandb to disable W&B logging
```

Produces `runs/<TS>_det/` and `runs/<TS>_cls/`, then prints the `export_model.py` command to run next.

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

### Adding a new dataset

Add a registry entry in `configs/datasets.yaml` (read exclusively by `download_all_datasets.sh` — no code change needed):

```yaml
- name: my-dataset
  host: neuro              # neuro | github | spineimage | zenodo
  url_ssh: git@...
  url_https: https://...
  commit: <pinned-sha>      # reproducibility
  mask_suffix: _label-SC_seg.nii.gz
```

Host isn't git/git-annex (e.g. Zenodo)? Write `scripts/download_<name>.sh` producing a BIDS-shaped tree under `data/raw/<name>/` — see `scripts/download_totalsegmentator.sh`.


### Pipeline steps

| # | Step | Output |
|---|---|---|
| 1 | Download datasets | `data/raw/<dataset>/` |
| 2 | Preprocess | `processed/<variant>/<dataset>/<patient>/png,txt,volume/` |
| 3 | Make splits | `<run-dir>/datasplits/` |
| 4 | Build dataset | detection: YOLO format / classification: `sc`/`no_sc` folders, in `<run-dir>/dataset/` |
| 5 | Train | `<run-dir>/checkpoints/weights/{best,last}.pt`, logged to W&B (project `spine_detection`) |
| 6 | Evaluate | detection: bbox IoU / classification: `gap_mm_S`, `gap_mm_I` — written to `<run-dir>/predictions/` |
| 7 | Compute metrics | `iou_3d_mm`, `gap_mm_R/L/P/A/I/S` per patient (`patients.csv`) |
| 8 | Plot metrics | violin plots per split/metric |
| 9 | Find failures | worst patients per metric, ranked |

### Repository structure

```
data/
  raw/                      ← BIDS datasets (read-only, gitignored)
  datasplits_seed50/        ← tracked reference train/val/test split YAMLs
processed/                  ← preprocessed PNG slices + YOLO labels (gitignored)
runs/<TS>/                  ← one full pipeline run: configs snapshot, dataset, checkpoints, predictions, pipeline.log (gitignored)
scripts/                    ← all pipeline scripts
```

`data/` (except the tracked split/summary files above), `processed/`, `runs/`, `checkpoints/`, `predictions/`, `datasets/`, `wandb/` are all gitignored.

### Release

One command publishes a trained detector + classifier as a `sc-crop` release:

Edit the variables at the top of `scripts/release.sh` (this is the file content to change, not a command to run):

```bash
DET_RUN="runs/YYYYMMDD_XXXXXX"        # detector run
CLS_RUN="runs/YYYYMMDD_XXXXXX"        # classifier run
MODEL_VERSION="0.0.X"                 # next model tag on ivadomed/sc-crop
PACKAGE_VERSION="0.1.X"               # next package tag on ivadomed/sc-crop
DET_CHECKPOINT="best.pt"              # best.pt | last.pt
CLS_CHECKPOINT="loss_best.pt"         # best.pt | loss_best.pt | last.pt
```

Then run:

```bash
bash scripts/release.sh
```

See [VERSIONS.md](https://github.com/ivadomed/sc-crop/blob/main/VERSIONS.md) on the `sc-crop` repo for how package/model versions map to training runs.
