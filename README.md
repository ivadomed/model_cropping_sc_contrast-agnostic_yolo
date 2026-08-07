# Spinal cord detection model — training

Trains the YOLO26n detector + classifier that find the spinal cord on **axial** MRI slices, contrast-agnostic (T1, T2, MP2RAGE, DWI…), across field strengths and pathologies, cervical and lumbar. Per-slice detections are aggregated into a 3D bounding box.
<img width="955" height="420" alt="image" src="https://github.com/user-attachments/assets/dbfdb2aa-2f0b-46ea-a333-6e995196a96b" />

<img width="1900" height="757" alt="image" src="https://github.com/user-attachments/assets/0bc628b4-e441-4ca0-b5ac-6b55bbb5d8c6" />

*(example crop produced by `sc-crop`, the inference package built from a model trained here)*

This repository only trains the model. **It does not run inference.** To crop a volume with an already-trained model, use [`sc-crop`](https://github.com/ivadomed/sc-crop) — a separate, standalone Python package/repository.

<img width="8192" height="884" alt="Dataset Export and Release-2026-08-06-202731" src="https://github.com/user-attachments/assets/82b7ef05-d8ba-4bfd-9488-c9186ef497e7" />

---

### Method

- Spinal cord detected on 2.5D **axial** slices using YOLO26n
- A YOLO26n image classifier gives a second opinion to say whether there is spinal cord or not in the image (used to avoid detections of the spinal cord in the brain)
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

### Train (the one command) that downloads datasets, trains a detector and classifier

Data downloading and preprocessing are skipped if already done.

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

Publishing a trained detector + classifier involves two repos: this one exports
the model, [sc-crop](https://github.com/ivadomed/sc-crop) publishes it. Full
walkthrough in [MIGRATION.md](MIGRATION.md) — summary here:

**1. Here — export ONNX + tag this repo:**

```bash
python scripts/export_model.py \
    --run-dir     runs/YYYYMMDD_XXXXXX \
    --cls-run-dir runs/YYYYMMDD_XXXXXX \
    --version     0.0.X
```

Produces `release_export/` (`model.pt`, `model.onnx`, `cls_model.pt`, `cls_model.onnx`,
`config.yaml`, `sha256.yaml`) and tags this repo `model-v0.0.X` at the current commit.
`--det-checkpoint`/`--cls-checkpoint` default to `best.pt`/`loss_best.pt` — pass
`--det-checkpoint last.pt` etc. to use a different weight file.

**2. In [sc-crop](https://github.com/ivadomed/sc-crop) — publish:**

```bash
bash scripts/publish_release.sh \
    --export-dir release_export/ \
    --package-version 0.1.X
```

Creates the GitHub release (model weights), deploys `config.yaml`, updates
`download.py` and `VERSIONS.md`, bumps the package version, commits + tags + pushes,
and publishes to PyPI. See that script's own `--skip-pypi` flag to defer the PyPI step.

See [VERSIONS.md](https://github.com/ivadomed/sc-crop/blob/main/VERSIONS.md) on the `sc-crop` repo for how package/model versions map to training runs.
