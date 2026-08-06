# Releasing a new model — full walkthrough

Two repos, two steps. This repo (`model_cropping_sc_contrast-agnostic_yolo`) trains and
exports; [`sc-crop`](https://github.com/ivadomed/sc-crop) publishes and is what users
actually install. Neither repo can do a release alone.

## Prerequisites

- Both repos cloned locally, `sc-crop` as a sibling directory of this one (or set
  `SCCROP_REPO=/path/to/sc-crop` — `scripts/publish_release.sh` in `sc-crop` reads it).
- [`gh`](https://cli.github.com/) installed and authenticated (`gh auth login`) — needs
  push access to `ivadomed/sc-crop`.
- A PyPI API token in `~/.pypirc`:
  ```ini
  [pypi]
  username = __token__
  password = <your PyPI token>
  ```
  Skip this if you'll pass `--skip-pypi` (see below) and publish to PyPI separately, later.
- `pip install build twine` in whichever environment runs `sc-crop`'s publish step.

## Step 1 — Export (this repo)

```bash
python scripts/export_model.py \
    --run-dir     runs/YYYYMMDD_XXXXXX \
    --cls-run-dir runs/YYYYMMDD_XXXXXX \
    --version     0.0.X
```

`--run-dir`/`--cls-run-dir` are the detector and classifier run directories produced by
`scripts/train_all.sh` (or `run_pipeline.py --mode detection` / `--mode classification`
run separately). `--version` is the **model** version — see "Two version numbers" below.

This command:
1. Exports both `best.pt` checkpoints to ONNX (`--det-checkpoint`/`--cls-checkpoint`
   override which weight file, default `best.pt`/`loss_best.pt`).
2. Writes `release_export/{model.pt, model.onnx, cls_model.pt, cls_model.onnx}`.
3. Writes `release_export/config.yaml` — every inference parameter (`si_res`,
   `inplane_res`, `channels`, `norm_scope`, `imgsz`, `conf`, `regularization`,
   `cls_conf`) read straight from this training run's own snapshotted
   `configs/preprocess.yaml`, plus full provenance (git hashes, run ids, wandb ids).
4. Writes `release_export/sha256.yaml` — SHA256 of the 4 model files, for `sc-crop`'s
   download-integrity check.
5. **Tags this repo** `model-v0.0.X` at the current commit and pushes the tag — this is
   the permanent record of exactly which training-repo state produced this model.
   Fails loudly if that tag already exists (you already exported this version once).

Nothing is published yet — `release_export/` is gitignored, local-only.

## Step 2 — Publish (sc-crop repo)

```bash
cd ../sc-crop   # or wherever your sc-crop clone is
bash scripts/publish_release.sh \
    --export-dir /path/to/model_cropping_sc_contrast-agnostic_yolo/release_export \
    --package-version 0.1.X
```

`--package-version` is the **package** version — see "Two version numbers" below.
Add `--skip-pypi` to do everything except the PyPI upload (useful if you don't have a
token handy yet, or want to review the GitHub release first).

This command:
1. Refuses to continue if `--package-version` or the model version (read from
   `sha256.yaml`) isn't strictly greater than the last row of `VERSIONS.md` — this is
   the guardrail against re-publishing a version by accident.
2. Creates the GitHub release `vX.Y.Z` on `ivadomed/sc-crop` with the 4 model files
   attached.
3. Deploys `config.yaml` into `sc_crop/config.yaml` (only the inference-relevant keys —
   strips the training-side provenance fields).
4. Updates `sc_crop/download.py`'s `_MODEL_TAG`/`_ASSETS` with the new release URL and
   SHA256 hashes.
5. Inserts a new row in `VERSIONS.md`.
6. Bumps `pyproject.toml` and `sc_crop/__init__.py` to the new package version.
7. Commits, tags (`vX.Y.Z`), and pushes all of the above.
8. Builds the package from a **clean git archive** (not the working directory — avoids
   accidentally bundling gitignored local files) and uploads to PyPI with `twine`,
   after checking no `.pt`/`.onnx` weight file ended up in the built sdist by mistake.

## Two version numbers — don't confuse them

Every release involves **two independent version numbers**:

| | Bumped by | Tag | Meaning |
|---|---|---|---|
| **Model version** | `export_model.py --version` | `model-vX.Y.Z` (here) + `vX.Y.Z` (sc-crop, GitHub release) | Identifies the detector+classifier weights |
| **Package version** | `publish_release.sh --package-version` | `vX.Y.Z` (sc-crop, git tag + PyPI) | Identifies the `sc-crop` code (CLI, API, padding defaults, bugfixes) |

They don't move together. A package-only fix (no new model) bumps only the package
version, reusing the existing model version — several rows in `VERSIONS.md` can point to
the same model. A new model bumps both. `VERSIONS.md` records the mapping; read its
"Lecture du tableau" section for the full explanation.

## Verifying a release worked

```bash
pip install --upgrade sc-crop==<PACKAGE_VERSION>
python -c "import sc_crop; print(sc_crop.__version__, sc_crop.__model_version__)"
sc_crop download   # forces a fresh download + SHA256 check of the new model
```

## If something goes wrong partway through

`publish_release.sh` runs its 7 phases in order but isn't restartable from an arbitrary
phase. If it fails after Step 6 (already pushed to `sc-crop`) but before Step 7 (PyPI),
publish manually from the `sc-crop` repo:

```bash
cd sc-crop && python -m build && twine check dist/* && twine upload dist/*
```

(This is exactly what `publish.sh` at the root of `sc-crop` does, and what
`publish_release.sh`'s own Phase 7 delegates to.)

If it fails before Step 6 (nothing pushed), fix the problem and re-run the whole command
— `git add`/`commit` on an already-clean tree is a no-op, so this is safe to retry.
