#!/usr/bin/env python3
"""
Train YOLO classification model (yolo26n-cls) for SC / no-SC axial slice classification.

Reads configs/training_cls.yaml (mode: classification).
Saves to <run-dir>/checkpoints_cls/weights/{best,last,loss_best}.pt.
  best.pt      — highest val accuracy_top1 (ultralytics default)
  loss_best.pt — lowest val/loss (saved via on_fit_epoch_end callback)

Dataset must have the YOLO classification structure built by build_class_dataset.py:
  <dataset-dir>/train/sc/   <dataset-dir>/train/no_sc/
  <dataset-dir>/val/sc/     <dataset-dir>/val/no_sc/

Augmentations :
  YOLO built-in : degrees=15, scale=0.2, translate=0.1, fliplr=0.5, flipud=0.5,
                  hsv_v=0.15, erasing=0.4 (patches noirs)
  albumentations: GaussNoise (p=0.1), GaussianBlur (p=0.2), Downscale (p=0.25),
                  RandomGamma (p=0.1), RandomBrightnessContrast contrast-only (p=0.15)

Usage:
    python scripts/train_cls.py \\
        --run-dir runs/20260601_120000 \\
        --dataset-dir runs/20260601_120000/dataset_cls
    python scripts/train_cls.py \\
        --run-dir runs/20260601_120000 \\
        --dataset-dir runs/20260601_120000/dataset_cls \\
        --no-wandb
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import albumentations as A
import numpy as np
import yaml

from ultralytics import YOLO
from ultralytics.utils import SETTINGS


def patch_albumentations_cls() -> None:
    """Monkeypatch ultralytics Albumentations.__init__ to inject standard augmentations for classification.

    Must be called BEFORE YOLO() so the patch is active when the dataset is built.
    No bbox_params — classification pipeline does not use bounding boxes.
    Rotation/scaling are handled by YOLO's own degrees/scale/translate params.
    """
    from ultralytics.data.augment import Albumentations

    extra = [
        A.GaussNoise(std_range=(0.05, 0.15), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Downscale(scale_range=(0.25, 1.0),
                    interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR},
                    p=0.25),
        A.RandomGamma(gamma_limit=(80, 120), p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.3, 0.3), p=0.15),
    ]

    _original_init = Albumentations.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        existing = list(self.transform.transforms) if self.transform is not None else []
        self.transform = A.Compose(existing + extra)
        names = [type(e).__name__ for e in self.transform.transforms]
        print(f"\n[AUG CLS] Pipeline patché ({len(names)} transforms): {names}\n")

    Albumentations.__init__ = _patched_init


def run(config: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        run_dir: str | Path | None = None,
        no_wandb: bool = False) -> None:
    """Train YOLO classification model. Hyper-parameters read from config file."""
    model_name    = "yolo26n-cls.pt"
    epochs        = 50
    imgsz         = 320
    batch         = -1
    device        = "0"
    patience      = 20
    workers       = 8
    wandb_project = "spine_detection"
    wandb_entity  = None

    if config:
        cfg = yaml.safe_load(Path(config).read_text())
        model_name    = cfg.get("model",          model_name)
        epochs        = cfg.get("epochs",         epochs)
        imgsz         = cfg.get("imgsz",          imgsz)
        batch         = cfg.get("batch",          batch)
        device        = str(cfg.get("device",     device))
        patience      = cfg.get("patience",       patience)
        workers       = cfg.get("workers",        workers)
        wandb_project = cfg.get("wandb_project",  wandb_project)
        wandb_entity  = cfg.get("wandb_entity",   wandb_entity)

    seed = int(os.environ.get("SEED", 50))
    random.seed(seed)
    np.random.seed(seed)

    if no_wandb:
        SETTINGS["wandb"] = False
        os.environ["WANDB_MODE"] = "disabled"
    else:
        SETTINGS["wandb"] = True
        import importlib
        import ultralytics.utils.callbacks.wb as _wb_mod
        importlib.reload(_wb_mod)
        import wandb
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=Path(run_dir).name + "_cls",
            tags=["yolo", "spine", "mri", "classification"],
            config={
                "model":    model_name,
                "dataset":  str(dataset_dir),
                "imgsz":    imgsz,
                "batch":    batch,
                "epochs":   epochs,
                "patience": patience,
                "seed":     seed,
                "task":     "classify",
            },
        )

    patch_albumentations_cls()
    model = YOLO(model_name)

    best_val_loss = [float("inf")]

    def _save_loss_best(trainer) -> None:
        loss = trainer.metrics.get("val/loss")
        if loss is None or loss >= best_val_loss[0]:
            return
        best_val_loss[0] = loss
        last = Path(trainer.save_dir) / "weights" / "last.pt"
        if last.exists():
            shutil.copy2(str(last), str(last.parent / "loss_best.pt"))

    model.add_callback("on_fit_epoch_end", _save_loss_best)

    results = model.train(
        data=str(Path(dataset_dir).resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(Path(run_dir).resolve()),
        name="checkpoints_cls",
        patience=patience,
        workers=workers,
        seed=seed,
        save=True,
        val=True,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.15,
        degrees=15.0, scale=0.2, translate=0.1,
        fliplr=0.5, flipud=0.5,
    )

    save_dir = Path(results.save_dir) if results is not None else Path(model.trainer.save_dir)
    weights  = save_dir / "weights"
    print(f"\nBest weights (accuracy): {weights / 'best.pt'}")
    print(f"Best weights (val loss): {weights / 'loss_best.pt'}  (val/loss={best_val_loss[0]:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO classification model for SC/no-SC slice classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",      default="configs/training_cls.yaml",
                        help="YAML config file (configs/training_cls.yaml)")
    parser.add_argument("--dataset-dir", required=True,
                        help="Classification dataset directory (built by build_class_dataset.py)")
    parser.add_argument("--run-dir",     required=True,
                        help="Run directory; checkpoints saved to <run-dir>/checkpoints_cls/")
    parser.add_argument("--no-wandb",    action="store_true")
    args = parser.parse_args()
    run(config=args.config, dataset_dir=args.dataset_dir,
        run_dir=args.run_dir, no_wandb=args.no_wandb)


if __name__ == "__main__":
    main()
