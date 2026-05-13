#!/usr/bin/env python3
"""
Train a YOLO model for spinal cord detection or classification on axial MRI slices.

Reads configs/training.yaml. Selects detection or classification via --mode (or mode override in config).

Detection (mode: detection)
  Model  : yolo26{n,s,m,l,x}.pt
  Loss   : standard BCE (no focal)
  Aug    : YOLO built-in (hsv_v=0.15, degrees=15, scale=0.2, translate=0.1, fliplr=0.5, flipud=0.5)
             + optional extra MRI albumentations when extra_augment=true
             (GaussNoise, GaussianBlur, Downscale, RandomGamma, BiasField, RandomInvert, Contrast)
  Output : <run-dir>/checkpoints/weights/{best,last}.pt
             best.pt = highest fitness = 0.1·mAP50 + 0.9·mAP50-95 on val

Classification (mode: classification)
  Model  : yolo26n-cls.pt
  Aug    : Same pipeline as detection — letterbox (pad to imgsz, no crop) + affine (scale=0.2,
             degrees=15, translate=0.1) + HSV brightness (hsv_v=0.15) + fliplr=0.5 + flipud=0.5.
             ClassificationDataset is patched to replace torchvision RandomResizedCrop+RandAugment
             (which crop content and include Grayscale, destroying pseudo-RGB channel encoding).
  Output : <run-dir>/checkpoints_cls/weights/{best,last,loss_best}.pt
             best.pt      = highest val accuracy_top1
             loss_best.pt = lowest val/loss

Usage:
    python scripts/train.py --mode detection  --dataset runs/20260504/dataset/dataset.yaml --run-dir runs/20260601_120000
    python scripts/train.py --mode classification --dataset runs/20260601_120000/dataset_cls --run-dir runs/20260601_120000
    python scripts/train.py --mode detection  --dataset ... --run-dir ... --no-augment
    python scripts/train.py --mode detection  --dataset ... --run-dir ... --no-wandb
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




# ── Augmentation helpers ───────────────────────────────────────────────────────

class BiasField(A.ImageOnlyTransform):
    """Polynomial multiplicative bias field simulating MRI B1 inhomogeneity."""

    def __init__(self, coef_range=(-0.4, 0.4), order=3, p=0.2):
        super().__init__(p=p)
        self.coef_range = coef_range
        self.order = order

    def apply(self, img, **params):
        h, w = img.shape[:2]
        x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        field = np.ones((h, w), dtype=np.float32)
        for i in range(self.order):
            for j in range(self.order - i):
                field += np.random.uniform(*self.coef_range) * (x**i) * (y**j)
        field = np.clip(field, 0.1, 3.0)
        if img.ndim == 3:
            field = field[:, :, None]
        return np.clip(img.astype(np.float32) * field, 0, 255).astype(np.uint8)

    def get_transform_init_args_dict(self):
        return {"coef_range": self.coef_range, "order": self.order}


class RandomInvert(A.ImageOnlyTransform):
    """Pixel inversion (255 - img) simulating T1↔T2 contrast swap."""

    def apply(self, img, **params):
        return 255 - img

    def get_transform_init_args_dict(self):
        return {}


def patch_albumentations_detection() -> None:
    """Monkeypatch ultralytics Albumentations to inject extra MRI transforms for detection.

    Must be called BEFORE YOLO() so workers inherit the patch at fork time.
    Only call when extra_augment=True.
    """
    from ultralytics.data.augment import Albumentations

    extra = [
        A.Affine(scale=(0.25, 4.0), translate_percent=(-0.3, 0.3), p=0.2),
        A.GaussNoise(std_range=(0.05, 0.15), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Downscale(scale_range=(0.25, 1.0),
                    interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR},
                    p=0.25),
        A.RandomGamma(gamma_limit=(80, 120), p=0.1),
        BiasField(coef_range=(-0.4, 0.4), order=3, p=0.2),
        RandomInvert(p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.3, 0.3), p=0.15),
    ]
    bbox_params = A.BboxParams(format="yolo", label_fields=["class_labels"])

    _orig = Albumentations.__init__

    def _patched(self, *args, **kwargs):
        _orig(self, *args, **kwargs)
        existing = list(self.transform.transforms) if self.transform is not None else []
        self.transform = A.Compose(existing + extra, bbox_params=bbox_params)
        self.contains_spatial = True
        names = [type(e).__name__ for e in self.transform.transforms]
        print(f"\n[DET AUG] Pipeline ({len(names)} transforms): {names}\n")

    Albumentations.__init__ = _patched


def patch_classification_letterbox(imgsz: int, hsv_v: float = 0.15, scale: float = 0.2,
                                    degrees: float = 15.0, translate: float = 0.1,
                                    fliplr: float = 0.5, flipud: float = 0.5) -> None:
    """Patch ClassificationDataset to use letterbox + detection-style augmentations.

    Replaces torchvision RandomResizedCrop + RandAugment with:
      - Letterbox: resize maintaining aspect ratio, pad to imgsz×imgsz with grey (114)
      - Affine (train only): random scale/rotation/translation matching detection RandomPerspective
      - HSV-V brightness (train only)
      - Random flips (train only)

    Must be called BEFORE YOLO() so workers inherit the patch at fork time.
    """
    import torch as _torch
    from ultralytics.data import ClassificationDataset

    class _LBTransform:
        def __init__(self, augment: bool):
            self.augment = augment

        def __call__(self, pil_img):
            img = np.array(pil_img, dtype=np.uint8)  # RGB HWC
            # Letterbox: maintain aspect ratio, pad with grey (114)
            h, w = img.shape[:2]
            r = imgsz / max(h, w)
            nh, nw = round(h * r), round(w * r)
            if nh != h or nw != w:
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            ph, pw = imgsz - nh, imgsz - nw
            img = cv2.copyMakeBorder(img, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
            if self.augment:
                # Affine: scale + rotation + translation (mirrors detection RandomPerspective)
                s  = np.random.uniform(1 - scale, 1 + scale)
                a  = np.random.uniform(-degrees, degrees)
                tx = np.random.uniform(-translate, translate) * imgsz
                ty = np.random.uniform(-translate, translate) * imgsz
                M  = cv2.getRotationMatrix2D((imgsz / 2, imgsz / 2), a, s)
                M[0, 2] += tx
                M[1, 2] += ty
                img = cv2.warpAffine(img, M, (imgsz, imgsz),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
                # HSV-V brightness (matches detection RandomHSV with hgain=sgain=0)
                if hsv_v > 0:
                    gain = np.random.uniform(-hsv_v, hsv_v)
                    lut  = np.clip(np.arange(256, dtype=np.float32) * (1 + gain), 0, 255).astype(np.uint8)
                    img  = cv2.LUT(img, lut)
                if flipud > 0 and np.random.random() < flipud:
                    img = img[::-1]
                if fliplr > 0 and np.random.random() < fliplr:
                    img = img[:, ::-1]
            # CHW float32 [0, 1]
            return _torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0

    _orig = ClassificationDataset.__init__

    def _patched(self, root, args, augment=False, prefix=""):
        _orig(self, root, args, augment=augment, prefix=prefix)
        self.torch_transforms = _LBTransform(augment=augment)
        mode = "train" if augment else "val"
        aug_desc = (f"affine(s={scale},d={degrees},t={translate}) + hsv_v={hsv_v} + "
                    f"flip(lr={fliplr},ud={flipud})") if augment else "none"
        print(f"\n[CLS LB] {mode}: letterbox({imgsz}) + {aug_desc}\n")

    ClassificationDataset.__init__ = _patched



# ── Config loading ─────────────────────────────────────────────────────────────

def _load_config(config_path: str | Path, mode_override: str | None = None) -> tuple[str, dict]:
    """Load config, return (mode, flat_dict) merging common keys + mode-specific section.

    mode_override (CLI --mode) takes precedence over the mode key in the file.
    """
    raw  = yaml.safe_load(Path(config_path).read_text())
    mode = mode_override or raw.get("mode")
    assert mode in ("detection", "classification"), \
        f"mode must be detection|classification — set in config or via --mode, got {mode!r}"
    common  = {k: v for k, v in raw.items() if k not in ("detection", "classification", "mode")}
    section = raw.get(mode, {})
    return mode, {**common, **section}



# ── Detection training ─────────────────────────────────────────────────────────

def _train_detection(cfg: dict, dataset: str | Path, run_dir: Path,
                     no_wandb: bool, no_augment: bool) -> None:
    model_name    = cfg.get("model",         "yolo26n.pt")
    epochs        = cfg.get("epochs",        200)
    imgsz         = cfg.get("imgsz",         320)
    batch         = cfg.get("batch",         -1)
    device        = str(cfg.get("device",    "0"))
    patience      = cfg.get("patience",      100)
    workers       = cfg.get("workers",       2)
    fraction      = cfg.get("fraction",      1.0)
    extra_augment = cfg.get("extra_augment", False)
    wandb_project = cfg.get("wandb_project", "spine_detection")
    wandb_entity  = cfg.get("wandb_entity",  None)
    seed          = cfg.get("seed",          50)

    dataset_dir = Path(dataset).parent
    pipeline_config, build_stats = {}, {}
    for fname, target in [("pipeline_config.yaml", pipeline_config),
                           ("build_stats.yaml",     build_stats)]:
        p = dataset_dir / fname
        if p.exists():
            target.update(yaml.safe_load(p.read_text()) or {})

    if no_wandb:
        SETTINGS["wandb"] = False
        os.environ["WANDB_MODE"] = "disabled"
    else:
        SETTINGS["wandb"] = True
        import importlib
        import ultralytics.utils.callbacks.wb as _wb_mod
        importlib.reload(_wb_mod)
        import wandb
        wb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_dir.name,
            tags=["yolo", "spine", "mri", "detection"],
            config={
                "model":          model_name,
                "dataset":        str(dataset),
                "imgsz":          imgsz,
                "batch":          batch,
                "epochs":         epochs,
                "patience":       patience,
                "seed":           seed,
                "augment":        not no_augment,
                "extra_augment":  extra_augment and not no_augment,
                "pipeline":       pipeline_config,
                "data": {
                    **{k: v for k, v in build_stats.items() if not isinstance(v, dict)},
                    "factor":       build_stats.get("dataset_factors", {}),
                    "train_slices": build_stats.get("slices_train_final", {}),
                    "sc_bg_raw":    build_stats.get("sc_bg_slices_raw", {}),
                    "sc_bg_final":  build_stats.get("sc_bg_slices_final", {}),
                },
            },
        )
        (run_dir / "wandb_run_id.txt").write_text(wb_run.id)

    if not no_augment and extra_augment:
        patch_albumentations_detection()

    model = YOLO(model_name)

    aug = dict(
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, scale=0.0, shear=0.0, perspective=0.0, translate=0.0,
        fliplr=0.0, flipud=0.0,
        mosaic=0.0, copy_paste=0.0,
    ) if no_augment else dict(
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.15,
        degrees=15.0, scale=0.2, shear=0.0, perspective=0.0, translate=0.1,
        fliplr=0.5, flipud=0.5,
        mosaic=0.0, copy_paste=0.0,
    )

    results = model.train(
        resume=False,
        data=str(dataset),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(run_dir.resolve()),
        name="checkpoints",
        patience=patience,
        workers=workers,
        fraction=fraction,
        seed=seed,
        save=True,
        val=True,
        rect=False,
        **aug,
    )

    save_dir = Path(results.save_dir) if results is not None else Path(model.trainer.save_dir)
    print(f"\nBest weights: {save_dir / 'weights' / 'best.pt'}")


# ── Classification training ────────────────────────────────────────────────────

def _train_classification(cfg: dict, dataset: str | Path, run_dir: Path,
                           no_wandb: bool) -> None:
    model_name    = cfg.get("model",         "yolo26n-cls.pt")
    epochs        = cfg.get("epochs",        200)
    imgsz         = cfg.get("imgsz",         320)
    batch         = cfg.get("batch",         -1)
    device        = str(cfg.get("device",    "0"))
    patience      = cfg.get("patience",      100)
    workers       = cfg.get("workers",       2)
    wandb_project = cfg.get("wandb_project", "spine_detection")
    wandb_entity  = cfg.get("wandb_entity",  None)
    seed          = cfg.get("seed",          50)

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
            name=run_dir.name + "_cls",
            tags=["yolo", "spine", "mri", "classification"],
            config={
                "model":    model_name,
                "dataset":  str(dataset),
                "imgsz":    imgsz,
                "batch":    batch,
                "epochs":   epochs,
                "patience": patience,
                "seed":     seed,
                "task":     "classify",
            },
        )

    # Patch ClassificationDataset before YOLO() so workers inherit it at fork time.
    # Replaces torchvision RandomResizedCrop + RandAugment with letterbox + detection-style aug.
    patch_classification_letterbox(
        imgsz=imgsz, hsv_v=0.15, scale=0.2, degrees=15.0, translate=0.1, fliplr=0.5, flipud=0.5,
    )
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
        data=str(Path(dataset).resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(run_dir.resolve()),
        name="checkpoints_cls",
        patience=patience,
        workers=workers,
        seed=seed,
        save=True,
        val=True,
        mosaic=0.0,
    )

    save_dir = Path(results.save_dir) if results is not None else Path(model.trainer.save_dir)
    weights  = save_dir / "weights"
    print(f"\nBest weights (accuracy): {weights / 'best.pt'}")
    print(f"Best weights (val loss):  {weights / 'loss_best.pt'}  (val/loss={best_val_loss[0]:.4f})")


# ── Public API ────────────────────────────────────────────────────────────────

def run(config: str | Path,
        dataset: str | Path,
        run_dir: str | Path,
        mode: str | None = None,
        no_wandb: bool = False,
        no_augment: bool = False) -> None:
    """Train detection or classification model. All hyper-parameters read from config.

    mode: override the mode key in the config file (optional).
    """
    mode, cfg = _load_config(config, mode)
    run_dir   = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 50)
    random.seed(seed)
    np.random.seed(seed)

    if mode == "detection":
        _train_detection(cfg, dataset, run_dir, no_wandb, no_augment)
    else:
        _train_classification(cfg, dataset, run_dir, no_wandb)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO for spinal cord detection or classification on axial MRI slices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     default="configs/training.yaml",
                        help="YAML config file")
    parser.add_argument("--mode",       default=None, choices=["detection", "classification"],
                        help="Override mode from config (default: read from config file)")
    parser.add_argument("--dataset",    required=True,
                        help="Detection: path to dataset.yaml. Classification: path to dataset directory.")
    parser.add_argument("--run-dir",    required=True,
                        help="Run directory; checkpoints saved to <run-dir>/checkpoints[_cls]/")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable all augmentation (detection only)")
    parser.add_argument("--no-wandb",   action="store_true")
    args = parser.parse_args()

    run(config=args.config, dataset=args.dataset, run_dir=args.run_dir,
        mode=args.mode, no_wandb=args.no_wandb, no_augment=args.no_augment)


if __name__ == "__main__":
    main()
