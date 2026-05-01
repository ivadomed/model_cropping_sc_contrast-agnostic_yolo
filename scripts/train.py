#!/usr/bin/env python3
"""
Train a YOLO 2D model for spinal cord / spinal canal detection on axial MRI slices.

Defaults: yolo26n, imgsz=320, batch=-1 (auto), device=0, epochs=100, patience=20.
Dataset par défaut : datasets/10mm_SI_1mm_axial_3ch_sc_only_region_balanced_all_datasets/dataset.yaml

Augmentations (désactivables via --no-augment) issues du papier contrast-agnostic :
  - Affine (rotation + scaling)           → YOLO: degrees=15, scale=0.2
  - Gaussian noise                        → albumentations: GaussNoise       (p=0.1)
  - Gaussian smoothing                    → albumentations: GaussianBlur     (p=0.2)
  - Brightness augmentation               → YOLO: hsv_v=0.15
  - Low-resolution simulation [0.25, 1.0] → albumentations: Downscale       (p=0.25)
  - Gamma correction                      → albumentations: RandomGamma      (p=0.1)
  - Mirroring across all axes             → YOLO: fliplr=0.5, flipud=0.5

Augmentations MRI supplémentaires (albumentations) :
  - Affine zoom+translation : scale [0.25, 4.0], translate ±30%                            (p=0.2)
  - BiasField       : inhomogénéité multiplicative polynomiale (simulation B1 IRM)          (p=0.2)
  - RandomInvert    : inversion pixel-à-pixel 255-img (simulation T1↔T2)                   (p=0.25)
  - Contrast        : RandomBrightnessContrast contrast uniquement                          (p=0.15)

Seed : lit la variable d'environnement SEED (défaut 50) — initialisée dans run_pipeline.sh.
       Propagée à random, numpy et ultralytics (model.train seed=).
W&B : wandb.init() avant model.train() pour contrôler projet/run.
Sauvegarde : checkpoints/<run-id>/weights/{best,last}.pt
             best.pt = meilleur fitness = 0.1·mAP50 + 0.9·mAP50-95 sur val

Usage:
    python scripts/train.py --run-id yolo26n_sc_only
    python scripts/train.py --run-id yolo26n_sc_only --no-augment
    python scripts/train.py --run-id yolo26s_sc_only --model yolo26s.pt --epochs 150
    python scripts/train.py --run-id yolo26n_sc_only_all_datasets \
        --dataset-yaml datasets/10mm_SI_1mm_axial_3ch_sc_only_region_balanced_all_datasets/dataset.yaml
    python scripts/train.py --run-id yolo26n_sc_and_canal \
        --dataset-yaml datasets/10mm_SI_1mm_axial_3ch_sc_and_canal/dataset.yaml
"""

import argparse
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils import LOGGER, SETTINGS


def make_focal_bce(gamma: float):
    """Return a focal-weighted BCE function with reduction='none' (compatible with detection loss .sum())."""
    import torch.nn.functional as F

    def focal_bce(pred, target):
        loss  = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t   = target * pred.sigmoid() + (1 - target) * (1 - pred.sigmoid())
        return loss * (1.0 - p_t) ** gamma

    return focal_bce


def inject_focal_loss(trainer) -> None:
    """Replace the BCE cls loss in the detection head with focal-weighted BCE."""
    gamma = trainer.args.fl_gamma if hasattr(trainer.args, "fl_gamma") else 2.0
    if hasattr(trainer, "compute_loss") and hasattr(trainer.compute_loss, "bce"):
        trainer.compute_loss.bce = make_focal_bce(gamma)
        print(f"\n[FOCAL LOSS] Injected focal BCE (gamma={gamma}) on cls loss\n")
        return
    print("\n[FOCAL LOSS] WARNING: compute_loss.bce not found — focal loss not injected\n")


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



def inject_mri_augmentations(trainer) -> None:
    """Inject MRI-specific albumentations after the dataloader is built."""
    from ultralytics.data.augment import Albumentations

    extra = [
        A.Affine(scale=(0.25, 4.0), translate_percent=(-0.3, 0.3), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.1),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Downscale(scale_range=(0.25, 1.0),
                    interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR},
                    p=0.25),
        A.RandomGamma(gamma_limit=(80, 120), p=0.1),
        BiasField(coef_range=(-0.4, 0.4), order=3, p=0.2),
        RandomInvert(p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.3, 0.3), p=0.15),
    ]

    dataset = trainer.train_loader.dataset
    for t in dataset.transforms.transforms:
        if isinstance(t, Albumentations) and t.transform is not None:
            bbox_params = None
            if hasattr(t.transform, "processors") and "bboxes" in t.transform.processors:
                bbox_params = A.BboxParams(format="yolo", label_fields=["class_labels"])
            t.transform = A.Compose(list(t.transform.transforms) + extra, bbox_params=bbox_params)
            names = [type(e).__name__ for e in t.transform.transforms]
            print(f"\n[MRI AUG] Pipeline actif ({len(names)} transforms): {names}\n")
            return

    print("\n[MRI AUG] WARNING: pipeline albumentations non trouvé — augmentations MRI ignorées\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO for spinal cord detection on axial MRI slices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-yaml",
                        default="datasets/10mm_SI_1mm_axial_3ch_sc_only_region_balanced_all_datasets/dataset.yaml")
    parser.add_argument("--run-id",  required=True, help="Run name used for checkpoints/ and W&B")
    parser.add_argument("--model",   default="yolo26n.pt",
                        help="Model weights: yolo26{n,s,m,l,x}.pt (latest) | yolo12{n,s,m,l,x}.pt | "
                             "yolo11{n,s,m,l,x}.pt | yolov8{n,s,m,l,x}.pt | path/to/custom.pt")
    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--imgsz",    type=int,   default=320)
    parser.add_argument("--batch",    type=int,   default=-1,
                        help="Batch size (-1 = auto-detect optimal for GPU memory)")
    parser.add_argument("--device",   default="0")
    parser.add_argument("--patience", type=int,   default=100)
    parser.add_argument("--workers",  type=int,   default=8)
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use (e.g. 0.01 for quick test)")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from last.pt (set --model to last.pt path)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable all augmentations (YOLO built-in + albumentations MRI)")
    parser.add_argument("--fl-gamma",  type=float, default=0.0,
                        help="Focal loss gamma (0.0 = BCE, 2.0 = standard focal loss)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="spine_detection")
    parser.add_argument("--wandb-entity",  default=None,
                        help="W&B entity (user or team). None = W&B default for the logged-in account.")
    args = parser.parse_args()

    seed = int(os.environ.get("SEED", 50))
    random.seed(seed)
    np.random.seed(seed)

    # Load pipeline + dataset build context for W&B logging
    dataset_dir = Path(args.dataset_yaml).parent
    pipeline_config, build_stats = {}, {}
    for fname, target in [("pipeline_config.yaml", pipeline_config),
                           ("build_stats.yaml",     build_stats)]:
        p = dataset_dir / fname
        if p.exists():
            import yaml as _yaml
            target.update(_yaml.safe_load(p.read_text()) or {})

    if args.no_wandb:
        SETTINGS["wandb"] = False
        os.environ["WANDB_MODE"] = "disabled"
    else:
        SETTINGS["wandb"] = True
        # wb.py is imported once at Python startup — if SETTINGS["wandb"] was False then,
        # wb=None is frozen. Force reload so the assert passes and callbacks are registered.
        import importlib
        import ultralytics.utils.callbacks.wb as _wb_mod
        importlib.reload(_wb_mod)
        import wandb
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_id,
            tags=["yolo", "spine", "mri"],
            config={
                "model":    args.model,
                "dataset":  args.dataset_yaml,
                "imgsz":    args.imgsz,
                "batch":    args.batch,
                "epochs":   args.epochs,
                "patience": args.patience,
                "seed":     seed,
                "augment":  not args.no_augment,
                "fl_gamma": args.fl_gamma,
                "aug": {
                    # ── YOLO built-in ──────────────────────────────
                    "yolo_brightness":          0.0 if args.no_augment else 0.15,
                    "yolo_rotation_deg":        0.0 if args.no_augment else 15.0,
                    "yolo_scale":               0.0 if args.no_augment else 0.2,
                    "yolo_translate":           0.0 if args.no_augment else 0.1,
                    "yolo_flip_lr":             0.0 if args.no_augment else 0.5,
                    "yolo_flip_ud":             0.0 if args.no_augment else 0.5,
                    # ── Albumentations ─────────────────────────────
                    "affine_p":                 0.0 if args.no_augment else 0.2,
                    "affine_scale_range":       [0.25, 4.0],
                    "affine_translate_range":   [-0.3, 0.3],
                    "gauss_noise_p":            0.0 if args.no_augment else 0.1,
                    "gauss_noise_std_range":    [0.01, 0.03],
                    "gaussian_blur_p":          0.0 if args.no_augment else 0.2,
                    "gaussian_blur_limit":      [3, 7],
                    "downscale_p":              0.0 if args.no_augment else 0.25,
                    "downscale_scale_range":    [0.25, 1.0],
                    "random_gamma_p":           0.0 if args.no_augment else 0.1,
                    "random_gamma_limit":       [80, 120],
                    "bias_field_p":             0.0 if args.no_augment else 0.2,
                    "bias_field_coef_range":    [-0.4, 0.4],
                    "pixel_invert_p":           0.0 if args.no_augment else 0.25,
                    "contrast_p":               0.0 if args.no_augment else 0.15,
                    "contrast_limit":           [-0.3, 0.3],
                },
                "pipeline": pipeline_config,
                "data": {
                    **{k: v for k, v in build_stats.items() if not isinstance(v, dict)},
                    "factor":       build_stats.get("dataset_factors", {}),
                    "train_slices": build_stats.get("slices_train_final", {}),
                    "sc_bg_raw":    build_stats.get("sc_bg_slices_raw", {}),
                    "sc_bg_final":  build_stats.get("sc_bg_slices_final", {}),
                },
            },
        )

    if not args.no_wandb:
        # Save W&B run ID so metrics.py can resume this exact run later
        wandb_id_file = Path("checkpoints") / args.run_id / "wandb_run_id.txt"
        wandb_id_file.parent.mkdir(parents=True, exist_ok=True)
        wandb_id_file.write_text(wb_run.id)

    model = YOLO(args.model)
    if not args.no_augment:
        model.add_callback("on_train_start", inject_mri_augmentations)
    if args.fl_gamma > 0.0:
        model.add_callback("on_train_start", inject_focal_loss)

    aug = dict(
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, scale=0.0, shear=0.0, perspective=0.0, translate=0.0,
        fliplr=0.0, flipud=0.0,
        mosaic=0.0, copy_paste=0.0,
    ) if args.no_augment else dict(
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.15,
        degrees=15.0, scale=0.2, shear=0.0, perspective=0.0, translate=0.1,
        fliplr=0.5, flipud=0.5,
        mosaic=0.0, copy_paste=0.0,
    )

    results = model.train(
        resume=args.resume,
        data=args.dataset_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(Path("checkpoints").resolve()),
        name=args.run_id,
        patience=args.patience,
        workers=args.workers,
        fraction=args.fraction,
        seed=seed,
        save=True,
        val=True,
        rect=False,
        **aug,
    )

    save_dir = Path(results.save_dir) if results is not None else Path(model.trainer.save_dir)
    print(f"\nBest weights: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
