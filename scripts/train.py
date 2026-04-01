#!/usr/bin/env python3
"""
Train a YOLO 2D model for spinal cord detection on axial MRI slices.

Augmentations from the contrast-agnostic paper:
  - Affine (rotation + scaling)           → YOLO: degrees, scale
  - Gaussian noise                        → albumentations: GaussNoise       (p=0.1)
  - Gaussian smoothing                    → albumentations: GaussianBlur     (p=0.2)
  - Brightness augmentation               → YOLO: hsv_v                     (p=0.15)
  - Low-resolution simulation [0.5, 1.0] → albumentations: Downscale        (p=0.25)
  - Gamma correction                      → albumentations: RandomGamma      (p=0.1)
  - Mirroring across all axes             → YOLO: fliplr, flipud             (p=0.5)

W&B: ultralytics native integration — configured via WANDB_PROJECT / WANDB_RUN_ID env vars.
Saves weights to: checkpoints/<run-id>/weights/best.pt

Usage:
    python scripts/train.py --dataset-yaml datasets/dataset.yaml --run-id yolo_spine_v1
    python scripts/train.py ... --model yolo26s.pt --epochs 150 --no-wandb
    python scripts/train.py ... --model yolo12n.pt --run-id yolo12_spine_v1
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import LOGGER, SETTINGS


def inject_mri_augmentations(trainer) -> None:
    """Inject MRI-specific albumentations after the dataloader is built."""
    import albumentations as A
    from ultralytics.data.augment import Albumentations

    extra = [
        A.GaussNoise(std_range=(0.01, 0.03), p=0.1),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Downscale(scale_range=(0.5, 1.0), p=0.25),
        A.RandomGamma(gamma_limit=(80, 120), p=0.1),
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
    parser.add_argument("--dataset-yaml", required=True)
    parser.add_argument("--run-id",  required=True, help="Run name used for checkpoints/ and W&B")
    parser.add_argument("--model",   default="yolo26n.pt",
                        help="Model weights: yolo26{n,s,m,l,x}.pt (latest) | yolo12{n,s,m,l,x}.pt | "
                             "yolo11{n,s,m,l,x}.pt | yolov8{n,s,m,l,x}.pt | path/to/custom.pt")
    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--imgsz",    type=int,   default=640)
    parser.add_argument("--batch",    type=int,   default=-1,
                        help="Batch size (-1 = auto-detect optimal for GPU memory)")
    parser.add_argument("--device",   default="0")
    parser.add_argument("--patience", type=int,   default=20)
    parser.add_argument("--workers",  type=int,   default=8)
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use (e.g. 0.01 for quick test)")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from last.pt (set --model to last.pt path)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable all augmentations (YOLO built-in + albumentations MRI)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="spine_detection")
    args = parser.parse_args()

    if args.no_wandb:
        SETTINGS["wandb"] = False
        os.environ["WANDB_MODE"] = "disabled"
    else:
        SETTINGS["wandb"] = True  # reset in case a previous --no-wandb run persisted False
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_id,
            tags=["yolo", "spine", "mri"],
        )

    model = YOLO(args.model)
    if not args.no_augment:
        model.add_callback("on_train_start", inject_mri_augmentations)

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
        save=True,
        val=True,
        rect=False,
        **aug,
    )

    save_dir = Path(results.save_dir) if results is not None else Path(model.trainer.save_dir)
    print(f"\nBest weights: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
