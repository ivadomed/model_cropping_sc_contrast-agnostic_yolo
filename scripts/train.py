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

Saves weights to: checkpoints/<run-id>/weights/best.pt
W&B run id = run-id → resume continues the same run.

Usage:
    python scripts/train.py --dataset-yaml datasets/dataset.yaml --run-id yolo_spine_v1
    python scripts/train.py ... --model yolo11s.pt --epochs 150 --no-wandb
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
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Downscale(scale_min=0.5, scale_max=1.0, p=0.25),
        A.RandomGamma(gamma_limit=(80, 120), p=0.1),
    ]

    dataset = trainer.train_loader.dataset
    for t in dataset.transforms.transforms:
        if isinstance(t, Albumentations) and t.transform is not None:
            bbox_params = None
            if hasattr(t.transform, "processors") and "bboxes" in t.transform.processors:
                bbox_params = A.BboxParams(format="yolo", label_fields=["class_labels"])
            t.transform = A.Compose(list(t.transform.transforms) + extra, bbox_params=bbox_params)
            LOGGER.info(f"MRI augmentations injected: {[type(e).__name__ for e in extra]}")
            return

    LOGGER.warning("Albumentations pipeline not found — MRI augmentations skipped.")


def make_wandb_callback(wandb_project: str, run_id: str, args):
    def _on_pretrain_routine_start(trainer):
        import wandb
        config = {
            "model": Path(args.model).stem,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "patience": args.patience,
            "aug/hsv_v": 0.15, "aug/degrees": 15.0, "aug/scale": 0.2,
            "aug/fliplr": 0.5, "aug/flipud": 0.5,
            "aug/gauss_noise_p": 0.1, "aug/gaussian_blur_p": 0.2,
            "aug/downscale_p": 0.25, "aug/random_gamma_p": 0.1,
            "aug/mosaic": 0.0, "aug/copy_paste": 0.0,
            "aug/hsv_h": 0.0, "aug/hsv_s": 0.0,
        }
        if wandb.run is not None:
            wandb.config.update(config, allow_val_change=True)
            return
        run = wandb.init(
            project=wandb_project,
            name=run_id,
            id=run_id,
            resume="allow",
            config=config,
            tags=["yolo", "spine", "mri"],
        )
        LOGGER.info(f"W&B run: {run.url}")

    return _on_pretrain_routine_start


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO for spinal cord detection on axial MRI slices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-yaml", required=True)
    parser.add_argument("--run-id",  required=True, help="Run name used for checkpoints/ and W&B")
    parser.add_argument("--model",   default="yolo11n.pt",
                        help="Starting weights: yolo11n.pt | yolo11s.pt | path/to/custom.pt")
    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--imgsz",    type=int,   default=320)
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--device",   default="0")
    parser.add_argument("--patience", type=int,   default=20)
    parser.add_argument("--workers",  type=int,   default=8)
    parser.add_argument("--resume",   action="store_true",
                        help="Resume from last.pt (set --model to last.pt path)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="spine_detection")
    args = parser.parse_args()

    if args.no_wandb:
        SETTINGS["wandb"] = False
        os.environ["WANDB_MODE"] = "disabled"

    model = YOLO(args.model)
    model.add_callback("on_train_start", inject_mri_augmentations)

    if not args.no_wandb:
        model.add_callback(
            "on_pretrain_routine_start",
            make_wandb_callback(args.wandb_project, args.run_id, args),
        )

    results = model.train(
        resume=args.resume,
        data=args.dataset_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="checkpoints",
        name=args.run_id,
        patience=args.patience,
        workers=args.workers,
        save=True,
        val=True,
        rect=False,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.15,
        degrees=15.0, scale=0.2, shear=0.0, perspective=0.0, translate=0.1,
        fliplr=0.5, flipud=0.5,
        mosaic=0.0, copy_paste=0.0,
    )

    save_dir = Path(results.save_dir) if results is not None else Path(model.trainer.save_dir)
    print(f"\nBest weights: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
