#!/usr/bin/env python3
"""
Reconstruct NIfTI volumes from predictions and raw data.

For each patient in predictions/<run-id>/:
  - original.nii.gz       → symlink to raw image
  - resampled.nii.gz      → raw image reoriented to RPI + resampled
  - mask_resampled.nii.gz → raw mask reoriented to RPI + resampled (order=1)
  - pred_slices_stacked.nii.gz → binary 3D volume: filled predicted bboxes per slice
  - gt_slices_stacked.nii.gz   → binary 3D volume: filled GT bboxes per slice

All volumes share the affine/spacing of resampled.nii.gz.

Usage:
    python scripts/reconstruct.py --run-id yolo_spine_v1
    python scripts/reconstruct.py --run-id yolo_spine_v1 --predictions predictions --out reconstructions
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import reorient_to_rpi, resample_nifti, stack_bbox_volume


def reconstruct_patient(
    pred_patient_dir: Path,
    processed_dir: Path,
    out_dir: Path,
) -> None:
    # Infer site/stem from path: predictions/<run-id>/<site>/<stem>
    stem = pred_patient_dir.name
    site = pred_patient_dir.parent.name
    patient_rel = f"{site}/{stem}"

    with open(processed_dir / patient_rel / "meta.yaml") as f:
        meta = yaml.safe_load(f)

    raw_image = Path(meta["raw_image"])
    raw_mask  = Path(meta["raw_mask"])
    res_mm    = tuple(meta["resolution_mm"])
    H, W, Z   = meta["shape"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Symlink to original raw image
    orig_link = out_dir / "original.nii.gz"
    if orig_link.is_symlink():
        orig_link.unlink()
    orig_link.symlink_to(raw_image.resolve())

    # Resampled image and mask (generated from raw, never copied from processed/)
    img_r = resample_nifti(reorient_to_rpi(nib.load(str(raw_image))), res_mm, order=3)
    nib.save(img_r, str(out_dir / "resampled.nii.gz"))

    mask_r = resample_nifti(reorient_to_rpi(nib.load(str(raw_mask))), res_mm, order=1)
    nib.save(mask_r, str(out_dir / "mask_resampled.nii.gz"))

    affine = img_r.affine
    header = img_r.header

    # Predicted bbox slices stacked into a 3D binary volume
    pred_vol = stack_bbox_volume(pred_patient_dir / "slices" / "txt", H, W, Z)
    nib.save(nib.Nifti1Image(pred_vol, affine, header), str(out_dir / "pred_slices_stacked.nii.gz"))

    # GT bbox slices stacked (from processed/txt/)
    gt_vol = stack_bbox_volume(processed_dir / patient_rel / "txt", H, W, Z)
    nib.save(nib.Nifti1Image(gt_vol, affine, header), str(out_dir / "gt_slices_stacked.nii.gz"))


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct NIfTI volumes from YOLO predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-id",       required=True)
    parser.add_argument("--predictions",  default="predictions")
    parser.add_argument("--processed",    default="processed")
    parser.add_argument("--out",          default="reconstructions")
    args = parser.parse_args()

    pred_run_dir = Path(args.predictions) / args.run_id
    processed_dir = Path(args.processed)
    recon_run_dir = Path(args.out) / args.run_id

    # Collect all patient dirs: predictions/<run-id>/<site>/<stem>/
    patient_dirs = [p for p in pred_run_dir.rglob("volume/bbox_3d.txt")]
    patient_dirs = [p.parent.parent for p in patient_dirs]  # <site>/<stem>/

    for patient_dir in tqdm(patient_dirs, desc="Patients"):
        site = patient_dir.parent.name
        stem = patient_dir.name
        out_dir = recon_run_dir / site / stem
        reconstruct_patient(patient_dir, processed_dir, out_dir)

    print(f"\nReconstructions saved to: {recon_run_dir}")


if __name__ == "__main__":
    main()
