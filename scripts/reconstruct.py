#!/usr/bin/env python3
"""
Reconstruct NIfTI volumes from predictions and raw data.

For each patient in predictions/<run-id>/:
  - original.nii.gz           → symlink to raw image
  - mask_original.nii.gz      → symlink to raw mask
  - pred_slices_stacked.nii.gz → binary 3D volume from predicted bboxes (LAS space)
  - gt_slices_stacked.nii.gz  → binary 3D volume from GT bboxes (LAS space)

All stacked volumes share the affine of the LAS-reoriented original image.
Bbox coords are normalised to [0,1] relative to the native LAS slice dimensions.

Usage:
    python scripts/reconstruct.py --run-id yolo_spine_v1
    python scripts/reconstruct.py --run-id yolo_spine_v1 --predictions predictions --out reconstructions
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import reorient_to_las, resample_z, stack_bbox_volume


def reconstruct_patient(
    pred_patient_dir: Path,
    processed_dir: Path,
    out_dir: Path,
) -> None:
    stem = pred_patient_dir.name
    site = pred_patient_dir.parent.name
    patient_rel = f"{site}/{stem}"

    with open(processed_dir / patient_rel / "meta.yaml") as f:
        meta = yaml.safe_load(f)

    raw_image = Path(meta["raw_image"])
    raw_mask  = Path(meta["raw_mask"])
    H, W, Z   = meta["shape_las"]
    si_res_mm = meta["si_res_mm"]

    out_dir.mkdir(parents=True, exist_ok=True)

    for link_name, target in [("original.nii.gz", raw_image), ("mask_original.nii.gz", raw_mask)]:
        link = out_dir / link_name
        if link.is_symlink():
            link.unlink()
        link.symlink_to(target.resolve())

    # Affine/header from LAS-reoriented + SI-resampled image (matches processed space)
    img_las = resample_z(reorient_to_las(nib.load(str(raw_image))), si_res_mm, order=3)
    affine, header = img_las.affine, img_las.header

    pred_vol = stack_bbox_volume(pred_patient_dir / "slices" / "txt", H, W, Z)
    nib.save(nib.Nifti1Image(pred_vol, affine, header), str(out_dir / "pred_slices_stacked.nii.gz"))

    gt_vol = stack_bbox_volume(processed_dir / patient_rel / "txt", H, W, Z)
    nib.save(nib.Nifti1Image(gt_vol, affine, header), str(out_dir / "gt_slices_stacked.nii.gz"))


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct NIfTI volumes from YOLO predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-id",      required=True)
    parser.add_argument("--predictions", default="predictions")
    parser.add_argument("--processed",   default="processed/10mm_SI")
    parser.add_argument("--out",         default="reconstructions")
    args = parser.parse_args()

    pred_run_dir  = Path(args.predictions) / args.run_id
    processed_dir = Path(args.processed)
    recon_run_dir = Path(args.out) / args.run_id

    patient_dirs = [p.parent.parent for p in pred_run_dir.rglob("volume/bbox_3d.txt")]

    for patient_dir in tqdm(patient_dirs, desc="Patients"):
        site    = patient_dir.parent.name
        stem    = patient_dir.name
        out_dir = recon_run_dir / site / stem
        reconstruct_patient(patient_dir, processed_dir, out_dir)

    print(f"\nReconstructions saved to: {recon_run_dir}")


if __name__ == "__main__":
    main()
