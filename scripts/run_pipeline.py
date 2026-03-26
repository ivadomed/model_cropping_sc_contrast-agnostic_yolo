#!/usr/bin/env python3
"""
End-to-end pipeline on a single volume for testing.

Runs the full pipeline in sandbox/<stem>/ (overwritten on each call):
  png/             axial slices (from GT mask)
  txt/             GT YOLO labels
  volume/bbox_3d.txt  GT 3D bbox
  slices/txt/      YOLO predictions per slice
  slices/png/      predictions visualised (bbox in red)
  volume_pred/bbox_3d.txt  predicted 3D bbox
  resampled.nii.gz
  mask_resampled.nii.gz
  pred_slices_stacked.nii.gz
  gt_slices_stacked.nii.gz

Usage:
    python scripts/run_pipeline.py \
        --image  data/raw/<site>/sub-XX/anat/sub-XX_T1w.nii.gz \
        --mask   data/raw/<site>/derivatives/labels/sub-XX/anat/sub-XX_T1w_label-SC_seg.nii.gz \
        --checkpoint checkpoints/yolo_spine_v1/weights/best.pt
"""

import argparse
import shutil
import sys
from pathlib import Path

import nibabel as nib
import yaml
from PIL import Image as PILImage, ImageDraw
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    bbox_3d_from_txts, nifti_stem, normalize_to_uint8,
    reorient_to_rpi, resample_nifti, seg_to_yolo_bbox,
    stack_bbox_volume, write_bbox_3d,
)


def preprocess_volume(img_path: Path, mask_path: Path, res_mm: tuple, out_dir: Path) -> tuple:
    """Preprocess one volume into out_dir/png/ + txt/ + volume/. Returns (H, W, Z, affine, header)."""
    (out_dir / "png").mkdir(parents=True, exist_ok=True)
    (out_dir / "txt").mkdir(parents=True, exist_ok=True)
    (out_dir / "volume").mkdir(parents=True, exist_ok=True)

    img_r  = resample_nifti(reorient_to_rpi(nib.load(str(img_path))), res_mm, order=3)
    mask_r = resample_nifti(reorient_to_rpi(nib.load(str(mask_path))), res_mm, order=1)
    nib.save(img_r,  str(out_dir / "resampled.nii.gz"))
    nib.save(mask_r, str(out_dir / "mask_resampled.nii.gz"))

    import numpy as np
    img_data  = img_r.get_fdata(dtype=np.float32)
    mask_data = mask_r.get_fdata().astype(np.uint8)
    H, W, Z   = img_data.shape[:3]

    for z in range(Z):
        PILImage.fromarray(normalize_to_uint8(img_data[:, :, z])).save(
            str(out_dir / "png" / f"slice_{z:03d}.png")
        )
        bbox = seg_to_yolo_bbox(mask_data[:, :, z])
        txt_path = out_dir / "txt" / f"slice_{z:03d}.txt"
        if bbox is not None:
            cx, cy, w, h = bbox
            txt_path.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        else:
            txt_path.write_text("")

    box = bbox_3d_from_txts(out_dir / "txt", H, W)
    if box is not None:
        write_bbox_3d(out_dir / "volume" / "bbox_3d.txt", **box)

    return H, W, Z, img_r.affine, img_r.header


def infer_volume(model: YOLO, out_dir: Path, H: int, W: int, conf: float) -> None:
    """Run YOLO on all slices in out_dir/png/ and save to slices/txt/ + slices/png/."""
    (out_dir / "slices" / "txt").mkdir(parents=True, exist_ok=True)
    (out_dir / "slices" / "png").mkdir(parents=True, exist_ok=True)
    (out_dir / "volume_pred").mkdir(parents=True, exist_ok=True)

    for png in tqdm(sorted((out_dir / "png").glob("slice_*.png")), desc="Slices"):
        img = PILImage.open(png).convert("RGB")
        results = model.predict(img, conf=conf, verbose=False)
        boxes = results[0].boxes
        txt_line = ""
        if boxes is not None and len(boxes) > 0:
            best = int(boxes.conf.argmax())
            cx, cy, w, h = boxes.xywhn[best].tolist()
            txt_line = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

        (out_dir / "slices" / "txt" / png.stem).with_suffix(".txt").write_text(
            txt_line + "\n" if txt_line else ""
        )

        vis = PILImage.open(png).convert("RGB")
        if txt_line:
            iW, iH = vis.size
            _, cx, cy, w, h = map(float, txt_line.split())
            x1, x2 = int((cx - w / 2) * iW), int((cx + w / 2) * iW)
            y1, y2 = int((cy - h / 2) * iH), int((cy + h / 2) * iH)
            ImageDraw.Draw(vis).rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        vis.save(str(out_dir / "slices" / "png" / png.name))

    box = bbox_3d_from_txts(out_dir / "slices" / "txt", H, W)
    if box is not None:
        write_bbox_3d(out_dir / "volume_pred" / "bbox_3d.txt", **box)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end YOLO spine crop pipeline on a single volume",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image",      required=True)
    parser.add_argument("--mask",       required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--res", nargs=3, type=float, metavar=("R", "P", "I"),
                        default=[0.9, 0.7, 1.0], help="Resampling resolution in mm")
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--out",        default="sandbox")
    args = parser.parse_args()

    img_path  = Path(args.image)
    mask_path = Path(args.mask)
    res_mm    = tuple(args.res)
    stem      = nifti_stem(img_path)
    out_dir   = Path(args.out) / stem

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"Preprocessing {stem} …")
    H, W, Z, affine, header = preprocess_volume(img_path, mask_path, res_mm, out_dir)

    print(f"Running YOLO inference …")
    model = YOLO(args.checkpoint)
    infer_volume(model, out_dir, H, W, args.conf)

    import nibabel as nib
    import numpy as np
    pred_vol = stack_bbox_volume(out_dir / "slices" / "txt", H, W, Z)
    gt_vol   = stack_bbox_volume(out_dir / "txt", H, W, Z)
    nib.save(nib.Nifti1Image(pred_vol, affine, header), str(out_dir / "pred_slices_stacked.nii.gz"))
    nib.save(nib.Nifti1Image(gt_vol,   affine, header), str(out_dir / "gt_slices_stacked.nii.gz"))

    print(f"\nSandbox output: {out_dir}")


if __name__ == "__main__":
    main()
