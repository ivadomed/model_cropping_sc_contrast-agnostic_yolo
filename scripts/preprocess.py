#!/usr/bin/env python3
"""
Preprocess BIDS datasets: data/raw/ → processed_{res}mm_SI/<site>/<stem>/

For each (image, mask) pair in each BIDS dataset:
  1. Reorient to LAS
  2. Resample along SI axis (axis 2) to --si-res mm
  3. Export axial slices → png/slice_NNN.png  (native in-plane resolution, normalised uint8)
  4. Compute YOLO GT bbox per slice → txt/slice_NNN.txt
  5. Compute 3D GT bbox → volume/bbox_3d.txt
  6. Write meta.yaml (raw paths, LAS shape after resampling, SI resolution)

No in-plane resampling — YOLO handles HxW resize via imgsz.
Mask discovery: per-dataset explicit suffix table in DATASET_MASK_SUFFIX.
  Crashes on unknown dataset name.

Usage:
    python scripts/preprocess.py --si-res 1.0
    python scripts/preprocess.py --si-res 10.0 --raw data/raw --workers 8
"""

import argparse
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from PIL import Image as PILImage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    bbox_3d_from_txts, nifti_stem, normalize_to_uint8,
    reorient_to_las, resample_z, seg_to_yolo_bbox, write_bbox_3d,
)

# Explicit SC mask suffix per dataset — crashes on unknown dataset name
DATASET_MASK_SUFFIX = {
    "data-multi-subject":           "_label-SC_seg.nii.gz",
    "basel-mp2rage":                "_label-SC_seg.nii.gz",
    "dcm-zurich":                   "_label-SC_seg.nii.gz",
    "lumbar-vanderbilt":            "_label-SC_seg.nii.gz",
    "nih-ms-mp2rage":               "_label-SC_seg.nii.gz",
    "canproco":                     "_seg-manual.nii.gz",
    "sci-colorado":                 "_seg-manual.nii.gz",
    "sci-paris":                    "_seg-manual.nii.gz",
    "sci-zurich":                   "_seg-manual.nii.gz",
    "sct-testing-large":            "_seg-manual.nii.gz",
    "lumbar-epfl":                  "_seg-manual.nii.gz",
    "dcm-brno":                     "_seg.nii.gz",
    "dcm-zurich-lesions":           "_label-SC_mask-manual.nii.gz",
    "dcm-zurich-lesions-20231115":  "_label-SC_mask-manual.nii.gz",
}


def find_pairs(dataset_root: Path):
    """Return (image_path, mask_path) pairs for all valid SC volumes in a BIDS dataset.
    Crashes if the dataset name is not in DATASET_MASK_SUFFIX.
    Skips pairs where the corresponding image file does not exist.
    """
    mask_suffix = DATASET_MASK_SUFFIX[dataset_root.name]
    labels_root = dataset_root / "derivatives" / "labels"

    pairs = []
    for sub_dir in sorted(labels_root.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        for anat_seg in list(sub_dir.glob("ses-*/anat")) + list(sub_dir.glob("anat")):
            for mask in sorted(anat_seg.glob(f"*{mask_suffix}")):
                img_path = dataset_root / sub_dir.name / anat_seg.relative_to(sub_dir) / (mask.name[: -len(mask_suffix)] + ".nii.gz")
                if img_path.exists():
                    pairs.append((img_path, mask))
    return pairs


def process_pair(args: tuple):
    img_path_str, mask_path_str, dataset_name, processed_str, si_res_mm = args
    img_path = Path(img_path_str)
    patient_dir = Path(processed_str) / dataset_name / nifti_stem(img_path)

    if (patient_dir / "meta.yaml").exists():
        return "skipped"

    for d in ("png", "txt", "volume"):
        (patient_dir / d).mkdir(parents=True, exist_ok=True)

    img_r  = resample_z(reorient_to_las(nib.load(str(img_path))),    si_res_mm, order=3)
    mask_r = resample_z(reorient_to_las(nib.load(str(mask_path_str))), si_res_mm, order=0)

    img_data  = img_r.get_fdata(dtype=np.float32)
    mask_data = mask_r.get_fdata().astype(np.uint8)
    H, W = img_data.shape[:2]
    Z = min(img_data.shape[2], mask_data.shape[2])  # zoom rounding can differ by 1 voxel
    # YOLO rejects images with any dimension < 10px — pad to at least 10 if needed
    H_out, W_out = max(H, 10), max(W, 10)

    for z in range(Z):
        arr = normalize_to_uint8(img_data[:, :, z])
        if H_out != H or W_out != W:
            padded = np.zeros((H_out, W_out), dtype=np.uint8)
            padded[:H, :W] = arr
            arr = padded
        PILImage.fromarray(arr).save(str(patient_dir / "png" / f"slice_{z:03d}.png"))
        bbox = seg_to_yolo_bbox(mask_data[:, :, z])
        # rescale bbox coords to padded dimensions
        if bbox is not None and (H_out != H or W_out != W):
            cx, cy, w, h = bbox
            bbox = (cx * W / W_out, cy * H / H_out, w * W / W_out, h * H / H_out)
        (patient_dir / "txt" / f"slice_{z:03d}.txt").write_text(
            "" if bbox is None else f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
        )

    box = bbox_3d_from_txts(patient_dir / "txt", H_out, W_out)
    if box is not None:
        write_bbox_3d(patient_dir / "volume" / "bbox_3d.txt", **box)

    (patient_dir / "meta.yaml").write_text(yaml.dump({
        "raw_image": str(img_path),
        "raw_mask":  str(mask_path_str),
        "shape_las": [H, W, Z],
        "si_res_mm": si_res_mm,
    }))

    return "processed"


def main():
    parser = argparse.ArgumentParser(
        description="BIDS raw → processed_{res}mm_SI/ (LAS slices resampled on SI + YOLO labels)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--si-res",  type=float, required=True, help="Target SI (Z) resolution in mm")
    parser.add_argument("--raw",     default="data/raw",  help="BIDS root directory")
    parser.add_argument("--out",     default=None,        help="Output directory (default: processed_{si_res}mm_SI)")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers")
    args = parser.parse_args()

    si_res = args.si_res
    processed_dir = str(Path(args.out or f"processed_{si_res:g}mm_SI"))

    worker_args = []
    for dataset_dir in sorted(Path(args.raw).iterdir()):
        if not dataset_dir.is_dir():
            continue
        pairs = find_pairs(dataset_dir)
        print(f"{dataset_dir.name}: {len(pairs)} pairs")
        worker_args.extend(
            (str(img), str(mask), dataset_dir.name, processed_dir, si_res) for img, mask in pairs
        )

    if not worker_args:
        print("No pairs found. Check --raw path.")
        return

    counts = {"processed": 0, "skipped": 0}
    with Pool(processes=args.workers or cpu_count()) as pool:
        for status in tqdm(
            pool.imap_unordered(process_pair, worker_args),
            total=len(worker_args), desc="Volumes", unit="vol",
        ):
            counts[status] += 1

    print(f"\nDone — processed: {counts['processed']}  skipped: {counts['skipped']}")


if __name__ == "__main__":
    main()
