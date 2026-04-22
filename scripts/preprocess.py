#!/usr/bin/env python3
"""
Preprocess BIDS datasets: data/raw/ → processed/<variant>/<dataset>/<stem>/

For each (image, mask) pair in each BIDS dataset:
  1. Reorient to LAS (nibabel axis permutation, no interpolation)
  2. Resample SI axis (axis 2) to --si-res mm  (image: order=1, mask: order=0)
  3. Optionally resample RL and AP axes (0 and 1) to --axial-res mm isotropically
     (nibabel.processing.resample_to_output, image: order=1, mask: order=0)
  4. Export axial slices → png/slice_NNN.png
       2D mode  : grayscale uint8 (H×W), min-max normalised per slice
       2.5D mode: pseudo-RGB uint8 (H×W×3), R=slice z-1, G=slice z, B=slice z+1
                  border slices (z=0, z=Z-1) use a black frame for missing neighbours
  5. Compute YOLO GT bbox per slice → txt/slice_NNN.txt
       without --with-canal: "0 cx cy w h" (SC only, class 0)
       with --with-canal:    up to 2 lines per file — "0 cx cy w h" (SC) and/or "1 cx cy w h" (canal)
  6. Compute 3D GT bbox → volume/bbox_3d.txt  ("row1 row2 col1 col2 z1 z2", voxels)
     with --with-canal: also volume/bbox_3d_canal.txt (same format, canal class)
  7. Write meta.yaml (raw_image, raw_mask, shape_las, si_res_mm, rl_res_mm, ap_res_mm,
                      axial_res_mm if --axial-res, channels=3 if --3ch,
                      raw_canal_mask if --with-canal and canal mask found)

Mask discovery: per-dataset explicit suffix tables DATASET_MASK_SUFFIX (SC) and
DATASET_CANAL_SUFFIX (canal, only for datasets that have it) — crashes on unknown dataset.
Output dir named automatically: processed/<si_res>mm_SI[_<axial_res>mm_axial][_3ch][_canal]

Usage:
    # 10mm SI, native axial resolution, grayscale → processed/10mm_SI/
    python scripts/preprocess.py --si-res 10.0

    # 10mm SI + 1mm isotropic axial resampling, grayscale → processed/10mm_SI_1mm_axial/
    python scripts/preprocess.py --si-res 10.0 --axial-res 1.0

    # 10mm SI + 1mm isotropic axial resampling, pseudo-RGB 2.5D → processed/10mm_SI_1mm_axial_3ch/
    python scripts/preprocess.py --si-res 10.0 --axial-res 1.0 --3ch

    # SC + canal dual-class, 3 datasets only → processed/10mm_SI_1mm_axial_3ch_canal/
    python scripts/preprocess.py --si-res 10.0 --axial-res 1.0 --3ch --with-canal \\
        --datasets data-multi-subject spider-challenge-2023 whole-spine

    # Patch existing meta.yaml with rl_res_mm/ap_res_mm without re-preprocessing
    python scripts/preprocess.py --update-meta --out processed/10mm_SI

    # Process a single dataset only
    python scripts/preprocess.py --si-res 10.0 --axial-res 1.0 --datasets spider-challenge-2023
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import yaml
from PIL import Image as PILImage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from nibabel.processing import resample_to_output

from utils import (
    bbox_3d_from_txts, nifti_stem, normalize_to_uint8,
    reorient_to_las, seg_to_yolo_bbox, write_bbox_3d,
)

# Explicit SC mask suffix per dataset — crashes on unknown dataset name
DATASET_MASK_SUFFIX = {  # noqa: E241
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
    "spider-challenge-2023":        "_label-SC_seg.nii.gz",
    "data-single-subject":          "_label-SC_seg.nii.gz",
    "whole-spine":                  "_label-SC_seg.nii.gz",
    "site_006":                     "_label-SC_seg.nii.gz",
    "site_007":                     "_label-SC_seg.nii.gz"
}

# Canal mask suffix — only for datasets that have canal segmentations (class 1)
DATASET_CANAL_SUFFIX = {
    "data-multi-subject":    "_label-canal_seg.nii.gz",
    "spider-challenge-2023": "_label-canal_seg.nii.gz",
    "whole-spine":           "_label-canal_seg.nii.gz",
}


def find_pairs(dataset_root: Path, with_canal: bool = False):
    """Return (image_path, sc_mask_path, canal_mask_path_or_None) triples for a BIDS dataset.
    Crashes if the dataset name is not in DATASET_MASK_SUFFIX.
    Skips pairs where the image file does not exist.
    canal_mask_path is None when with_canal=False or no canal mask exists for the pair.
    """
    mask_suffix = DATASET_MASK_SUFFIX[dataset_root.name]
    canal_suffix = DATASET_CANAL_SUFFIX.get(dataset_root.name) if with_canal else None
    labels_root = dataset_root / "derivatives" / "labels"

    pairs = []
    for sub_dir in sorted(labels_root.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        for anat_seg in list(sub_dir.glob("ses-*/anat")) + list(sub_dir.glob("anat")):
            for mask in sorted(anat_seg.glob(f"*{mask_suffix}")):
                img_path = dataset_root / sub_dir.name / anat_seg.relative_to(sub_dir) / (mask.name[: -len(mask_suffix)] + ".nii.gz")
                if not img_path.exists():
                    continue
                canal_mask = None
                if canal_suffix:
                    canal_path = anat_seg / (mask.name[: -len(mask_suffix)] + canal_suffix)
                    if canal_path.exists():
                        canal_mask = canal_path
                pairs.append((img_path, mask, canal_mask))
    return pairs


def _rescale_bbox(bbox, H, W, H_out, W_out):
    """Rescale normalised bbox coords when image was padded to H_out×W_out."""
    if bbox is None or (H_out == H and W_out == W):
        return bbox
    cx, cy, w, h = bbox
    return (cx * W / W_out, cy * H / H_out, w * W / W_out, h * H / H_out)


def _bbox3d_from_mask(mask_data: np.ndarray, H: int, W: int, Z: int,
                      H_out: int, W_out: int) -> Optional[dict]:
    """Compute 3D bbox directly from a 3D mask array. Returns {row1,row2,col1,col2,z1,z2} or None."""
    rows1, rows2, cols1, cols2, zs = [], [], [], [], []
    for z in range(Z):
        bbox = seg_to_yolo_bbox(mask_data[:, :, z])
        bbox = _rescale_bbox(bbox, H, W, H_out, W_out)
        if bbox is None:
            continue
        cx, cy, w, h = bbox
        rows1.append(max(0, int((cy - h / 2) * H_out)))
        rows2.append(min(H_out, int((cy + h / 2) * H_out)))
        cols1.append(max(0, int((cx - w / 2) * W_out)))
        cols2.append(min(W_out, int((cx + w / 2) * W_out)))
        zs.append(z)
    if not zs:
        return None
    return {"row1": min(rows1), "row2": max(rows2),
            "col1": min(cols1), "col2": max(cols2),
            "z1":   min(zs),    "z2":   max(zs)}


def process_pair(args: tuple):
    img_path_str, mask_path_str, canal_mask_path_str, dataset_name, processed_str, si_res_mm, axial_res_mm, three_ch = args
    img_path = Path(img_path_str)
    patient_dir = Path(processed_str) / dataset_name / nifti_stem(img_path)

    if (patient_dir / "meta.yaml").exists():
        return "skipped"

    for d in ("png", "txt", "volume"):
        (patient_dir / d).mkdir(parents=True, exist_ok=True)

    img_las  = reorient_to_las(nib.load(str(img_path)))
    mask_las = reorient_to_las(nib.load(str(mask_path_str)))
    rl_res_mm, ap_res_mm = float(img_las.header.get_zooms()[0]), float(img_las.header.get_zooms()[1])

    voxel_sizes = (axial_res_mm or rl_res_mm, axial_res_mm or ap_res_mm, si_res_mm)
    img_r  = resample_to_output(img_las,  voxel_sizes=voxel_sizes, order=1)
    mask_r = resample_to_output(mask_las, voxel_sizes=voxel_sizes, order=0)

    img_data  = img_r.get_fdata(dtype=np.float32)
    mask_data = np.round(mask_r.get_fdata()).astype(np.uint8)
    H, W = img_data.shape[:2]
    Z = min(img_data.shape[2], mask_data.shape[2])  # zoom rounding can differ by 1 voxel
    # YOLO rejects images with any dimension < 10px — pad to at least 10 if needed
    H_out, W_out = max(H, 10), max(W, 10)

    canal_data = None
    if canal_mask_path_str is not None:
        canal_r    = resample_to_output(reorient_to_las(nib.load(str(canal_mask_path_str))),
                                        voxel_sizes=voxel_sizes, order=0)
        canal_data = np.round(canal_r.get_fdata()).astype(np.uint8)

    blank = np.zeros((H_out, W_out), dtype=np.uint8)

    def get_slice(z: int) -> np.ndarray:
        if z < 0 or z >= Z:
            return blank
        arr = normalize_to_uint8(img_data[:, :, z])
        if H_out != H or W_out != W:
            padded = blank.copy()
            padded[:H, :W] = arr
            return padded
        return arr

    for z in range(Z):
        arr = get_slice(z)
        if three_ch:
            rgb = np.stack([get_slice(z - 1), arr, get_slice(z + 1)], axis=2)
            PILImage.fromarray(rgb).save(str(patient_dir / "png" / f"slice_{z:03d}.png"))
        else:
            PILImage.fromarray(arr).save(str(patient_dir / "png" / f"slice_{z:03d}.png"))

        sc_bbox     = _rescale_bbox(seg_to_yolo_bbox(mask_data[:, :, z]), H, W, H_out, W_out)
        canal_bbox  = None
        if canal_data is not None and z < canal_data.shape[2]:
            canal_bbox = _rescale_bbox(seg_to_yolo_bbox(canal_data[:, :, z]), H, W, H_out, W_out)

        lines = ""
        if sc_bbox is not None:
            lines += f"0 {sc_bbox[0]:.6f} {sc_bbox[1]:.6f} {sc_bbox[2]:.6f} {sc_bbox[3]:.6f}\n"
        if canal_bbox is not None:
            lines += f"1 {canal_bbox[0]:.6f} {canal_bbox[1]:.6f} {canal_bbox[2]:.6f} {canal_bbox[3]:.6f}\n"
        (patient_dir / "txt" / f"slice_{z:03d}.txt").write_text(lines)

    box = bbox_3d_from_txts(patient_dir / "txt", H_out, W_out)
    if box is not None:
        write_bbox_3d(patient_dir / "volume" / "bbox_3d.txt", **box)

    if canal_data is not None:
        canal_box = _bbox3d_from_mask(canal_data, H, W, min(Z, canal_data.shape[2]), H_out, W_out)
        if canal_box is not None:
            write_bbox_3d(patient_dir / "volume" / "bbox_3d_canal.txt", **canal_box)

    meta = {
        "raw_image": str(img_path),
        "raw_mask":  str(mask_path_str),
        "shape_las": [H, W, Z],
        "si_res_mm": si_res_mm,
        "rl_res_mm": round(axial_res_mm or rl_res_mm, 4),
        "ap_res_mm": round(axial_res_mm or ap_res_mm, 4),
    }
    if axial_res_mm is not None:
        meta["axial_res_mm"] = axial_res_mm
    if three_ch:
        meta["channels"] = 3
    if canal_mask_path_str is not None:
        meta["raw_canal_mask"] = str(canal_mask_path_str)
    (patient_dir / "meta.yaml").write_text(yaml.dump(meta))

    return "processed"


def update_meta_one(meta_path_str: str) -> str:
    """Add rl_res_mm/ap_res_mm to one meta.yaml. Returns 'updated' or 'skipped'."""
    meta_path = Path(meta_path_str)
    meta = yaml.safe_load(meta_path.read_text())
    if "rl_res_mm" in meta:
        return "skipped"
    img_las = reorient_to_las(nib.load(meta["raw_image"]))
    rl_res_mm, ap_res_mm = float(img_las.header.get_zooms()[0]), float(img_las.header.get_zooms()[1])
    meta["rl_res_mm"] = round(rl_res_mm, 4)
    meta["ap_res_mm"] = round(ap_res_mm, 4)
    meta_path.write_text(yaml.dump(meta))
    return "updated"


def update_meta_resolutions(processed_dir: Path) -> None:
    """Patch all existing meta.yaml in processed_dir to add rl_res_mm and ap_res_mm."""
    metas = sorted(processed_dir.rglob("meta.yaml"))
    print(f"Found {len(metas)} meta.yaml files in {processed_dir}")
    counts = {"updated": 0, "skipped": 0}
    for meta_path in tqdm(metas, desc="meta.yaml", unit="file"):
        counts[update_meta_one(str(meta_path))] += 1
    print(f"Done — updated: {counts['updated']}  skipped (already had fields): {counts['skipped']}")


def main():
    parser = argparse.ArgumentParser(
        description="BIDS raw → processed_{res}mm_SI/ (LAS slices resampled on SI + YOLO labels)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--si-res",      type=float, default=None, help="Target SI (Z) resolution in mm (required unless --update-meta)")
    parser.add_argument("--axial-res",   type=float, default=None, help="Target in-plane (RL, AP) isotropic resolution in mm (optional)")
    parser.add_argument("--raw",         default="data/raw",  help="BIDS root directory")
    parser.add_argument("--out",         default=None,        help="Output directory (default: processed/{si_res}mm_SI[_{axial_res}mm_axial])")
    parser.add_argument("--datasets",   nargs="+", default=None, help="Restrict to these dataset names (default: all)")
    parser.add_argument("--3ch",          action="store_true", dest="three_ch",
                        help="Export pseudo-RGB PNG (R=prev slice, G=current, B=next). Appends '_3ch' to output dir name.")
    parser.add_argument("--with-canal",  action="store_true", dest="with_canal",
                        help="Also extract canal bbox (class 1) alongside SC (class 0). "
                             "Requires dataset to be in DATASET_CANAL_SUFFIX. Appends '_canal' to output dir name.")
    parser.add_argument("--update-meta", action="store_true",
                        help="Only patch existing meta.yaml with rl_res_mm/ap_res_mm (no re-preprocessing). Requires --out.")
    args = parser.parse_args()

    if args.update_meta:
        assert args.out, "--out is required with --update-meta"
        update_meta_resolutions(Path(args.out))
        return

    assert args.si_res is not None, "--si-res is required"
    si_res     = args.si_res
    axial_res  = args.axial_res
    three_ch   = args.three_ch
    with_canal = args.with_canal
    if args.out:
        processed_dir = str(Path(args.out))
    else:
        name = f"{si_res:g}mm_SI"
        if axial_res is not None:
            name += f"_{axial_res:g}mm_axial"
        if three_ch:
            name += "_3ch"
        if with_canal:
            name += "_canal"
        processed_dir = str(Path("processed") / name)

    worker_args = []
    for dataset_dir in sorted(Path(args.raw).iterdir()):
        if not dataset_dir.is_dir():
            continue
        if args.datasets and dataset_dir.name not in args.datasets:
            continue
        pairs = find_pairs(dataset_dir, with_canal=with_canal)
        print(f"{dataset_dir.name}: {len(pairs)} pairs")
        worker_args.extend(
            (str(img), str(mask), str(canal) if canal else None,
             dataset_dir.name, processed_dir, si_res, axial_res, three_ch)
            for img, mask, canal in pairs
        )

    if not worker_args:
        print("No pairs found. Check --raw path.")
        return

    counts = {"processed": 0, "skipped": 0}
    for args_tuple in tqdm(worker_args, desc="Volumes", unit="vol"):
        counts[process_pair(args_tuple)] += 1

    print(f"\nDone — processed: {counts['processed']}  skipped: {counts['skipped']}")


if __name__ == "__main__":
    main()
