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
  5. Compute YOLO GT bbox per slice → txt/slice_NNN.txt  ("0 cx cy w h", normalised [0,1])
  6. Compute 3D GT bbox → volume/bbox_3d.txt  ("row1 row2 col1 col2 z1 z2", voxels)
  7. Write meta.yaml (raw_image, raw_mask, shape_las, si_res_mm, rl_res_mm, ap_res_mm,
                      axial_res_mm if --axial-res, channels=3 if --3ch)

Mask discovery: per-dataset explicit suffix table DATASET_MASK_SUFFIX — crashes on unknown dataset.
Output dir named automatically: processed/<si_res>mm_SI[_<axial_res>mm_axial][_3ch]

Usage:
    # 10mm SI, native axial resolution, grayscale → processed/10mm_SI/
    python scripts/preprocess.py --si-res 10.0

    # 10mm SI + 1mm isotropic axial resampling, grayscale → processed/10mm_SI_1mm_axial/
    python scripts/preprocess.py --si-res 10.0 --axial-res 1.0

    # 10mm SI + 1mm isotropic axial resampling, pseudo-RGB 2.5D → processed/10mm_SI_1mm_axial_3ch/
    python scripts/preprocess.py --si-res 10.0 --axial-res 1.0 --3ch

    # Patch existing meta.yaml with rl_res_mm/ap_res_mm without re-preprocessing
    python scripts/preprocess.py --update-meta --out processed/10mm_SI
"""

import argparse
import sys
from pathlib import Path

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
    img_path_str, mask_path_str, dataset_name, processed_str, si_res_mm, axial_res_mm, three_ch = args
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
    mask_data = mask_r.get_fdata().astype(np.uint8)
    H, W = img_data.shape[:2]
    Z = min(img_data.shape[2], mask_data.shape[2])  # zoom rounding can differ by 1 voxel
    # YOLO rejects images with any dimension < 10px — pad to at least 10 if needed
    H_out, W_out = max(H, 10), max(W, 10)

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
    parser.add_argument("--3ch",          action="store_true", dest="three_ch",
                        help="Export pseudo-RGB PNG (R=prev slice, G=current, B=next). Appends '_3ch' to output dir name.")
    parser.add_argument("--update-meta", action="store_true",
                        help="Only patch existing meta.yaml with rl_res_mm/ap_res_mm (no re-preprocessing). Requires --out.")
    args = parser.parse_args()

    if args.update_meta:
        assert args.out, "--out is required with --update-meta"
        update_meta_resolutions(Path(args.out))
        return

    assert args.si_res is not None, "--si-res is required"
    si_res    = args.si_res
    axial_res = args.axial_res
    three_ch  = args.three_ch
    if args.out:
        processed_dir = str(Path(args.out))
    else:
        name = f"{si_res:g}mm_SI"
        if axial_res is not None:
            name += f"_{axial_res:g}mm_axial"
        if three_ch:
            name += "_3ch"
        processed_dir = str(Path("processed") / name)

    worker_args = []
    for dataset_dir in sorted(Path(args.raw).iterdir()):
        if not dataset_dir.is_dir():
            continue
        pairs = find_pairs(dataset_dir)
        print(f"{dataset_dir.name}: {len(pairs)} pairs")
        worker_args.extend(
            (str(img), str(mask), dataset_dir.name, processed_dir, si_res, axial_res, three_ch) for img, mask in pairs
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
