#!/usr/bin/env python3
"""
Preprocess BIDS datasets: data/raw/ → processed/<site>/<stem>/

For each (image, mask) pair in each BIDS dataset:
  1. Reorient to RPI
  2. Resample (image: cubic spline order=3 / mask: linear order=1, binarised at 0.5)
  3. Extract axial slices → png/slice_NNN.png  +  txt/slice_NNN.txt (YOLO bbox)
  4. Compute 3D GT bbox  → volume/bbox_3d.txt
  5. Write meta.yaml (raw paths, resolution, resampled shape)

Target resolution: median over all dataset pairs (axis-0 R, axis-1 P, axis-2 I).
Override with --res to fix resolution explicitly.

Mask discovery rules:
  - data-multi-subject: derivatives/labels_softseg_bin/*_desc-softseg_label-SC_seg.nii.gz
  - others:             derivatives/labels/*_label-SC_seg.nii.gz
                        (excluding: canal, disc, spine, rootlets, softseg)

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --raw data/raw --out processed --res 0.9 0.7 1.0 --workers 8
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
    reorient_to_rpi, resample_nifti, seg_to_yolo_bbox, write_bbox_3d,
)

_EXCLUDE = {"canal", "disc", "spine", "rootlets", "softseg"}


# ── BIDS pair discovery ────────────────────────────────────────────────────────

def find_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    """Return (image_path, mask_path) pairs for all valid volumes in a BIDS dataset."""
    name = dataset_root.name
    if name == "data-multi-subject":
        labels_root = dataset_root / "derivatives" / "labels_softseg_bin"
        mask_tag = "_desc-softseg_label-SC_seg"
    else:
        labels_root = dataset_root / "derivatives" / "labels"
        mask_tag = "_label-SC_seg"

    if not labels_root.exists():
        return []

    pairs = []
    for sub_dir in sorted(labels_root.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        subject = sub_dir.name
        anat_dirs = list(sub_dir.glob("ses-*/anat")) + list(sub_dir.glob("anat"))
        for anat_seg in anat_dirs:
            for mask in sorted(anat_seg.glob(f"*{mask_tag}.nii.gz")):
                if name != "data-multi-subject" and any(kw in mask.name.lower() for kw in _EXCLUDE):
                    continue
                img_fname = mask.name.replace(f"{mask_tag}.nii.gz", ".nii.gz")
                rel_anat = anat_seg.relative_to(labels_root / subject)
                img_path = dataset_root / subject / rel_anat / img_fname
                if img_path.exists():
                    pairs.append((img_path, mask))
    return pairs


# ── resolution helpers ─────────────────────────────────────────────────────────

def volume_resolution(img_path: Path) -> tuple:
    return tuple(float(z) for z in nib.load(str(img_path)).header.get_zooms()[:3])


def median_resolution(pairs: list[tuple[Path, Path]]) -> tuple:
    resolutions = [volume_resolution(img) for img, _ in pairs]
    arr = np.array(resolutions)
    return tuple(float(v) for v in np.median(arr, axis=0))


# ── per-pair processing ────────────────────────────────────────────────────────

def process_pair(args: tuple) -> tuple[str, str]:
    img_path_str, mask_path_str, dataset_name, processed_str, res_mm = args
    img_path = Path(img_path_str)
    mask_path = Path(mask_path_str)
    processed_dir = Path(processed_str)

    stem = nifti_stem(img_path)
    patient_dir = processed_dir / dataset_name / stem

    if (patient_dir / "meta.yaml").exists():
        return stem, "skipped"

    (patient_dir / "png").mkdir(parents=True, exist_ok=True)
    (patient_dir / "txt").mkdir(parents=True, exist_ok=True)
    (patient_dir / "volume").mkdir(parents=True, exist_ok=True)

    img_r = resample_nifti(reorient_to_rpi(nib.load(str(img_path))), res_mm, order=3)
    mask_r = resample_nifti(reorient_to_rpi(nib.load(str(mask_path))), res_mm, order=1)

    img_data = img_r.get_fdata(dtype=np.float32)
    mask_data = mask_r.get_fdata().astype(np.uint8)
    H, W, Z = img_data.shape[:3]

    for z in range(Z):
        PILImage.fromarray(normalize_to_uint8(img_data[:, :, z])).save(
            str(patient_dir / "png" / f"slice_{z:03d}.png")
        )
        bbox = seg_to_yolo_bbox(mask_data[:, :, z])
        txt_path = patient_dir / "txt" / f"slice_{z:03d}.txt"
        if bbox is not None:
            cx, cy, w, h = bbox
            txt_path.write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        else:
            txt_path.write_text("")

    box = bbox_3d_from_txts(patient_dir / "txt", H, W)
    if box is not None:
        write_bbox_3d(patient_dir / "volume" / "bbox_3d.txt", **box)

    (patient_dir / "meta.yaml").write_text(yaml.dump({
        "raw_image":     str(img_path),
        "raw_mask":      str(mask_path),
        "resolution_mm": list(res_mm),
        "shape":         [H, W, Z],
    }))

    return stem, "processed"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BIDS raw → processed/ (slices + YOLO labels + 3D GT bbox)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw",     default="data/raw",   help="BIDS root directory")
    parser.add_argument("--out",     default="processed",  help="Output directory")
    parser.add_argument("--res",     nargs=3, type=float,  metavar=("R", "P", "I"),
                        help="Target resolution in mm (default: median over dataset)")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    processed_dir = Path(args.out)
    workers = args.workers or cpu_count()

    # Collect all (image, mask, dataset_name) pairs across datasets
    all_pairs = []
    for dataset_dir in sorted(raw_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        pairs = find_pairs(dataset_dir)
        print(f"{dataset_dir.name}: {len(pairs)} pairs")
        all_pairs.extend((img, mask, dataset_dir.name) for img, mask in pairs)

    if not all_pairs:
        print("No pairs found. Check --raw path.")
        return

    # Compute or use provided resolution
    if args.res:
        res_mm = tuple(args.res)
        print(f"Resolution (provided): {res_mm} mm")
    else:
        print("Computing median resolution (first pass)...")
        res_mm = median_resolution([(img, mask) for img, mask, _ in all_pairs])
        print(f"Resolution (median): {[round(r, 3) for r in res_mm]} mm")

    # Build worker args
    worker_args = [
        (str(img), str(mask), dataset, str(processed_dir), res_mm)
        for img, mask, dataset in all_pairs
    ]

    counts = {"processed": 0, "skipped": 0}
    with Pool(processes=workers) as pool:
        for _, status in tqdm(
            pool.imap_unordered(process_pair, worker_args),
            total=len(worker_args), desc="Volumes", unit="vol",
        ):
            counts[status] += 1

    print(f"\nDone — processed: {counts['processed']}  skipped: {counts['skipped']}")


if __name__ == "__main__":
    main()
