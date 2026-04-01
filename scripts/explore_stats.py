#!/usr/bin/env python3
"""
Compute dataset statistics and write dataset_stats.csv.

For each dataset in data/raw/, scans all NIfTI volumes under
sub-*/[ses-*/]anat/ (derivatives excluded).

Shape and resolution are reported in LAS space (Left × Anterior × Superior)
so dimensions are anatomically comparable across datasets regardless of their
native storage orientation.

The reorientation is virtual: no voxel data is loaded or moved.
ornt_transform() computes the axis permutation (which stored axis maps to L,
which to A, which to S), then shape and zooms are remapped by that permutation.

Output columns per dataset:
  n                        : number of volumes
  shape min/med/max LAS    : image size along L, A, S axes (px)
  res min/med/max LAS      : voxel size along L, A, S axes (mm)
  orientations (native)    : storage orientation codes with volume counts

One row per dataset + one ALL DATASETS summary row.

Usage:
    python scripts/explore_stats.py
    python scripts/explore_stats.py --raw data/raw --out dataset_stats.csv
"""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes, axcodes2ornt, io_orientation, ornt_transform


def find_images(dataset_root: Path) -> list:
    images = []
    for sub in sorted(dataset_root.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("sub-"):
            continue
        for anat in list(sub.glob("ses-*/anat")) + list(sub.glob("anat")):
            images.extend(sorted(anat.glob("*.nii.gz")))
    return images


def image_stats(path: Path) -> dict:
    img = nib.load(str(path))
    # Compute axis permutation from native → RPI (no data loading needed)
    transform = ornt_transform(io_orientation(img.affine), axcodes2ornt(("L", "A", "S")))
    axes = [int(transform[i, 0]) for i in range(3)]
    shape = img.shape[:3]
    zooms = img.header.get_zooms()[:3]
    L, A, S = (shape[a] for a in axes)
    rL, rA, rS = (float(zooms[a]) for a in axes)
    return {"L": L, "A": A, "S": S, "rL": rL, "rA": rA, "rS": rS,
            "Lmm": L * rL, "Amm": A * rA, "Smm": S * rS,
            "orientation": "".join(aff2axcodes(img.affine))}


def summarise(dataset: str, records: list) -> dict:
    def col(k):
        return [r[k] for r in records]

    def fmt(a, b, c, p=1):
        return f"{a:.{p}f}x{b:.{p}f}x{c:.{p}f}"

    def stats3(ka, kb, kc):
        a, b, c = col(ka), col(kb), col(kc)
        return (fmt(min(a), min(b), min(c)),
                fmt(float(np.median(a)), float(np.median(b)), float(np.median(c))),
                fmt(max(a), max(b), max(c)))

    ornt_counts = {}
    for r in records:
        ornt_counts[r["orientation"]] = ornt_counts.get(r["orientation"], 0) + 1
    ornts = ", ".join(f"{k} ({v})" for k, v in sorted(ornt_counts.items()))

    sh_min, sh_med, sh_max   = stats3("L",   "A",   "S")
    rs_min, rs_med, rs_max   = stats3("rL",  "rA",  "rS")
    fov_min, fov_med, fov_max = stats3("Lmm", "Amm", "Smm")

    return {
        "dataset":                   dataset,
        "n":                         len(records),
        "shape_min LAS (LxAxS px)":  sh_min,
        "shape_med LAS (LxAxS px)":  sh_med,
        "shape_max LAS (LxAxS px)":  sh_max,
        "res_min LAS (mm)":          rs_min,
        "res_med LAS (mm)":          rs_med,
        "res_max LAS (mm)":          rs_max,
        "fov_min LAS (LxAxS mm)":    fov_min,
        "fov_med LAS (LxAxS mm)":    fov_med,
        "fov_max LAS (LxAxS mm)":    fov_max,
        "orientations (native)":     ornts,
    }


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="dataset_stats.csv")
    args = parser.parse_args()

    all_records, rows = [], []
    for ds in sorted(Path(args.raw).iterdir()):
        if not ds.is_dir() or ds.name.startswith("."):
            continue
        images = find_images(ds)
        if not images:
            continue
        print(f"  {ds.name} ({len(images)} images)...")
        records = [image_stats(p) for p in images]
        rows.append(summarise(ds.name, records))
        all_records.extend(records)

    rows.append(summarise("ALL DATASETS", all_records))

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
