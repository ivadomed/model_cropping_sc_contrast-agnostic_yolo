#!/usr/bin/env python3
"""
Build YOLO classification dataset from processed/ slices.

Classes: sc (GT non-empty) / no_sc (GT empty).
Creates symlinks in <out>/<split>/sc/ and <out>/<split>/no_sc/.
Slices in 'unknown' split (no datasplit yaml or subject not listed) are skipped.

--superior-only: restrict each patient to slices [0, z2_GT] where z2_GT is the
  inferior SC boundary from volume/bbox_3d.txt. Only the superior transition
  (brain/neck → SC) is represented; inferior SC→lumbar transition is excluded.

Output structure:
  <out>/
    train/sc/    ← symlinks: <dataset>_<patient>_slice_NNN.png
    train/no_sc/
    val/sc/
    val/no_sc/
    test/sc/
    test/no_sc/

Usage:
    python scripts/build_class_dataset.py \\
        --processed processed/10mm_SI_1mm_axial_3ch \\
        --splits-dir runs/20260601_120000/datasplits \\
        --out runs/20260601_120000/dataset_cls \\
        --superior-only
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


def _subject_split(splits_dir: Path, dataset: str, stem: str) -> str:
    split_file = splits_dir / f"datasplit_{dataset}_seed50.yaml"
    if not split_file.exists():
        return "unknown"
    data = yaml.safe_load(split_file.read_text()) or {}
    m = re.match(r"(sub-[^_]+)", stem)
    subject = m.group(1) if m else stem
    for split_name in ("train", "val", "test"):
        if subject in (data.get(split_name) or []):
            return split_name
    return "unknown"


def _superior_z2(patient_dir: Path) -> int | None:
    """Return z2 (inferior SC boundary, inclusive) from volume/bbox_3d.txt, or None."""
    bbox_path = patient_dir / "volume" / "bbox_3d.txt"
    if not bbox_path.exists():
        return None
    return int(bbox_path.read_text().split()[5])


def run(processed: str | Path, splits_dir: str | Path, out: str | Path,
        test_datasets: list[str] | None = None,
        superior_only: bool = False) -> None:
    processed_dir  = Path(processed)
    splits_dir     = Path(splits_dir)
    out_dir        = Path(out)
    test_datasets  = set(test_datasets or [])

    counts: dict[str, dict[str, int]] = {
        s: {"sc": 0, "no_sc": 0} for s in ("train", "val", "test")
    }

    for dataset_dir in sorted(processed_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for patient_dir in sorted(dataset_dir.iterdir()):
            if not (patient_dir / "png").is_dir():
                continue

            if dataset in test_datasets:
                split = "test"
            else:
                split = _subject_split(splits_dir, dataset, patient_dir.name)

            if split == "unknown":
                continue

            z2 = _superior_z2(patient_dir) if superior_only else None

            txt_dir = patient_dir / "txt"
            for png in sorted((patient_dir / "png").glob("slice_*.png")):
                z = int(png.stem.split("_")[1])
                if z2 is not None and z > z2:
                    continue
                txt = txt_dir / (png.stem + ".txt")
                cls = "sc" if (txt.exists() and txt.stat().st_size > 0) else "no_sc"
                link_dir = out_dir / split / cls
                link_dir.mkdir(parents=True, exist_ok=True)
                link = link_dir / f"{dataset}_{patient_dir.name}_{png.name}"
                if not link.exists():
                    link.symlink_to(png.resolve())
                counts[split][cls] += 1

    total = sum(v for s in counts.values() for v in s.values())
    print(f"Classification dataset → {out_dir}  ({total} slices)")
    for split, cls_counts in counts.items():
        n_sc    = cls_counts["sc"]
        n_no_sc = cls_counts["no_sc"]
        ratio   = n_sc / (n_sc + n_no_sc) if (n_sc + n_no_sc) > 0 else 0.0
        print(f"  {split}: sc={n_sc}  no_sc={n_no_sc}  sc_ratio={ratio:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Build YOLO classification dataset (sc / no_sc) from processed/ slices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--processed",     required=True, help="processed/<variant> dir")
    parser.add_argument("--splits-dir",    required=True, help="Directory with datasplit_*.yaml")
    parser.add_argument("--out",           required=True, help="Output dataset directory")
    parser.add_argument("--test-datasets",  nargs="*", default=None,
                        help="Dataset names where all subjects go to test partition")
    parser.add_argument("--superior-only", action="store_true", dest="superior_only",
                        help="Restrict each patient to slices [0, z2_GT] (superior SC boundary only)")
    args = parser.parse_args()
    run(processed=args.processed, splits_dir=args.splits_dir, out=args.out,
        test_datasets=args.test_datasets, superior_only=args.superior_only)


if __name__ == "__main__":
    main()
