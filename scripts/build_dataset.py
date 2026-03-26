#!/usr/bin/env python3
"""
Build the YOLO dataset from processed slices and a split YAML.

Creates per-slice symlinks (not copies) into datasets/:
  datasets/images/{train,val,test}/<site>_<stem>_slice_NNN.png → processed/<site>/<stem>/png/slice_NNN.png
  datasets/labels/{train,val,test}/<site>_<stem>_slice_NNN.txt → processed/<site>/<stem>/txt/slice_NNN.txt
  datasets/dataset.yaml

Split YAML format (paths relative to processed/):
  train: [canproco/sub-001_T1w, ...]
  val:   [canproco/sub-002_T1w, ...]
  test:  [canproco/sub-003_T1w, ...]

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --splits data/splits/global.yaml --processed processed --out datasets
"""

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm


def link_patient(patient_rel: str, processed_dir: Path, images_out: Path, labels_out: Path) -> int:
    """Create per-slice symlinks for one patient. Returns number of slices linked."""
    patient_dir = processed_dir / patient_rel
    prefix = patient_rel.replace("/", "_")
    n = 0
    for png in sorted((patient_dir / "png").glob("slice_*.png")):
        link = images_out / f"{prefix}_{png.name}"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(png.resolve())
        n += 1
    for txt in sorted((patient_dir / "txt").glob("slice_*.txt")):
        link = labels_out / f"{prefix}_{txt.name}"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(txt.resolve())
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Build YOLO dataset (symlinks) from processed/ + split YAML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--splits",    default="data/splits/global.yaml")
    parser.add_argument("--processed", default="processed")
    parser.add_argument("--out",       default="datasets")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    out_dir = Path(args.out)

    with open(args.splits) as f:
        splits = yaml.safe_load(f)

    for partition in ("train", "val", "test"):
        (out_dir / "images" / partition).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / partition).mkdir(parents=True, exist_ok=True)

    counts = {}
    for partition, patients in splits.items():
        n_slices = 0
        for patient_rel in tqdm(patients, desc=partition):
            n_slices += link_patient(
                patient_rel, processed_dir,
                out_dir / "images" / partition,
                out_dir / "labels" / partition,
            )
        counts[partition] = n_slices

    (out_dir / "dataset.yaml").write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n"
        f"\nnc: 1\n"
        f"names:\n"
        f"  0: spine\n"
    )

    for partition, n in counts.items():
        print(f"  {partition:5s}: {n} slices")
    print(f"Dataset YAML: {out_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
