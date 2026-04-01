#!/usr/bin/env python3
"""
Build the YOLO dataset from processed slices and per-dataset split YAMLs.

Iterates all datasplit_<dataset>_seed50.yaml in --splits-dir.
For each file, resolves the dataset name and maps each listed subject to all
its acquisition folders in processed/<dataset>/<subject>_*/.
Creates flat per-slice symlinks (not copies) into datasets/:
  datasets/images/{train,val,test}/<dataset>_<subject>_<acq>_slice_NNN.png
  datasets/labels/{train,val,test}/<dataset>_<subject>_<acq>_slice_NNN.txt
  datasets/dataset.yaml

Split YAML format (paths from data/datasplits/):
  train: [sub-xxx, sub-yyy, ...]
  val:   [sub-zzz, ...]
  test:  [sub-www, ...]

Splits whose dataset name has no matching processed/ folder are skipped.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --processed processed_1mm_SI --out datasets_1mm_SI
"""

import argparse
import re
from pathlib import Path

import yaml
from tqdm import tqdm


def dataset_name_from_yaml(path: Path) -> str:
    """Extract dataset name from datasplit_<dataset>_seed<N>.yaml filename."""
    return re.sub(r"_seed\d+$", "", path.stem[len("datasplit_"):])


def find_patient_dirs(dataset_dir: Path, subject: str):
    """All processed dirs for a subject across all its acquisitions."""
    return [d for d in sorted(dataset_dir.iterdir())
            if d.is_dir() and (d.name == subject or d.name.startswith(subject + "_"))]


def link_patient(patient_dir: Path, prefix: str, images_out: Path, labels_out: Path) -> int:
    """Symlink all slices for one patient. Returns number of slices linked."""
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
        description="Build YOLO dataset (flat symlinks) from processed/ + datasplit YAMLs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--splits-dir", default="data/datasplits", help="Directory with datasplit_*.yaml files")
    parser.add_argument("--processed",  default="processed",       help="Processed data root")
    parser.add_argument("--out",        default="datasets",        help="Output dataset directory")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    out_dir = Path(args.out)

    for partition in ("train", "val", "test"):
        (out_dir / "images" / partition).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / partition).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}

    split_files = sorted(Path(args.splits_dir).glob("datasplit_*.yaml"))
    for split_yaml in split_files:
        dataset = dataset_name_from_yaml(split_yaml)
        dataset_dir = processed_dir / dataset
        if not dataset_dir.is_dir():
            print(f"  skip {dataset} (no processed dir)")
            continue

        with open(split_yaml) as f:
            splits = yaml.safe_load(f)

        n_subjects = sum(len(v) for v in splits.values())
        print(f"{dataset}: {n_subjects} subjects")

        for partition, subjects in splits.items():
            images_out = out_dir / "images" / partition
            labels_out = out_dir / "labels" / partition
            for subject in tqdm(subjects, desc=f"  {partition}", leave=False):
                for patient_dir in find_patient_dirs(dataset_dir, subject):
                    prefix = f"{dataset}_{patient_dir.name}"
                    counts[partition] += link_patient(patient_dir, prefix, images_out, labels_out)

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
