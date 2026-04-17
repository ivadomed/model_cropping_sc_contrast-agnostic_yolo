#!/usr/bin/env python3
"""
Build the YOLO dataset from processed slices and per-dataset split YAMLs.

Iterates all datasplit_<dataset>_seed50.yaml in --splits-dir.
For each file, resolves the dataset name and maps each listed subject to all
its acquisition folders in processed/<dataset>/<subject>_*/.
Creates flat per-slice symlinks (not copies) into datasets/:
  datasets/images/{train,val,test}/<dataset>_<subject>_<acq>_slice_NNN[_rN].png
  datasets/labels/{train,val,test}/<dataset>_<subject>_<acq>_slice_NNN[_rN].txt
  datasets/dataset.yaml

Oversampling (train split only, requires --regions-csv):
  --balance-regions : oversample lumbar slices so lumbar ≈ cervical slice count
  --balance-classes : oversample sc_tip slices to reach --tip-ratio (default 0.2)
                      requires convert_labels.py to have been run first
  --dataset-factors : per-dataset multipliers, e.g. sci-zurich:5 lumbar-epfl:3
  Oversampling creates additional symlinks _r2, _r3, ... pointing to the same files.
  All factors are multiplicative (region_factor × dataset_factor × tip_factor).
  Val and test splits are never oversampled.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --processed processed/10mm_SI_1mm_axial_3ch \\
        --out datasets/10mm_SI_1mm_axial_3ch_2cls \\
        --nc 2 --class-names sc_mid sc_tip \\
        --regions-csv data/dataset_regions.csv \\
        --balance-regions --balance-classes --tip-ratio 0.2 \\
        --dataset-factors sci-zurich:5
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


def dataset_name_from_yaml(path: Path) -> str:
    """Extract dataset name from datasplit_<dataset>_seed<N>.yaml filename."""
    return re.sub(r"_seed\d+$", "", path.stem[len("datasplit_"):])


def find_stem_dirs(dataset_dir: Path, subject: str) -> list[Path]:
    """All processed dirs for a subject across all its acquisitions."""
    return [d for d in sorted(dataset_dir.iterdir())
            if d.is_dir() and (d.name == subject or d.name.startswith(subject + "_"))]


def collect_slices(stem_dir: Path, prefix: str, region: str, dataset: str) -> list[dict]:
    """Return one dict per slice: {png, txt, prefix, region, dataset, class_id}."""
    slices = []
    for png in sorted((stem_dir / "png").glob("slice_*.png")):
        txt     = stem_dir / "txt" / (png.stem + ".txt")
        content = txt.read_text().strip() if txt.exists() else ""
        class_id = int(content.split()[0]) if content else -1
        slices.append({"png": png, "txt": txt, "prefix": prefix,
                        "region": region, "dataset": dataset, "class_id": class_id})
    return slices


def make_symlinks(slices: list[dict], images_out: Path, labels_out: Path,
                  n_copies_fn) -> int:
    """Create symlinks for each slice, n_copies_fn(slice) times. Returns total links."""
    n = 0
    for s in slices:
        n_copies = n_copies_fn(s)
        for i in range(n_copies):
            suffix   = f"_r{i + 1}" if i > 0 else ""
            img_link = images_out / f"{s['prefix']}_{s['png'].stem}{suffix}.png"
            lbl_link = labels_out / f"{s['prefix']}_{s['txt'].stem}{suffix}.txt"
            for link, target in ((img_link, s["png"]), (lbl_link, s["txt"])):
                if link.is_symlink():
                    link.unlink()
                link.symlink_to(target.resolve())
            n += 1
    return n


def compute_factors(train_slices: list[dict], balance_regions: bool,
                    balance_classes: bool, tip_ratio: float,
                    dataset_factors: dict) -> tuple[float, float]:
    """
    Returns (region_factor, tip_factor).
    region_factor : multiplier applied to every lumbar slice
    tip_factor    : additional multiplier applied to sc_tip slices (on top of region_factor)
    dataset_factors are applied multiplicatively on top of region_factor in n_copies_fn.
    """
    region_factor = 1.0
    tip_factor    = 1.0

    if balance_regions:
        n_cervical = sum(1 for s in train_slices if s["region"] == "cervical")
        n_lumbar   = sum(1 for s in train_slices if s["region"] == "lumbar")
        if n_lumbar > 0:
            region_factor = n_cervical / n_lumbar
            print(f"  Region balance: cervical={n_cervical}  lumbar={n_lumbar}  "
                  f"→ lumbar oversample ×{region_factor:.2f}")
        else:
            print("  Region balance: no lumbar slices found, skipping.")

    if balance_classes:
        # Effective counts after region + dataset oversampling
        def eff(s):
            return (region_factor if s["region"] == "lumbar" else 1.0) * dataset_factors.get(s["dataset"], 1.0)

        eff_mid = sum(eff(s) for s in train_slices if s["class_id"] == 0)
        eff_tip = sum(eff(s) for s in train_slices if s["class_id"] == 1)
        if eff_tip > 0:
            current_ratio = eff_tip / (eff_mid + eff_tip)
            needed_tip    = eff_mid * tip_ratio / (1 - tip_ratio)
            tip_factor    = needed_tip / eff_tip
            print(f"  Class balance: eff_mid={eff_mid:.0f}  eff_tip={eff_tip:.0f}  "
                  f"current_tip%={100*current_ratio:.1f}  target={100*tip_ratio:.0f}%  "
                  f"→ tip oversample ×{tip_factor:.2f}")
        else:
            print("  Class balance: no sc_tip slices found — run convert_labels.py first.")

    return region_factor, tip_factor


def main():
    parser = argparse.ArgumentParser(
        description="Build YOLO dataset (flat symlinks) from processed/ + datasplit YAMLs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--splits-dir",      default="data/datasplits/from_raw")
    parser.add_argument("--processed",       default="processed/10mm_SI")
    parser.add_argument("--out",             default="datasets/10mm_SI")
    parser.add_argument("--nc",              type=int, default=1)
    parser.add_argument("--class-names",     nargs="+", default=["spine"])
    parser.add_argument("--regions-csv",     default=None,
                        help="data/dataset_regions.csv (required for --balance-*)")
    parser.add_argument("--balance-regions", action="store_true",
                        help="Oversample lumbar slices in train to match cervical count")
    parser.add_argument("--balance-classes", action="store_true",
                        help="Oversample sc_tip slices in train to reach --tip-ratio")
    parser.add_argument("--tip-ratio",       type=float, default=0.2,
                        help="Target tip/(mid+tip) fraction for --balance-classes")
    parser.add_argument("--dataset-factors", nargs="*", default=[],
                        metavar="DATASET:N",
                        help="Per-dataset oversampling multipliers, e.g. sci-zurich:5 lumbar-epfl:3")
    args = parser.parse_args()

    assert not (args.balance_regions or args.balance_classes) or args.regions_csv, \
        "--regions-csv is required when using --balance-regions or --balance-classes"

    dataset_factors = {k: float(v) for k, v in (f.split(":") for f in args.dataset_factors)}
    if dataset_factors:
        print(f"Dataset factors: {dataset_factors}")

    regions = {}
    if args.regions_csv:
        df = pd.read_csv(args.regions_csv)
        regions = dict(zip(df["dataset"], df["region"]))

    processed_dir = Path(args.processed)
    out_dir       = Path(args.out)

    for partition in ("train", "val", "test"):
        (out_dir / "images" / partition).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / partition).mkdir(parents=True, exist_ok=True)

    split_files = sorted(Path(args.splits_dir).glob("datasplit_*.yaml"))

    # Collect all slices per partition
    all_slices: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for split_yaml in split_files:
        dataset     = dataset_name_from_yaml(split_yaml)
        dataset_dir = processed_dir / dataset
        if not dataset_dir.is_dir():
            print(f"  skip {dataset} (no processed dir)")
            continue

        region = regions.get(dataset, "unknown")
        splits = yaml.safe_load(split_yaml.read_text())
        n_subjects = sum(len(v or []) for v in splits.values())
        print(f"{dataset} [{region}]: {n_subjects} subjects")

        for partition, subjects in splits.items():
            for subject in (subjects or []):
                for stem_dir in find_stem_dirs(dataset_dir, subject):
                    prefix = f"{dataset}_{stem_dir.name}"
                    all_slices[partition].extend(
                        collect_slices(stem_dir, prefix, region, dataset)
                    )

    # Compute oversampling factors from train set
    region_factor, tip_factor = 1.0, 1.0
    if args.balance_regions or args.balance_classes:
        region_factor, tip_factor = compute_factors(
            all_slices["train"], args.balance_regions, args.balance_classes, args.tip_ratio,
            dataset_factors
        )

    # Link all partitions
    counts = {}
    for partition, slices in all_slices.items():
        images_out = out_dir / "images" / partition
        labels_out = out_dir / "labels" / partition

        if partition == "train":
            def n_copies_fn(s):
                base = region_factor if s["region"] == "lumbar" else 1.0
                base *= dataset_factors.get(s["dataset"], 1.0)
                if s["class_id"] == 1:
                    base *= tip_factor
                return max(1, round(base))
        else:
            def n_copies_fn(s):
                return 1

        counts[partition] = make_symlinks(slices, images_out, labels_out, n_copies_fn)

    names_str = "\n".join(f"  {i}: {n}" for i, n in enumerate(args.class_names))
    (out_dir / "dataset.yaml").write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n"
        f"\nnc: {args.nc}\n"
        f"names:\n{names_str}\n"
    )

    for partition, n in counts.items():
        print(f"  {partition:5s}: {n} slices (with oversampling)")
    print(f"Dataset YAML: {out_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
