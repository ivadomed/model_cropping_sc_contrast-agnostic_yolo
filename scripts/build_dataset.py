#!/usr/bin/env python3
"""
Build the YOLO dataset from processed slices and per-dataset split YAMLs.

Iterates all datasplit_<dataset>_seed<N>.yaml in --splits-dir.
For each file, resolves the dataset name and maps each listed subject to all
its acquisition folders in processed/<dataset>/<subject>_*/.
Datasets with role=test in data/datasets.yaml are placed entirely in test,
bypassing the datasplit yaml (out-of-domain hold-out).
Creates flat per-slice symlinks (not copies) into datasets/:
  datasets/images/{train,val,test}/<dataset>_<subject>_<acq>_slice_NNN[_rN].png
  datasets/labels/{train,val,test}/<dataset>_<subject>_<acq>_slice_NNN[_rN].txt
  datasets/dataset.yaml
  datasets/build_stats.yaml  ← slice counts per dataset (raw + after oversampling)

Oversampling / balancing (train split only):
  --balance-regions : oversample lumbar slices so lumbar ≈ cervical slice count (default: off)
  --dataset-factors : per-dataset multipliers, e.g. sci-zurich:5 lumbar-epfl:3
  --sc-ratio N      : per-volume SC balance — keep all SC slices, subsample empty slices to N×n_sc
                      e.g. --sc-ratio 3 → at most 3 empty slices per SC slice per volume (train only)
  Oversampling creates additional symlinks _r2, _r3, ... pointing to the same files.
  All factors are multiplicative (region_factor × dataset_factor).
  Val and test splits are never oversampled or subsampled.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --processed processed/10mm_SI_1mm_axial_3ch_sc_and_canal \\
        --out datasets/10mm_SI_1mm_axial_3ch_sc_and_canal \\
        --dataset-factors sci-zurich:5 lumbar-epfl:3
    # spinal cord only (drop spinal canal labels):
    python scripts/build_dataset.py --keep-classes 0 \\
        --out datasets/10mm_SI_1mm_axial_3ch_sc_only_all_datasets
"""

import argparse
import random
import re
from pathlib import Path
from typing import Optional

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
    """Return one dict per slice: {png, txt, prefix, region, dataset, has_sc, slice_idx}."""
    slices = []
    for png in sorted((stem_dir / "png").glob("slice_*.png")):
        txt    = stem_dir / "txt" / (png.stem + ".txt")
        has_sc = txt.exists() and bool(txt.read_text().strip())
        slices.append({"png": png, "txt": txt, "prefix": prefix,
                        "region": region, "dataset": dataset, "has_sc": has_sc,
                        "slice_idx": int(png.stem.split("_")[1])})
    return slices


def mark_border_slices(slices: list[dict], border_mm: float, stem_dir: Path) -> None:
    """Mark slices near the SC boundary or without SC as border (is_border=True).

    Border = no SC anywhere in volume, OR within border_mm of the first/last SC slice
    on the SI axis. Mutates slices in place.
    """
    si_res = yaml.safe_load((stem_dir / "meta.yaml").read_text())["si_res_mm"]
    sc_idxs = [s["slice_idx"] for s in slices if s["has_sc"]]
    if not sc_idxs:
        for s in slices:
            s["is_border"] = True
        return
    z1, z2 = min(sc_idxs), max(sc_idxs)
    for s in slices:
        dist_mm = min(abs(s["slice_idx"] - z1), abs(s["slice_idx"] - z2)) * si_res
        s["is_border"] = not s["has_sc"] or dist_mm <= border_mm


def subsample_empty_slices(slices: list[dict], sc_ratio: int, seed: int) -> list[dict]:
    """Keep all SC slices; subsample empty slices to at most n_sc * sc_ratio per volume.

    Applied per volume (all slices passed here belong to a single stem_dir).
    Subsampling is deterministic via seed.
    """
    sc_slices    = [s for s in slices if s["has_sc"]]
    empty_slices = [s for s in slices if not s["has_sc"]]
    max_empty    = len(sc_slices) * sc_ratio
    if len(empty_slices) > max_empty:
        rng          = random.Random(seed)
        empty_slices = rng.sample(empty_slices, max_empty)
    return sc_slices + empty_slices


def make_symlinks(slices: list[dict], images_out: Path, labels_out: Path,
                  n_copies_fn, keep_classes: Optional[set] = None) -> int:
    """Create symlinks for each slice, n_copies_fn(slice) times. Returns total links."""
    n = 0
    for s in slices:
        n_copies = n_copies_fn(s)
        for i in range(n_copies):
            suffix   = f"_r{i + 1}" if i > 0 else ""
            img_link = images_out / f"{s['prefix']}_{s['png'].stem}{suffix}.png"
            lbl_out  = labels_out  / f"{s['prefix']}_{s['txt'].stem}{suffix}.txt"

            if img_link.is_symlink():
                img_link.unlink()
            img_link.symlink_to(s["png"].resolve())

            if keep_classes is None:
                if lbl_out.is_symlink():
                    lbl_out.unlink()
                lbl_out.symlink_to(s["txt"].resolve())
            else:
                raw = s["txt"].read_text() if s["txt"].exists() else ""
                filtered = "\n".join(
                    line for line in raw.splitlines()
                    if line and int(line.split()[0]) in keep_classes
                )
                lbl_out.write_text(filtered + "\n" if filtered else "")
            n += 1
    return n


def compute_region_factor(train_slices: list[dict]) -> float:
    """Returns lumbar oversample multiplier so lumbar ≈ cervical slice count."""
    n_cervical = sum(1 for s in train_slices if s["region"] == "cervical")
    n_lumbar   = sum(1 for s in train_slices if s["region"] == "lumbar")
    if n_lumbar == 0:
        print("  Region balance: no lumbar slices found, skipping.")
        return 1.0
    factor = n_cervical / n_lumbar
    print(f"  Region balance: cervical={n_cervical}  lumbar={n_lumbar}  "
          f"→ lumbar oversample ×{factor:.2f}")
    return factor


def main():
    parser = argparse.ArgumentParser(
        description="Build YOLO dataset (flat symlinks) from processed/ + datasplit YAMLs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--splits-dir",      default="data/datasplits_seed50")
    parser.add_argument("--processed",       default="processed/10mm_SI_1mm_axial_3ch_sc_and_canal")
    parser.add_argument("--out",             default="datasets/10mm_SI_1mm_axial_3ch_sc_and_canal")
    parser.add_argument("--nc",              type=int, default=1)
    parser.add_argument("--class-names",     nargs="+", default=["spine"])
    parser.add_argument("--regions-csv",     default="data/dataset_regions.csv")
    parser.add_argument("--balance-regions", action="store_true", default=False,
                        help="Oversample lumbar slices in train to match cervical count (default: off)")
    parser.add_argument("--no-balance-regions", dest="balance_regions", action="store_false")
    parser.add_argument("--keep-classes",    nargs="+", type=int, default=None,
                        metavar="ID",
                        help="Keep only these class IDs in labels (e.g. --keep-classes 0 to drop spinal canal)")
    parser.add_argument("--dataset-factors", nargs="*", default=[],
                        metavar="DATASET:N",
                        help="Per-dataset oversampling multipliers, e.g. sci-zurich:5 lumbar-epfl:3 (default: 1 for all)")
    parser.add_argument("--sc-ratio",        type=int, default=None,
                        metavar="N",
                        help="Per-volume SC balance: keep all SC slices, subsample empty slices to N×n_sc (train only).")
    parser.add_argument("--border-oversample", type=float, default=None,
                        metavar="MM",
                        help="Oversample ×2 slices with no SC or within MM of the superior/inferior SC boundary (train only).")
    args = parser.parse_args()

    assert not args.balance_regions or Path(args.regions_csv).exists(), \
        f"--regions-csv not found: {args.regions_csv}"

    dataset_factors = {k: float(v) for k, v in (f.split(":") for f in args.dataset_factors)}
    if dataset_factors:
        print(f"Dataset factors: {dataset_factors}")

    regions = {}
    if Path(args.regions_csv).exists():
        df = pd.read_csv(args.regions_csv)
        regions = dict(zip(df["dataset"], df["region"]))

    # Datasets with role=test go entirely to test (out-of-domain hold-out)
    _registry_path = Path(__file__).parent.parent / "data" / "datasets.yaml"
    _registry = yaml.safe_load(_registry_path.read_text())["datasets"]
    test_only = {d["name"] for d in _registry if d.get("role") == "test"}
    if test_only:
        print(f"Test-only datasets (role=test): {sorted(test_only)}")

    processed_dir = Path(args.processed)
    out_dir       = Path(args.out)

    for partition in ("train", "val", "test"):
        (out_dir / "images" / partition).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / partition).mkdir(parents=True, exist_ok=True)

    split_files = sorted(Path(args.splits_dir).glob("datasplit_*.yaml"))

    all_slices: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for split_yaml in split_files:
        dataset     = dataset_name_from_yaml(split_yaml)
        if dataset in test_only:
            print(f"  skip {dataset} (role=test, handled separately)")
            continue
        dataset_dir = processed_dir / dataset
        if not dataset_dir.is_dir():
            print(f"  skip {dataset} (no processed dir)")
            continue

        region = regions.get(dataset, "unknown")
        splits = yaml.safe_load(split_yaml.read_text())
        splits = {k: v for k, v in splits.items() if isinstance(v, list)}
        n_subjects = sum(len(v) for v in splits.values())
        print(f"{dataset} [{region}]: {n_subjects} subjects")

        for partition, subjects in splits.items():
            for subject in (subjects or []):
                for stem_dir in find_stem_dirs(dataset_dir, subject):
                    prefix = f"{dataset}_{stem_dir.name}"
                    slices = collect_slices(stem_dir, prefix, region, dataset)
                    if args.sc_ratio is not None and partition == "train":
                        seed   = hash(stem_dir.name) & 0xFFFFFFFF
                        slices = subsample_empty_slices(slices, args.sc_ratio, seed)
                    if args.border_oversample is not None and partition == "train":
                        mark_border_slices(slices, args.border_oversample, stem_dir)
                    all_slices[partition].extend(slices)

    # Test-only datasets: all subjects → test partition, no datasplit yaml needed
    for dataset in sorted(test_only):
        dataset_dir = processed_dir / dataset
        if not dataset_dir.is_dir():
            print(f"  skip {dataset} (role=test, no processed dir)")
            continue
        region = regions.get(dataset, "unknown")
        n_subjects = 0
        for stem_dir in sorted(dataset_dir.iterdir()):
            if not stem_dir.is_dir():
                continue
            prefix = f"{dataset}_{stem_dir.name}"
            all_slices["test"].extend(collect_slices(stem_dir, prefix, region, dataset))
            n_subjects += 1
        print(f"{dataset} [role=test, {region}]: {n_subjects} subjects → test")

    region_factor = 1.0
    if args.balance_regions:
        region_factor = compute_region_factor(all_slices["train"])

    # Per-dataset raw slice counts
    raw_counts: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for partition, slices in all_slices.items():
        for s in slices:
            raw_counts[partition][s["dataset"]] = raw_counts[partition].get(s["dataset"], 0) + 1

    border_factor = 2 if args.border_oversample is not None else 1

    # Link all partitions
    counts = {}
    for partition, slices in all_slices.items():
        images_out = out_dir / "images" / partition
        labels_out = out_dir / "labels" / partition

        if partition == "train":
            def n_copies_fn(s):
                base = region_factor if s["region"] == "lumbar" else 1.0
                base *= dataset_factors.get(s["dataset"], 1.0)
                if s.get("is_border", False):
                    base *= border_factor
                return max(1, round(base))
        else:
            def n_copies_fn(s):
                return 1

        keep = set(args.keep_classes) if args.keep_classes is not None else None
        counts[partition] = make_symlinks(slices, images_out, labels_out, n_copies_fn, keep)

    names_str = "\n".join(f"  {i}: {n}" for i, n in enumerate(args.class_names))
    (out_dir / "dataset.yaml").write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n"
        f"\nnc: {args.nc}\n"
        f"names:\n{names_str}\n"
    )

    # Final slice counts and SC/BG balance — single pass over train slices
    slices_final: dict[str, int] = {}
    sc_bg_raw    = {"sc": 0, "bg": 0}
    sc_bg_final  = {"sc": 0, "bg": 0}
    for s in all_slices["train"]:
        ds = s["dataset"]
        f  = region_factor if s["region"] == "lumbar" else 1.0
        f *= dataset_factors.get(ds, 1.0)
        if s.get("is_border", False):
            f *= border_factor
        copies = max(1, round(f))
        slices_final[ds] = slices_final.get(ds, 0) + copies
        key = "sc" if s["has_sc"] else "bg"
        sc_bg_raw[key]   += 1
        sc_bg_final[key] += copies

    all_datasets = sorted(slices_final)
    build_stats = {
        "border_oversample_mm": args.border_oversample,
        "dataset_factors":      {ds: dataset_factors.get(ds, 1.0) for ds in all_datasets},
        "sc_bg_slices_raw":     sc_bg_raw,
        "sc_bg_slices_final":   sc_bg_final,
        "slices_raw":           raw_counts,
        "slices_train_final":   dict(sorted(slices_final.items())),
        "total":                counts,
    }
    with open(out_dir / "build_stats.yaml", "w") as f:
        yaml.dump(build_stats, f, sort_keys=False, default_flow_style=False)

    for partition, n in counts.items():
        print(f"  {partition:5s}: {n} slices (with oversampling)")
    print(f"Dataset YAML:      {out_dir / 'dataset.yaml'}")
    print(f"Build stats YAML:  {out_dir / 'build_stats.yaml'}")


if __name__ == "__main__":
    main()
