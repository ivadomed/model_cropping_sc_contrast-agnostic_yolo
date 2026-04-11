#!/usr/bin/env python3
"""
Three grouped bar charts of dataset content (train / val / test),
counted from the symlink filenames in datasets/<variant>/images/{train,val,test}/.

Does not rely on any CSV or split YAML — counts directly from filenames.
Filename format: <dataset>_<stem>_slice_NNN.png  (stem = subject[_contrast])

Produces:
  patients_per_split.png  — unique subjects (sub-XXX, contrast ignored)
  volumes_per_split.png   — unique volumes  (sub-XXX[_contrast])
  slices_per_split.png    — total slices    (all PNG files)

Usage:
    python scripts/plot_dataset_content.py --dataset datasets/10mm_SI_1mm_axial
    python scripts/plot_dataset_content.py --dataset datasets/10mm_SI_1mm_axial --dpi 150
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
SPLITS = ["train", "val", "test"]


def collect(images_dir: Path) -> dict:
    """Returns {dataset: {"patients": set, "volumes": set, "slices": int}}."""
    data = defaultdict(lambda: {"patients": set(), "volumes": set(), "slices": 0})
    for f in images_dir.iterdir():
        if not f.name.endswith(".png"):
            continue
        # filename: <dataset>_<stem>_slice_NNN.png  (stem = sub-XXX[_contrast])
        m = re.match(r"^(.+?)_(sub-[^_]+(?:_[^_]+)*)_slice_\d+\.png$", f.name)
        if not m:
            continue
        dataset, stem = m.group(1), m.group(2)
        subject = re.match(r"(sub-[^_]+)", stem).group(1)
        data[dataset]["patients"].add(subject)
        data[dataset]["volumes"].add(stem)
        data[dataset]["slices"] += 1
    return data


def make_counts(images_dirs: dict) -> dict:
    """Returns {split: {dataset: {"patients": n, "volumes": n, "slices": n}}}."""
    counts = {}
    for split, images_dir in images_dirs.items():
        raw = collect(images_dir)
        counts[split] = {
            ds: {"patients": len(v["patients"]), "volumes": len(v["volumes"]), "slices": v["slices"]}
            for ds, v in raw.items()
        }
    return counts


def plot_bar(counts: dict, key: str, ylabel: str, title: str, out: Path, dpi: int) -> None:
    all_datasets = sorted({ds for split_data in counts.values() for ds in split_data})
    n     = len(all_datasets)
    x     = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for split, offset in zip(SPLITS, [-width, 0, width]):
        values = [counts[split].get(ds, {}).get(key, 0) for ds in all_datasets]
        bars = ax.bar(x + offset, values, width, label=split,
                      color=SPLIT_COLORS[split], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(v), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(all_datasets, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(handles=[mpatches.Patch(color=SPLIT_COLORS[s], label=s) for s in SPLITS],
              fontsize=9, loc="upper right")

    plt.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Bar charts of patients / volumes / slices per dataset and split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="datasets/10mm_SI_1mm_axial",
                        help="Path to YOLO dataset directory")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    images_dirs = {split: dataset_dir / "images" / split for split in SPLITS}
    counts = make_counts(images_dirs)

    name = dataset_dir.name
    plot_bar(counts, "patients", "Number of patients", f"Patients per split — {name}",
             dataset_dir / "patients_per_split.png", args.dpi)
    plot_bar(counts, "volumes",  "Number of volumes",  f"Volumes per split — {name}",
             dataset_dir / "volumes_per_split.png",  args.dpi)
    plot_bar(counts, "slices",   "Number of slices",   f"Slices per split — {name}",
             dataset_dir / "slices_per_split.png",   args.dpi)


if __name__ == "__main__":
    main()
