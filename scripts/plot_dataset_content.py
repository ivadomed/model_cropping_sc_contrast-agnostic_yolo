#!/usr/bin/env python3
"""
Grouped bar chart of actual patient counts in a YOLO dataset (train / val / test),
counted from the symlink filenames in datasets/<variant>/images/{train,val,test}/.

Does not rely on any CSV or split YAML — counts directly from filenames.
Filename format: <dataset>_<stem>_slice_NNN.png  (stem = subject[_contrast])

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


def count_patients(images_dir: Path) -> dict:
    """Returns {dataset: n_unique_subjects} from filenames in images_dir.
    Counts unique subjects (sub-XXX), not acquisitions.
    Filename format: <dataset>_<stem>_slice_NNN.png  (stem = sub-XXX[_contrast])
    """
    counts = defaultdict(set)
    for f in images_dir.iterdir():
        if not f.name.endswith(".png"):
            continue
        # extract dataset and subject (sub-XXX only, ignore contrast suffix)
        m = re.match(r"^(.+?)_(sub-[^_]+)", f.name)
        if m:
            counts[m.group(1)].add(m.group(2))
    return {k: len(v) for k, v in counts.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Bar chart of actual patient counts per dataset and split from symlink filenames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="datasets/10mm_SI_1mm_axial",
                        help="Path to YOLO dataset directory")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    splits      = ["train", "val", "test"]

    # Count patients per dataset per split
    all_datasets = set()
    counts = {}
    for split in splits:
        c = count_patients(dataset_dir / "images" / split)
        counts[split] = c
        all_datasets.update(c)

    datasets = sorted(all_datasets)
    n        = len(datasets)
    x        = np.arange(n)
    width    = 0.25

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 6))
    fig.suptitle(f"Actual patient counts — {dataset_dir.name}", fontsize=13, fontweight="bold")

    for split, offset in zip(splits, [-width, 0, width]):
        values = [counts[split].get(d, 0) for d in datasets]
        bars = ax.bar(x + offset, values, width, label=split,
                      color=SPLIT_COLORS[split], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(v), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Number of patients", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(handles=[mpatches.Patch(color=SPLIT_COLORS[s], label=s) for s in splits],
              fontsize=9, loc="upper right")

    plt.tight_layout()
    out = dataset_dir / "patients_per_split.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
