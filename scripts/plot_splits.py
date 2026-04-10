#!/usr/bin/env python3
"""
Two plots about dataset splits:
  1. Grouped bar chart: number of patients per dataset and split (train / val / test)
  2. Bar chart: percentage of patients in data/raw/ not assigned to any split

Reads all datasplit_<dataset>_seed<N>.yaml from --splits-dir.
Reads subject folders from --raw.

Usage:
    python scripts/plot_splits.py
    python scripts/plot_splits.py --splits-dir data/datasplits --raw data/raw --dpi 150
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml


SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}


def load_splits(splits_dir: Path) -> dict:
    """Returns {dataset: {split: [subjects]}}."""
    splits = {}
    for f in sorted(splits_dir.glob("datasplit_*.yaml")):
        dataset = re.sub(r"_seed\d+$", "", f.stem[len("datasplit_"):])
        splits[dataset] = yaml.safe_load(f.read_text())
    return splits


def load_raw_subjects(raw_dir: Path) -> dict:
    """Returns {dataset: set of subject names} from data/raw/ folder structure."""
    subjects = {}
    for dataset_dir in sorted(raw_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        subjects[dataset_dir.name] = {
            d.name for d in dataset_dir.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        }
    return subjects


def main():
    parser = argparse.ArgumentParser(
        description="Split distribution plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--splits-dir", default="data/datasplits")
    parser.add_argument("--raw",        default="data/raw")
    parser.add_argument("--dpi",        type=int, default=150)
    args = parser.parse_args()

    splits       = load_splits(Path(args.splits_dir))
    raw_subjects = load_raw_subjects(Path(args.raw))
    datasets     = sorted(raw_subjects)

    # ── Plot 1 : grouped bar chart train/val/test ─────────────────────────────
    split_datasets = sorted(splits)
    n      = len(split_datasets)
    x      = np.arange(n)
    width  = 0.25

    fig1, ax1 = plt.subplots(figsize=(max(10, n * 1.2), 6))
    fig1.suptitle("Number of patients per dataset and split", fontsize=13, fontweight="bold")

    for split_name, offset in zip(["train", "val", "test"], [-width, 0, width]):
        values = [len(splits[d].get(split_name) or []) for d in split_datasets]
        bars = ax1.bar(x + offset, values, width, label=split_name,
                       color=SPLIT_COLORS[split_name], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, values):
            if v > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         str(v), ha="center", va="bottom", fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(split_datasets, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("Number of patients", fontsize=10)
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.1)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.legend(handles=[mpatches.Patch(color=SPLIT_COLORS[s], label=s) for s in ["train", "val", "test"]],
               fontsize=9, loc="upper right")
    fig1.tight_layout()

    out1 = Path("data/splits_distribution.png")
    out1.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(out1, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig1)
    print(f"→ {out1}")

    # ── Plot 2 : % unknown (not in any split) ────────────────────────────────
    pcts, n_unknown_list, n_total_list = [], [], []
    for dataset in datasets:
        total = raw_subjects[dataset]
        if dataset in splits:
            assigned = {s for subjs in splits[dataset].values() for s in (subjs or [])}
        else:
            assigned = set()
        n_total   = len(total)
        n_unknown = len(total - assigned)
        pcts.append(100 * n_unknown / n_total if n_total else 0)
        n_unknown_list.append(n_unknown)
        n_total_list.append(n_total)

    n2   = len(datasets)
    x2   = np.arange(n2)
    fig2, ax2 = plt.subplots(figsize=(max(10, n2 * 1.2), 6))
    fig2.suptitle("Percentage of patients not assigned to any split", fontsize=13, fontweight="bold")

    bars = ax2.bar(x2, pcts, color="#C44E52", edgecolor="white", linewidth=0.5)
    for bar, pct, n_unk, n_tot in zip(bars, pcts, n_unknown_list, n_total_list):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{pct:.0f}%\n({n_unk}/{n_tot})", ha="center", va="bottom", fontsize=7)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(datasets, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("Unknown patients (%)", fontsize=10)
    ax2.set_ylim(0, 115)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    fig2.tight_layout()

    out2 = Path("data/splits_unknown.png")
    fig2.savefig(out2, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig2)
    print(f"→ {out2}")


if __name__ == "__main__":
    main()
