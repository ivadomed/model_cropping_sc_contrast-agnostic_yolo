#!/usr/bin/env python3
"""
Generate IoU violin plots per dataset from per-patient slices.csv files.

Produces 3 figures (train / val / test+unknown), each with one violin per dataset.
All slices are included (iou=0 for FP, FN, and slices with neither GT nor pred).
Mean and median are displayed on each violin.
n_patients (unique subjects) and n_slices are shown below each violin.

Reads slices.csv from: predictions/<run_id>/<dataset>/<patient>/metrics/slices.csv
Splits assigned from: data/datasplits/datasplit_*.yaml

Output: predictions/<run_id>/plots/violin_<split>.png  (one per split group)

Usage:
    python scripts/plot_metrics.py \
        --inference predictions/yolo_spine_v1 \
        [--splits-dir data/datasplits] [--dpi 150]
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))


def load_splits_map(splits_dir: Path) -> dict:
    """Returns {(dataset, subject): split_name}."""
    mapping = {}
    for f in sorted(splits_dir.glob("datasplit_*.yaml")):
        dataset = re.sub(r"_seed\d+$", "", f.stem[len("datasplit_"):])
        for split_name, subjects in yaml.safe_load(f.read_text()).items():
            for subj in (subjects or []):
                mapping[(dataset, subj)] = split_name
    return mapping


def load_all_slices(inference_dir: Path, splits_map: dict) -> pd.DataFrame:
    """Load all slices.csv and tag with dataset/subject/split."""
    frames = []
    for slices_csv in sorted(inference_dir.rglob("metrics/slices.csv")):
        patient_dir = slices_csv.parent.parent
        dataset     = patient_dir.parent.name
        stem        = patient_dir.name
        m           = re.match(r"(sub-[^_]+)_?(.*)", stem)
        subject     = m.group(1) if m else stem
        split       = splits_map.get((dataset, subject), "unknown")
        contrast    = m.group(2) if m and m.group(2) else "default"
        df          = pd.read_csv(slices_csv, usecols=["iou", "has_gt", "has_pred"])
        df["dataset"]  = dataset
        df["subject"]  = subject
        df["contrast"] = contrast
        df["split"]    = split
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_violins(df: pd.DataFrame, split_label: str, out_path: Path, dpi: int,
                 group_col: str = "dataset") -> None:
    datasets = sorted(df[group_col].unique())
    n        = len(datasets)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.4), 7))
    fig.suptitle(f"IoU distribution per dataset — {split_label}", fontsize=14, fontweight="bold")

    positions = list(range(1, n + 1))
    data_per_dataset = []
    labels            = []
    n_patients_list   = []
    n_slices_list     = []

    for dataset in datasets:
        sub        = df[df[group_col] == dataset]["iou"].values
        n_patients = df[df[group_col] == dataset]["subject"].nunique()
        data_per_dataset.append(sub)
        labels.append(dataset)
        n_patients_list.append(n_patients)
        n_slices_list.append(len(sub))

    # Violin plot
    parts = ax.violinplot(data_per_dataset, positions=positions,
                          showmeans=False, showmedians=False, showextrema=False)

    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_edgecolor("#2a4a80")
        pc.set_alpha(0.75)

    # Overlay mean and median
    for i, (pos, data) in enumerate(zip(positions, data_per_dataset)):
        if len(data) == 0:
            continue
        mean   = np.mean(data)
        median = np.median(data)
        ax.scatter(pos, mean,   color="red",    zorder=5, s=40, marker="D", label="Mean"   if i == 0 else "")
        ax.scatter(pos, median, color="orange", zorder=5, s=40, marker="o", label="Median" if i == 0 else "")
        ax.vlines(pos, np.percentile(data, 25), np.percentile(data, 75),
                  color="white", linewidth=2.5, zorder=4)

    # x-axis labels with dataset name + stats below
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{lbl}\n{np}\u00a0pts\n{ns}\u00a0sl"
         for lbl, np, ns in zip(labels, n_patients_list, n_slices_list)],
        rotation=30, ha="right", fontsize=9,
    )

    ax.set_ylabel("IoU", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    mean_patch   = mpatches.Patch(color="red",    label="Mean")
    median_patch = mpatches.Patch(color="orange", label="Median")
    ax.legend(handles=[mean_patch, median_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Violin plots of IoU per dataset from slices.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference",  required=True, help="Path to inference run directory")
    parser.add_argument("--splits-dir", default="data/datasplits")
    parser.add_argument("--dpi",        type=int, default=150)
    args = parser.parse_args()

    inference_dir = Path(args.inference)
    splits_map    = load_splits_map(Path(args.splits_dir))
    plots_dir     = inference_dir / "plots"

    print("Loading slices.csv files…")
    df = load_all_slices(inference_dir, splits_map)
    if df.empty:
        print("No slices.csv found. Run metrics.py first.")
        return

    print(f"Loaded {len(df)} slices from {df['subject'].nunique()} patients")

    # Merge test + unknown
    df["split_group"] = df["split"].replace({"unknown": "test/unknown", "test": "test/unknown"})

    split_groups = [
        ("train",        "Train"),
        ("val",          "Validation"),
        ("test/unknown", "Test + Unknown"),
    ]

    for split_key, split_label in split_groups:
        sub = df[df["split_group"] == split_key]
        if sub.empty:
            print(f"  No data for {split_label}, skipping.")
            continue
        # All slices (IoU=0 for FP/FN)
        out_path = plots_dir / f"violin_{split_key.replace('/', '_')}.png"
        print(f"Plotting {split_label} — all slices ({len(sub)} slices, {sub['subject'].nunique()} patients)…")
        plot_violins(sub, f"{split_label} — all slices", out_path, args.dpi)
        # Matched slices only (has_gt AND has_pred)
        matched  = sub[sub["has_gt"] & sub["has_pred"]]
        out_path = plots_dir / f"violin_{split_key.replace('/', '_')}_matched.png"
        print(f"Plotting {split_label} — matched only ({len(matched)} slices, {matched['subject'].nunique()} patients)…")
        plot_violins(matched, f"{split_label} — matched only (GT ∩ pred)", out_path, args.dpi)

    # Per-dataset contrast plots — test/unknown only
    test_unk = df[df["split_group"] == "test/unknown"]
    multi_contrast_datasets = [
        d for d in sorted(test_unk["dataset"].unique())
        if test_unk[test_unk["dataset"] == d]["contrast"].nunique() > 1
    ]
    if multi_contrast_datasets:
        print("\nContrast violin plots (test/unknown)…")
        for dataset in multi_contrast_datasets:
            sub      = test_unk[test_unk["dataset"] == dataset]
            n_contrasts = sub["contrast"].nunique()
            # All slices
            out_path = plots_dir / f"violin_contrast_{dataset}.png"
            print(f"  {dataset} ({n_contrasts} contrasts) — all slices…")
            plot_violins(sub, f"{dataset} — Test/Unknown by contrast", out_path, args.dpi,
                         group_col="contrast")
            # Matched only
            matched  = sub[sub["has_gt"] & sub["has_pred"]]
            out_path = plots_dir / f"violin_contrast_{dataset}_matched.png"
            print(f"  {dataset} ({n_contrasts} contrasts) — matched only…")
            plot_violins(matched, f"{dataset} — Test/Unknown by contrast (matched only)", out_path,
                         args.dpi, group_col="contrast")


if __name__ == "__main__":
    main()
