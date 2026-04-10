#!/usr/bin/env python3
"""
Violin plots of per-patient IoU from patients.csv.

Each violin = one dataset, each point = one patient.

Two metrics available via --metric:
  iou_gt_mean  (default) : mean IoU over all GT slices (FN slices counted as 0)
  iou_3d                 : 3D IoU between predicted bbox_3d and GT bbox_3d

Default splits: test + unknown.

Usage:
    python scripts/plot_metrics.py --inference predictions/yolo26_1mm_axial
    python scripts/plot_metrics.py --inference predictions/yolo26_1mm_axial --metric iou_3d
    python scripts/plot_metrics.py --inference predictions/yolo26_1mm_axial --splits train val
    python scripts/plot_metrics.py --inference predictions/yolo26_1mm_axial --datasets data-multi-subject canproco
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

METRIC_LABELS = {
    "iou_gt_mean": "Mean of IoU slices per patient",
    "iou_3d":      "3D IoU (smallest enclosing box pred vs GT)",
}


def plot_violins(df: pd.DataFrame, metric: str, title: str, out_path: Path, dpi: int) -> None:
    datasets = sorted(df["dataset"].unique())
    n        = len(datasets)

    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    positions        = list(range(1, n + 1))
    data_per_dataset = []
    labels           = []

    for dataset in datasets:
        vals = df[df["dataset"] == dataset][metric].dropna().values
        data_per_dataset.append(vals)
        labels.append(f"{dataset}\n(n={len(vals)})")

    violin_idx = [i for i, d in enumerate(data_per_dataset) if len(d) >= 2]
    single_idx = [i for i, d in enumerate(data_per_dataset) if len(d) == 1]
    empty_idx  = [i for i, d in enumerate(data_per_dataset) if len(d) == 0]

    if violin_idx:
        parts = ax.violinplot(
            [data_per_dataset[i] for i in violin_idx],
            positions=[positions[i] for i in violin_idx],
            showmeans=False, showmedians=False, showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0")
            pc.set_edgecolor("#2a4a80")
            pc.set_alpha(0.75)

    for i in violin_idx:
        d   = data_per_dataset[i]
        pos = positions[i]
        ax.vlines(pos, np.percentile(d, 25), np.percentile(d, 75), color="white", linewidth=2.5, zorder=4)
        ax.scatter(pos, np.mean(d),   color="red",    zorder=5, s=40, marker="D", label="Mean"   if i == violin_idx[0] else "")
        ax.scatter(pos, np.median(d), color="orange", zorder=5, s=40, marker="o", label="Median" if i == violin_idx[0] else "")

    for i in single_idx:
        ax.scatter([positions[i]], data_per_dataset[i], color="#4C72B0", zorder=5, s=40)

    for i in empty_idx:
        ax.text(positions[i], 0.5, "—", ha="center", va="center", color="#aaa", fontsize=10)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    if violin_idx:
        ax.legend(handles=[mpatches.Patch(color="red", label="Mean"),
                            mpatches.Patch(color="orange", label="Median")],
                  loc="lower right", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-patient IoU violin plots per dataset from patients.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference", required=True, help="Inference run directory (predictions/<run_id>/)")
    parser.add_argument("--metric",    default="iou_gt_mean", choices=list(METRIC_LABELS),
                        help="Metric to plot")
    parser.add_argument("--splits",    nargs="+", default=["test", "unknown"],
                        choices=["train", "val", "test", "unknown"],
                        help="Splits to include (default: test unknown)")
    parser.add_argument("--datasets",  nargs="+", default=None,
                        help="Restrict to these datasets (default: all)")
    parser.add_argument("--dpi",       type=int, default=150)
    args = parser.parse_args()

    pred_root    = Path(args.inference)
    df           = pd.read_csv(pred_root / "patients.csv")
    df           = df[df["split"].isin(args.splits)]
    if args.datasets:
        df = df[df["dataset"].isin(args.datasets)]

    splits_label = "+".join(args.splits)
    title        = f"{METRIC_LABELS[args.metric]} — {pred_root.name} [{splits_label}]"
    out_path     = pred_root / "plots" / f"violin_{args.metric}_{splits_label}.png"

    print(f"{len(df)} patients — {df['dataset'].nunique()} datasets [{splits_label}] ({args.metric})")
    plot_violins(df, args.metric, title, out_path, args.dpi)


if __name__ == "__main__":
    main()
