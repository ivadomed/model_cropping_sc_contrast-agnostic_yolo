#!/usr/bin/env python3
"""
Border metrics: IoU, FP%, FN% at both extremities of the spinal cord.

Reads per-patient slices.csv produced by metrics.py (must be run first).

For each patient (splits: test + unknown) two boundaries are analysed independently:

  Superior boundary (brain-cord junction, top of cervicals)
    boundary_z = max GT slice (highest z with has_gt=True)
    Level  0   = last GT slice
    Level -k   = k slices below (cord present, GT exists)
    Level +k   = k slices above (no cord, no GT — FP zone)

  Inferior boundary (bottom of cord / start of lumbars)
    boundary_z = min GT slice (lowest z with has_gt=True)
    Level  0   = first GT slice
    Level -k   = k slices below (no cord, no GT — FP zone)
    Level +k   = k slices above (cord present, GT exists)

Outputs per dataset in <inference>/<dataset>/metrics/:
  - border_metrics_superior.csv / border_metrics_inferior.csv
  - border_iou_superior.png    / border_iou_inferior.png
  - border_fp_fn_superior.png  / border_fp_fn_inferior.png

Usage:
    python scripts/border_metrics.py \\
        --inference predictions/yolo26_10mm_aug_320_tassan \\
        [--datasets data-multi-subject basel-mp2rage ...] \\
        [--splits-dir data/datasplits] \\
        [--n 5] [--conf 0.5] [--iou-thresh 0.5]
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

SPLITS = {"test", "unknown"}


def load_split_map(splits_dir: Path, dataset: str) -> dict:
    """Returns {subject: split_name} from the dataset's datasplit yaml."""
    mapping = {}
    for f in sorted(splits_dir.glob(f"datasplit_{dataset}_seed*.yaml")):
        for split_name, subjects in yaml.safe_load(f.read_text()).items():
            for subj in (subjects or []):
                mapping[subj] = split_name
    return mapping


def patient_border_rows_from_csv(
    slices_csv: Path,
    n: int,
    conf_thresh: float,
    iou_thresh: float,
    boundary: str,
) -> list:
    """One row per level in [-n, +n] relative to the boundary slice.

    Reads slices.csv produced by metrics.py.
    boundary='superior' → boundary_z = max GT slice
    boundary='inferior' → boundary_z = min GT slice
    """
    df = pd.read_csv(slices_csv)
    gt_zs = df.loc[df["has_gt"], "slice_idx"].values
    if len(gt_zs) == 0:
        return []
    boundary_z = int(gt_zs.max()) if boundary == "superior" else int(gt_zs.min())
    df = df.set_index("slice_idx")

    rows = []
    for level in range(-n, n + 1):
        z = boundary_z + level
        if z < 0 or z not in df.index:
            continue
        row      = df.loc[z]
        has_gt   = bool(row["has_gt"])
        has_pred = bool(row["has_pred"]) and float(row["pred_conf"]) >= conf_thresh
        iou = float(row["iou"]) if (has_gt and has_pred) else (0.0 if has_gt else None)
        fp  = has_pred and (iou is None or iou < iou_thresh)
        fn  = has_gt and not has_pred
        rows.append({"level": level, "has_gt": has_gt, "iou": iou, "is_fp": fp, "is_fn": fn})
    return rows


def collect_rows(
    pred_root: Path,
    split_map: dict,
    dataset: str,
    n: int,
    conf_thresh: float,
    iou_thresh: float,
    boundary: str,
) -> pd.DataFrame:
    dataset_dir = pred_root / dataset
    rows = []
    for patient_dir in sorted(dataset_dir.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name == "metrics":
            continue
        stem = patient_dir.name
        m = re.match(r"(sub-[^_]+)", stem)
        if not m:
            continue
        subject = m.group(1)
        split   = split_map.get(subject, "unknown")
        if split not in SPLITS:
            continue

        slices_csv = patient_dir / "metrics" / "slices.csv"
        if not slices_csv.exists():
            print(f"  WARNING: {slices_csv} not found — run metrics.py first, skipping {stem}")
            continue

        for row in patient_border_rows_from_csv(slices_csv, n, conf_thresh, iou_thresh, boundary):
            row["subject"] = stem
            row["split"]   = split
            rows.append(row)
    return pd.DataFrame(rows)


def plot_violin(ax, df: pd.DataFrame, col: str, levels: list, color: str, title: str, ylabel: str, xlabel: str):
    data = [df.loc[df["level"] == l, col].dropna().values for l in levels]

    violin_idx = [i for i, d in enumerate(data) if len(d) >= 2]
    single_idx = [i for i, d in enumerate(data) if len(d) == 1]

    if violin_idx:
        plot_data = [data[i] for i in violin_idx]
        parts = ax.violinplot(plot_data, positions=violin_idx, showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(0.8)

    for i in single_idx:
        ax.scatter([i], data[i], color=color, zorder=5, s=30)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([str(l) for l in levels], rotation=0, ha="center", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    y_min = ax.get_ylim()[0]
    for i, d in enumerate(data):
        ax.text(i, y_min, f"n={len(d)}", ha="center", va="bottom", fontsize=6, color="#555")


def plot_bar(ax, df: pd.DataFrame, col: str, levels: list, color: str, title: str, ylabel: str,
             no_gt_side: str, xlabel: str):
    """no_gt_side: 'right' (superior boundary) or 'left' (inferior boundary)."""
    pcts, ns = [], []
    for lvl in levels:
        sub = df[df["level"] == lvl]
        n   = len(sub)
        pct = 100 * int(sub[col].sum()) / n if n > 0 else 0.0
        pcts.append(pct)
        ns.append(n)

    ax.bar(range(len(levels)), pcts, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)

    zero_pos = levels.index(0)
    if no_gt_side == "right":
        boundary_x = zero_pos + 0.5
        ax.text(boundary_x + 0.1, 108, "no GT →", fontsize=8, color="red", va="top")
    else:
        boundary_x = zero_pos - 0.5
        ax.text(boundary_x - 0.1, 108, "← no GT", fontsize=8, color="red", va="top", ha="right")
    ax.axvline(x=boundary_x, color="red", linestyle="--", linewidth=1.2)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(levels, ns)], rotation=0, ha="center", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylim(0, 120)
    ax.grid(axis="y", alpha=0.3)

    for i, pct in enumerate(pcts):
        if pct > 0:
            ax.text(i, pct + 1, f"{pct:.1f}%", ha="center", va="bottom", fontsize=6, rotation=90)


def run_boundary(
    boundary: str,
    dataset: str,
    pred_root: Path,
    split_map: dict,
    out_dir: Path,
    n: int,
    conf: float,
    iou_thresh: float,
):
    print(f"  [{dataset}] collecting rows ({boundary})...")
    df = collect_rows(pred_root, split_map, dataset, n, conf, iou_thresh, boundary)
    if df.empty:
        print(f"  [{dataset}] no rows ({boundary}) — skipping.")
        return

    print(f"  [{dataset}] {boundary}: {len(df)} rows, {df['subject'].nunique()} patients")

    csv_path = out_dir / f"border_metrics_{boundary}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [{dataset}] → {csv_path}")

    title_base = (
        f"{dataset}  ({boundary.upper()})  (test + unknown)\n"
        f"model: {pred_root.name} | conf ≥ {conf} | IoU thresh = {iou_thresh} | N = {n}"
    )

    if boundary == "superior":
        levels_iou  = list(range(-n, 1))
        no_gt_side  = "right"
        xlabel_base = "Slice index relative to last GT slice (0 = last slice with spinal cord)"
    else:
        levels_iou  = list(range(0, n + 1))
        no_gt_side  = "left"
        xlabel_base = "Slice index relative to first GT slice (0 = first slice with spinal cord)"

    levels_bar = list(range(-n, n + 1))
    gt_df = df[df["has_gt"]].copy()

    iou_width = max(10, len(levels_iou) * 0.4)
    bar_width  = max(14, len(levels_bar) * 0.4)

    fig1, ax_iou = plt.subplots(figsize=(iou_width, 5))
    fig1.suptitle(f"IoU — {title_base}", fontsize=10)
    plot_violin(ax_iou, gt_df, "iou", levels_iou, "#4C72B0", "IoU (GT slices only)", "IoU", xlabel_base)
    fig1.tight_layout()
    fig1_path = out_dir / f"border_iou_{boundary}.png"
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    print(f"  [{dataset}] → {fig1_path}")
    plt.close(fig1)

    fig2, (ax_fp, ax_fn) = plt.subplots(1, 2, figsize=(bar_width, 6))
    fig2.suptitle(f"FP / FN — {title_base}", fontsize=10)
    plot_bar(ax_fp, df, "is_fp", levels_bar, "#C44E52", "FP", "FP (%)", no_gt_side, xlabel_base)
    plot_bar(ax_fn, df, "is_fn", levels_bar, "#DD8452", "FN (no prediction at all)", "FN (%)", no_gt_side, xlabel_base)
    fig2.tight_layout()
    fig2_path = out_dir / f"border_fp_fn_{boundary}.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    print(f"  [{dataset}] → {fig2_path}")
    plt.close(fig2)


def run_dataset(
    dataset: str,
    pred_root: Path,
    splits_dir: Path,
    n: int,
    conf: float,
    iou_thresh: float,
):
    split_map = load_split_map(splits_dir, dataset)
    out_dir   = pred_root / dataset / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    for boundary in ("superior", "inferior"):
        run_boundary(boundary, dataset, pred_root, split_map, out_dir, n, conf, iou_thresh)


def main():
    parser = argparse.ArgumentParser(
        description="Border metrics at both extremities of the spinal cord (requires metrics.py slices.csv)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference",  required=True, help="Path to predictions/<run-id>/")
    parser.add_argument("--datasets",   nargs="+", default=None,
                        help="Datasets to process (default: all subdirs in <inference>/)")
    parser.add_argument("--splits-dir", default="data/datasplits")
    parser.add_argument("--n",          type=int,   default=5,   help="Border window size (N slices each side)")
    parser.add_argument("--conf",       type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for TP")
    args = parser.parse_args()

    pred_root  = Path(args.inference)
    splits_dir = Path(args.splits_dir)

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted(
            d.name for d in pred_root.iterdir()
            if d.is_dir() and any((p / "metrics" / "slices.csv").exists()
                                  for p in d.iterdir() if p.is_dir())
        )

    print(f"Running border metrics on {len(datasets)} dataset(s): {datasets}")
    for dataset in datasets:
        run_dataset(dataset, pred_root, splits_dir, args.n, args.conf, args.iou_thresh)


if __name__ == "__main__":
    main()
