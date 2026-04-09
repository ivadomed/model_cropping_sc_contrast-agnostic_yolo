#!/usr/bin/env python3
"""
Border metrics: IoU, dIoU, FP%, FN% at the brain-cord junction (superior end).

For each patient in each dataset (splits: test + unknown), finds the last GT slice
(highest z with a non-empty GT bbox = superior boundary of the spinal cord), then
computes per-slice metrics for levels -N to +N relative to that boundary.

Level 0  = last GT slice (junction with brain)
Level -k = k slices below the boundary (cord still visible, GT present)
Level +k = k slices above the boundary (no cord, no GT — potential FP zone)

Outputs per dataset in <inference>/<dataset>/metrics/:
  - border_metrics.csv  : per-slice rows with level, iou, is_fp, is_fn
  - border_iou.png      : IoU violin plot (levels -N to 0, GT slices only)
  - border_fp_fn.png    : FP% and FN% bar plots (levels -N to +N)

Usage:
    python scripts/border_metrics.py \\
        --inference predictions/yolo26_10mm_aug_320_tassan \\
        [--processed processed_10mm_SI] \\
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


def read_gt_box(txt_path: Path):
    """Returns (cx, cy, w, h) or None if file is empty or missing."""
    if not txt_path.exists():
        return None
    content = txt_path.read_text().strip()
    if not content:
        return None
    p = content.split()
    return (float(p[1]), float(p[2]), float(p[3]), float(p[4]))


def read_pred(txt_path: Path):
    """Returns ((cx, cy, w, h), conf) or (None, 0.0) if file is empty or missing."""
    if not txt_path.exists():
        return None, 0.0
    content = txt_path.read_text().strip()
    if not content:
        return None, 0.0
    p = content.split()
    return (float(p[1]), float(p[2]), float(p[3]), float(p[4])), float(p[5] if len(p) > 5 else 1.0)


def bbox_iou(a, b) -> float:
    """IoU between two (cx, cy, w, h) normalised bboxes."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0., min(ax2, bx2) - max(ax1, bx1)) * max(0., min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.



def patient_border_rows(
    gt_txt_dir: Path,
    pred_txt_dir: Path,
    n: int,
    conf_thresh: float,
    iou_thresh: float,
) -> list:
    """One row per level in [-n, +n] relative to the last GT slice of this patient."""
    gt_txts = {int(p.stem.split("_")[1]): p for p in sorted(gt_txt_dir.glob("slice_*.txt"))}
    gt_with_bbox = [z for z, p in gt_txts.items() if read_gt_box(p) is not None]
    if not gt_with_bbox:
        return []
    last_gt = max(gt_with_bbox)

    rows = []
    for level in range(-n, n + 1):
        z = last_gt + level
        if z < 0:
            continue

        gt_box = read_gt_box(gt_txts[z]) if z in gt_txts else None
        pred_txt = pred_txt_dir / f"slice_{z:03d}.txt"
        pred_box, pred_conf = read_pred(pred_txt) if pred_txt_dir.is_dir() else (None, 0.0)

        has_gt   = gt_box is not None
        has_pred = pred_box is not None and pred_conf >= conf_thresh

        # IoU only when GT is present
        if has_gt and has_pred:
            iou = bbox_iou(gt_box, pred_box)
        elif has_gt:
            iou = 0.0   # GT present, no valid prediction
        else:
            iou = None  # no GT — not applicable

        # FP = pred present and IoU < iou_thresh (covers spurious preds and bad localisation)
        # FN = GT present and no prediction at all
        fp = has_pred and (iou is None or iou < iou_thresh)
        fn = has_gt and not has_pred

        rows.append({
            "level":  level,
            "has_gt": has_gt,
            "iou":    iou,
            "is_fp":  fp,
            "is_fn":  fn,
        })
    return rows


def collect_rows(
    processed_dir: Path,
    pred_root: Path,
    split_map: dict,
    dataset: str,
    n: int,
    conf_thresh: float,
    iou_thresh: float,
) -> pd.DataFrame:
    dataset_dir = processed_dir / dataset
    rows = []
    for patient_dir in sorted(dataset_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        stem = patient_dir.name
        m = re.match(r"(sub-[^_]+)", stem)
        if not m:
            continue
        subject = m.group(1)
        split   = split_map.get(subject, "unknown")
        if split not in SPLITS:
            continue

        gt_txt_dir   = patient_dir / "txt"
        pred_txt_dir = pred_root / dataset / stem / "txt"

        for row in patient_border_rows(gt_txt_dir, pred_txt_dir, n, conf_thresh, iou_thresh):
            row["subject"] = stem
            row["split"]   = split
            rows.append(row)
    return pd.DataFrame(rows)


def level_label(level: int) -> str:
    return str(level)


def plot_violin(ax, df: pd.DataFrame, col: str, levels: list, color: str, title: str, ylabel: str):
    data = [df.loc[df["level"] == l, col].dropna().values for l in levels]

    # violinplot requires >= 2 points per group — plot single points separately
    violin_idx  = [i for i, d in enumerate(data) if len(d) >= 2]
    single_idx  = [i for i, d in enumerate(data) if len(d) == 1]

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
    ax.set_xticklabels([level_label(l) for l in levels], rotation=0, ha="center", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Slice index relative to last slice containing spinal cord (0 = last GT slice)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    y_min = ax.get_ylim()[0]
    for i, d in enumerate(data):
        ax.text(i, y_min, f"n={len(d)}", ha="center", va="bottom", fontsize=6, color="#555")


def plot_bar(ax, df: pd.DataFrame, col: str, levels: list, color: str, title: str, ylabel: str):
    pcts, ns = [], []
    for lvl in levels:
        sub = df[df["level"] == lvl]
        n   = len(sub)
        pct = 100 * int(sub[col].sum()) / n if n > 0 else 0.0
        pcts.append(pct)
        ns.append(n)

    ax.bar(range(len(levels)), pcts, color=color, alpha=0.7, edgecolor="black", linewidth=0.5)

    # Vertical line between level 0 and level +1 (GT boundary)
    boundary_x = levels.index(0) + 0.5
    ax.axvline(x=boundary_x, color="red", linestyle="--", linewidth=1.2, label="GT boundary")
    ax.text(boundary_x + 0.05, 95, "↑ no GT", fontsize=8, color="red", va="top")

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([level_label(l) for l in levels], rotation=0, ha="center", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Slice index relative to last slice containing spinal cord (0 = last GT slice)")
    ax.set_title(title)
    ax.set_ylim(0, 120)
    ax.grid(axis="y", alpha=0.3)

    for i, (pct, n) in enumerate(zip(pcts, ns)):
        label = f"{pct:.2f}%\n(n={n})"
        ax.text(i, pct + 1, label, ha="center", va="bottom", fontsize=6, rotation=90)


def run_dataset(
    dataset: str,
    pred_root: Path,
    processed_dir: Path,
    splits_dir: Path,
    n: int,
    conf: float,
    iou_thresh: float,
):
    split_map = load_split_map(splits_dir, dataset)
    out_dir   = pred_root / dataset / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{dataset}] collecting rows...")
    df = collect_rows(processed_dir, pred_root, split_map, dataset, n, conf, iou_thresh)
    if df.empty:
        print(f"  [{dataset}] no rows — skipping.")
        return

    print(f"  [{dataset}] {len(df)} rows, {df['subject'].nunique()} patients")

    csv_path = out_dir / "border_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [{dataset}] → {csv_path}")

    levels_iou = list(range(-n, 1))        # -N … 0
    levels_bar = list(range(-n, n + 1))    # -N … +N
    title_base = (
        f"{dataset}  (test + unknown)\n"
        f"model: {pred_root.name} | conf ≥ {conf} | IoU thresh = {iou_thresh} | N = {n}"
    )
    gt_df = df[df["has_gt"]].copy()

    # Scale figure width with number of levels (min 10, 0.4 inch per level)
    iou_width = max(10, len(levels_iou) * 0.4)
    bar_width  = max(14, len(levels_bar) * 0.4)

    # Figure 1: IoU violin
    fig1, ax_iou = plt.subplots(figsize=(iou_width, 5))
    fig1.suptitle(f"IoU — {title_base}", fontsize=10)
    plot_violin(ax_iou, gt_df, "iou", levels_iou, "#4C72B0", "IoU (GT slices only)", "IoU")
    fig1.tight_layout()
    fig1_path = out_dir / "border_iou.png"
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    print(f"  [{dataset}] → {fig1_path}")
    plt.close(fig1)

    # Figure 2: FP and FN bars
    fig2, (ax_fp, ax_fn) = plt.subplots(1, 2, figsize=(bar_width, 6))
    fig2.suptitle(f"FP / FN — {title_base}", fontsize=10)
    plot_bar(ax_fp, df, "is_fp", levels_bar, "#C44E52", "FP", "FP (%)")
    plot_bar(ax_fn, df, "is_fn", levels_bar, "#DD8452", "FN without any prediction", "FN (%)")
    fig2.tight_layout()
    fig2_path = out_dir / "border_fp_fn.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    print(f"  [{dataset}] → {fig2_path}")
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(
        description="Border metrics at the brain-cord junction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference",  required=True, help="Path to predictions/<run-id>/")
    parser.add_argument("--processed",  default="processed_10mm_SI")
    parser.add_argument("--datasets",   nargs="+", default=None,
                        help="Datasets to process (default: all subdirs in <inference>/)")
    parser.add_argument("--splits-dir", default="data/datasplits")
    parser.add_argument("--n",          type=int,   default=5,   help="Border window size (N slices each side)")
    parser.add_argument("--conf",       type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for TP")
    args = parser.parse_args()

    pred_root     = Path(args.inference)
    processed_dir = Path(args.processed)
    splits_dir    = Path(args.splits_dir)

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted(
            d.name for d in pred_root.iterdir()
            if d.is_dir() and (processed_dir / d.name).is_dir()
        )

    print(f"Running border metrics on {len(datasets)} dataset(s): {datasets}")
    for dataset in datasets:
        run_dataset(dataset, pred_root, processed_dir, splits_dir, args.n, args.conf, args.iou_thresh)


if __name__ == "__main__":
    main()
