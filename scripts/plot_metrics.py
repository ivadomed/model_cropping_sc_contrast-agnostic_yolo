#!/usr/bin/env python3
"""
Violin plots of per-patient metrics, one violin per dataset.

Reads patients.csv (index: dataset, stem) + per-patient patient.csv
(metrics at each conf threshold). Split assignment is resolved at
runtime from --splits-dir YAMLs.

Metrics:
  iou_gt_mean   : mean IoU over GT slices (FN counted as 0)
  iou_all_mean  : mean IoU over all slices (FP and FN counted as 0)
  iou_3d        : 3D IoU between predicted bbox_3d and GT bbox_3d
  fp_rate       : FP slices / total slices per patient
  fn_rate       : FN slices / GT slices per patient
  fp_iou_rate   : pred with IoU < iou-thresh / total pred slices
  fn_iou_rate   : GT slices with IoU < iou-thresh / GT slices
  fp_on_gt_rate       : pred with IoU == 0 / GT slices with a detection (conf >= thresh)
  fp_on_gt_inner_rate : same, restricted to inner GT slices (excluding first and last GT slice in Z)

Usage:
    python scripts/plot_metrics.py --inference predictions/yolo26_1mm_axial
    python scripts/plot_metrics.py --inference predictions/yolo26_1mm_axial --conf-sweep
    python scripts/plot_metrics.py --inference predictions/yolo26_1mm_axial --splits val train test --conf-sweep
    python scripts/plot_metrics.py --inference predictions/yolo26_1mm_axial --metric iou_3d --splits test
"""

import argparse
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

METRIC_LABELS = {
    "iou_gt_mean":        "Mean IoU on SC slices",
    "iou_all_mean":       "Mean IoU on all slices",
    "iou_3d":             "3D IoU voxel (full spine box)",
    "iou_3d_mm":          "3D IoU mm³ (full spine box, slice thickness = si_res_mm)",
    "iou_3d_mm_filt":     "3D IoU mm³ filtered (outlier slices with IoU=0 vs all others removed)",
    "iou_3d_mm_ransac":   "3D IoU mm³ RANSAC (linear z→cx,cy fit, outlier slices rejected)",
    "iou_3d_mm_pad10":    "3D IoU mm³ with 10mm padding on all faces",
    "gt_in_pad10":        "GT fully inside pred + 10mm padding (proportion per dataset)",
    "iou_3d_mm_padz20":   "3D IoU mm³ with 10mm xy + 20mm Z padding",
    "gt_in_padz20":       "GT fully inside pred + 10mm xy / 20mm Z padding (proportion per dataset)",
    "pred_vol_ratio":     "Pred bbox volume / total image volume",
    "iou_sc_mid_box":     "3D IoU (sc_mid expansion box)",
    "fp_rate":            "FP rate (pred on non-SC slices / total slices)",
    "fn_rate":            "FN rate (SC slices missed / SC slices)",
    "fp_iou_rate":        "FP IoU rate (pred with IoU < thresh / total pred slices)",
    "fn_iou_rate":        "FN IoU rate (SC slices with IoU < thresh / SC slices)",
    "fp_on_gt_rate":       "FP on GT slices (pred with IoU = 0 / GT slices with pred)",
    "fp_on_gt_inner_rate": "FP on inner GT slices (excl. first & last GT slice)",
    "gap_mm_R":            "Gap Right face — mm to expand pred to cover GT (+=expand, −=already covers)",
    "gap_mm_L":            "Gap Left face — mm to expand pred to cover GT (+=expand, −=already covers)",
    "gap_mm_P":            "Gap Posterior face — mm to expand pred to cover GT (+=expand, −=already covers)",
    "gap_mm_A":            "Gap Anterior face — mm to expand pred to cover GT (+=expand, −=already covers)",
    "gap_mm_I":            "Gap Inferior face — mm to expand pred to cover GT (+=expand, −=already covers)",
    "gap_mm_S":            "Gap Superior face — mm to expand pred to cover GT (+=expand, −=already covers)",
}

SWEEP_METRICS      = ["iou_gt_mean", "iou_all_mean", "iou_3d", "iou_3d_mm", "iou_3d_mm_filt",
                      "iou_3d_mm_ransac", "iou_3d_mm_pad10", "gt_in_pad10",
                      "iou_3d_mm_padz20", "gt_in_padz20", "pred_vol_ratio",
                      "iou_sc_mid_box", "fp_on_gt_rate", "fp_on_gt_inner_rate",
                      "gap_mm_R", "gap_mm_L", "gap_mm_P", "gap_mm_A", "gap_mm_I", "gap_mm_S"]
PCT_METRICS        = {"fp_rate", "fn_rate", "fp_iou_rate", "fn_iou_rate", "fp_on_gt_rate", "fp_on_gt_inner_rate"}
PROPORTION_METRICS = {"gt_in_pad10", "gt_in_padz20"}   # binary 0/1 → bar chart of % per dataset
FREE_SCALE_METRICS = {"pred_vol_ratio",
                      "gap_mm_R", "gap_mm_L", "gap_mm_P", "gap_mm_A", "gap_mm_I", "gap_mm_S"}
CONF_STEPS    = np.round(np.array([0.0, 0.001, 0.01, 0.05] + list(np.arange(0.1, 1.01, 0.1))), 3)


def load_splits(splits_dir: Path) -> dict:
    """Returns {(dataset, subject): split_name}."""
    mapping = {}
    for f in sorted(Path(splits_dir).glob("datasplit_*.yaml")):
        dataset = re.sub(r"_seed\d+$", "", f.stem[len("datasplit_"):])
        for split_name, subjects in yaml.safe_load(f.read_text()).items():
            for subj in (subjects or []):
                mapping[(dataset, subj)] = split_name
    return mapping


def load_patients_at_conf(pred_root: Path, patients_idx: pd.DataFrame, splits_map: dict,
                           conf_thresh: float, splits_filter: list, datasets_filter) -> pd.DataFrame:
    """Build per-patient DataFrame by reading patient.csv at the given conf threshold."""
    rows = []
    for _, row in patients_idx.iterrows():
        dataset, stem = row["dataset"], row["stem"]
        m       = re.match(r"(sub-[^_]+)", stem)
        subject = m.group(1) if m else stem
        split   = splits_map.get((dataset, subject), "unknown")

        if split not in splits_filter:
            continue
        if datasets_filter and dataset not in datasets_filter:
            continue

        patient_csv = pred_root / dataset / stem / "metrics" / "patient.csv"
        if not patient_csv.exists():
            continue

        df      = pd.read_csv(patient_csv)
        matched = df[np.isclose(df["conf_thresh"], conf_thresh, atol=0.0005)]
        if matched.empty:
            continue
        rows.append({"dataset": dataset, "stem": stem, "split": split,
                     **matched.iloc[0].to_dict()})
    return pd.DataFrame(rows)


def plot_violins(df: pd.DataFrame, metric: str, title: str, out_path: Path, dpi: int) -> None:
    is_pct   = metric in PCT_METRICS
    datasets = sorted(df["dataset"].unique())
    n        = len(datasets)

    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    positions        = list(range(1, n + 1))
    data_per_dataset = []
    labels           = []

    for dataset in datasets:
        vals = df[df["dataset"] == dataset][metric].dropna().values
        if is_pct:
            vals = vals * 100
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
        center = 50 if is_pct else 0.5
        ax.text(positions[i], center, "—", ha="center", va="center", color="#aaa", fontsize=10)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    if is_pct:
        ax.set_ylabel(METRIC_LABELS[metric] + " (%)", fontsize=10)
        ax.set_ylim(-2, 102)
    elif metric in FREE_SCALE_METRICS:
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        if metric.startswith("gap_mm_"):
            ax.axhline(0, color="#888", linewidth=1.0, linestyle="--", zorder=1)
    else:
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_ylim(-0.05, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    if violin_idx:
        ax.legend(handles=[mpatches.Patch(color="red", label="Mean"),
                            mpatches.Patch(color="orange", label="Median")],
                  loc="upper right" if is_pct else "lower right", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def plot_bars(df: pd.DataFrame, metric: str, title: str, out_path: Path, dpi: int) -> None:
    """Bar chart of proportion (mean of 0/1 metric) per dataset."""
    datasets  = sorted(df["dataset"].unique())
    positions = list(range(1, len(datasets) + 1))

    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.4), 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for pos, dataset in zip(positions, datasets):
        vals = df[df["dataset"] == dataset][metric].dropna().values
        if len(vals) == 0:
            ax.text(pos, 1, "—", ha="center", va="bottom", color="#aaa", fontsize=10)
            continue
        pct   = float(vals.mean()) * 100
        color = "#1a7a1a" if pct >= 80 else "#7a6a00" if pct >= 50 else "#aa1a1a"
        ax.bar(pos, pct, color=color, alpha=0.8)
        ax.text(pos, pct + 1, f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{d}\n(n={len(df[df['dataset']==d][metric].dropna())})"
                        for d in datasets], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% patients", fontsize=10)
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-patient metric violin plots per dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference",  required=True,
                        help="Inference run directory (predictions/<run_id>/)")
    parser.add_argument("--metric",     default="iou_gt_mean", choices=list(METRIC_LABELS),
                        metavar="{" + ",".join(METRIC_LABELS) + "}",
                        help="Metric to plot (ignored with --conf-sweep)")
    parser.add_argument("--splits",      nargs="+", default=["val", "train", "test"],
                        choices=["train", "val", "test", "unknown"])
    parser.add_argument("--datasets",   nargs="+", default=None,
                        help="Restrict to these datasets (default: all)")
    parser.add_argument("--splits-dir", default="data/datasplits/from_raw",
                        help="Directory with datasplit_*.yaml for split assignment")
    parser.add_argument("--conf",       type=float, default=0.1,
                        help="Confidence threshold (ignored with --conf-sweep)")
    parser.add_argument("--conf-sweep", action="store_true",
                        help="Generate one violin plot per metric per conf threshold (0.0→1.0)")
    parser.add_argument("--exclude",    nargs="+", default=None,
                        help="Stems to exclude (e.g. sub-nih184_UNIT1)")
    parser.add_argument("--suffix",     default="",
                        help="Suffix appended to output filename (e.g. _no_nih)")
    parser.add_argument("--dpi",        type=int, default=150)
    args = parser.parse_args()

    pred_root    = Path(args.inference)
    splits_map   = load_splits(Path(args.splits_dir))
    patients_idx = pd.read_csv(pred_root / "patients.csv")
    if args.exclude:
        patients_idx = patients_idx[~patients_idx["stem"].isin(args.exclude)]

    def conf_label(c: float) -> str:
        return f"conf{c:.3f}".rstrip("0").rstrip(".")

    if not args.conf_sweep:
        for split in args.splits:
            df = load_patients_at_conf(pred_root, patients_idx, splits_map,
                                       args.conf, [split], args.datasets)
            title    = f"{METRIC_LABELS[args.metric]} — {pred_root.name} [{split}] conf≥{args.conf}"
            out_path = pred_root / "plots" / split / args.metric / f"{conf_label(args.conf)}{args.suffix}.png"
            print(f"{len(df)} patients — {df['dataset'].nunique()} datasets [{split}] "
                  f"({args.metric}) conf≥{args.conf}")
            if args.metric in PROPORTION_METRICS:
                plot_bars(df, args.metric, title, out_path, args.dpi)
            else:
                plot_violins(df, args.metric, title, out_path, args.dpi)
        return

    print(f"Sweeping {len(CONF_STEPS)} thresholds × {len(SWEEP_METRICS)} metrics {args.splits}...")
    for split in args.splits:
        for conf in CONF_STEPS:
            df = load_patients_at_conf(pred_root, patients_idx, splits_map,
                                       conf, [split], args.datasets)
            for metric in SWEEP_METRICS:
                title    = f"{METRIC_LABELS[metric]} — {pred_root.name} [{split}] conf≥{conf}"
                out_path = pred_root / "plots" / split / metric / f"{conf_label(conf)}{args.suffix}.png"
                if metric in PROPORTION_METRICS:
                    plot_bars(df, metric, title, out_path, args.dpi)
                else:
                    plot_violins(df, metric, title, out_path, args.dpi)


if __name__ == "__main__":
    main()
