#!/usr/bin/env python3
"""
Violin plots of per-patient metrics, one violin per dataset.

Reads patients.csv (index: dataset, stem) + per-patient patient.csv
(metrics at each conf threshold). Split assignment is resolved at
runtime from --splits-dir YAMLs.

Metrics:
  iou_gt_mean  : mean IoU over GT slices (FN counted as 0)
  iou_3d       : 3D IoU between predicted bbox_3d and GT bbox_3d
  fp_rate      : FP slices / total slices per patient
  fn_rate      : FN slices / GT slices per patient
  fp_iou_rate  : pred with IoU < iou-thresh / total pred slices
  fn_iou_rate  : GT slices with IoU < iou-thresh / GT slices

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
    "iou_gt_mean":  "Mean IoU on SC slices",
    "iou_all_mean": "Mean IoU on all slices",
    "iou_3d":       "3D IoU",
    "fp_rate":      "FP rate (pred on non-SC slices / total slices)",
    "fn_rate":      "FN rate (SC slices missed / SC slices)",
    "fp_iou_rate":  "FP IoU rate (pred with IoU < thresh / total pred slices)",
    "fn_iou_rate":  "FN IoU rate (SC slices with IoU < thresh / SC slices)",
}

SWEEP_METRICS = ["iou_gt_mean", "iou_all_mean"]
PCT_METRICS   = {"fp_rate", "fn_rate", "fp_iou_rate", "fn_iou_rate"}
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
    parser.add_argument("--dpi",        type=int, default=150)
    args = parser.parse_args()

    pred_root    = Path(args.inference)
    splits_map   = load_splits(Path(args.splits_dir))
    patients_idx = pd.read_csv(pred_root / "patients.csv")

    def conf_label(c: float) -> str:
        return f"conf{c:.3f}".rstrip("0").rstrip(".")

    if not args.conf_sweep:
        for split in args.splits:
            df = load_patients_at_conf(pred_root, patients_idx, splits_map,
                                       args.conf, [split], args.datasets)
            title    = f"{METRIC_LABELS[args.metric]} — {pred_root.name} [{split}] conf≥{args.conf}"
            out_path = pred_root / "plots" / split / args.metric / f"{conf_label(args.conf)}.png"
            print(f"{len(df)} patients — {df['dataset'].nunique()} datasets [{split}] "
                  f"({args.metric}) conf≥{args.conf}")
            plot_violins(df, args.metric, title, out_path, args.dpi)
        return

    print(f"Sweeping {len(CONF_STEPS)} thresholds × {len(SWEEP_METRICS)} metrics {args.splits}...")
    for split in args.splits:
        for conf in CONF_STEPS:
            df = load_patients_at_conf(pred_root, patients_idx, splits_map,
                                       conf, [split], args.datasets)
            for metric in SWEEP_METRICS:
                title    = f"{METRIC_LABELS[metric]} — {pred_root.name} [{split}] conf≥{conf}"
                out_path = pred_root / "plots" / split / metric / f"{conf_label(conf)}.png"
                plot_violins(df, metric, title, out_path, args.dpi)


if __name__ == "__main__":
    main()
