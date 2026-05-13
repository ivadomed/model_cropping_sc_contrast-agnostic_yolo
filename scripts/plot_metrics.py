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
    python scripts/plot_metrics.py --inference predictions/yolo26_1mm_axial --metrics iou_3d_mm gap_mm_R gap_mm_L --splits val test
    python scripts/plot_metrics.py --inference predictions/yolo26_1mm_axial --exclude-csv bad_gt.csv

exclude-csv format (two columns, no index):
    dataset,stem
    nih-ms-mp2rage,sub-nih062_UNIT1
    dcm-zurich,sub-001
"""

from __future__ import annotations

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
    "iou_3d_mm_reg30mm":   "3D IoU mm³ (reg30mm: isolated preds >30mm from all others removed)",
    "gap_mm_R_reg30mm":    "Gap Right face reg30mm — isolated preds >30mm removed",
    "gap_mm_L_reg30mm":    "Gap Left face reg30mm — isolated preds >30mm removed",
    "gap_mm_P_reg30mm":    "Gap Posterior face reg30mm — isolated preds >30mm removed",
    "gap_mm_A_reg30mm":    "Gap Anterior face reg30mm — isolated preds >30mm removed",
    "gap_mm_I_reg30mm":    "Gap Inferior face reg30mm — isolated preds >30mm removed",
    "gap_mm_S_reg30mm":    "Gap Superior face reg30mm — isolated preds >30mm removed",
    "iou_3d_mm_trim50":    "3D IoU mm³ (trim50: pred boundary slices >50mm 3D dist removed)",
    "gap_mm_R_trim50":     "Gap Right face trim50 — pred boundary slices >50mm 3D dist removed",
    "gap_mm_L_trim50":     "Gap Left face trim50 — pred boundary slices >50mm 3D dist removed",
    "gap_mm_P_trim50":     "Gap Posterior face trim50 — pred boundary slices >50mm 3D dist removed",
    "gap_mm_A_trim50":     "Gap Anterior face trim50 — pred boundary slices >50mm 3D dist removed",
    "gap_mm_I_trim50":     "Gap Inferior face trim50 — pred boundary slices >50mm 3D dist removed",
    "gap_mm_S_trim50":     "Gap Superior face trim50 — pred boundary slices >50mm 3D dist removed",
    "iou_3d_mm_trim40":    "3D IoU mm³ (trim40: pred boundary slices >40mm 3D dist removed)",
    "gap_mm_R_trim40":     "Gap Right face trim40 — pred boundary slices >40mm 3D dist removed",
    "gap_mm_L_trim40":     "Gap Left face trim40 — pred boundary slices >40mm 3D dist removed",
    "gap_mm_P_trim40":     "Gap Posterior face trim40 — pred boundary slices >40mm 3D dist removed",
    "gap_mm_A_trim40":     "Gap Anterior face trim40 — pred boundary slices >40mm 3D dist removed",
    "gap_mm_I_trim40":     "Gap Inferior face trim40 — pred boundary slices >40mm 3D dist removed",
    "gap_mm_S_trim40":     "Gap Superior face trim40 — pred boundary slices >40mm 3D dist removed",
    "iou_3d_mm_trim30":    "3D IoU mm³ (trim30: pred boundary slices >30mm 3D dist removed)",
    "gap_mm_R_trim30":     "Gap Right face trim30 — pred boundary slices >30mm 3D dist removed",
    "gap_mm_L_trim30":     "Gap Left face trim30 — pred boundary slices >30mm 3D dist removed",
    "gap_mm_P_trim30":     "Gap Posterior face trim30 — pred boundary slices >30mm 3D dist removed",
    "gap_mm_A_trim30":     "Gap Anterior face trim30 — pred boundary slices >30mm 3D dist removed",
    "gap_mm_I_trim30":     "Gap Inferior face trim30 — pred boundary slices >30mm 3D dist removed",
    "gap_mm_S_trim30":     "Gap Superior face trim30 — pred boundary slices >30mm 3D dist removed",
    "iou_3d_mm_graphreg":  "3D IoU mm³ (graphreg: keep best-confidence connected component)",
    "gap_mm_R_graphreg":   "Gap Right face graphreg — best-confidence connected component kept",
    "gap_mm_L_graphreg":   "Gap Left face graphreg — best-confidence connected component kept",
    "gap_mm_P_graphreg":   "Gap Posterior face graphreg — best-confidence connected component kept",
    "gap_mm_A_graphreg":   "Gap Anterior face graphreg — best-confidence connected component kept",
    "gap_mm_I_graphreg":   "Gap Inferior face graphreg — best-confidence connected component kept",
    "gap_mm_S_graphreg":   "Gap Superior face graphreg — best-confidence connected component kept",
    "iou_3d_mm_graphtrim": "3D IoU mm³ (graphtrim: remove outermost det. if boundary edge breaks graphreg criterion)",
    "gap_mm_R_graphtrim":  "Gap Right face graphtrim — boundary graphreg trim",
    "gap_mm_L_graphtrim":  "Gap Left face graphtrim — boundary graphreg trim",
    "gap_mm_P_graphtrim":  "Gap Posterior face graphtrim — boundary graphreg trim",
    "gap_mm_A_graphtrim":  "Gap Anterior face graphtrim — boundary graphreg trim",
    "gap_mm_I_graphtrim":  "Gap Inferior face graphtrim — boundary graphreg trim",
    "gap_mm_S_graphtrim":  "Gap Superior face graphtrim — boundary graphreg trim",
    "iou_3d_mm_facetrim":  "3D IoU mm³ (facetrim: per-face outlier trim A=30mm P=40mm R=L=10mm)",
    "gap_mm_R_facetrim":   "Gap Right face facetrim — per-face trim (R=10mm)",
    "gap_mm_L_facetrim":   "Gap Left face facetrim — per-face trim (L=10mm)",
    "gap_mm_P_facetrim":   "Gap Posterior face facetrim — per-face trim (P=40mm)",
    "gap_mm_A_facetrim":   "Gap Anterior face facetrim — per-face trim (A=30mm)",
    "gap_mm_I_facetrim":   "Gap Inferior face facetrim — no in-plane trim",
    "gap_mm_S_facetrim":   "Gap Superior face facetrim — no in-plane trim",
    "iou_3d_mm_clsfilt":   "3D IoU mm³ (clsfilt: det preds filtered to cls z-range)",
    "gap_mm_R_clsfilt":    "Gap Right face clsfilt — det preds filtered to cls z-range",
    "gap_mm_L_clsfilt":    "Gap Left face clsfilt — det preds filtered to cls z-range",
    "gap_mm_P_clsfilt":    "Gap Posterior face clsfilt — det preds filtered to cls z-range",
    "gap_mm_A_clsfilt":    "Gap Anterior face clsfilt — det preds filtered to cls z-range",
    "gap_mm_I_clsfilt":    "Gap Inferior face clsfilt — det preds filtered to cls z-range",
    "gap_mm_S_clsfilt":    "Gap Superior face clsfilt — det preds filtered to cls z-range",
    "iou_3d_mm_clscomp":   "3D IoU mm³ (clscomp: first cls-validated SI component + all below)",
    "gap_mm_R_clscomp":    "Gap Right face clscomp — first cls-validated component kept",
    "gap_mm_L_clscomp":    "Gap Left face clscomp — first cls-validated component kept",
    "gap_mm_P_clscomp":    "Gap Posterior face clscomp — first cls-validated component kept",
    "gap_mm_A_clscomp":    "Gap Anterior face clscomp — first cls-validated component kept",
    "gap_mm_I_clscomp":    "Gap Inferior face clscomp — first cls-validated component kept",
    "gap_mm_S_clscomp":    "Gap Superior face clscomp — first cls-validated component kept",
}

SWEEP_METRICS      = ["iou_3d_mm",
                      "gap_mm_R", "gap_mm_L", "gap_mm_P", "gap_mm_A", "gap_mm_I", "gap_mm_S"]
PCT_METRICS        = {"fp_rate", "fn_rate", "fp_iou_rate", "fn_iou_rate", "fp_on_gt_rate", "fp_on_gt_inner_rate"}
PROPORTION_METRICS = {"gt_in_pad10", "gt_in_padz20"}   # binary 0/1 → bar chart of % per dataset
FREE_SCALE_METRICS = {"pred_vol_ratio",
                      "gap_mm_R", "gap_mm_L", "gap_mm_P", "gap_mm_A", "gap_mm_I", "gap_mm_S",
                      "gap_mm_R_reg30mm", "gap_mm_L_reg30mm", "gap_mm_P_reg30mm",
                      "gap_mm_A_reg30mm", "gap_mm_I_reg30mm", "gap_mm_S_reg30mm",
                      "gap_mm_R_trim50", "gap_mm_L_trim50", "gap_mm_P_trim50",
                      "gap_mm_A_trim50", "gap_mm_I_trim50", "gap_mm_S_trim50",
                      "gap_mm_R_trim40", "gap_mm_L_trim40", "gap_mm_P_trim40",
                      "gap_mm_A_trim40", "gap_mm_I_trim40", "gap_mm_S_trim40",
                      "gap_mm_R_trim30", "gap_mm_L_trim30", "gap_mm_P_trim30",
                      "gap_mm_A_trim30", "gap_mm_I_trim30", "gap_mm_S_trim30",
                      "gap_mm_R_facetrim", "gap_mm_L_facetrim", "gap_mm_P_facetrim",
                      "gap_mm_A_facetrim", "gap_mm_I_facetrim", "gap_mm_S_facetrim",
                      "gap_mm_R_clsfilt",  "gap_mm_L_clsfilt",  "gap_mm_P_clsfilt",
                      "gap_mm_A_clsfilt",  "gap_mm_I_clsfilt",  "gap_mm_S_clsfilt",
                      "gap_mm_R_clscomp",  "gap_mm_L_clscomp",  "gap_mm_P_clscomp",
                      "gap_mm_A_clscomp",  "gap_mm_I_clscomp",  "gap_mm_S_clscomp"}
CONF_STEPS    = np.round(np.array([0.0, 0.001, 0.01, 0.05] + list(np.arange(0.1, 1.01, 0.1))), 3)
SPLIT_COLORS  = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868", "unknown": "#8172B2"}


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

        patient_csv = pred_root / "predictions" / dataset / stem / "metrics" / "patient.csv"
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
    is_pct    = metric in PCT_METRICS
    is_gap_mm = metric.startswith("gap_mm_")
    datasets  = sorted(df["dataset"].unique())
    n         = len(datasets)

    if is_gap_mm:
        all_vals = df[metric].dropna().values
        y_min = np.floor(all_vals.min() / 10) * 10 if len(all_vals) else -10
        y_max = np.ceil(all_vals.max()  / 10) * 10 if len(all_vals) else  10
        y_range   = max(y_max - y_min, 10)
        fig_height = max(3, y_range / 10 * 0.9)
    else:
        fig_height = 6

    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), fig_height))
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

    rng = np.random.default_rng(0)
    for i in violin_idx:
        d   = data_per_dataset[i]
        pos = positions[i]
        ax.vlines(pos, np.percentile(d, 25), np.percentile(d, 75), color="white", linewidth=2.5, zorder=4)
        ax.scatter(pos, np.mean(d),   color="red",    zorder=6, s=40, marker="D", label="Mean"   if i == violin_idx[0] else "")
        ax.scatter(pos, np.median(d), color="orange", zorder=6, s=40, marker="o", label="Median" if i == violin_idx[0] else "")
        jitter = rng.uniform(-0.08, 0.08, size=len(d))
        ax.scatter(pos + jitter, d, color="black", alpha=0.35, zorder=5, s=12, linewidths=0)
        if is_gap_mm:
            ax.text(pos, y_max, f"max={d.max():.1f}mm", ha="center", va="bottom",
                    fontsize=7, color="#333", clip_on=False)

    for i in single_idx:
        ax.scatter([positions[i]], data_per_dataset[i], color="black", zorder=5, s=20)

    for i in empty_idx:
        center = 50 if is_pct else 0.5
        ax.text(positions[i], center, "—", ha="center", va="center", color="#aaa", fontsize=10)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    if is_pct:
        ax.set_ylabel(METRIC_LABELS[metric] + " (%)", fontsize=10)
        ax.set_ylim(-2, 102)
    elif is_gap_mm:
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 1, 10))
        ax.axhline(0, color="#888", linewidth=1.0, linestyle="--", zorder=1)
    elif metric in FREE_SCALE_METRICS:
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
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


def plot_global_violins(split_dfs: dict, metric: str, title: str, out_path: Path, dpi: int) -> None:
    """One violin per split (all datasets aggregated), colored by split, with max dashed lines."""
    is_pct    = metric in PCT_METRICS
    is_gap_mm = metric.startswith("gap_mm_")

    splits    = [s for s in ("train", "val", "test", "unknown") if s in split_dfs and not split_dfs[s].empty]
    if not splits:
        return

    all_vals = np.concatenate([split_dfs[s][metric].dropna().values for s in splits])
    if is_gap_mm and len(all_vals) > 0:
        y_min      = np.floor(all_vals.min() / 10) * 10
        y_max      = np.ceil(all_vals.max()  / 10) * 10
        fig_height = max(3, max(y_max - y_min, 10) / 10 * 0.9)
    else:
        fig_height = 6
        y_min = y_max = None

    n   = len(splits)
    fig, ax = plt.subplots(figsize=(max(4, n * 2.5), fig_height))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    rng = np.random.default_rng(0)
    for i, split in enumerate(splits):
        pos   = i + 1
        color = SPLIT_COLORS.get(split, "#999")
        vals  = split_dfs[split][metric].dropna().values
        if is_pct:
            vals = vals * 100

        if len(vals) >= 2:
            parts = ax.violinplot([vals], positions=[pos], showmeans=False, showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(0.75)
            ax.vlines(pos, np.percentile(vals, 25), np.percentile(vals, 75), color="white", linewidth=2.5, zorder=4)
            ax.scatter(pos, np.mean(vals),   color="red",    zorder=6, s=40, marker="D")
            ax.scatter(pos, np.median(vals), color="orange", zorder=6, s=40, marker="o")
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(pos + jitter, vals, color="black", alpha=0.35, zorder=5, s=12, linewidths=0)
        elif len(vals) == 1:
            ax.scatter([pos], vals, color=color, zorder=5, s=20)

        if len(vals) > 0:
            max_val  = float(vals.max())
            unit_str = "%" if is_pct else ("mm" if is_gap_mm else "")
            ax.axhline(max_val, color=color, linestyle="--", linewidth=1.5, alpha=0.9, zorder=3,
                       label=f"{split} max = {max_val:.1f}{unit_str}")

    ax.set_xticks(list(range(1, n + 1)))
    ax.set_xticklabels(
        [f"{s}\n(n={len(split_dfs[s][metric].dropna())})" for s in splits], fontsize=10)

    if is_pct:
        ax.set_ylabel(METRIC_LABELS[metric] + " (%)", fontsize=10)
        ax.set_ylim(-2, 102)
    elif is_gap_mm:
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + 1, 10))
        ax.axhline(0, color="#888", linewidth=1.0, linestyle="--", zorder=1)
    elif metric in FREE_SCALE_METRICS:
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
    else:
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_ylim(-0.05, 1.05)

    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right" if is_pct else "lower right", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def run(inference: str | Path, splits_dir: str | Path) -> None:
    """Plot per-patient metric violin plots from saved predictions."""
    main(["--inference", str(inference), "--splits-dir", str(splits_dir)])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Per-patient metric violin plots per dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference",  required=True,
                        help="Inference run directory (predictions/<run_id>/)")
    parser.add_argument("--metric",     default=None, choices=list(METRIC_LABELS),
                        metavar="{" + ",".join(METRIC_LABELS) + "}",
                        help="Single metric to plot (ignored with --conf-sweep or --metrics)")
    parser.add_argument("--metrics",    nargs="+", default=None, choices=list(METRIC_LABELS),
                        metavar="METRIC",
                        help="One or more metrics to plot (ignored with --conf-sweep)")
    parser.add_argument("--splits",      nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test", "unknown"])
    parser.add_argument("--datasets",   nargs="+", default=None,
                        help="Restrict to these datasets (default: all)")
    parser.add_argument("--splits-dir", default="data/datasplits_seed50",
                        help="Directory with datasplit_*.yaml for split assignment")
    parser.add_argument("--conf",       type=float, default=0.1,
                        help="Confidence threshold (ignored with --conf-sweep)")
    parser.add_argument("--conf-sweep", action="store_true",
                        help="Generate one violin plot per metric per conf threshold (0.0→1.0)")
    parser.add_argument("--exclude",    nargs="+", default=None,
                        help="Stems to exclude (e.g. sub-nih184_UNIT1)")
    parser.add_argument("--exclude-csv", default=str(Path.home() / "bad_gt.csv"),
                        help="CSV with columns 'dataset' and 'stem' — matching (dataset, stem) pairs are excluded (default: ~/bad_gt.csv if it exists)")
    parser.add_argument("--suffix",     default="",
                        help="Suffix appended to output filename (e.g. _no_nih)")
    parser.add_argument("--dpi",        type=int, default=150)
    args = parser.parse_args(argv)

    pred_root    = Path(args.inference)
    splits_map   = load_splits(Path(args.splits_dir))
    patients_idx = pd.read_csv(pred_root / "patients.csv")
    if args.exclude:
        patients_idx = patients_idx[~patients_idx["stem"].isin(args.exclude)]
    if args.exclude_csv and Path(args.exclude_csv).exists():
        excl = pd.read_csv(args.exclude_csv)
        excl_set = set(zip(excl["dataset"], excl["stem"]))
        mask = patients_idx.apply(lambda r: (r["dataset"], r["stem"]) not in excl_set, axis=1)
        n_excl = (~mask).sum()
        patients_idx = patients_idx[mask]
        print(f"Excluded {n_excl} patient(s) from {args.exclude_csv}")

    def conf_label(c: float) -> str:
        return f"conf{c:.3f}".rstrip("0").rstrip(".")

    if not args.conf_sweep:
        metrics_to_plot = args.metrics or ([args.metric] if args.metric else SWEEP_METRICS)
        for split in args.splits:
            df = load_patients_at_conf(pred_root, patients_idx, splits_map,
                                       args.conf, [split], args.datasets)
            for metric in metrics_to_plot:
                title    = f"{METRIC_LABELS[metric]} — {pred_root.name} [{split}] conf≥{args.conf}"
                out_path = (pred_root / "metrics" / "per_split" / split / metric
                            / conf_label(args.conf) / f"{metric}_{conf_label(args.conf)}{args.suffix}.png")
                print(f"{len(df)} patients — {df['dataset'].nunique()} datasets [{split}] "
                      f"({metric}) conf≥{args.conf}")
                if metric in PROPORTION_METRICS:
                    plot_bars(df, metric, title, out_path, args.dpi)
                else:
                    plot_violins(df, metric, title, out_path, args.dpi)
        # Global plots — all splits on one figure
        df_all = load_patients_at_conf(pred_root, patients_idx, splits_map,
                                       args.conf, args.splits, args.datasets)
        for metric in metrics_to_plot:
            if metric in PROPORTION_METRICS:
                continue
            split_dfs = {s: df_all[df_all["split"] == s] for s in args.splits}
            title     = f"{METRIC_LABELS[metric]} — {pred_root.name} [all splits] conf≥{args.conf}"
            out_path  = (pred_root / "metrics" / "globals" / metric
                         / conf_label(args.conf) / f"{metric}_globals_{conf_label(args.conf)}{args.suffix}.png")
            plot_global_violins(split_dfs, metric, title, out_path, args.dpi)
        return

    print(f"Sweeping {len(CONF_STEPS)} thresholds × {len(SWEEP_METRICS)} metrics {args.splits}...")
    for conf in CONF_STEPS:
        for split in args.splits:
            df = load_patients_at_conf(pred_root, patients_idx, splits_map,
                                       conf, [split], args.datasets)
            for metric in SWEEP_METRICS:
                title    = f"{METRIC_LABELS[metric]} — {pred_root.name} [{split}] conf≥{conf}"
                out_path = (pred_root / "metrics" / "per_split" / split / metric
                            / conf_label(conf) / f"{metric}_{conf_label(conf)}{args.suffix}.png")
                if metric in PROPORTION_METRICS:
                    plot_bars(df, metric, title, out_path, args.dpi)
                else:
                    plot_violins(df, metric, title, out_path, args.dpi)
        # Global for this conf step
        df_all = load_patients_at_conf(pred_root, patients_idx, splits_map,
                                       conf, args.splits, args.datasets)
        for metric in SWEEP_METRICS:
            if metric in PROPORTION_METRICS:
                continue
            split_dfs = {s: df_all[df_all["split"] == s] for s in args.splits}
            title     = f"{METRIC_LABELS[metric]} — {pred_root.name} [all splits] conf≥{conf}"
            out_path  = (pred_root / "metrics" / "globals" / metric
                         / conf_label(conf) / f"{metric}_globals_{conf_label(conf)}{args.suffix}.png")
            plot_global_violins(split_dfs, metric, title, out_path, args.dpi)


if __name__ == "__main__":
    main()
