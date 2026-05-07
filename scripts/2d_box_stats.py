#!/usr/bin/env python3
"""
2D bounding box edge-gap statistics from GT labels in processed/.

For each patient and each of the 4 in-plane edges (R, L, A, P):
  - Collect the edge position (in mm) for every detected slice
  - Sort by most extreme value and compute |most_extreme - second_most_extreme|

This gap measures how isolated the most extreme detection is. A large gap
indicates a likely outlier detection and can inform a rejection threshold.

Edge convention (axial LAS slices, rows=AP col=RL):
  R : cx + w/2  (rightmost extent,  mm from left border)
  L : cx - w/2  (leftmost extent,   mm from left border → sort ascending)
  A : cy - h/2  (most anterior,     mm from top border  → sort ascending)
  P : cy + h/2  (most posterior,    mm from top border)
  W = shape_las[0] * rl_res_mm,  H = shape_las[1] * ap_res_mm

Outputs (in processed_stats/<processed_variant>/):
  2d_box_stats.csv              — one row per patient
  violin_2d_box_stats.png       — violin plot of the 4 gaps, one violin per dataset

Usage:
    python scripts/2d_box_stats.py --processed processed/10mm_SI_1mm_axial
    python scripts/2d_box_stats.py --processed processed/10mm_SI_1mm_axial --splits-dir data/datasplits
    python scripts/2d_box_stats.py --processed processed/10mm_SI_1mm_axial --exclude-csv bad_gt.csv
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
from tqdm import tqdm

SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868", "unknown": "#8172B2"}
EDGES        = ["R_gap_mm", "L_gap_mm", "A_gap_mm", "P_gap_mm"]
EDGE_LABELS  = {"R_gap_mm": "Right", "L_gap_mm": "Left", "A_gap_mm": "Anterior", "P_gap_mm": "Posterior"}


def load_splits(splits_dir: Path) -> dict:
    mapping = {}
    for f in sorted(splits_dir.glob("datasplit_*.yaml")):
        dataset = re.sub(r"_seed\d+$", "", f.stem[len("datasplit_"):])
        for split_name, subjects in yaml.safe_load(f.read_text()).items():
            for subj in (subjects or []):
                mapping[(dataset, subj)] = split_name
    return mapping


def edge_gap(values: list[float], ascending: bool) -> float:
    """Gap between most extreme and second most extreme value."""
    if len(values) < 2:
        return 0.0
    s = sorted(values, reverse=not ascending)
    return abs(s[0] - s[1])


def patient_gaps(txt_dir: Path, rl_res: float, ap_res: float, W: int, H: int) -> dict | None:
    """Compute the 4 edge gaps for one patient. Returns None if fewer than 2 detections."""
    R_vals, L_vals, A_vals, P_vals = [], [], [], []
    W_mm = W * rl_res
    H_mm = H * ap_res

    for txt_file in sorted(txt_dir.glob("slice_*.txt")):
        lines = [l for l in txt_file.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        parts = lines[0].split()
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        R_vals.append((cx + w / 2) * W_mm)
        L_vals.append((cx - w / 2) * W_mm)
        A_vals.append((cy - h / 2) * H_mm)
        P_vals.append((cy + h / 2) * H_mm)

    if len(R_vals) < 2:
        return None

    return {
        "R_gap_mm": edge_gap(R_vals, ascending=False),
        "L_gap_mm": edge_gap(L_vals, ascending=True),
        "A_gap_mm": edge_gap(A_vals, ascending=True),
        "P_gap_mm": edge_gap(P_vals, ascending=False),
    }


def plot_violin(df: pd.DataFrame, out_path: Path, dpi: int = 150) -> None:
    datasets = sorted(df["dataset"].unique())
    n        = len(datasets)
    fig, axes = plt.subplots(1, 4, figsize=(max(10, n * 1.2) * 4 / 4 + 2, 6), sharey=False)
    fig.suptitle("2D bounding box edge gap (|most extreme − 2nd most extreme|)", fontsize=13, fontweight="bold")

    rng = np.random.default_rng(0)

    for ax, edge in zip(axes, EDGES):
        positions, data_per_dataset, labels = [], [], []
        for pos, dataset in enumerate(datasets, 1):
            vals = df[df["dataset"] == dataset][edge].dropna().values
            positions.append(pos)
            data_per_dataset.append(vals)
            labels.append(f"{dataset}\n(n={len(vals)})")

        violin_idx = [i for i, d in enumerate(data_per_dataset) if len(d) >= 2]
        single_idx = [i for i, d in enumerate(data_per_dataset) if len(d) == 1]

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
            ax.scatter(pos, np.mean(d),   color="red",    zorder=6, s=40, marker="D")
            ax.scatter(pos, np.median(d), color="orange", zorder=6, s=40, marker="o")
            jitter = rng.uniform(-0.08, 0.08, size=len(d))
            ax.scatter(pos + jitter, d, color="black", alpha=0.35, zorder=5, s=12, linewidths=0)
            ax.text(pos, d.max() * 1.02, f"max={d.max():.1f}mm", ha="center", va="bottom",
                    fontsize=6, color="#333", clip_on=False)

        for i in single_idx:
            ax.scatter([positions[i]], data_per_dataset[i], color="black", zorder=5, s=20)

        ax.set_title(EDGE_LABELS[edge], fontsize=11)
        ax.set_ylabel("Gap (mm)", fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    axes[-1].legend(handles=[mpatches.Patch(color="red", label="Mean"),
                              mpatches.Patch(color="orange", label="Median")],
                    loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="2D bounding box edge-gap statistics from GT labels.")
    parser.add_argument("--processed",    required=True, help="processed/<variant>/ directory")
    parser.add_argument("--splits-dir",   default="data/datasplits", help="directory with datasplit_*.yaml files")
    parser.add_argument("--exclude-csv",  default=None, help="CSV with columns dataset,stem to exclude")
    parser.add_argument("--dpi",          type=int, default=150)
    args = parser.parse_args()

    processed = Path(args.processed)
    variant   = processed.name
    out_dir   = Path("processed_stats") / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_map   = load_splits(Path(args.splits_dir)) if Path(args.splits_dir).exists() else {}
    exclude_set  = set()
    if args.exclude_csv:
        exc = pd.read_csv(args.exclude_csv)
        exclude_set = {(r["dataset"], r["stem"]) for _, r in exc.iterrows()}
        print(f"Excluding {len(exclude_set)} subjects from {args.exclude_csv}")

    rows = []
    datasets = sorted(p.name for p in processed.iterdir() if p.is_dir())

    for dataset in tqdm(datasets, desc="datasets"):
        for patient_dir in sorted((processed / dataset).iterdir()):
            if not patient_dir.is_dir():
                continue
            meta_path = patient_dir / "meta.yaml"
            txt_dir   = patient_dir / "txt"
            if not meta_path.exists() or not txt_dir.exists():
                continue

            meta     = yaml.safe_load(meta_path.read_text())
            rl_res   = float(meta["rl_res_mm"])
            ap_res   = float(meta["ap_res_mm"])
            W, H     = int(meta["shape_las"][0]), int(meta["shape_las"][1])

            stem    = patient_dir.name
            if (dataset, stem) in exclude_set:
                continue

            gaps = patient_gaps(txt_dir, rl_res, ap_res, W, H)
            if gaps is None:
                continue
            subject = re.match(r"(sub-[^_]+)", stem)
            subject = subject.group(1) if subject else stem
            split   = splits_map.get((dataset, subject), "unknown")

            rows.append({"dataset": dataset, "stem": stem, "split": split, **gaps})

    df = pd.DataFrame(rows)
    csv_path = out_dir / "2d_box_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} patients → {csv_path}")

    plot_violin(df, out_dir / "violin_2d_box_stats.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
