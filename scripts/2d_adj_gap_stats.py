#!/usr/bin/env python3
"""
3D Cartesian adjacent-gap statistics from GT labels in processed/.

For each patient and k=1..max_k:
  max 3D Cartesian distance (mm) between the centers of detected bounding boxes
  up to k hops apart along the SI axis (hops 1..k all included, cumulative).

  Center of a detection in mm (LAS space):
    RL : cx * W_mm   (W_mm = shape_las[0] * rl_res_mm)
    AP : cy * H_mm   (H_mm = shape_las[1] * ap_res_mm)
    SI : slice_idx * si_res_mm

  distance = sqrt(dRL² + dAP² + dSI²)

  k=1 : only adjacent slices (hop=1)
  k=2 : pairs 1 and 2 hops apart
  k=3 : pairs 1, 2 and 3 hops apart
  ...

Outputs (in processed_stats/<variant>/adj_gap/):
  adj_gap_k{k}.csv         — one row per patient: dataset, stem, split, adj_gap_mm
  violin_adj_gap_k{k}.png  — one violin per dataset

Usage:
    python scripts/2d_adj_gap_stats.py --processed processed/10mm_SI_1mm_axial --max-k 3
    python scripts/2d_adj_gap_stats.py --processed processed/10mm_SI_1mm_axial --max-k 5 --exclude-csv /home/quentinr/bad_gt.csv
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


def load_splits(splits_dir: Path) -> dict:
    mapping = {}
    for f in sorted(splits_dir.glob("datasplit_*.yaml")):
        dataset = re.sub(r"_seed\d+$", "", f.stem[len("datasplit_"):])
        for split_name, subjects in yaml.safe_load(f.read_text()).items():
            for subj in (subjects or []):
                mapping[(dataset, subj)] = split_name
    return mapping


def read_detections(txt_dir: Path, W_mm: float, H_mm: float,
                    si_res: float) -> list[tuple[float, float, float]]:
    """Return list of (rl_mm, ap_mm, si_mm) for each detected slice, sorted by SI."""
    detections = []
    for txt_file in sorted(txt_dir.glob("slice_*.txt")):
        lines = [l for l in txt_file.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        m = re.search(r"slice_(\d+)\.txt$", txt_file.name)
        if not m:
            continue
        parts = lines[0].split()
        cx, cy = float(parts[1]), float(parts[2])
        si_mm  = int(m.group(1)) * si_res
        detections.append((cx * W_mm, cy * H_mm, si_mm))
    detections.sort(key=lambda d: d[2])
    return detections


def patient_adj_gap(detections: list[tuple[float, float, float]], k: int) -> float | None:
    """Max 3D Cartesian distance between bbox centers up to k hops apart.

    Returns None if fewer than 2 detections.
    """
    if len(detections) < 2:
        return None
    return max(
        np.sqrt((detections[i + hop][0] - detections[i][0]) ** 2
              + (detections[i + hop][1] - detections[i][1]) ** 2
              + (detections[i + hop][2] - detections[i][2]) ** 2)
        for hop in range(1, k + 1)
        for i in range(len(detections) - hop)
    )


def plot_violin(df: pd.DataFrame, out_path: Path, k: int, dpi: int) -> None:
    datasets         = sorted(df["dataset"].unique())
    n                = len(datasets)
    fig, ax          = plt.subplots(figsize=(max(8, n * 1.4), 6))
    fig.suptitle(f"Max 3D Cartesian gap between bbox centers (hops 1..{k})",
                 fontsize=13, fontweight="bold")

    rng              = np.random.default_rng(0)
    positions        = list(range(1, n + 1))
    data_per_dataset = [df[df["dataset"] == d]["adj_gap_mm"].dropna().values for d in datasets]
    violin_idx       = [i for i, d in enumerate(data_per_dataset) if len(d) >= 2]
    single_idx       = [i for i, d in enumerate(data_per_dataset) if len(d) == 1]

    if violin_idx:
        parts = ax.violinplot([data_per_dataset[i] for i in violin_idx],
                              positions=[positions[i] for i in violin_idx],
                              showmeans=False, showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0"); pc.set_edgecolor("#2a4a80"); pc.set_alpha(0.75)

    for i in violin_idx:
        d, pos = data_per_dataset[i], positions[i]
        ax.vlines(pos, np.percentile(d, 25), np.percentile(d, 75), color="white", linewidth=2.5, zorder=4)
        ax.scatter(pos, np.mean(d),   color="red",    zorder=6, s=40, marker="D")
        ax.scatter(pos, np.median(d), color="orange", zorder=6, s=40, marker="o")
        ax.scatter(pos + rng.uniform(-0.08, 0.08, len(d)), d,
                   color="black", alpha=0.35, zorder=5, s=12, linewidths=0)
        ax.text(pos, d.max() * 1.02, f"max={d.max():.1f}mm",
                ha="center", va="bottom", fontsize=6, color="#333", clip_on=False)
    for i in single_idx:
        ax.scatter([positions[i]], data_per_dataset[i], color="black", zorder=5, s=20)

    ax.set_ylabel("3D Cartesian distance (mm)", fontsize=10)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{d}\n(n={len(df[df['dataset']==d]['adj_gap_mm'].dropna())})"
                        for d in datasets], rotation=30, ha="right", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=[mpatches.Patch(color="red", label="Mean"),
                        mpatches.Patch(color="orange", label="Median")],
              loc="upper right", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="3D Cartesian adjacent-gap statistics from GT labels.")
    parser.add_argument("--processed",   required=True)
    parser.add_argument("--splits-dir",  default="data/datasplits")
    parser.add_argument("--exclude-csv", default="/home/quentinr/bad_gt.csv")
    parser.add_argument("--max-k",       type=int, required=True, dest="max_k",
                        help="compute for k=1..max_k hops")
    parser.add_argument("--dpi",         type=int, default=150)
    args = parser.parse_args()

    processed = Path(args.processed)
    out_dir   = Path("processed_stats") / processed.name / "adj_gap"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_map  = load_splits(Path(args.splits_dir)) if Path(args.splits_dir).exists() else {}
    exclude_set = set()
    if args.exclude_csv and Path(args.exclude_csv).exists():
        exc = pd.read_csv(args.exclude_csv)
        exclude_set = {(r["dataset"], r["stem"]) for _, r in exc.iterrows()}
        print(f"Excluding {len(exclude_set)} subjects from {args.exclude_csv}")

    # Read all patients once
    patient_records = []
    for dataset in tqdm(sorted(p.name for p in processed.iterdir() if p.is_dir()), desc="datasets"):
        for patient_dir in sorted((processed / dataset).iterdir()):
            if not patient_dir.is_dir():
                continue
            meta_path = patient_dir / "meta.yaml"
            txt_dir   = patient_dir / "txt"
            if not meta_path.exists() or not txt_dir.exists():
                continue
            stem = patient_dir.name
            if (dataset, stem) in exclude_set:
                continue
            meta       = yaml.safe_load(meta_path.read_text())
            W_mm       = int(meta["shape_las"][0]) * float(meta["rl_res_mm"])
            H_mm       = int(meta["shape_las"][1]) * float(meta["ap_res_mm"])
            si_res     = float(meta["si_res_mm"])
            detections = read_detections(txt_dir, W_mm, H_mm, si_res)
            if len(detections) < 2:
                continue
            subject = re.match(r"(sub-[^_]+)", stem)
            subject = subject.group(1) if subject else stem
            patient_records.append({
                "dataset":    dataset,
                "stem":       stem,
                "split":      splits_map.get((dataset, subject), "unknown"),
                "detections": detections,
            })

    for k in range(1, args.max_k + 1):
        rows = []
        for rec in patient_records:
            gap = patient_adj_gap(rec["detections"], k)
            if gap is None:
                continue
            rows.append({"dataset": rec["dataset"], "stem": rec["stem"],
                         "split": rec["split"], "adj_gap_mm": gap})

        df       = pd.DataFrame(rows)
        csv_path = out_dir / f"adj_gap_k{k}.csv"
        df.to_csv(csv_path, index=False)
        print(f"k={k}: saved {len(df)} patients -> {csv_path}")
        plot_violin(df, out_dir / f"violin_adj_gap_k{k}.png", k=k, dpi=args.dpi)


if __name__ == "__main__":
    main()
