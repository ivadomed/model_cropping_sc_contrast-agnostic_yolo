#!/usr/bin/env python3
"""
2D bounding box statistics from GT labels in processed/.

Two independent modes, selected by the argument used:

  --rank R   (edge-gap mode)
    For each patient and each of the 4 in-plane edges (R, L, A, P):
      |most_extreme_edge - R-th_most_extreme_edge| in mm.
    Output: 2d_box_stats_rank{R}.csv  /  violin_2d_box_stats_rank{R}.png

  --max-k K  (adjacent-gap mode)
    For each patient and k=1..K:
      max distance (mm) between any two detected slices up to k hops apart
      along the SI axis (all hops 1..k are included, cumulative).
    Output: 2d_box_stats_adjgap_k{k}.csv  /  violin_2d_box_stats_adjgap_k{k}.png
      (one CSV and one plot per k value)

Edge convention (axial LAS slices, rows=AP col=RL):
  R : cx + w/2  (rightmost, mm from left),  L : cx - w/2  (leftmost, ascending sort)
  A : cy - h/2  (most anterior, ascending), P : cy + h/2  (most posterior)
  W = shape_las[0] * rl_res_mm,  H = shape_las[1] * ap_res_mm

Usage:
    python scripts/2d_box_stats.py --processed processed/10mm_SI_1mm_axial --rank 2
    python scripts/2d_box_stats.py --processed processed/10mm_SI_1mm_axial --rank 3
    python scripts/2d_box_stats.py --processed processed/10mm_SI_1mm_axial --max-k 3
    python scripts/2d_box_stats.py --processed processed/10mm_SI_1mm_axial --max-k 3 --exclude-csv bad_gt.csv
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


# ─── Edge-gap mode ────────────────────────────────────────────────────────────

def edge_gap(values: list[float], ascending: bool, rank: int) -> float:
    """Gap between most extreme value and the rank-th most extreme (rank=2 → 2nd, …)."""
    if len(values) < rank:
        return 0.0
    s = sorted(values, reverse=not ascending)
    return abs(s[0] - s[rank - 1])


def patient_edge_gaps(txt_dir: Path, rl_res: float, ap_res: float,
                      W: int, H: int, rank: int) -> dict | None:
    """Compute the 4 edge gaps for one patient. Returns None if fewer than rank detections."""
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

    if len(R_vals) < rank:
        return None

    return {
        "R_gap_mm": edge_gap(R_vals, ascending=False, rank=rank),
        "L_gap_mm": edge_gap(L_vals, ascending=True,  rank=rank),
        "A_gap_mm": edge_gap(A_vals, ascending=True,  rank=rank),
        "P_gap_mm": edge_gap(P_vals, ascending=False, rank=rank),
    }


def plot_edge_violin(df: pd.DataFrame, out_path: Path, rank: int, dpi: int = 150) -> None:
    datasets = sorted(df["dataset"].unique())
    n        = len(datasets)
    fig, axes = plt.subplots(1, 4, figsize=(max(10, n * 1.2) * 4 / 4 + 2, 6), sharey=False)
    ordinal   = {2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")
    fig.suptitle(f"2D edge gap (|most extreme - {ordinal} most extreme|)",
                 fontsize=13, fontweight="bold")

    rng = np.random.default_rng(0)
    for ax, edge in zip(axes, EDGES):
        _draw_violin_ax(ax, df, edge, datasets, rng, ylabel="Gap (mm)",
                        title=EDGE_LABELS[edge])

    axes[-1].legend(handles=[mpatches.Patch(color="red", label="Mean"),
                              mpatches.Patch(color="orange", label="Median")],
                    loc="upper right", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ─── Adjacent-gap mode ────────────────────────────────────────────────────────

def patient_adj_gap(txt_dir: Path, si_res: float, k: int) -> float | None:
    """Max SI distance (mm) between detected slices up to k hops apart.

    Returns None if fewer than 2 detections.
    Slice index extracted from filename slice_NNN.txt; distance = |idx_j - idx_i| * si_res.
    """
    indices = []
    for txt_file in sorted(txt_dir.glob("slice_*.txt")):
        if any(l.strip() for l in txt_file.read_text().splitlines()):
            m = re.search(r"slice_(\d+)\.txt$", txt_file.name)
            if m:
                indices.append(int(m.group(1)))

    if len(indices) < 2:
        return None

    indices.sort()
    return max(
        (indices[i + hop] - indices[i]) * si_res
        for hop in range(1, k + 1)
        for i in range(len(indices) - hop)
    )


def plot_adjgap_violin(df: pd.DataFrame, out_path: Path, k: int, dpi: int = 150) -> None:
    datasets = sorted(df["dataset"].unique())
    fig, ax  = plt.subplots(figsize=(max(8, len(datasets) * 1.4), 6))
    fig.suptitle(f"Max SI gap between detected slices (hops 1..{k})",
                 fontsize=13, fontweight="bold")

    rng = np.random.default_rng(0)
    _draw_violin_ax(ax, df, "adj_gap_mm", datasets, rng, ylabel="Gap (mm)", title="")

    ax.legend(handles=[mpatches.Patch(color="red", label="Mean"),
                        mpatches.Patch(color="orange", label="Median")],
              loc="upper right", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ─── Shared violin drawing ─────────────────────────────────────────────────────

def _draw_violin_ax(ax, df: pd.DataFrame, col: str, datasets: list,
                    rng, ylabel: str, title: str) -> None:
    positions, data_per_dataset, labels = [], [], []
    for pos, dataset in enumerate(datasets, 1):
        vals = df[df["dataset"] == dataset][col].dropna().values
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

    if title:
        ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="2D bounding box statistics from GT labels.")
    parser.add_argument("--processed",   required=True, help="processed/<variant>/ directory")
    parser.add_argument("--splits-dir",  default="data/datasplits")
    parser.add_argument("--exclude-csv", default=None, help="CSV with columns dataset,stem to exclude")
    parser.add_argument("--dpi",         type=int, default=150)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rank",  type=int,
                      help="edge-gap mode: compare most extreme edge vs rank-th most extreme (e.g. 2)")
    mode.add_argument("--max-k", type=int, dest="max_k",
                      help="adj-gap mode: max k hops; produces outputs for k=1..K")
    args = parser.parse_args()

    processed  = Path(args.processed)
    variant    = processed.name
    out_dir    = Path("processed_stats") / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_map  = load_splits(Path(args.splits_dir)) if Path(args.splits_dir).exists() else {}
    exclude_set = set()
    if args.exclude_csv:
        exc = pd.read_csv(args.exclude_csv)
        exclude_set = {(r["dataset"], r["stem"]) for _, r in exc.iterrows()}
        print(f"Excluding {len(exclude_set)} subjects from {args.exclude_csv}")

    datasets = sorted(p.name for p in processed.iterdir() if p.is_dir())

    # ── Edge-gap mode ──
    if args.rank:
        rank = args.rank
        rows = []
        for dataset in tqdm(datasets, desc="datasets"):
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
                meta   = yaml.safe_load(meta_path.read_text())
                gaps   = patient_edge_gaps(txt_dir, float(meta["rl_res_mm"]),
                                           float(meta["ap_res_mm"]),
                                           int(meta["shape_las"][0]),
                                           int(meta["shape_las"][1]), rank)
                if gaps is None:
                    continue
                subject = re.match(r"(sub-[^_]+)", stem)
                subject = subject.group(1) if subject else stem
                rows.append({"dataset": dataset, "stem": stem,
                             "split": splits_map.get((dataset, subject), "unknown"), **gaps})

        df       = pd.DataFrame(rows)
        suffix   = f"_rank{rank}"
        csv_path = out_dir / f"2d_box_stats{suffix}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} patients -> {csv_path}")
        plot_edge_violin(df, out_dir / f"violin_2d_box_stats{suffix}.png", rank=rank, dpi=args.dpi)

    # ── Adjacent-gap mode ──
    else:
        max_k = args.max_k

        # Collect all patients once, then compute per k
        patient_records = []
        for dataset in tqdm(datasets, desc="datasets"):
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
                meta    = yaml.safe_load(meta_path.read_text())
                si_res  = float(meta["si_res_mm"])
                subject = re.match(r"(sub-[^_]+)", stem)
                subject = subject.group(1) if subject else stem
                split   = splits_map.get((dataset, subject), "unknown")
                patient_records.append((dataset, stem, split, txt_dir, si_res))

        for k in range(1, max_k + 1):
            rows = []
            for dataset, stem, split, txt_dir, si_res in patient_records:
                gap = patient_adj_gap(txt_dir, si_res, k)
                if gap is None:
                    continue
                rows.append({"dataset": dataset, "stem": stem, "split": split, "adj_gap_mm": gap})

            df       = pd.DataFrame(rows)
            suffix   = f"_adjgap_k{k}"
            csv_path = out_dir / f"2d_box_stats{suffix}.csv"
            df.to_csv(csv_path, index=False)
            print(f"k={k}: saved {len(df)} patients -> {csv_path}")
            plot_adjgap_violin(df, out_dir / f"violin_2d_box_stats{suffix}.png", k=k, dpi=args.dpi)


if __name__ == "__main__":
    main()
