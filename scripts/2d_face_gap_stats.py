#!/usr/bin/env python3
"""
Per-face ordered-gap statistics from GT labels in processed/.

For each of the 6 faces (R, L, A, P, I, S) and k=1..max_k:
  - Collect the face value for every detected slice
  - Sort by extremity (most extreme first)
  - Compute max |v_i - v_{i+hop}| for hop=1..k over all i (cumulative)

Face definitions (axial LAS slices):
  R : (cx + w/2) * W_mm   — sort descending
  L : (cx - w/2) * W_mm   — sort ascending
  A : (cy - h/2) * H_mm   — sort ascending  (cy=0 = Anterior)
  P : (cy + h/2) * H_mm   — sort descending
  S : slice_idx * si_res   — sort ascending  (slice_idx=0 = Superior)
  I : slice_idx * si_res   — sort descending

Outputs (in processed_stats/<variant>/face_gap/):
  face_gap_k{k}.csv         — one row per patient: dataset, stem, split, R/L/A/P/I/S_gap_mm
  violin_face_gap_k{k}.png  — 6-subplot violin, one violin per dataset

Usage:
    python scripts/2d_face_gap_stats.py --processed processed/10mm_SI_1mm_axial --max-k 3
    python scripts/2d_face_gap_stats.py --processed processed/10mm_SI_1mm_axial --max-k 5 --exclude-csv /home/quentinr/bad_gt.csv
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

FACES = ["R_gap_mm", "L_gap_mm", "A_gap_mm", "P_gap_mm", "I_gap_mm", "S_gap_mm"]
FACE_LABELS = {
    "R_gap_mm": "Right",
    "L_gap_mm": "Left",
    "A_gap_mm": "Anterior",
    "P_gap_mm": "Posterior",
    "I_gap_mm": "Inferior",
    "S_gap_mm": "Superior",
}


def load_splits(splits_dir: Path) -> dict:
    mapping = {}
    for f in sorted(splits_dir.glob("datasplit_*.yaml")):
        dataset = re.sub(r"_seed\d+$", "", f.stem[len("datasplit_"):])
        for split_name, subjects in yaml.safe_load(f.read_text()).items():
            for subj in (subjects or []):
                mapping[(dataset, subj)] = split_name
    return mapping


def ordered_gap(values: list[float], descending: bool, k: int) -> float:
    """Max |v_i - v_{i+hop}| for hop=1..k over all i, values sorted by extremity."""
    if len(values) < 2:
        return 0.0
    s = sorted(values, reverse=descending)
    return max(
        abs(s[i] - s[i + hop])
        for hop in range(1, min(k, len(s) - 1) + 1)
        for i in range(len(s) - hop)
    )


def read_patient_faces(txt_dir: Path, W_mm: float, H_mm: float,
                       si_res: float) -> dict[str, list[float]] | None:
    """Collect per-face values across all detected slices. Returns None if < 2 detections."""
    R, L, A, P, SI = [], [], [], [], []

    for txt_file in sorted(txt_dir.glob("slice_*.txt")):
        lines = [l for l in txt_file.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        m = re.search(r"slice_(\d+)\.txt$", txt_file.name)
        if not m:
            continue
        parts = lines[0].split()
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        idx = int(m.group(1))
        R.append((cx + w / 2) * W_mm)
        L.append((cx - w / 2) * W_mm)
        A.append((cy - h / 2) * H_mm)
        P.append((cy + h / 2) * H_mm)
        SI.append(idx * si_res)

    if len(R) < 2:
        return None
    return {"R": R, "L": L, "A": A, "P": P, "SI": SI}


def patient_gaps(faces: dict[str, list[float]], k: int) -> dict:
    return {
        "R_gap_mm": ordered_gap(faces["R"],  descending=True,  k=k),
        "L_gap_mm": ordered_gap(faces["L"],  descending=False, k=k),
        "A_gap_mm": ordered_gap(faces["A"],  descending=False, k=k),
        "P_gap_mm": ordered_gap(faces["P"],  descending=True,  k=k),
        "I_gap_mm": ordered_gap(faces["SI"], descending=True,  k=k),
        "S_gap_mm": ordered_gap(faces["SI"], descending=False, k=k),
    }


def plot_violin(df: pd.DataFrame, out_path: Path, k: int, dpi: int) -> None:
    datasets  = sorted(df["dataset"].unique())
    n         = len(datasets)
    fig, axes = plt.subplots(2, 3, figsize=(max(10, n * 1.2) + 2, 10), sharey=False)
    axes      = axes.flatten()
    fig.suptitle(f"Per-face ordered gap (hops 1..{k})", fontsize=13, fontweight="bold")

    rng = np.random.default_rng(0)
    for ax, face in zip(axes, FACES):
        data_per_dataset = [df[df["dataset"] == d][face].dropna().values for d in datasets]
        positions        = list(range(1, n + 1))
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
            ax.vlines(pos, np.percentile(d, 25), np.percentile(d, 75),
                      color="white", linewidth=2.5, zorder=4)
            ax.scatter(pos, np.mean(d),   color="red",    zorder=6, s=40, marker="D")
            ax.scatter(pos, np.median(d), color="orange", zorder=6, s=40, marker="o")
            ax.scatter(pos + rng.uniform(-0.08, 0.08, len(d)), d,
                       color="black", alpha=0.35, zorder=5, s=12, linewidths=0)
            ax.text(pos, d.max() * 1.02, f"max={d.max():.1f}mm",
                    ha="center", va="bottom", fontsize=6, color="#333", clip_on=False)
        for i in single_idx:
            ax.scatter([positions[i]], data_per_dataset[i], color="black", zorder=5, s=20)

        ax.set_title(FACE_LABELS[face], fontsize=11)
        ax.set_ylabel("Gap (mm)", fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{d}\n(n={len(df[df['dataset']==d][face].dropna())})"
                            for d in datasets], rotation=30, ha="right", fontsize=7)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    axes[-1].legend(handles=[mpatches.Patch(color="red", label="Mean"),
                              mpatches.Patch(color="orange", label="Median")],
                    loc="upper right", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-face ordered-gap statistics from GT labels.")
    parser.add_argument("--processed",   required=True)
    parser.add_argument("--splits-dir",  default="data/datasplits")
    parser.add_argument("--exclude-csv", default="/home/quentinr/bad_gt.csv")
    parser.add_argument("--max-k",       type=int, required=True, dest="max_k",
                        help="compute for k=1..max_k")
    parser.add_argument("--dpi",         type=int, default=150)
    args = parser.parse_args()

    processed = Path(args.processed)
    out_dir   = Path("processed_stats") / processed.name / "face_gap"
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
            meta  = yaml.safe_load(meta_path.read_text())
            W_mm  = int(meta["shape_las"][0]) * float(meta["rl_res_mm"])
            H_mm  = int(meta["shape_las"][1]) * float(meta["ap_res_mm"])
            faces = read_patient_faces(txt_dir, W_mm, H_mm, float(meta["si_res_mm"]))
            if faces is None:
                continue
            subject = re.match(r"(sub-[^_]+)", stem)
            subject = subject.group(1) if subject else stem
            patient_records.append({
                "dataset": dataset, "stem": stem,
                "split":   splits_map.get((dataset, subject), "unknown"),
                "faces":   faces,
            })

    for k in range(1, args.max_k + 1):
        rows = []
        for rec in patient_records:
            rows.append({"dataset": rec["dataset"], "stem": rec["stem"],
                         "split": rec["split"], **patient_gaps(rec["faces"], k)})

        df       = pd.DataFrame(rows)
        csv_path = out_dir / f"face_gap_k{k}.csv"
        df.to_csv(csv_path, index=False)
        print(f"k={k}: saved {len(df)} patients -> {csv_path}")
        plot_violin(df, out_dir / f"violin_face_gap_k{k}.png", k=k, dpi=args.dpi)


if __name__ == "__main__":
    main()
