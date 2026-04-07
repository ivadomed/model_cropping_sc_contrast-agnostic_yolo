#!/usr/bin/env python3
"""
Collect the N worst matched failures and symlink their overlay PNGs into a single folder.

Reads failures_matched.csv (produced by find_failures.py) and links the overlay images
(GT green + pred red) from predictions/<run_id>/<dataset>/<patient>/png/slice_NNN.png
into a flat output directory, named by rank for easy browsing.

Usage:
    python scripts/show_failures.py \
        --inference predictions/yolo26_10mm_aug_320_tassan \
        [--n 10] [--out-dir predictions/yolo26_10mm_aug_320_tassan/top_failures]
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def patient_stem(subject: str, contrast: str) -> str:
    return subject if contrast == "default" else f"{subject}_{contrast}"


def main():
    parser = argparse.ArgumentParser(
        description="Symlink overlay PNGs of the N worst matched failures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference", required=True, help="Path to inference run directory")
    parser.add_argument("--n",         type=int, default=10, help="Number of failures to collect")
    parser.add_argument("--out-dir",   default=None, help="Output directory (default: <inference>/top_failures)")
    args = parser.parse_args()

    pred_root = Path(args.inference)
    csv_path  = pred_root / "failures_matched.csv"
    out_dir   = Path(args.out_dir) if args.out_dir else pred_root / "top_failures"

    df = pd.read_csv(csv_path).head(args.n)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale symlinks from a previous run
    for existing in sorted(out_dir.glob("*.png")):
        existing.unlink()

    created = 0
    for rank, row in enumerate(df.itertuples(), start=1):
        stem    = patient_stem(row.subject, row.contrast)
        src     = pred_root / row.dataset / stem / "png" / f"slice_{int(row.slice_idx):03d}.png"
        if not src.exists():
            print(f"  [WARN] missing: {src}")
            continue
        name    = f"{rank:03d}_{row.dataset}_{stem}_slice{int(row.slice_idx):03d}_iou{row.iou:.3f}.png"
        dst     = out_dir / name
        os.symlink(src.resolve(), dst)
        created += 1

    print(f"{created}/{args.n} symlinks → {out_dir}")


if __name__ == "__main__":
    main()
