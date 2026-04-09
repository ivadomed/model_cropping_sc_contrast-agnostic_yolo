#!/usr/bin/env python3
"""
Rank volumes by failure category and create per-dataset failures/ directories.

Requires patients.csv produced by metrics.py.

Three independent failure categories (a volume can appear in multiple):
  fn_dominant  : highest fn_rate  = n_fn / n_gt_slices   (GT slices with no prediction)
  fp_dominant  : highest fp_rate  = n_fp / n_slices       (predictions on non-GT slices)
  low_iou      : lowest iou_gt_mean = mean IoU over all GT slices (FN counted as 0)

Outputs per dataset in <inference>/<dataset>/failures/:
  - failures.csv          : full patient table with all metrics
  - fn_dominant/          : top-K symlinks + fn_failures.csv
  - fp_dominant/          : top-K symlinks + fp_failures.csv
  - low_iou/              : top-K symlinks + low_iou_failures.csv

Symlinks: NNN_<stem> -> ../../<stem>  (relative, pointing to prediction dir)

Usage:
    python scripts/find_failures.py \\
        --inference predictions/yolo26_10mm_aug_320_tassan \\
        [--top-k 20] [--split test]
"""

import argparse
from pathlib import Path

import pandas as pd


def write_category(failures_dir: Path, subdir: str, top: pd.DataFrame, csv_name: str):
    """Create <subdir>/ with ranked symlinks and a CSV."""
    cat_dir = failures_dir / subdir
    cat_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale symlinks
    for p in cat_dir.iterdir():
        if p.is_symlink():
            p.unlink()

    top.to_csv(cat_dir / csv_name, index=False)

    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        target = Path("../..") / row["stem"]   # relative to cat_dir
        link   = cat_dir / f"{rank:03d}_{row['stem']}"
        link.symlink_to(target)


def main():
    parser = argparse.ArgumentParser(
        description="Rank volumes by failure category using patients.csv from metrics.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference", required=True, help="Path to predictions/<run-id>/")
    parser.add_argument("--top-k",     type=int, default=20, help="Number of worst patients per category")
    parser.add_argument("--split",     default=None, choices=["train", "val", "test", "unknown"],
                        help="Restrict to a single split (default: all)")
    args = parser.parse_args()

    pred_root    = Path(args.inference)
    patients_csv = pred_root / "patients.csv"
    assert patients_csv.exists(), f"patients.csv not found at {patients_csv} — run metrics.py first"

    df = pd.read_csv(patients_csv)
    if args.split:
        df = df[df["split"] == args.split]

    for dataset, group in df.groupby("dataset"):
        failures_dir = pred_root / dataset / "failures"
        failures_dir.mkdir(parents=True, exist_ok=True)

        # Remove stale root-level symlinks from previous runs
        for p in failures_dir.iterdir():
            if p.is_symlink():
                p.unlink()

        group.to_csv(failures_dir / "failures.csv", index=False)

        # FN dominant — highest fn_rate
        fn_top = group.dropna(subset=["fn_rate"]).sort_values("fn_rate", ascending=False).head(args.top_k)
        write_category(failures_dir, "fn_dominant", fn_top, "fn_failures.csv")

        # FP dominant — highest fp_rate
        fp_top = group.dropna(subset=["fp_rate"]).sort_values("fp_rate", ascending=False).head(args.top_k)
        write_category(failures_dir, "fp_dominant", fp_top, "fp_failures.csv")

        # Low IoU — lowest iou_gt_mean (GT slices only, FN counted as 0)
        iou_top = group.dropna(subset=["iou_gt_mean"]).sort_values("iou_gt_mean", ascending=True).head(args.top_k)
        write_category(failures_dir, "low_iou", iou_top, "low_iou_failures.csv")

        print(
            f"[{dataset}] failures → {failures_dir}\n"
            f"  fn_dominant : worst {fn_top.iloc[0]['stem']}  fn_rate={fn_top.iloc[0]['fn_rate']:.3f}\n"
            f"  fp_dominant : worst {fp_top.iloc[0]['stem']}  fp_rate={fp_top.iloc[0]['fp_rate']:.3f}\n"
            f"  low_iou     : worst {iou_top.iloc[0]['stem']}  iou_gt_mean={iou_top.iloc[0]['iou_gt_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
