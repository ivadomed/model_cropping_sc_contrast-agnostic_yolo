#!/usr/bin/env python3
"""
Find worst-performing volumes per dataset for iou_gt_mean and iou_all_mean.

Requires patients.csv (index) and per-patient metrics/patient.csv from metrics.py.
Split assignment resolved at runtime from --splits-dir YAMLs.

Output structure:
  predictions/<run_id>/failures/<split>/<metric>/<dataset>/
    ranking.csv            ← top-K worst volumes with metric value
    001_<stem>             ← symlink → ../../../../<dataset>/<stem>

Usage:
    python scripts/find_failures.py --inference predictions/yolo26_1mm_axial_v2
    python scripts/find_failures.py --inference predictions/yolo26_1mm_axial_v2 --conf 0.1
    python scripts/find_failures.py --inference predictions/yolo26_1mm_axial_v2 \\
        --splits val train test --top-k 10
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

METRICS = {
    "iou_gt_mean":  True,   # ascending=True → lowest (worst) first
    "iou_all_mean": True,
}


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
                           conf_thresh: float, split: str) -> pd.DataFrame:
    """Build per-patient DataFrame by reading patient.csv at the given conf threshold."""
    rows = []
    for _, row in patients_idx.iterrows():
        dataset, stem = row["dataset"], row["stem"]
        m       = re.match(r"(sub-[^_]+)", stem)
        subject = m.group(1) if m else stem
        if splits_map.get((dataset, subject), "unknown") != split:
            continue
        patient_csv = pred_root / dataset / stem / "metrics" / "patient.csv"
        if not patient_csv.exists():
            continue
        df      = pd.read_csv(patient_csv)
        matched = df[np.isclose(df["conf_thresh"], conf_thresh, atol=0.0005)]
        if matched.empty:
            continue
        rows.append({"dataset": dataset, "stem": stem, **matched.iloc[0].to_dict()})
    return pd.DataFrame(rows)


def write_failures(out_dir: Path, top: pd.DataFrame, metric: str) -> None:
    """Create ranking.csv + symlinks for one (split, metric, dataset) group."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.iterdir():
        if p.is_symlink():
            p.unlink()
    top[["stem", metric]].to_csv(out_dir / "ranking.csv", index=False)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        link = out_dir / f"{rank:03d}_{row['stem']}"
        # out_dir: pred_root/failures/<split>/<metric>/<dataset>/
        # target:  pred_root/<dataset>/<stem>  →  ../../../../<dataset>/<stem>
        link.symlink_to(Path("../../../../") / row["dataset"] / row["stem"])


def main():
    parser = argparse.ArgumentParser(
        description="Top-K worst volumes per dataset for iou_gt_mean and iou_all_mean",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference",  required=True, help="Path to predictions/<run-id>/")
    parser.add_argument("--splits-dir", default="data/datasplits/from_raw")
    parser.add_argument("--splits",     nargs="+", default=["val", "train", "test"],
                        choices=["train", "val", "test", "unknown"])
    parser.add_argument("--conf",       type=float, default=0.001,
                        help="Confidence threshold")
    parser.add_argument("--top-k",      type=int, default=10)
    args = parser.parse_args()

    pred_root    = Path(args.inference)
    splits_map   = load_splits(Path(args.splits_dir))
    patients_idx = pd.read_csv(pred_root / "patients.csv")

    for split in args.splits:
        df = load_patients_at_conf(pred_root, patients_idx, splits_map, args.conf, split)
        if df.empty:
            print(f"  [{split}] no data at conf={args.conf}")
            continue
        for metric, ascending in METRICS.items():
            for dataset, group in df.groupby("dataset"):
                top = group.dropna(subset=[metric]).sort_values(metric, ascending=ascending).head(args.top_k)
                if top.empty:
                    continue
                write_failures(pred_root / "failures" / split / metric / dataset, top, metric)
        print(f"  [{split}] conf={args.conf} → done")


if __name__ == "__main__":
    main()
