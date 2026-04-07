#!/usr/bin/env python3
"""
Detect bad YOLO predictions by comparing each prediction to the nearest GT bbox.

For slices where GT exists: IoU computed directly against same-slice GT.
For slices without GT (cord extremities): nearest slice with GT is used as reference.
This catches predictions that are spatially inconsistent with the cord trajectory,
even on slices where no GT annotation was made.

Only slices with a prediction are reported. Results sorted by iou_nearest_gt ascending
(worst detections first).

Reads predictions from predictions/<run-id>/ (output of evaluate.py).
Reads GT from processed/.

Usage:
    python scripts/find_failures.py \
        --inference predictions/yolo26_10mm_aug_320_tassan \
        --processed processed_10mm_SI \
        [--iou-thresh 0.5] [--splits-dir data/datasplits]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

IOU_THRESH = 0.5  # default threshold below which a prediction is flagged as failure


def load_splits(splits_dir: Path) -> dict:
    """Returns {(dataset, subject): split_name} from all datasplit_*.yaml."""
    mapping = {}
    for f in sorted(splits_dir.glob("datasplit_*.yaml")):
        dataset = re.sub(r"_seed\d+$", "", f.stem[len("datasplit_"):])
        for split_name, subjects in yaml.safe_load(f.read_text()).items():
            for subj in (subjects or []):
                mapping[(dataset, subj)] = split_name
    return mapping


def read_box(txt_path: Path):
    """Returns (cx,cy,w,h) or None if file empty/missing. Handles optional conf field."""
    if not txt_path.exists():
        return None
    content = txt_path.read_text().strip()
    if not content:
        return None
    p = content.split()
    return (float(p[1]), float(p[2]), float(p[3]), float(p[4]))


def read_conf(txt_path: Path) -> float:
    """Returns confidence from pred txt (field 5), or 1.0 if absent."""
    if not txt_path.exists():
        return 0.0
    content = txt_path.read_text().strip()
    if not content:
        return 0.0
    p = content.split()
    return float(p[5]) if len(p) > 5 else 1.0


def bbox_iou(a, b) -> float:
    """IoU between two (cx,cy,w,h) normalised bboxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2]/2, a[1] - a[3]/2, a[0] + a[2]/2, a[1] + a[3]/2
    bx1, by1, bx2, by2 = b[0] - b[2]/2, b[1] - b[3]/2, b[0] + b[2]/2, b[1] + b[3]/2
    inter = max(0., min(ax2, bx2) - max(ax1, bx1)) * max(0., min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1)*(ay2 - ay1) + (bx2 - bx1)*(by2 - by1) - inter
    return inter / union if union > 0 else 0.


def process_patient(gt_txt_dir: Path, pred_txt_dir: Path) -> list:
    """
    For each predicted slice, find the nearest GT slice and compute IoU.
    Returns list of dicts with slice_idx, z_dist_to_ref_gt, ref_gt_slice, iou_nearest_gt.
    """
    # Build z → GT box mapping for this patient
    gt_by_z = {}
    for txt in sorted(gt_txt_dir.glob("slice_*.txt")):
        z = int(txt.stem.split("_")[1])
        box = read_box(txt)
        if box is not None:
            gt_by_z[z] = box

    if not gt_by_z:
        return []  # no GT at all for this patient — skip

    gt_slices = np.array(sorted(gt_by_z.keys()))
    records = []

    for txt in sorted(pred_txt_dir.glob("slice_*.txt")):
        pred_box = read_box(txt)
        if pred_box is None:
            continue  # no prediction on this slice — not a failure
        z = int(txt.stem.split("_")[1])
        pred_conf = read_conf(txt)

        # Find nearest GT slice
        dists = np.abs(gt_slices - z)
        nearest_z = int(gt_slices[dists.argmin()])
        z_dist = int(dists.min())
        ref_gt_box = gt_by_z[nearest_z]

        iou = bbox_iou(pred_box, ref_gt_box)
        records.append({
            "slice_idx":         z,
            "z_dist_to_ref_gt":  z_dist,
            "ref_gt_slice":      nearest_z,
            "iou_nearest_gt":    round(iou, 4),
            "pred_conf":         round(pred_conf, 4),
        })

    return records


def collect_failures(processed_dir: Path, pred_root: Path, splits_map: dict) -> list:
    rows = []
    for dataset_dir in sorted(processed_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for patient_dir in sorted(dataset_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            gt_txt_dir = patient_dir / "txt"
            if not gt_txt_dir.is_dir():
                continue
            stem = patient_dir.name
            m = re.match(r"(sub-[^_]+)_?(.*)", stem)
            subject = m.group(1)
            contrast = m.group(2) or "default"
            split = splits_map.get((dataset, subject), "unknown")

            pred_txt_dir = pred_root / dataset / stem / "txt"
            if not pred_txt_dir.is_dir():
                continue  # no predictions for this patient

            for rec in process_patient(gt_txt_dir, pred_txt_dir):
                rows.append({
                    "dataset":          dataset,
                    "subject":          subject,
                    "contrast":         contrast,
                    "split":            split,
                    **rec,
                })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Find worst YOLO predictions using nearest-GT IoU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference",   required=True, help="Path to inference run directory (predictions/<run-id>/)")
    parser.add_argument("--processed",   default="processed")
    parser.add_argument("--splits-dir",  default="data/datasplits")
    parser.add_argument("--iou-thresh",  type=float, default=IOU_THRESH,
                        help="IoU below this value is flagged as failure")
    parser.add_argument("--split",       default=None, choices=["train", "val", "test"],
                        help="Restrict to a single split (default: all)")
    args = parser.parse_args()

    pred_root  = Path(args.inference)
    splits_map = load_splits(Path(args.splits_dir))
    rows       = collect_failures(Path(args.processed), pred_root, splits_map)

    if args.split:
        rows = [r for r in rows if r["split"] == args.split]

    df = pd.DataFrame(rows).sort_values("iou_nearest_gt", ascending=True)

    n_total    = len(df)
    n_failures = int((df["iou_nearest_gt"] < args.iou_thresh).sum())
    print(f"{n_total} predicted slices — {n_failures} with IoU < {args.iou_thresh} "
          f"({100*n_failures/n_total:.1f}%)" if n_total else "No predicted slices found.")

    out_csv = pred_root / "failures.csv"
    df.to_csv(out_csv, index=False)
    print(f"Failures (nearest-GT) → {out_csv}")

    if n_failures:
        print("\nWorst 10 (nearest-GT):")
        print(df.head(10).to_string(index=False))

    # Matched failures: slices where both GT and pred exist but IoU is low
    # Read from slices.csv which already has has_gt, has_pred, iou
    matched_rows = []
    for slices_csv in sorted(pred_root.rglob("metrics/slices.csv")):
        patient_dir = slices_csv.parent.parent
        dataset     = patient_dir.parent.name
        stem        = patient_dir.name
        m_          = re.match(r"(sub-[^_]+)_?(.*)", stem)
        subject     = m_.group(1) if m_ else stem
        contrast    = m_.group(2) if m_ and m_.group(2) else "default"
        split       = splits_map.get((dataset, subject), "unknown")
        if args.split and split != args.split:
            continue
        sub = pd.read_csv(slices_csv)
        sub = sub[sub["has_gt"] & sub["has_pred"]][["slice_idx", "iou", "pred_conf"]].copy()
        if sub.empty:
            continue
        sub["dataset"]  = dataset
        sub["subject"]  = subject
        sub["contrast"] = contrast
        sub["split"]    = split
        matched_rows.append(sub)

    if matched_rows:
        df_matched   = pd.concat(matched_rows, ignore_index=True).sort_values("iou", ascending=True)
        n_matched    = len(df_matched)
        n_fail_match = int((df_matched["iou"] < args.iou_thresh).sum())
        print(f"\n{n_matched} matched slices (GT ∩ pred) — "
              f"{n_fail_match} with IoU < {args.iou_thresh} ({100*n_fail_match/n_matched:.1f}%)")
        out_matched = pred_root / "failures_matched.csv"
        df_matched.to_csv(out_matched, index=False)
        print(f"Matched failures → {out_matched}")
        if n_fail_match:
            print("\nWorst 10 (matched):")
            print(df_matched.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
