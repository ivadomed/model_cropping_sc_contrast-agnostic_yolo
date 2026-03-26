#!/usr/bin/env python3
"""
Evaluate predicted 3D bounding boxes against ground truth.

For each patient in predictions/<run-id>/ that has a bbox_3d.txt:
  - reads predictions/<run-id>/<site>/<stem>/volume/bbox_3d.txt  (predicted)
  - reads processed/<site>/<stem>/volume/bbox_3d.txt             (GT)
  - computes: 3D IoU, coverage (fraction of GT covered by pred), per-axis deltas

Outputs:
  - CSV: <out-csv>  (one row per patient)
  - W&B summary (if available)

Usage:
    python scripts/evaluate.py --run-id yolo_spine_v1
    python scripts/evaluate.py --run-id yolo_spine_v1 --out results/eval_v1.csv --no-wandb
"""

import argparse
import csv
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import read_bbox_3d


# ── metrics ───────────────────────────────────────────────────────────────────

def iou_3d(pred: dict, gt: dict) -> float:
    """Volumetric IoU between two bboxes {row1,row2,col1,col2,z1,z2}."""
    inter_r = max(0, min(pred["row2"], gt["row2"]) - max(pred["row1"], gt["row1"]))
    inter_c = max(0, min(pred["col2"], gt["col2"]) - max(pred["col1"], gt["col1"]))
    inter_z = max(0, min(pred["z2"],   gt["z2"])   - max(pred["z1"],   gt["z1"]))
    intersection = inter_r * inter_c * inter_z
    if intersection == 0:
        return 0.0
    vol_pred = (pred["row2"] - pred["row1"]) * (pred["col2"] - pred["col1"]) * (pred["z2"] - pred["z1"])
    vol_gt   = (gt["row2"]   - gt["row1"])   * (gt["col2"]   - gt["col1"])   * (gt["z2"]   - gt["z1"])
    return intersection / (vol_pred + vol_gt - intersection)


def coverage(pred: dict, gt: dict) -> float:
    """Fraction of GT bbox volume that is covered by pred bbox (recall-like)."""
    inter_r = max(0, min(pred["row2"], gt["row2"]) - max(pred["row1"], gt["row1"]))
    inter_c = max(0, min(pred["col2"], gt["col2"]) - max(pred["col1"], gt["col1"]))
    inter_z = max(0, min(pred["z2"],   gt["z2"])   - max(pred["z1"],   gt["z1"]))
    intersection = inter_r * inter_c * inter_z
    vol_gt = (gt["row2"] - gt["row1"]) * (gt["col2"] - gt["col1"]) * (gt["z2"] - gt["z1"])
    return intersection / vol_gt if vol_gt > 0 else 0.0


def axis_deltas(pred: dict, gt: dict) -> dict:
    """Signed delta (pred - gt) on each of the 6 faces (voxels). Negative = pred is inside GT."""
    return {
        "d_row1": pred["row1"] - gt["row1"],
        "d_row2": pred["row2"] - gt["row2"],
        "d_col1": pred["col1"] - gt["col1"],
        "d_col2": pred["col2"] - gt["col2"],
        "d_z1":   pred["z1"]   - gt["z1"],
        "d_z2":   pred["z2"]   - gt["z2"],
    }


# ── per-patient evaluation ────────────────────────────────────────────────────

def evaluate_patient(site: str, stem: str, pred_run_dir: Path, processed_dir: Path) -> dict | None:
    pred_txt = pred_run_dir / site / stem / "volume" / "bbox_3d.txt"
    gt_txt   = processed_dir / site / stem / "volume" / "bbox_3d.txt"

    if not pred_txt.exists():
        return {"site": site, "stem": stem, "status": "no_prediction"}
    if not gt_txt.exists():
        return {"site": site, "stem": stem, "status": "no_gt"}

    pred = read_bbox_3d(pred_txt)
    gt   = read_bbox_3d(gt_txt)

    row = {
        "site": site,
        "stem": stem,
        "status": "ok",
        "iou_3d":    round(iou_3d(pred, gt), 4),
        "coverage":  round(coverage(pred, gt), 4),
    }
    row.update(axis_deltas(pred, gt))
    return row


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted 3D bboxes vs GT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-id",      required=True)
    parser.add_argument("--predictions", default="predictions")
    parser.add_argument("--processed",   default="processed")
    parser.add_argument("--out",         default=None, help="Output CSV path (default: results/<run-id>.csv)")
    parser.add_argument("--no-wandb",    action="store_true")
    parser.add_argument("--wandb-project", default="spine_detection")
    args = parser.parse_args()

    pred_run_dir  = Path(args.predictions) / args.run_id
    processed_dir = Path(args.processed)
    out_csv = Path(args.out) if args.out else Path("results") / f"{args.run_id}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Find all patients that have predictions
    patient_paths = sorted({p.parent.parent for p in pred_run_dir.rglob("volume/bbox_3d.txt")})

    rows = []
    for patient_dir in tqdm(patient_paths, desc="Patients"):
        site = patient_dir.parent.name
        stem = patient_dir.name
        row = evaluate_patient(site, stem, pred_run_dir, processed_dir)
        if row:
            rows.append(row)

    # Write CSV
    fieldnames = ["site", "stem", "status", "iou_3d", "coverage",
                  "d_row1", "d_row2", "d_col1", "d_col2", "d_z1", "d_z2"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    print(f"\nEvaluated: {len(ok_rows)} / {len(rows)} patients")
    if ok_rows:
        import statistics
        ious = [r["iou_3d"] for r in ok_rows]
        covs = [r["coverage"] for r in ok_rows]
        print(f"  IoU 3D  — mean: {statistics.mean(ious):.3f}  median: {statistics.median(ious):.3f}  min: {min(ious):.3f}")
        print(f"  Coverage— mean: {statistics.mean(covs):.3f}  median: {statistics.median(covs):.3f}  min: {min(covs):.3f}")

        if not args.no_wandb:
            import wandb
            wandb.init(project=args.wandb_project, id=args.run_id, resume="allow")
            wandb.summary.update({
                "eval/iou_3d_mean":   statistics.mean(ious),
                "eval/iou_3d_median": statistics.median(ious),
                "eval/coverage_mean": statistics.mean(covs),
                "eval/n_patients":    len(ok_rows),
            })
            wandb.finish()

    print(f"Results: {out_csv}")


if __name__ == "__main__":
    main()
