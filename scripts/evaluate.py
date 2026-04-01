#!/usr/bin/env python3
"""
Evaluate YOLO 2D bbox predictions: IoU, Dice, AP50, AP50:95.

Runs inference directly from processed/ slices (conf=0.001 to capture full PR curve).
Reports broken down by: global / split / dataset / dataset×contrast / dataset×contrast×split.

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/yolo_spine_v1/weights/best.pt \
        --run-id yolo_spine_v1 \
        --processed processed_10mm_SI \
        --splits-dir data/datasplits
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))


def load_splits(splits_dir: Path) -> dict:
    """Returns {(dataset, subject): split_name} from all datasplit_*.yaml."""
    mapping = {}
    for f in sorted(splits_dir.glob("datasplit_*.yaml")):
        dataset = re.sub(r"_seed\d+$", "", f.stem[len("datasplit_"):])
        for split_name, subjects in yaml.safe_load(f.read_text()).items():
            for subj in (subjects or []):
                mapping[(dataset, subj)] = split_name
    return mapping


def bbox_iou(a, b) -> float:
    """IoU between two (cx,cy,w,h) normalised bboxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def read_gt_box(txt_path: Path):
    """Returns (cx,cy,w,h) or None if file empty/missing."""
    if not txt_path.exists():
        return None
    content = txt_path.read_text().strip()
    if not content:
        return None
    p = content.split()
    return (float(p[1]), float(p[2]), float(p[3]), float(p[4]))


def collect_gt_records(processed_dir: Path, splits_map: dict) -> list:
    """Walk processed/ and return one record per PNG slice with GT info."""
    records = []
    for dataset_dir in sorted(processed_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for patient_dir in sorted(dataset_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            stem = patient_dir.name
            m = re.match(r"(sub-[^_]+)_?(.*)", stem)
            subject = m.group(1)
            contrast = m.group(2) or "default"
            split = splits_map.get((dataset, subject), "unknown")

            png_dir = patient_dir / "png"
            txt_dir = patient_dir / "txt"
            if not png_dir.is_dir() or not txt_dir.is_dir():
                continue

            for png in sorted(png_dir.glob("slice_*.png")):
                gt_box = read_gt_box(txt_dir / (png.stem + ".txt"))
                records.append({
                    "dataset":   dataset,
                    "subject":   subject,
                    "contrast":  contrast,
                    "split":     split,
                    "png_path":  str(png),
                    "gt_box":    gt_box,
                    "has_gt":    gt_box is not None,
                    "pred_box":  None,
                    "pred_conf": 0.0,
                })
    return records


def run_inference(model: YOLO, records: list, conf_thresh: float, batch: int) -> None:
    """Run YOLO inference and fill pred_box / pred_conf in records in-place."""
    paths = [r["png_path"] for r in records]
    for i in tqdm(range(0, len(paths), batch), desc="Inference", unit="batch"):
        results = model.predict(paths[i:i + batch], conf=conf_thresh, verbose=False)
        for rec, res in zip(records[i:i + batch], results):
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            best = int(boxes.conf.argmax())
            cx, cy, w, h = boxes.xywhn[best].tolist()
            rec["pred_box"]  = (cx, cy, w, h)
            rec["pred_conf"] = float(boxes.conf[best])


def ap_at_iou(df: pd.DataFrame, iou_thresh: float) -> float:
    """AP at a single IoU threshold, predictions ranked by confidence."""
    n_gt = int(df["has_gt"].sum())
    if n_gt == 0:
        return float("nan")
    preds = df[df["has_pred"]].sort_values("pred_conf", ascending=False)
    if len(preds) == 0:
        return 0.0
    tp = (preds["has_gt"] & (preds["iou"] >= iou_thresh)).astype(int).values
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    precision = cum_tp / (cum_tp + cum_fp)
    recall    = cum_tp / n_gt
    precision = np.concatenate([[1.0], precision])
    recall    = np.concatenate([[0.0], recall])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    return float(np.trapz(precision, recall))


def summarise_group(df: pd.DataFrame) -> dict:
    n_gt   = int(df["has_gt"].sum())
    n_pred = int(df["has_pred"].sum())

    matched   = df[df["has_gt"] & df["has_pred"]]
    iou_mean  = float(matched["iou"].mean()) if len(matched) else float("nan")
    dice_mean = float((2 * matched["iou"] / (1 + matched["iou"])).mean()) if len(matched) else float("nan")

    tp50        = int((df["has_gt"] & df["has_pred"] & (df["iou"] >= 0.5)).sum())
    recall50    = tp50 / n_gt   if n_gt   else float("nan")
    precision50 = tp50 / n_pred if n_pred else float("nan")
    denom       = (precision50 + recall50) if (not np.isnan(precision50) and not np.isnan(recall50)) else 0.0
    f1_50       = 2 * precision50 * recall50 / denom if denom > 0 else float("nan")

    ap50    = ap_at_iou(df, 0.50)
    ap50_95 = float(np.nanmean([ap_at_iou(df, t) for t in np.arange(0.50, 1.00, 0.05)]))

    return {
        "n_slices":    len(df),
        "n_gt":        n_gt,
        "n_pred":      n_pred,
        "iou_mean":    round(iou_mean,    4),
        "dice_mean":   round(dice_mean,   4),
        "recall50":    round(recall50,    4),
        "precision50": round(precision50, 4),
        "f1_50":       round(f1_50,       4),
        "ap50":        round(ap50,        4),
        "ap50_95":     round(ap50_95,     4),
    }


def build_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(g, split="ALL", dataset="ALL", contrast="ALL"):
        rows.append({"split": split, "dataset": dataset, "contrast": contrast, **summarise_group(g)})

    add(df)
    for split, g in df.groupby("split"):
        add(g, split=split)
    for dataset, g in df.groupby("dataset"):
        add(g, dataset=dataset)
        for split, gg in g.groupby("split"):
            add(gg, split=split, dataset=dataset)
        for contrast, gg in g.groupby("contrast"):
            add(gg, dataset=dataset, contrast=contrast)
            for split, ggg in gg.groupby("split"):
                add(ggg, split=split, dataset=dataset, contrast=contrast)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 2D bbox: IoU, Dice, AP50, AP50:95 — per dataset/contrast/split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--run-id",     required=True, help="Name used for output CSV")
    parser.add_argument("--processed",  default="processed")
    parser.add_argument("--splits-dir", default="data/datasplits")
    parser.add_argument("--conf",       type=float, default=0.001,
                        help="Min confidence for inference (low = full PR curve for AP computation)")
    parser.add_argument("--batch",      type=int,   default=64)
    parser.add_argument("--out",        default=".",  help="Output directory for CSV")
    args = parser.parse_args()

    splits_map = load_splits(Path(args.splits_dir))
    records = collect_gt_records(Path(args.processed), splits_map)
    print(f"Collected {len(records)} slices — "
          f"{len(set(r['dataset'] for r in records))} datasets, "
          f"{len(set(r['subject'] for r in records))} subjects")

    model = YOLO(args.checkpoint)
    run_inference(model, records, conf_thresh=args.conf, batch=args.batch)

    for r in records:
        r["has_pred"] = r["pred_box"] is not None
        if r["has_gt"] and r["has_pred"]:
            r["iou"] = bbox_iou(r["gt_box"], r["pred_box"])
        elif not r["has_gt"] and not r["has_pred"]:
            r["iou"] = float("nan")
        else:
            r["iou"] = 0.0

    df = pd.DataFrame(records).drop(columns=["png_path", "gt_box", "pred_box"])
    report = build_report(df)

    out_csv = Path(args.out) / f"metrics_{args.run_id}.csv"
    report.to_csv(out_csv, index=False)
    print(f"\nReport → {out_csv}")
    print(report[report["dataset"] == "ALL"].to_string(index=False))


if __name__ == "__main__":
    main()
