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
from PIL import Image, ImageDraw
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


def draw_boxes(png_path: str, gt_box, pred_box) -> Image.Image:
    """Draw GT (green) and pred (red) bboxes on the slice. Returns RGB image."""
    img = Image.open(png_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    def bbox_pixels(box):
        cx, cy, w, h = box
        return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]

    if gt_box is not None:
        draw.rectangle(bbox_pixels(gt_box),   outline=(0, 255, 0), width=2)  # green = GT
    if pred_box is not None:
        draw.rectangle(bbox_pixels(pred_box), outline=(255, 0, 0), width=2)  # red = pred
    return img


def auto_batch(model: YOLO, conf_thresh: float, start: int = 512) -> int:
    """Binary search for largest inference batch that fits in GPU memory."""
    import torch
    dummy = [np.zeros((640, 640, 3), dtype=np.uint8)] * start
    batch = start
    while batch >= 1:
        try:
            model.predict(dummy[:batch], conf=conf_thresh, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Auto batch size: {batch}")
            return batch
        except RuntimeError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch //= 2
    return 1


def run_inference(model: YOLO, records: list, conf_thresh: float, batch: int,
                  inference_dir: Path = None, processed_dir: Path = None,
                  viz_conf: float = 0.1) -> None:
    """Run YOLO inference batch by batch, fill pred_box/pred_conf, and optionally save visualisations."""
    paths = [r["png_path"] for r in records]
    for i in tqdm(range(0, len(paths), batch), desc="Inference", unit="batch"):
        batch_records = records[i:i + batch]
        results = model.predict(paths[i:i + batch], conf=conf_thresh, verbose=False)
        for rec, res in zip(batch_records, results):
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                best = int(boxes.conf.argmax())
                cx, cy, w, h = boxes.xywhn[best].tolist()
                rec["pred_box"]  = (cx, cy, w, h)
                rec["pred_conf"] = float(boxes.conf[best])

        if inference_dir is not None:
            for rec in batch_records:
                png_path = Path(rec["png_path"])
                out = inference_dir / png_path.relative_to(processed_dir)
                out.parent.mkdir(parents=True, exist_ok=True)
                # only draw pred box if confidence exceeds viz threshold
                viz_pred = rec["pred_box"] if rec["pred_conf"] >= viz_conf else None
                draw_boxes(rec["png_path"], rec["gt_box"], viz_pred).save(str(out))


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
    n_pred_thresh = int(df["has_pred_thresh"].sum())

    matched   = df[df["has_gt"] & df["has_pred"]]
    iou_mean  = float(matched["iou"].mean()) if len(matched) else float("nan")
    dice_mean = float((2 * matched["iou"] / (1 + matched["iou"])).mean()) if len(matched) else float("nan")

    # fixed-threshold metrics use has_pred_thresh (conf >= metrics_conf) to avoid counting
    # very low-confidence noise predictions on slices where the cord is absent
    tp50        = int((df["has_gt"] & df["has_pred_thresh"] & (df["iou"] >= 0.5)).sum())
    recall50    = tp50 / n_gt          if n_gt          else float("nan")
    precision50 = tp50 / n_pred_thresh if n_pred_thresh else float("nan")
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
    parser.add_argument("--conf",         type=float, default=0.25,
                        help="Min confidence for inference (ultralytics default)")
    parser.add_argument("--metrics-conf", type=float, default=0.25,
                        help="Confidence threshold for precision/recall/f1 fixed-threshold metrics")
    parser.add_argument("--viz-conf",     type=float, default=0.25,
                        help="Min confidence to draw a pred box in visualisations")
    parser.add_argument("--batch",       type=int,   default=-1,
                        help="Inference batch size (-1 = auto-detect)")
    parser.add_argument("--split",       default=None, choices=["train", "val", "test"],
                        help="Restrict to a single split (default: all splits)")
    parser.add_argument("--out",         default=".", help="Output directory for CSV")
    parser.add_argument("--inference-dir", default="inference", help="Root dir for visualisation output")
    parser.add_argument("--no-viz",      action="store_true", help="Skip saving visualisation images")
    args = parser.parse_args()

    splits_map = load_splits(Path(args.splits_dir))
    records = collect_gt_records(Path(args.processed), splits_map)
    if args.split:
        records = [r for r in records if r["split"] == args.split]
    print(f"Collected {len(records)} slices — "
          f"{len(set(r['dataset'] for r in records))} datasets, "
          f"{len(set(r['subject'] for r in records))} subjects"
          + (f" [{args.split}]" if args.split else ""))

    checkpoint = Path(args.checkpoint)
    train_args_yaml = checkpoint.parent.parent / "args.yaml"
    if train_args_yaml.exists():
        train_args = yaml.safe_load(train_args_yaml.read_text())
        print(f"Checkpoint : {checkpoint}")
        print(f"  model    : {train_args.get('model', '?')}")
        print(f"  data     : {train_args.get('data', '?')}")
        print(f"  epochs   : {train_args.get('epochs', '?')}  imgsz: {train_args.get('imgsz', '?')}")
    else:
        print(f"Checkpoint : {checkpoint}  (args.yaml introuvable)")

    inference_dir = None if args.no_viz else Path(args.inference_dir) / args.run_id
    if inference_dir:
        print(f"Visualisations → {inference_dir}")

    model = YOLO(args.checkpoint)
    batch = auto_batch(model, args.conf) if args.batch == -1 else args.batch
    run_inference(model, records, conf_thresh=args.conf, batch=batch,
                  inference_dir=inference_dir, processed_dir=Path(args.processed),
                  viz_conf=args.viz_conf)

    for r in records:
        r["has_pred"] = r["pred_box"] is not None
        # has_pred_thresh: used for precision/recall/f1 — only confident predictions count
        r["has_pred_thresh"] = r["pred_box"] is not None and r["pred_conf"] >= args.metrics_conf
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
    print(f"Report → {out_csv}")
    print(report[report["dataset"] == "ALL"].to_string(index=False))


if __name__ == "__main__":
    main()
