#!/usr/bin/env python3
"""
Run YOLO inference on processed slices → predictions/<run-id>/

For each patient in the split partition:
  - reads PNGs from processed/<site>/<stem>/png/
  - runs YOLO slice by slice
  - saves raw predictions: predictions/<run-id>/<site>/<stem>/slices/txt/slice_NNN.txt
  - saves visualisations:  predictions/<run-id>/<site>/<stem>/slices/png/slice_NNN.png
  - computes 3D predicted bbox: predictions/<run-id>/<site>/<stem>/volume/bbox_3d.txt

Usage:
    python scripts/infer.py \
        --checkpoint checkpoints/yolo_spine_v1/weights/best.pt \
        --run-id yolo_spine_v1 \
        --split data/splits/global.yaml \
        --partition val
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image as PILImage, ImageDraw
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from utils import bbox_3d_from_txts, write_bbox_3d


# ── per-slice helpers ──────────────────────────────────────────────────────────

def predict_slice(model: YOLO, png_path: Path, conf: float) -> str:
    """Run YOLO on one slice. Returns YOLO txt line (best detection) or empty string."""
    img = PILImage.open(png_path).convert("RGB")
    results = model.predict(img, conf=conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return ""
    best = int(boxes.conf.argmax())
    cx, cy, w, h = boxes.xywhn[best].tolist()
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def draw_bbox(png_path: Path, txt_line: str) -> PILImage.Image:
    """Draw predicted bbox (red) on slice PNG. Returns RGB PIL image."""
    img = PILImage.open(png_path).convert("RGB")
    if not txt_line.strip():
        return img
    W, H = img.size
    _, cx, cy, w, h = map(float, txt_line.split())
    x1, x2 = int((cx - w / 2) * W), int((cx + w / 2) * W)
    y1, y2 = int((cy - h / 2) * H), int((cy + h / 2) * H)
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
    return img


# ── per-patient inference ──────────────────────────────────────────────────────

def infer_patient(model: YOLO, patient_dir: Path, pred_dir: Path, conf: float) -> None:
    """Run inference on all slices of one patient and save results."""
    with open(patient_dir / "meta.yaml") as f:
        meta = yaml.safe_load(f)
    H, W, _ = meta["shape"]

    (pred_dir / "slices" / "txt").mkdir(parents=True, exist_ok=True)
    (pred_dir / "slices" / "png").mkdir(parents=True, exist_ok=True)
    (pred_dir / "volume").mkdir(parents=True, exist_ok=True)

    for png in sorted((patient_dir / "png").glob("slice_*.png")):
        txt_line = predict_slice(model, png, conf)
        (pred_dir / "slices" / "txt" / png.stem).with_suffix(".txt").write_text(
            txt_line + "\n" if txt_line else ""
        )
        draw_bbox(png, txt_line).save(str(pred_dir / "slices" / "png" / png.name))

    box = bbox_3d_from_txts(pred_dir / "slices" / "txt", H, W)
    if box is not None:
        write_bbox_3d(pred_dir / "volume" / "bbox_3d.txt", **box)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YOLO inference on processed slices → predictions/<run-id>/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--run-id",     required=True)
    parser.add_argument("--split",      default="data/splits/global.yaml")
    parser.add_argument("--partition",  default="val",  choices=["train", "val", "test", "all"])
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--processed",  default="processed")
    parser.add_argument("--out",        default="predictions")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    pred_root = Path(args.out) / args.run_id

    with open(args.split) as f:
        splits = yaml.safe_load(f)

    partitions = list(splits.keys()) if args.partition == "all" else [args.partition]
    patients = [p for part in partitions for p in splits.get(part, [])]

    model = YOLO(args.checkpoint)

    for patient_rel in tqdm(patients, desc="Patients", unit="pat"):
        patient_dir = processed_dir / patient_rel
        pred_dir = pred_root / patient_rel
        infer_patient(model, patient_dir, pred_dir, args.conf)

    print(f"\nPredictions saved to: {pred_root}")


if __name__ == "__main__":
    main()
