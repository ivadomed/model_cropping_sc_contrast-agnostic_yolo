#!/usr/bin/env python3
"""
Run YOLO inference on processed/ slices → predictions/<run-id>/

Mirrors processed/ hierarchy:
  predictions/<run-id>/<dataset>/<patient>/
    png/slice_NNN.png   ← GT (green) + pred (red) overlay
    txt/slice_NNN.txt   ← predicted bbox: "0 cx cy w h conf" (empty if no detection)
    volume/bbox_3d.txt  ← 3D bbox union from predicted slices

Run metrics.py afterwards to compute aggregated performance metrics from saved predictions.

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/yolo_spine_v1/weights/best.pt \
        --run-id yolo_spine_v1 \
        --processed processed_10mm_SI \
        --splits-dir data/datasplits \
        [--conf 0.25] [--split val] [--batch -1] [--no-viz] [--out predictions]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from utils import bbox_3d_from_txts, write_bbox_3d

CONF_THRESH = 0.25  # default confidence threshold for inference and visualisations


def load_split_subjects(splits_dir: Path, split: str) -> set:
    """Returns set of (dataset, subject) pairs assigned to the given split."""
    result = set()
    for f in sorted(splits_dir.glob("datasplit_*.yaml")):
        dataset = re.sub(r"_seed\d+$", "", f.stem[len("datasplit_"):])
        for subj in (yaml.safe_load(f.read_text()).get(split) or []):
            result.add((dataset, subj))
    return result


def read_gt_box(txt_path: Path):
    """Returns (cx,cy,w,h) or None if file empty/missing."""
    if not txt_path.exists():
        return None
    content = txt_path.read_text().strip()
    if not content:
        return None
    p = content.split()
    return (float(p[1]), float(p[2]), float(p[3]), float(p[4]))


def draw_boxes(png_path: str, gt_box, pred_box) -> Image.Image:
    """Draw GT (green) and pred (red) bboxes on the slice. Returns RGB image."""
    img = Image.open(png_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    def to_pixels(box):
        cx, cy, w, h = box
        return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]

    if gt_box is not None:
        draw.rectangle(to_pixels(gt_box), outline=(0, 255, 0), width=2)
    if pred_box is not None:
        draw.rectangle(to_pixels(pred_box), outline=(255, 0, 0), width=2)
    return img


def auto_batch(model: YOLO, conf: float, start: int = 512) -> int:
    """Binary search for largest inference batch that fits in GPU memory."""
    import torch
    dummy = [np.zeros((640, 640, 3), dtype=np.uint8)] * start
    batch = start
    while batch >= 1:
        try:
            model.predict(dummy[:batch], conf=conf, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Auto batch size: {batch}")
            return batch
        except RuntimeError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch //= 2
    return 1


def infer_patient(model: YOLO, patient_dir: Path, pred_dir: Path,
                  conf: float, batch_size: int, save_viz: bool) -> None:
    """Run inference on all slices of one patient and save predictions."""
    pred_txt_dir = pred_dir / "txt"
    pred_vol_dir = pred_dir / "volume"
    for d in (pred_txt_dir, pred_vol_dir):
        d.mkdir(parents=True, exist_ok=True)
    if save_viz:
        (pred_dir / "png").mkdir(parents=True, exist_ok=True)

    gt_txt_dir = patient_dir / "txt"
    pngs = sorted((patient_dir / "png").glob("slice_*.png"))

    for i in range(0, len(pngs), batch_size):
        chunk = pngs[i:i + batch_size]
        results = model.predict([str(p) for p in chunk], conf=conf, verbose=False)
        for png, res in zip(chunk, results):
            boxes = res.boxes
            pred_box, pred_conf = None, 0.0
            if boxes is not None and len(boxes) > 0:
                best = int(boxes.conf.argmax())
                cx, cy, w, h = boxes.xywhn[best].tolist()
                pred_conf = float(boxes.conf[best])
                pred_box = (cx, cy, w, h)

            txt_out = pred_txt_dir / (png.stem + ".txt")
            if pred_box is not None:
                txt_out.write_text(
                    f"0 {pred_box[0]:.6f} {pred_box[1]:.6f} {pred_box[2]:.6f} {pred_box[3]:.6f} {pred_conf:.6f}\n"
                )
            else:
                txt_out.write_text("")

            if save_viz:
                gt_box = read_gt_box(gt_txt_dir / (png.stem + ".txt"))
                draw_boxes(str(png), gt_box, pred_box).save(str(pred_dir / "png" / png.name))

    meta = yaml.safe_load((patient_dir / "meta.yaml").read_text())
    H, W = meta["shape_las"][0], meta["shape_las"][1]
    box = bbox_3d_from_txts(pred_txt_dir, H, W)
    if box is not None:
        write_bbox_3d(pred_vol_dir / "bbox_3d.txt", **box)


def main():
    parser = argparse.ArgumentParser(
        description="YOLO inference on processed/ slices → predictions/<run-id>/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",  required=True, help="Path to best.pt")
    parser.add_argument("--run-id",      required=True, help="Output run identifier")
    parser.add_argument("--processed",   default="processed", help="processed/ root dir")
    parser.add_argument("--splits-dir",  default="data/datasplits")
    parser.add_argument("--conf",        type=float, default=CONF_THRESH,
                        help="Confidence threshold for inference and visualisations")
    parser.add_argument("--batch",       type=int, default=-1,
                        help="Inference batch size (-1 = auto-detect)")
    parser.add_argument("--split",       default=None, choices=["train", "val", "test"],
                        help="Restrict inference to subjects in this split (default: all)")
    parser.add_argument("--out",         default="predictions", help="Output root directory")
    parser.add_argument("--no-viz",      action="store_true", help="Skip saving overlay images")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    pred_root = Path(args.out) / args.run_id

    split_subjects = None
    if args.split:
        split_subjects = load_split_subjects(Path(args.splits_dir), args.split)

    model = YOLO(args.checkpoint)
    batch_size = auto_batch(model, args.conf) if args.batch == -1 else args.batch

    patients = []
    for dataset_dir in sorted(processed_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for patient_dir in sorted(dataset_dir.iterdir()):
            if not (patient_dir / "png").is_dir():
                continue
            if split_subjects is not None:
                m = re.match(r"(sub-[^_]+)", patient_dir.name)
                subject = m.group(1) if m else patient_dir.name
                if (dataset_dir.name, subject) not in split_subjects:
                    continue
            patients.append((dataset_dir.name, patient_dir))

    print(f"Running inference on {len(patients)} patients → {pred_root}"
          + (f" [{args.split}]" if args.split else ""))

    for dataset, patient_dir in tqdm(patients, desc="Patients", unit="pat"):
        infer_patient(model, patient_dir, pred_root / dataset / patient_dir.name,
                      args.conf, batch_size, not args.no_viz)

    print(f"Predictions saved to: {pred_root}")
    print("Run metrics.py to compute aggregated metrics.")


if __name__ == "__main__":
    main()
