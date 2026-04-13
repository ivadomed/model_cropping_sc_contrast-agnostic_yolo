#!/usr/bin/env python3
"""
Run YOLO inference on processed/ slices → predictions/<run-id>/

Mirrors processed/ hierarchy:
  predictions/<run-id>/<dataset>/<patient>/
    png/slice_NNN.png   ← GT (green filled) + pred (red 1px outline + conf score) overlay
    txt/slice_NNN.txt   ← predicted bbox: "0 cx cy w h conf" (empty if no detection)
    volume/bbox_3d.txt  ← 3D bbox union from predicted slices

conf=0 by default: all boxes are saved regardless of confidence.
Filter by confidence at the metrics stage (metrics.py --conf).

Run metrics.py afterwards to compute aggregated performance metrics from saved predictions.

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/yolo26_1mm_axial/weights/best.pt \
        --processed processed/10mm_SI_1mm_axial

    # Custom run-id (default: derived from checkpoint parent dir name):
    python scripts/evaluate.py \
        --checkpoint checkpoints/yolo26_1mm_axial/weights/best.pt \
        --run-id yolo26_1mm_axial_conf25 \
        --processed processed/10mm_SI_1mm_axial

    # Regenerate overlay images from existing predictions (no inference):
    python scripts/evaluate.py \
        --viz-only \
        --run-id yolo26_1mm_axial \
        --processed processed/10mm_SI_1mm_axial
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from utils import bbox_3d_from_txts, write_bbox_3d

CONF_THRESH = 0.0  # default confidence threshold for inference (filter at metrics stage)


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


def draw_boxes(png_path: str, gt_box, pred_box, pred_conf: float = None) -> Image.Image:
    """Draw GT (green filled area) and pred (red 1px outline + conf score) on the slice."""
    img = Image.open(png_path).convert("RGB")
    W, H = img.size

    def to_pixels(box):
        cx, cy, w, h = box
        x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
        x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
        return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=10)
    except OSError:
        font = ImageFont.load_default()

    if gt_box is not None:
        overlay = Image.new("RGB", img.size, (0, 0, 0))
        ImageDraw.Draw(overlay).rectangle(to_pixels(gt_box), fill=(0, 255, 0))
        img = Image.blend(img, overlay, alpha=0.3)

    if pred_box is not None:
        draw = ImageDraw.Draw(img)
        coords = to_pixels(pred_box)
        draw.rectangle(coords, outline=(255, 0, 0), width=1)
        if pred_conf is not None:
            label = f"{pred_conf:.4f}"
            x_text = coords[0] + 2
            y_text = max(coords[1] - 12, 0)
            draw.text((x_text, y_text), label, fill=(255, 0, 0), font=font)

    # Legend
    draw = ImageDraw.Draw(img)
    entries = []
    if gt_box is not None:
        entries.append(("GT", (0, 200, 0)))
    if pred_box is not None:
        entries.append(("Pred", (255, 0, 0)))
    x, y = 4, 4
    for label, color in entries:
        draw.rectangle([x, y, x + 8, y + 8], fill=color)
        draw.text((x + 11, y - 1), label, fill=(255, 255, 255), font=font)
        x += 11 + draw.textlength(label, font=font) + 6

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
                draw_boxes(str(png), gt_box, pred_box, pred_conf if pred_box is not None else None).save(
                    str(pred_dir / "png" / png.name))

    meta = yaml.safe_load((patient_dir / "meta.yaml").read_text())
    H, W = meta["shape_las"][0], meta["shape_las"][1]
    box = bbox_3d_from_txts(pred_txt_dir, H, W)
    if box is not None:
        write_bbox_3d(pred_vol_dir / "bbox_3d.txt", **box)


def render_overlays(pred_root: Path, processed_dir: Path) -> None:
    """Regenerate overlay PNGs from existing txt predictions without re-running inference."""
    patients = [
        (pred_dataset_dir.name, pred_patient_dir)
        for pred_dataset_dir in sorted(pred_root.iterdir()) if pred_dataset_dir.is_dir()
        for pred_patient_dir in sorted(pred_dataset_dir.iterdir())
        if (pred_patient_dir / "txt").is_dir()
    ]
    print(f"Rendering overlays for {len(patients)} patients → {pred_root}")
    for dataset, pred_patient_dir in tqdm(patients, desc="Patients", unit="pat"):
        stem        = pred_patient_dir.name
        gt_txt_dir  = processed_dir / dataset / stem / "txt"
        src_png_dir = processed_dir / dataset / stem / "png"
        out_png_dir = pred_patient_dir / "png"
        out_png_dir.mkdir(parents=True, exist_ok=True)
        for pred_txt in sorted((pred_patient_dir / "txt").glob("slice_*.txt")):
            png_src = src_png_dir / (pred_txt.stem + ".png")
            if not png_src.exists():
                continue
            content   = pred_txt.read_text().strip()
            pred_box  = None
            pred_conf = None
            if content:
                p = content.split()
                pred_box  = (float(p[1]), float(p[2]), float(p[3]), float(p[4]))
                pred_conf = float(p[5]) if len(p) >= 6 else None
            gt_box = read_gt_box(gt_txt_dir / pred_txt.name)
            draw_boxes(str(png_src), gt_box, pred_box, pred_conf).save(str(out_png_dir / png_src.name))


def main():
    parser = argparse.ArgumentParser(
        description="YOLO inference on processed/ slices → predictions/<run-id>/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",  default=None, help="Path to best.pt (not needed with --viz-only)")
    parser.add_argument("--run-id",      default=None,
                        help="Output run identifier (default: derived from checkpoint directory name)")
    parser.add_argument("--processed",   default="processed/10mm_SI", help="processed/<variant> dir")
    parser.add_argument("--splits-dir",  default="data/datasplits")
    parser.add_argument("--conf",        type=float, default=CONF_THRESH,
                        help="Confidence threshold for inference and visualisations")
    parser.add_argument("--batch",       type=int, default=-1,
                        help="Inference batch size (-1 = auto-detect)")
    parser.add_argument("--split",       default=None, choices=["train", "val", "test"],
                        help="Restrict inference to subjects in this split (default: all)")
    parser.add_argument("--out",         default="predictions", help="Output root directory")
    parser.add_argument("--no-viz",      action="store_true", help="Skip saving overlay images")
    parser.add_argument("--viz-only",    action="store_true",
                        help="Regenerate overlay PNGs from existing predictions (no inference)")
    args = parser.parse_args()

    processed_dir = Path(args.processed)

    if args.viz_only:
        assert args.run_id is not None, "--run-id is required with --viz-only"
    else:
        assert args.checkpoint is not None, "--checkpoint is required unless --viz-only is set"
        if args.run_id is None:
            args.run_id = Path(args.checkpoint).parent.parent.name

    pred_root = Path(args.out) / args.run_id

    if args.viz_only:
        render_overlays(pred_root, processed_dir)
        return

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
