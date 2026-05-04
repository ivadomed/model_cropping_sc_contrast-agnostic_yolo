#!/usr/bin/env python3
"""
Run YOLO inference on processed/ slices → predictions/<run-id>/

Output structure:
  predictions/<run-id>/
    predictions/<dataset>/<patient>/
      png/slice_NNN.png   ← GT (green filled) + pred (red 1px outline + conf score) overlay
      txt/slice_NNN.txt   ← predicted bbox: "0 cx cy w h conf" per line (empty if no detection)
      volume/bbox_3d.txt  ← 3D bbox union from predicted slices

conf=0.1 by default: boxes below threshold are discarded.
Sagittal plane saves all boxes above threshold; axial saves only the best per class.

Axial plane  : one box per class saved (best confidence).
Sagittal plane: all boxes above conf threshold saved (multiple lines per class allowed).

Run metrics.py afterwards to compute aggregated performance metrics from saved predictions.
Patients that fail (e.g. missing files, GPU OOM) are caught individually, logged to
predictions/<run-id>/failed_patients.log (TSV: dataset/patient/error), and the script
exits with code 1 if any failed — other patients are still processed.

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

from __future__ import annotations

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

CONF_THRESH = 0.1  # default confidence threshold for inference


def load_split_subjects(splits_dir: Path, split: str) -> set:
    """Returns set of (dataset, subject) pairs assigned to the given split."""
    result = set()
    for f in sorted(splits_dir.glob("datasplit_*.yaml")):
        dataset = re.sub(r"_seed\d+$", "", f.stem[len("datasplit_"):])
        for subj in (yaml.safe_load(f.read_text()).get(split) or []):
            result.add((dataset, subj))
    return result


def read_gt_boxes(txt_path: Path) -> dict:
    """Returns {class_id: (cx,cy,w,h)} for all lines in the GT txt, or {} if missing/empty."""
    if not txt_path.exists():
        return {}
    boxes = {}
    for line in txt_path.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            boxes[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]), float(p[4]))
    return boxes


# GT fill colours per class (semi-transparent overlay)
GT_COLORS   = {0: (0, 255, 0),   1: (0, 200, 255)}   # sc=green, canal=cyan
# Pred outline colours per class
CLASS_COLORS = {0: (255, 0, 0),  1: (255, 220, 0)}    # sc=red, canal=yellow
CLASS_NAMES  = {0: "sc",         1: "canal"}


def draw_boxes(png_path: str, gt_boxes: dict,
               pred_boxes: list) -> Image.Image:
    """Draw GT (filled area) and pred (1px outline + conf score) for all classes.

    gt_boxes  : {class_id: (cx,cy,w,h)}
    pred_boxes: [(class_id, cx, cy, w, h, conf), ...]  — multiple boxes per class allowed
    """
    img = Image.open(png_path).convert("RGB")
    W, H = img.size

    def to_pixels(cx, cy, w, h):
        x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
        x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
        return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=10)
    except OSError:
        font = ImageFont.load_default()

    for cls_id, gt_box in gt_boxes.items():
        color = GT_COLORS.get(cls_id, (0, 255, 0))
        overlay = Image.new("RGB", img.size, (0, 0, 0))
        ImageDraw.Draw(overlay).rectangle(to_pixels(*gt_box), fill=color)
        img = Image.blend(img, overlay, alpha=0.3)

    draw = ImageDraw.Draw(img)
    for cls_id, cx, cy, w, h, conf in pred_boxes:
        color = CLASS_COLORS.get(cls_id, (255, 0, 0))
        coords = to_pixels(cx, cy, w, h)
        draw.rectangle(coords, outline=color, width=1)
        draw.text((coords[0] + 2, max(coords[1] - 12, 0)),
                  f"{conf:.4f}", fill=color, font=font)

    # Legend
    pred_cls_ids = sorted({b[0] for b in pred_boxes})
    entries = []
    for cls_id in sorted(gt_boxes):
        entries.append((f"GT {CLASS_NAMES.get(cls_id, str(cls_id))}", GT_COLORS.get(cls_id, (0,255,0))))
    for cls_id in pred_cls_ids:
        entries.append((CLASS_NAMES.get(cls_id, str(cls_id)), CLASS_COLORS.get(cls_id, (255,0,0))))
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
                  conf: float, batch_size: int, save_viz: bool,
                  flip_x: bool = False) -> None:
    """Run inference on all slices of one patient and save predictions.

    flip_x: horizontally flip each slice before inference, then unflip cx (cx → 1-cx).
    Overlay PNGs always show the original (unflipped) image with corrected prediction.
    """
    pred_txt_dir = pred_dir / "txt"
    pred_vol_dir = pred_dir / "volume"
    for d in (pred_txt_dir, pred_vol_dir):
        d.mkdir(parents=True, exist_ok=True)
    if save_viz:
        (pred_dir / "png").mkdir(parents=True, exist_ok=True)

    gt_txt_dir = patient_dir / "txt"
    pngs = sorted((patient_dir / "png").glob("slice_*.png"))

    meta      = yaml.safe_load((patient_dir / "meta.yaml").read_text())
    sagittal  = meta.get("plane", "axial") == "sagittal"

    for i in range(0, len(pngs), batch_size):
        chunk = pngs[i:i + batch_size]
        if flip_x:
            inputs = [np.array(Image.open(p).convert("RGB"))[:, ::-1, ::-1].copy() for p in chunk]
        else:
            inputs = [str(p) for p in chunk]
        results = model.predict(inputs, conf=conf, verbose=False)
        for png, res in zip(chunk, results):
            # Sagittal: keep all boxes above conf threshold.
            # Axial: keep best box per class.
            pred_boxes: list = []
            if res.boxes is not None and len(res.boxes) > 0:
                if sagittal:
                    for i_box in range(len(res.boxes)):
                        cls_id = int(res.boxes.cls[i_box].item())
                        cx, cy, w, h = res.boxes.xywhn[i_box].tolist()
                        if flip_x:
                            cx = 1.0 - cx
                        pred_boxes.append((cls_id, cx, cy, w, h, float(res.boxes.conf[i_box])))
                else:
                    for cls_id in res.boxes.cls.unique().tolist():
                        cls_id = int(cls_id)
                        mask   = res.boxes.cls == cls_id
                        best   = int(res.boxes.conf[mask].argmax())
                        idxs   = mask.nonzero(as_tuple=True)[0]
                        idx    = idxs[best].item()
                        cx, cy, w, h = res.boxes.xywhn[idx].tolist()
                        if flip_x:
                            cx = 1.0 - cx
                        pred_boxes.append((cls_id, cx, cy, w, h, float(res.boxes.conf[idx])))

            txt_out = pred_txt_dir / (png.stem + ".txt")
            txt_out.write_text("".join(
                f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {c:.6f}\n"
                for cls_id, cx, cy, w, h, c in pred_boxes
            ))

            if save_viz:
                gt_boxes = read_gt_boxes(gt_txt_dir / (png.stem + ".txt"))
                draw_boxes(str(png), gt_boxes, pred_boxes).save(
                    str(pred_dir / "png" / png.name))

    s = meta["shape_las"]
    if meta.get("plane", "axial") == "sagittal":
        H, W = s[2], s[1]  # sagittal: H=SI_dim, W=AP_dim
    else:
        H, W = s[1], s[0]  # axial: H=AP_dim, W=RL_dim (row 0=Anterior, col 0=Left)
    box = bbox_3d_from_txts(pred_txt_dir, H, W)
    if box is not None:
        write_bbox_3d(pred_vol_dir / "bbox_3d.txt", **box)

    # Copy GT data into pred dir so metrics.py needs no --processed argument
    import shutil
    shutil.copy2(patient_dir / "meta.yaml", pred_dir / "meta.yaml")
    gt_link = pred_dir / "gt"
    if not gt_link.exists():
        gt_link.symlink_to(patient_dir.resolve())


def render_overlays(pred_root: Path, processed_dir: Path = None) -> None:
    """Regenerate overlay PNGs from existing txt predictions without re-running inference."""
    patients = [
        (pred_dataset_dir.name, pred_patient_dir)
        for pred_dataset_dir in sorted((pred_root / "predictions").iterdir()) if pred_dataset_dir.is_dir()
        for pred_patient_dir in sorted(pred_dataset_dir.iterdir())
        if (pred_patient_dir / "txt").is_dir()
    ]
    print(f"Rendering overlays for {len(patients)} patients → {pred_root}")
    for dataset, pred_patient_dir in tqdm(patients, desc="Patients", unit="pat"):
        stem = pred_patient_dir.name
        if (pred_patient_dir / "gt").exists():
            gt_src = pred_patient_dir / "gt"
        else:
            assert processed_dir is not None, \
                f"No gt/ symlink in {pred_patient_dir} and --processed not provided"
            gt_src = processed_dir / dataset / stem
        gt_txt_dir  = gt_src / "txt"
        src_png_dir = gt_src / "png"
        out_png_dir = pred_patient_dir / "png"
        out_png_dir.mkdir(parents=True, exist_ok=True)
        for pred_txt in sorted((pred_patient_dir / "txt").glob("slice_*.txt")):
            png_src = src_png_dir / (pred_txt.stem + ".png")
            if not png_src.exists():
                continue
            pred_boxes: list = []
            for line in pred_txt.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    cls_id = int(p[0])
                    conf   = float(p[5]) if len(p) >= 6 else 0.0
                    pred_boxes.append((cls_id, float(p[1]), float(p[2]), float(p[3]), float(p[4]), conf))
            gt_boxes = read_gt_boxes(gt_txt_dir / pred_txt.name)
            draw_boxes(str(png_src), gt_boxes, pred_boxes).save(
                str(out_png_dir / png_src.name))


def run(checkpoint: str | Path, processed: str | Path, out: str | Path,
        splits_dir: str | Path | None = None, conf: float = CONF_THRESH) -> None:
    """Run YOLO inference on all patients in processed/ → out/."""
    processed_dir = Path(processed)
    pred_root     = Path(out)

    model      = YOLO(str(checkpoint))
    batch_size = auto_batch(model, conf)

    patients = []
    for dataset_dir in sorted(processed_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for patient_dir in sorted(dataset_dir.iterdir()):
            if not (patient_dir / "png").is_dir():
                continue
            patients.append((dataset_dir.name, patient_dir))

    print(f"Running inference on {len(patients)} patients → {pred_root}")

    failed: list[tuple[str, str, str]] = []
    skipped = 0
    for dataset, patient_dir in tqdm(patients, desc="Patients", unit="pat"):
        pred_dir = pred_root / "predictions" / dataset / patient_dir.name
        if (pred_dir / "meta.yaml").exists():
            skipped += 1
            continue
        try:
            infer_patient(model, patient_dir, pred_dir, conf, batch_size, True)
        except Exception as e:
            failed.append((dataset, patient_dir.name, str(e)))

    print(f"Predictions saved to: {pred_root}"
          + (f"  ({skipped} already done, skipped)" if skipped else ""))

    if failed:
        log_path = pred_root / "failed_patients.log"
        with open(log_path, "w") as f:
            f.write("dataset\tpatient\terror\n")
            for ds, pat, err in failed:
                f.write(f"{ds}\t{pat}\t{err}\n")
        print(f"WARNING: {len(failed)}/{len(patients)} patients failed — see {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO inference on processed/ slices → predictions/<run-id>/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint",  default=None, help="Path to best.pt (not needed with --viz-only)")
    parser.add_argument("--run-id",      default=None,
                        help="Output run identifier (default: derived from checkpoint directory name)")
    parser.add_argument("--processed",   default="processed/10mm_SI", help="processed/<variant> dir")
    parser.add_argument("--splits-dir",  default="data/datasplits/from_raw")
    parser.add_argument("--conf",        type=float, default=CONF_THRESH,
                        help="Confidence threshold for inference and visualisations")
    parser.add_argument("--batch",       type=int, default=-1,
                        help="Inference batch size (-1 = auto-detect)")
    parser.add_argument("--split",       default=None, choices=["train", "val", "test"],
                        help="Restrict inference to subjects in this split (default: all)")
    parser.add_argument("--out",         default="predictions", help="Output root directory")
    parser.add_argument("--datasets",    nargs="+", default=None,
                        help="Restrict to these dataset names (default: all)")
    parser.add_argument("--flip-x",      action="store_true",
                        help="Flip each slice horizontally before inference, unflip cx in predictions")
    parser.add_argument("--no-viz",      action="store_true", help="Skip saving overlay images")
    parser.add_argument("--viz-only",    action="store_true",
                        help="Regenerate overlay PNGs from existing predictions (no inference)")
    args = parser.parse_args()

    processed_dir = Path(args.processed)

    if args.viz_only:
        assert args.run_id is not None, "--run-id is required with --viz-only"
        render_overlays(Path(args.out) / args.run_id, processed_dir)
        return

    assert args.checkpoint is not None, "--checkpoint is required unless --viz-only is set"
    if args.run_id is None:
        base = Path(args.checkpoint).parent.parent.name
        args.run_id = base + "_flipx" if args.flip_x else base

    pred_root = Path(args.out) / args.run_id

    split_subjects_set = None
    if args.split:
        split_subjects_set = load_split_subjects(Path(args.splits_dir), args.split)

    model = YOLO(args.checkpoint)
    batch_size = auto_batch(model, args.conf) if args.batch == -1 else args.batch

    patients = []
    for dataset_dir in sorted(processed_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if args.datasets and dataset_dir.name not in args.datasets:
            continue
        for patient_dir in sorted(dataset_dir.iterdir()):
            if not (patient_dir / "png").is_dir():
                continue
            if split_subjects_set is not None:
                m = re.match(r"(sub-[^_]+)", patient_dir.name)
                subject = m.group(1) if m else patient_dir.name
                if (dataset_dir.name, subject) not in split_subjects_set:
                    continue
            patients.append((dataset_dir.name, patient_dir))

    print(f"Running inference on {len(patients)} patients → {pred_root}"
          + (f" [{args.split}]" if args.split else ""))

    failed: list[tuple[str, str, str]] = []
    skipped = 0
    for dataset, patient_dir in tqdm(patients, desc="Patients", unit="pat"):
        pred_dir = pred_root / "predictions" / dataset / patient_dir.name
        if (pred_dir / "meta.yaml").exists():
            skipped += 1
            continue
        try:
            infer_patient(model, patient_dir, pred_dir,
                          args.conf, batch_size, not args.no_viz, args.flip_x)
        except Exception as e:
            failed.append((dataset, patient_dir.name, str(e)))

    print(f"Predictions saved to: {pred_root}"
          + (f"  ({skipped} already done, skipped)" if skipped else ""))

    if failed:
        log_path = pred_root / "failed_patients.log"
        with open(log_path, "w") as f:
            f.write("dataset\tpatient\terror\n")
            for ds, pat, err in failed:
                f.write(f"{ds}\t{pat}\t{err}\n")
        print(f"WARNING: {len(failed)}/{len(patients)} patients failed — see {log_path}")
        sys.exit(1)

    print("Run metrics.py to compute aggregated metrics.")


if __name__ == "__main__":
    main()
