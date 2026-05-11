#!/usr/bin/env python3
"""
Evaluate YOLO classification model (yolo26n-cls) for SC / no-SC axial slice classification.

For each slice:
  - Always writes dummy bbox "0 0.5 0.5 1.0 1.0 prob" in txt (full SC probability stored).
  - metrics.py sweeps conf thresholds over the stored prob — no threshold baked in at inference.
  - cls_conf is used only for coloured-border visualisation and bbox_3d.txt boundary.

Compatible with metrics.py (gap_mm_S, gap_mm_I).

Overlay PNGs per slice (colored border):
  green  = TP  (GT=SC,    pred=SC)
  red    = FP  (GT=no-SC, pred=SC)   ← brain false positives
  orange = FN  (GT=SC,    pred=no-SC) ← missed SC slices
  none   = TN  (GT=no-SC, pred=no-SC)

Per-patient overview_boundaries.png: N slices around GT z1 (superior) and z2 (inferior).

Output structure:
  <inference>/predictions/<dataset>/<patient>/
    txt/slice_NNN.txt
    volume/bbox_3d.txt
    png/slice_NNN.png        ← colored border overlay
    overview_boundaries.png  ← strip of boundary slices
    gt/  meta.yaml

Usage:
    python scripts/evaluate_cls.py \\
        --cls-checkpoint runs/20260601_120000/checkpoints_cls/weights/best.pt \\
        --processed processed/10mm_SI_1mm_axial_3ch \\
        --inference runs/20260601_120000/predictions \\
        --cls-conf 0.5
    python scripts/evaluate_cls.py ... --no-viz
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from utils import write_bbox_3d

# colored border: (gt_is_sc, pred_is_sc) → RGB  (None = no border)
_BORDER_COLOR = {
    (True,  True):  (0, 220, 0),
    (False, True):  (220, 50, 50),
    (True,  False): (255, 160, 0),
    (False, False): None,
}
_BORDER_W    = 4
_OVERVIEW_H  = 160   # height in pixels for each slice in the boundary overview
_OVERVIEW_N  = 4     # slices shown on each side of the boundary


def _sc_class_idx(model: YOLO) -> int:
    for idx, name in model.names.items():
        if name == "sc":
            return int(idx)
    raise ValueError(f"Class 'sc' not found in model.names: {model.names}")


def _annotate(src_png: Path, gt_is_sc: bool, pred_is_sc: bool,
               z: int, sc_prob: float) -> Image.Image:
    img   = Image.open(src_png).convert("RGB")
    draw  = ImageDraw.Draw(img)
    color = _BORDER_COLOR[(gt_is_sc, pred_is_sc)]
    if color:
        w, h = img.size
        draw.rectangle([0, 0, w - 1, h - 1], outline=color, width=_BORDER_W)
    label = f"z={z:03d} GT={'SC' if gt_is_sc else '--'} P={'SC' if pred_is_sc else '--'} {sc_prob:.2f}"
    draw.text((2, 2), label, fill=(255, 255, 255), font=ImageFont.load_default())
    return img


def _find_slice_png(src_png_dir: Path, z: int) -> Path | None:
    for digits in (3, 4, 5):
        p = src_png_dir / f"slice_{z:0{digits}d}.png"
        if p.exists():
            return p
    return None


def _boundary_overview(gt_bbox_path: Path, src_png_dir: Path,
                        pred_sc: dict[int, bool], gt_sc: dict[int, bool],
                        sc_prob: dict[int, float]) -> Image.Image | None:
    if not gt_bbox_path.exists():
        return None
    vals      = list(map(int, gt_bbox_path.read_text().split()))
    z1, z2    = vals[4], vals[5]

    def strip(center_z: int) -> list[Image.Image]:
        imgs = []
        for z in range(center_z - _OVERVIEW_N, center_z + _OVERVIEW_N + 1):
            png = _find_slice_png(src_png_dir, z)
            if png is None:
                continue
            img = _annotate(png, gt_sc.get(z, False), pred_sc.get(z, False), z, sc_prob.get(z, 0.0))
            img = img.resize((int(img.width * _OVERVIEW_H / img.height), _OVERVIEW_H))
            imgs.append(img)
        return imgs

    sup = strip(z1)
    inf = strip(z2)
    if not sup and not inf:
        return None

    sep      = Image.new("RGB", (6, _OVERVIEW_H), (80, 80, 80))
    parts    = sup + ([sep] + inf if sup and inf else inf)
    total_w  = sum(p.width for p in parts)
    overview = Image.new("RGB", (total_w, _OVERVIEW_H), (30, 30, 30))
    x = 0
    for p in parts:
        overview.paste(p, (x, 0))
        x += p.width
    return overview


def infer_patient(cls_model: YOLO, sc_idx: int, patient_dir: Path, pred_dir: Path,
                  cls_conf: float, batch_size: int, save_viz: bool,
                  superior_only: bool = False) -> None:
    pred_txt_dir = pred_dir / "txt"
    pred_vol_dir = pred_dir / "volume"
    pred_txt_dir.mkdir(parents=True, exist_ok=True)
    pred_vol_dir.mkdir(parents=True, exist_ok=True)

    meta     = yaml.safe_load((patient_dir / "meta.yaml").read_text())
    gt_txt_dir = patient_dir / "txt"
    pngs     = sorted((patient_dir / "png").glob("slice_*.png"))

    if superior_only:
        bbox_path = patient_dir / "volume" / "bbox_3d.txt"
        if bbox_path.exists():
            z2 = int(bbox_path.read_text().split()[5])
            pngs = [p for p in pngs if int(p.stem.split("_")[1]) <= z2]

    # per-slice results keyed by z index (position in sorted list = slice filename number)
    pred_sc_per_z:  dict[int, bool]  = {}
    gt_sc_per_z:    dict[int, bool]  = {}
    sc_prob_per_z:  dict[int, float] = {}

    for i in range(0, len(pngs), batch_size):
        chunk   = pngs[i:i + batch_size]
        results = cls_model.predict([str(p) for p in chunk], verbose=False)
        for png, res in zip(chunk, results):
            z       = int(png.stem.split("_")[1])
            prob    = float(res.probs.data[sc_idx])
            is_pred = (res.probs.top1 == sc_idx and prob >= cls_conf)
            gt_txt  = gt_txt_dir / (png.stem + ".txt")
            is_gt   = gt_txt.exists() and gt_txt.stat().st_size > 0

            pred_sc_per_z[z]  = is_pred
            gt_sc_per_z[z]    = is_gt
            sc_prob_per_z[z]  = prob

            # Always write prob — metrics.py sweeps conf thresholds itself
            (pred_txt_dir / (png.stem + ".txt")).write_text(
                f"0 0.500000 0.500000 1.000000 1.000000 {prob:.6f}\n"
            )

    s    = meta["shape_las"]
    H, W = s[1], s[0]
    # bbox_3d bounded by cls_conf-positive slices only (dummy full-image in-plane)
    sc_zs = sorted(z for z, p in pred_sc_per_z.items() if p)
    if sc_zs:
        write_bbox_3d(pred_vol_dir / "bbox_3d.txt", 0, H, 0, W, sc_zs[0], sc_zs[-1])

    shutil.copy2(patient_dir / "meta.yaml", pred_dir / "meta.yaml")
    gt_link = pred_dir / "gt"
    if not gt_link.exists():
        gt_link.symlink_to(patient_dir.resolve())

    if save_viz:
        png_dir = pred_dir / "png"
        png_dir.mkdir(exist_ok=True)
        src_png_dir = patient_dir / "png"
        for z, is_pred in pred_sc_per_z.items():
            png = _find_slice_png(src_png_dir, z)
            if png is None:
                continue
            _annotate(png, gt_sc_per_z[z], is_pred, z, sc_prob_per_z[z]).save(
                str(png_dir / png.name))

        gt_bbox = patient_dir / "volume" / "bbox_3d.txt"
        overview = _boundary_overview(gt_bbox, src_png_dir,
                                      pred_sc_per_z, gt_sc_per_z, sc_prob_per_z)
        if overview is not None:
            overview.save(str(pred_dir / "overview_boundaries.png"))


def run(cls_checkpoint: str | Path, processed: str | Path, inference: str | Path,
        cls_conf: float = 0.5, batch_size: int = 32, save_viz: bool = True,
        superior_only: bool = False) -> None:
    processed_dir = Path(processed)
    pred_root     = Path(inference)

    cls_model = YOLO(str(cls_checkpoint))
    sc_idx    = _sc_class_idx(cls_model)

    patients = [
        (d.name, p)
        for d in sorted(processed_dir.iterdir()) if d.is_dir()
        for p in sorted(d.iterdir())
        if (p / "png").is_dir()
    ]
    print(f"Evaluating {len(patients)} patients [cls only] → {pred_root}")
    print(f"  cls_conf={cls_conf}  sc_class_idx={sc_idx}  viz={save_viz}")

    failed = []
    skipped = 0
    for dataset, patient_dir in tqdm(patients, desc="Patients", unit="pat"):
        pred_dir = pred_root / "predictions" / dataset / patient_dir.name
        if (pred_dir / "meta.yaml").exists():
            skipped += 1
            continue
        try:
            infer_patient(cls_model, sc_idx, patient_dir, pred_dir,
                          cls_conf, batch_size, save_viz, superior_only)
        except Exception as e:
            failed.append((dataset, patient_dir.name, str(e)))

    print(f"Predictions saved to: {pred_root}"
          + (f"  ({skipped} already done, skipped)" if skipped else ""))

    if failed:
        log = pred_root / "failed_patients.log"
        log.write_text("dataset\tpatient\terror\n" +
                       "".join(f"{d}\t{p}\t{e}\n" for d, p, e in failed))
        print(f"WARNING: {len(failed)} patients failed — see {log}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SC/no-SC classifier (gap_mm_S, gap_mm_I + colored PNGs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cls-checkpoint", required=True)
    parser.add_argument("--processed",      required=True)
    parser.add_argument("--inference",      required=True)
    parser.add_argument("--cls-conf",       type=float, default=0.5)
    parser.add_argument("--batch",          type=int,   default=32)
    parser.add_argument("--no-viz",         action="store_true",
                        help="Skip PNG generation (faster)")
    args = parser.parse_args()
    run(cls_checkpoint=args.cls_checkpoint, processed=args.processed,
        inference=args.inference, cls_conf=args.cls_conf,
        batch_size=args.batch, save_viz=not args.no_viz)


if __name__ == "__main__":
    main()
