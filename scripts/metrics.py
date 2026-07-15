#!/usr/bin/env python3
"""
Compute production bbox metrics (gap_mm_* and iou_3d_mm) from saved predictions.

For every patient, at each confidence threshold (CONF_STEPS), reconstructs the
3D pred bbox from per-slice detections (class=0, "sc") and compares it to the
GT 3D bbox (processed/.../volume/bbox_3d.txt):

  iou_3d_mm:          physical-space 3D IoU in mm³; each detected slice contributes
                       si_res_mm in Z depth, in-plane pixels scaled by rl_res_mm/ap_res_mm
  gap_mm_R/L/P/A/I/S: signed mm gap on each face (LAS: R=row_min, L=row_max, P=col_min,
                       A=col_max, I=z_min, S=z_max); positive = pred must expand to contain GT

Per-patient output: predictions/<run_id>/<dataset>/<patient>/metrics/patient.csv
  one row per conf threshold, columns: conf_thresh, iou_3d_mm, gap_mm_R, gap_mm_L,
  gap_mm_P, gap_mm_A, gap_mm_I, gap_mm_S

Run-level output: patients.csv — index of all patients: dataset, stem (no split column).

Usage:
    python scripts/metrics.py \\
        --inference predictions/yolo26_1mm_axial \\
        --processed processed/10mm_SI_1mm_axial

    # Restrict to one split (patients.csv still covers all splits):
    python scripts/metrics.py \\
        --inference predictions/yolo26_1mm_axial \\
        --processed processed/10mm_SI_1mm_axial --split val

    # Classifier predictions (gap_mm_S / gap_mm_I only, no 3D IoU — classifier has no in-plane bbox):
    python scripts/metrics.py \\
        --inference runs/20260601_120000/predictions \\
        --processed processed/10mm_SI_1mm_axial_3ch \\
        --metrics gap_mm_S gap_mm_I
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

CONF_STEPS = np.round(np.array([0.0, 0.001, 0.01, 0.05] + list(np.arange(0.1, 1.01, 0.1))), 3)
METRICS    = ["iou_3d_mm", "gap_mm_R", "gap_mm_L", "gap_mm_P", "gap_mm_A", "gap_mm_I", "gap_mm_S"]


def load_splits(splits_dir: Path) -> dict:
    """Returns {(dataset, subject): split_name} from all datasplit_*.yaml."""
    mapping = {}
    for f in sorted(splits_dir.glob("datasplit_*.yaml")):
        dataset = re.sub(r"_seed\d+$", "", f.stem[len("datasplit_"):])
        for split_name, subjects in yaml.safe_load(f.read_text()).items():
            if not isinstance(subjects, list):  # skip meta block
                continue
            for subj in (subjects or []):
                mapping[(dataset, subj)] = split_name
    return mapping


def read_pred_boxes(pred_txt_dir: Path, class_id: int = 0) -> dict:
    """Returns {z: (cx, cy, w, h, conf, class_id)} for predicted slices of a given class."""
    if not pred_txt_dir.is_dir():
        return {}
    boxes = {}
    for txt in sorted(pred_txt_dir.glob("slice_*.txt")):
        z = int(txt.stem.split("_")[1])
        for line in txt.read_text().splitlines():
            p = line.split()
            if p and int(p[0]) == class_id:
                boxes[z] = (float(p[1]), float(p[2]), float(p[3]), float(p[4]),
                            float(p[5]) if len(p) > 5 else 1.0,
                            class_id)
                break
    return boxes


def iou_3d_mm(b1: list, b2: list, row_res: float, col_res: float, z_res: float) -> float:
    """3D IoU in mm³ between two boxes [row1, row2, col1, col2, z1, z2].

    row_res: mm per pixel along the row axis (AP for axial, SI for sagittal)
    col_res: mm per pixel along the col axis (RL for axial, AP for sagittal)
    z_res:   mm per slice along the z axis  (SI for axial, RL for sagittal)
    A single-slice box (z1=z2) correctly has Z depth = z_res mm.
    """
    inter_r = max(0.0, (min(b1[1], b2[1]) - max(b1[0], b2[0])) * row_res)
    inter_c = max(0.0, (min(b1[3], b2[3]) - max(b1[2], b2[2])) * col_res)
    inter_z = max(0.0, (min(b1[5]+1, b2[5]+1) - max(b1[4], b2[4])) * z_res)
    inter   = inter_r * inter_c * inter_z
    vol1    = (b1[1]-b1[0]) * row_res * (b1[3]-b1[2]) * col_res * (b1[5]-b1[4]+1) * z_res
    vol2    = (b2[1]-b2[0]) * row_res * (b2[3]-b2[2]) * col_res * (b2[5]-b2[4]+1) * z_res
    union   = vol1 + vol2 - inter
    return inter / union if union > 0 else 0.0


def plane_res(meta: dict) -> tuple[float, float, float]:
    """Return (row_res, col_res, z_res) in mm for the plane stored in meta.

    Axial   : rows=AP, cols=RL, z=SI  (row 0=Anterior, col 0=Left, z=0=Superior)
    Sagittal: rows=SI, cols=AP, z=RL
    """
    rl = meta.get("rl_res_mm", 1.0)
    ap = meta.get("ap_res_mm", 1.0)
    si = meta["si_res_mm"]
    if meta.get("plane", "axial") == "sagittal":
        return si, ap, rl
    return ap, rl, si


def get_slice_dims(meta: dict) -> tuple[int, int, int]:
    """Return (H, W, Z) of the slice space from meta.yaml, plane-aware.

    Axial   : H=AP_dim, W=RL_dim, Z=SI_dim  (row 0=Anterior, col 0=Left, z=0=Superior)
    Sagittal: H=SI_dim, W=AP_dim, Z=RL_dim  (slice index along RL, rows=SI after transpose)
    """
    s = meta["shape_las"]
    if meta.get("plane", "axial") == "sagittal":
        return s[2], s[1], s[0]
    return s[1], s[0], s[2]  # axial: H=AP, W=RL


def reconstruct_bbox3d(boxes: dict, H: int, W: int) -> list:
    """Reconstruct 3D bbox union from {z: (cx,cy,w,h,...)}. Returns [row1,row2,col1,col2,z1,z2]."""
    rows1, rows2, cols1, cols2, zs = [], [], [], [], []
    for z, b in boxes.items():
        cx, cy, w, h = b[0], b[1], b[2], b[3]
        rows1.append(max(0, int((cy - h / 2) * H)))
        rows2.append(min(H, int((cy + h / 2) * H)))
        cols1.append(max(0, int((cx - w / 2) * W)))
        cols2.append(min(W, int((cx + w / 2) * W)))
        zs.append(z)
    return [min(rows1), max(rows2), min(cols1), max(cols2), min(zs), max(zs)]


def compute_gap_and_iou(pred_boxes: dict, conf_thresh: float, gt_bbox: list,
                        H: int, W: int, meta: dict) -> dict:
    """Compute iou_3d_mm + gap_mm_R/L/P/A/I/S at one conf threshold. NaN if no active pred or no GT."""
    active = {z: b for z, b in pred_boxes.items() if b[4] >= conf_thresh}
    if not active or gt_bbox is None:
        return {m: float("nan") for m in METRICS}

    row_res, col_res, z_res = plane_res(meta)
    b = reconstruct_bbox3d(active, H, W)

    if meta.get("plane", "axial") == "sagittal":
        # Sagittal: row=SI (0=Superior), col=AP (0=Posterior), z=RL (0=Right)
        gaps = {
            "gap_mm_R": (b[4] - gt_bbox[4]) * z_res,    # z=RL, z_min=Right
            "gap_mm_L": (gt_bbox[5] - b[5]) * z_res,
            "gap_mm_P": (b[2] - gt_bbox[2]) * col_res,  # col=AP, col_min=Posterior
            "gap_mm_A": (gt_bbox[3] - b[3]) * col_res,
            "gap_mm_S": (b[0] - gt_bbox[0]) * row_res,  # row=SI, row_min=Superior
            "gap_mm_I": (gt_bbox[1] - b[1]) * row_res,
        }
    else:
        # Axial: row=AP (0=Anterior), col=RL (0=Left), z=SI (0=Superior)
        gaps = {
            "gap_mm_A": (b[0] - gt_bbox[0]) * row_res,  # row_min=Anterior face
            "gap_mm_P": (gt_bbox[1] - b[1]) * row_res,  # row_max=Posterior face
            "gap_mm_L": (b[2] - gt_bbox[2]) * col_res,  # col_min=Left face
            "gap_mm_R": (gt_bbox[3] - b[3]) * col_res,  # col_max=Right face
            "gap_mm_S": (b[4] - gt_bbox[4]) * z_res,    # z_min=Superior face
            "gap_mm_I": (gt_bbox[5] - b[5]) * z_res,    # z_max=Inferior face
        }

    return {
        "iou_3d_mm": round(iou_3d_mm(b, gt_bbox, row_res, col_res, z_res), 4),
        **{k: round(v, 2) for k, v in gaps.items()},
    }


def run(inference: str | Path, splits_dir: str | Path, processed: str | Path | None = None) -> None:
    """Compute per-patient gap_mm_*/iou_3d_mm metrics from saved predictions."""
    argv = ["--inference", str(inference), "--splits-dir", str(splits_dir)]
    if processed is not None:
        argv += ["--processed", str(processed)]
    main(argv)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute gap_mm_*/iou_3d_mm metrics at all confidence thresholds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference",  required=True,
                        help="Path to inference run directory (predictions/<run-id>/)")
    parser.add_argument("--processed",  default=None,
                        help="processed/<variant> dir (GT source). If omitted, reads from gt/ symlink "
                             "in each patient pred dir (created by evaluate.py).")
    parser.add_argument("--splits-dir", default="data/datasplits_seed50",
                        help="Directory with datasplit_*.yaml (used for split assignment in patients)")
    parser.add_argument("--split",      default=None, choices=["train", "val", "test", "unknown"],
                        help="Restrict computation to subjects in this split (default: all)")
    parser.add_argument("--metrics",    nargs="+", default=METRICS, choices=METRICS,
                        help="Metrics to compute/patch into patient.csv")
    parser.add_argument("--datasets",   nargs="+", default=None,
                        help="Restrict to these dataset names (default: all)")
    args = parser.parse_args(argv)

    pred_root     = Path(args.inference)
    splits_map    = load_splits(Path(args.splits_dir))
    processed_dir = Path(args.processed) if args.processed else None

    def gt_dirs(pred_patient_dir: Path):
        """Return (meta_path, gt_dir): from --processed if given, else from gt/ symlink."""
        if processed_dir is not None:
            proc = processed_dir / pred_patient_dir.parent.name / pred_patient_dir.name
            return proc / "meta.yaml", proc
        return pred_patient_dir / "meta.yaml", pred_patient_dir / "gt"

    # patients: driven by pred_root/predictions/ (what has predictions); processed/ only for GT lookup
    patients = [
        (d.name, p.name)
        for d in sorted((pred_root / "predictions").iterdir()) if d.is_dir()
        and (not args.datasets or d.name in args.datasets)
        for p in sorted(d.iterdir()) if (p / "txt").is_dir()
    ]

    # patients.csv: full index — only (re)written when no --datasets filter is active
    if not args.datasets:
        pd.DataFrame([{"dataset": ds, "stem": st} for ds, st in patients]).to_csv(
            pred_root / "patients.csv", index=False)
        print(f"Patients index → {pred_root / 'patients.csv'} ({len(patients)} patients)")
    else:
        print(f"Skipping patients.csv rewrite (--datasets filter active): {len(patients)} patients")

    if args.split:
        patients = [
            (dataset, stem) for dataset, stem in patients
            if splits_map.get((dataset, re.match(r"(sub-[^_]+)", stem).group(1)), "unknown") == args.split
        ]

    for dataset, stem in tqdm(patients, desc="Metrics", unit="pat"):
        pred_txt_dir        = pred_root / "predictions" / dataset / stem / "txt"
        meta_path, proc_dir = gt_dirs(pred_root / "predictions" / dataset / stem)
        if not meta_path.exists():
            continue
        meta         = yaml.safe_load(meta_path.read_text())
        H, W, _      = get_slice_dims(meta)
        gt_bbox_path = proc_dir / "volume" / "bbox_3d.txt"
        gt_bbox      = list(map(int, gt_bbox_path.read_text().split())) if gt_bbox_path.exists() else None
        pred_boxes   = read_pred_boxes(pred_txt_dir)
        metrics_dir  = pred_root / "predictions" / dataset / stem / "metrics"
        patient_csv  = metrics_dir / "patient.csv"

        df = pd.read_csv(patient_csv) if patient_csv.exists() \
             else pd.DataFrame({"conf_thresh": [round(float(c), 3) for c in CONF_STEPS]})
        computed = [compute_gap_and_iou(pred_boxes, conf, gt_bbox, H, W, meta) for conf in CONF_STEPS]
        for metric in args.metrics:
            df[metric] = [row[metric] for row in computed]

        metrics_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(patient_csv, index=False)

    print(f"Done — {args.metrics}")


if __name__ == "__main__":
    main()
