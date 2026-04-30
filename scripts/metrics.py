#!/usr/bin/env python3
"""
Compute 2D bbox metrics from saved predictions.

For every slice of every patient:
  - iou            : IoU vs same-slice GT (0 if either is absent)
  - iou_nearest_gt : IoU vs nearest GT slice in z (0 if no pred)
  - is_fp          : pred present but no GT on this slice
  - is_fn          : GT present but no pred on this slice
  - pred_class     : predicted class id (-1 if no pred)
  - gt_class       : GT class id (-1 if no GT on this slice)

Per-patient outputs saved to predictions/<run_id>/<dataset>/<patient>/metrics/:
  slices.csv   — one row per slice (source of truth)
  patient.csv  — one row per conf threshold (CONF_STEPS: 0.0, 0.001, 0.01, 0.05, 0.1…1.0)
                 columns: conf_thresh, iou_gt_mean, iou_all_mean, fp_rate, fn_rate,
                          fp_iou_rate, fn_iou_rate, fp_on_gt_rate, iou_3d, iou_3d_mm, iou_sc_mid_box
                 fp_on_gt_rate:       among GT slices with a detection (conf >= thresh), fraction with IoU == 0
                 fp_on_gt_inner_rate: same but restricted to inner GT slices (excluding first and last GT slice in Z)
                 iou_3d:              voxel-space 3D IoU (broken for single-slice GT, kept for reference)
                 iou_3d_mm:           physical-space 3D IoU in mm³; each detected slice contributes
                                      si_res_mm in Z depth, in-plane pixels scaled by rl_res_mm/ap_res_mm
                 iou_sc_mid_box:      3D IoU between pred sc_mid expansion box and GT sc_mid box
                 gap_mm_R/L/P/A/I/S: signed mm gap on each face (LAS: R=row_min, L=row_max, P=col_min, A=col_max, I=z_min, S=z_max); positive = pred must expand to contain GT
                                      pred box: expand from max-conf sc_mid (class=0) slice until
                                                first sc_tip (class=1) or gap in each Z direction
                                      GT box:   union of all GT sc_mid (class=0) slices
                 gap_mm_*/iou_3d_mm with _trim50 suffix: same but pred boundary slices isolated by
                                      >50mm in Z from their neighbor are removed before computing
                                      (analogous to reg30mm but in the Z axis instead of x-y plane)

Run-level outputs:
  patients.csv — index of all patients: dataset, stem (no metrics, no split)
  report.html  — IoU summary per dataset/contrast/split (at --conf threshold)

Usage:
    # Full recompute (all splits):
    python scripts/metrics.py \\
        --inference predictions/yolo26_1mm_axial \\
        --processed processed/10mm_SI_1mm_axial

    # Full recompute restricted to one split (patients.csv still covers all splits):
    python scripts/metrics.py \\
        --inference predictions/yolo26_1mm_axial \\
        --processed processed/10mm_SI_1mm_axial --split val

    # Patch mode: recompute only bbox metrics in existing patient.csv (fast, no slices.csv):
    python scripts/metrics.py \\
        --inference predictions/yolo26_1mm_axial \\
        --processed processed/10mm_SI_1mm_axial --split val --metrics iou_3d_mm iou_3d
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

CONF_STEPS        = np.round(np.array([0.0, 0.001, 0.01, 0.05] + list(np.arange(0.1, 1.01, 0.1))), 3)
CONF_REPORT       = 0.1    # default threshold used only for report.html
SPLITS            = ["test", "val", "train", "unknown"]
REG_DIST_MM       = 30.0   # distance threshold for _reg30mm spatial outlier filter
REG_SUFFIX        = "_reg30mm"
# _trim<N> suffix: N is the 3D Euclidean distance threshold in mm encoded in the metric name
TRIM_DISTANCES   = [30, 40, 50]   # supported values; add more here to expose them
# Metrics that depend only on pred bbox + GT bbox + meta (no slices.csv needed → fast patch)
_TRIM_METRICS    = [f"{pfx}_trim{d}"
                     for d in TRIM_DISTANCES
                     for pfx in ("iou_3d_mm",
                                 "gap_mm_R", "gap_mm_L", "gap_mm_P",
                                 "gap_mm_A", "gap_mm_I", "gap_mm_S")]
BBOX_ONLY_METRICS = {"iou_3d", "iou_3d_mm", "iou_3d_mm_filt", "iou_3d_mm_ransac",
                     "iou_3d_mm_pad10", "gt_in_pad10",
                     "iou_3d_mm_padz20", "gt_in_padz20",
                     "pred_vol_ratio",
                     "gap_mm_R", "gap_mm_L", "gap_mm_P", "gap_mm_A",
                     "gap_mm_I", "gap_mm_S",
                     "iou_sc_mid_box",
                     "iou_3d_mm_reg30mm",
                     "gap_mm_R_reg30mm", "gap_mm_L_reg30mm", "gap_mm_P_reg30mm",
                     "gap_mm_A_reg30mm", "gap_mm_I_reg30mm", "gap_mm_S_reg30mm",
                     *_TRIM_METRICS}

CSS = """
<style>
  body { font-family: Arial, sans-serif; font-size: 13px; margin: 30px; background: #f9f9f9; }
  h1   { color: #222; }
  h2   { color: #444; margin-top: 40px; border-bottom: 2px solid #ccc; padding-bottom: 4px; }
  h3   { color: #555; margin-top: 30px; }
  table { border-collapse: collapse; margin-bottom: 20px; background: white;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  th   { background: #3a3a3a; color: white; padding: 7px 12px; text-align: center; }
  th.left { text-align: left; }
  td   { padding: 5px 12px; border-bottom: 1px solid #e0e0e0; text-align: center; }
  td.left { text-align: left; }
  tr:hover td { background: #f0f4ff; }
  .good  { color: #1a7a1a; font-weight: bold; }
  .ok    { color: #7a6a00; }
  .bad   { color: #aa1a1a; }
  .na    { color: #aaa; font-style: italic; }
  .meta  { font-size: 11px; color: #888; }
</style>
"""


def load_splits(splits_dir: Path) -> dict:
    """Returns {(dataset, subject): split_name} from all datasplit_*.yaml."""
    mapping = {}
    for f in sorted(splits_dir.glob("datasplit_*.yaml")):
        dataset = re.sub(r"_seed\d+$", "", f.stem[len("datasplit_"):])
        for split_name, subjects in yaml.safe_load(f.read_text()).items():
            for subj in (subjects or []):
                mapping[(dataset, subj)] = split_name
    return mapping


def read_gt_box(txt_path: Path):
    """Returns (cx,cy,w,h) or None if file empty/missing."""
    if not txt_path.exists():
        return None
    content = txt_path.read_text().strip()
    if not content:
        return None
    p = content.split()
    return (float(p[1]), float(p[2]), float(p[3]), float(p[4]))


def read_pred(txt_path: Path):
    """Returns ((cx,cy,w,h), conf) or (None, 0.0) if empty/missing."""
    if not txt_path.exists():
        return None, 0.0
    content = txt_path.read_text().strip()
    if not content:
        return None, 0.0
    p = content.split()
    return (float(p[1]), float(p[2]), float(p[3]), float(p[4])), (float(p[5]) if len(p) > 5 else 1.0)


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


def read_gt_boxes(gt_txt_dir: Path, class_id: int = 0) -> dict:
    """Returns {z: (cx, cy, w, h, class_id)} for GT slices of a given class."""
    boxes = {}
    for txt in sorted(gt_txt_dir.glob("slice_*.txt")):
        z = int(txt.stem.split("_")[1])
        for line in txt.read_text().splitlines():
            p = line.split()
            if p and int(p[0]) == class_id:
                boxes[z] = (float(p[1]), float(p[2]), float(p[3]), float(p[4]), class_id)
                break
    return boxes


def iou_3d(b1: list, b2: list) -> float:
    """3D IoU between two boxes [row1, row2, col1, col2, z1, z2] in voxel space.

    Note: returns 0 if either box has zero extent in any axis (e.g. single-slice GT with z1=z2).
    Use iou_3d_mm for physically correct results.
    """
    inter_r = max(0, min(b1[1], b2[1]) - max(b1[0], b2[0]))
    inter_c = max(0, min(b1[3], b2[3]) - max(b1[2], b2[2]))
    inter_z = max(0, min(b1[5], b2[5]) - max(b1[4], b2[4]))
    inter   = inter_r * inter_c * inter_z
    vol1    = (b1[1]-b1[0]) * (b1[3]-b1[2]) * (b1[5]-b1[4])
    vol2    = (b2[1]-b2[0]) * (b2[3]-b2[2]) * (b2[5]-b2[4])
    union   = vol1 + vol2 - inter
    return inter / union if union > 0 else 0.0


def iou_3d_mm(b1: list, b2: list, row_res: float, col_res: float, z_res: float) -> float:
    """3D IoU in mm³ between two boxes [row1, row2, col1, col2, z1, z2].

    row_res: mm per pixel along the row axis (RL for axial, SI for sagittal)
    col_res: mm per pixel along the col axis (AP for both planes)
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

    Axial   : rows=RL, cols=AP, z=SI
    Sagittal: rows=SI, cols=AP, z=RL
    """
    rl = meta.get("rl_res_mm", 1.0)
    ap = meta.get("ap_res_mm", 1.0)
    si = meta["si_res_mm"]
    if meta.get("plane", "axial") == "sagittal":
        return si, ap, rl
    return rl, ap, si


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


def bbox_iou(a, b) -> float:
    """IoU between two (cx,cy,w,h) normalised bboxes."""
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    inter = max(0., min(ax2,bx2)-max(ax1,bx1)) * max(0., min(ay2,by2)-max(ay1,by1))
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.


def patient_slices(gt_txt_dir: Path, pred_txt_dir: Path) -> pd.DataFrame:
    """Build one row per slice for a patient.

    pred_class / gt_class: predicted / GT class id (-1 if absent on this slice).
    """
    gt_txts   = {int(p.stem.split("_")[1]): p for p in sorted(gt_txt_dir.glob("slice_*.txt"))}
    pred_txts = {int(p.stem.split("_")[1]): p
                 for p in sorted(pred_txt_dir.glob("slice_*.txt"))} if pred_txt_dir.is_dir() else {}

    all_z = sorted(set(gt_txts) | set(pred_txts))
    if not all_z:
        return pd.DataFrame()

    gt_by_z, gt_class_by_z = {}, {}
    for z, p in gt_txts.items():
        raw = p.read_text().strip()
        if raw:
            parts = raw.split()
            gt_class_by_z[z] = int(parts[0])
            gt_by_z[z] = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))

    gt_slices = np.array(sorted(gt_by_z)) if gt_by_z else np.array([], dtype=int)

    rows = []
    for z in all_z:
        gt_box   = gt_by_z.get(z)
        gt_class = gt_class_by_z.get(z, -1)

        pred_box, pred_conf, pred_class = None, 0.0, -1
        if z in pred_txts:
            for line in pred_txts[z].read_text().splitlines():
                p = line.split()
                if p and int(p[0]) == 0:   # class 0 = SC
                    pred_class = 0
                    pred_box   = (float(p[1]), float(p[2]), float(p[3]), float(p[4]))
                    pred_conf  = float(p[5]) if len(p) > 5 else 1.0
                    break

        has_gt   = gt_box is not None
        has_pred = pred_box is not None
        iou      = bbox_iou(gt_box, pred_box) if (has_gt and has_pred) else 0.0

        if not has_pred or len(gt_slices) == 0:
            iou_nearest_gt, z_dist_to_ref, ref_gt_slice = 0.0, None, None
        else:
            dists          = np.abs(gt_slices - z)
            nearest_z      = int(gt_slices[dists.argmin()])
            z_dist_to_ref  = int(dists.min())
            ref_gt_slice   = nearest_z
            iou_nearest_gt = bbox_iou(gt_by_z[nearest_z], pred_box)

        rows.append({
            "slice_idx":        z,
            "has_gt":           has_gt,
            "has_pred":         has_pred,
            "pred_conf":        round(pred_conf, 4),
            "pred_class":       pred_class,
            "gt_class":         gt_class,
            "iou":              round(iou, 4),
            "iou_nearest_gt":   round(iou_nearest_gt, 4),
            "z_dist_to_ref_gt": z_dist_to_ref,
            "ref_gt_slice":     ref_gt_slice,
            "is_fp":            has_pred and not has_gt,
            "is_fn":            has_gt and not has_pred,
        })
    return pd.DataFrame(rows)


def sc_mid_box_iou(pred_boxes: dict, conf_thresh: float, gt_boxes: dict,
                   H: int, W: int) -> float:
    """3D IoU between pred sc_mid expansion box and GT sc_mid box.

    Pred: among detections with conf >= conf_thresh, start from max-conf sc_mid (class=0)
          slice, expand in both Z directions stopping at first sc_tip (class=1) or gap.
    GT:   union box of all GT slices.
    Returns nan if no sc_mid pred above threshold or no GT slices.
    """
    active = {z: b for z, b in pred_boxes.items() if b[4] >= conf_thresh}

    sc_mid_active = {z: b for z, b in active.items() if b[5] == 0}
    if not sc_mid_active:
        return float("nan")
    z_peak = max(sc_mid_active, key=lambda z: sc_mid_active[z][4])

    selected = {z_peak}
    for step in (+1, -1):
        z = z_peak + step
        while z in active:
            selected.add(z)
            if active[z][5] == 1:   # first sc_tip encountered → include and stop
                break
            z += step

    pred_3d = reconstruct_bbox3d({z: active[z] for z in selected}, H, W)

    if not gt_boxes:
        return float("nan")
    gt_3d = reconstruct_bbox3d(gt_boxes, H, W)

    return round(iou_3d(pred_3d, gt_3d), 4)


def ransac_filter_slices(boxes: dict) -> dict:
    """Keep RANSAC inliers of a linear z → (cx, cy) fit. Requires >= 3 slices.

    Fits two independent RANSAC linear regressors (z → cx, z → cy).
    A slice is kept if it is an inlier in both fits.
    Falls back to the full set if all slices would be removed.
    """
    if len(boxes) < 3:
        return boxes
    zs      = np.array(sorted(boxes)).reshape(-1, 1)
    cxs     = np.array([boxes[z][0] for z in zs.ravel()])
    cys     = np.array([boxes[z][1] for z in zs.ravel()])
    inliers = (RANSACRegressor(min_samples=2, random_state=0).fit(zs, cxs).inlier_mask_ &
               RANSACRegressor(min_samples=2, random_state=0).fit(zs, cys).inlier_mask_)
    return {z: boxes[z] for z, keep in zip(zs.ravel(), inliers) if keep} or boxes


def filter_outlier_slices(boxes: dict) -> dict:
    """Remove slices with IoU=0 against every other slice (only when >= 3 slices).

    Keeps a slice if it overlaps (IoU > 0) with at least one other slice.
    Falls back to the full set if filtering would remove everything.
    """
    if len(boxes) < 3:
        return boxes
    zs   = list(boxes)
    keep = {z: boxes[z] for z in zs
            if any(bbox_iou(boxes[z][:4], boxes[other][:4]) > 0 for other in zs if other != z)}
    return keep if keep else boxes


def reg_dist_filter_slices(boxes: dict, H: int, W: int,
                           rl_res: float, ap_res: float, max_dist_mm: float) -> dict:
    """Remove predictions whose center is farther than max_dist_mm from all other centers (requires >= 3 slices).

    Centers in mm: cx_mm = cx * W * ap_res, cy_mm = cy * H * rl_res.
    Falls back to the full set if filtering would remove everything.
    """
    if len(boxes) < 3:
        return boxes
    zs  = list(boxes)
    pts = np.array([(boxes[z][0] * W * ap_res, boxes[z][1] * H * rl_res) for z in zs])
    keep = [
        z for i, z in enumerate(zs)
        if np.any(np.sqrt(np.sum((pts[i] - np.delete(pts, i, axis=0)) ** 2, axis=1)) <= max_dist_mm)
    ]
    return {z: boxes[z] for z in keep} if keep else boxes


def trim_z_boundary(boxes: dict, trim_mm: float, si_res: float, H: int, W: int,
                    rl_res: float, ap_res: float) -> dict:
    """Remove topmost/bottommost pred slice if its 3D Euclidean distance to its neighbor exceeds trim_mm.

    Requires >= 3 slices to trim: with only 2 slices it is impossible to determine which one is the
    outlier, so the full set is returned unchanged.

    Distance between two detections = sqrt(dZ² + dRL² + dAP²) in mm, where:
      dZ  = (z2 - z1) * si_res
      dRL = (cy2 - cy1) * H * rl_res   (cy normalised by H)
      dAP = (cx2 - cx1) * W * ap_res   (cx normalised by W)
    """
    if len(boxes) < 3:
        return boxes

    def dist3d(z1, z2):
        b1, b2 = boxes[z1], boxes[z2]
        dz  = (z2 - z1) * si_res
        drl = (b2[1] - b1[1]) * H * rl_res
        dap = (b2[0] - b1[0]) * W * ap_res
        return (dz**2 + drl**2 + dap**2) ** 0.5

    zs      = sorted(boxes)
    trimmed = dict(boxes)
    if dist3d(zs[-2], zs[-1]) > trim_mm:
        del trimmed[zs[-1]]
        zs = zs[:-1]
    if len(zs) >= 3 and dist3d(zs[0], zs[1]) > trim_mm:
        del trimmed[zs[0]]
    return trimmed


def get_slice_dims(meta: dict) -> tuple[int, int, int]:
    """Return (H, W, Z) of the slice space from meta.yaml, plane-aware.

    Axial   : H=RL_dim, W=AP_dim, Z=SI_dim  (slice index along SI)
    Sagittal: H=SI_dim, W=AP_dim, Z=RL_dim  (slice index along RL, rows=SI after transpose)
    """
    s = meta["shape_las"]
    if meta.get("plane", "axial") == "sagittal":
        return s[2], s[1], s[0]
    return s[0], s[1], s[2]


def pad_bbox3d(bbox: list, pad_xy_mm: float, pad_z_mm: float, H: int, W: int, Z: int,
               si_res: float, rl_res: float, ap_res: float) -> list:
    """Expand bbox on all 6 faces (pad_xy_mm in-plane, pad_z_mm along Z), clamped to image bounds."""
    pad_r = pad_xy_mm / rl_res
    pad_c = pad_xy_mm / ap_res
    pad_z = pad_z_mm  / si_res
    return [max(0.0,      bbox[0] - pad_r), min(float(H), bbox[1] + pad_r),
            max(0.0,      bbox[2] - pad_c), min(float(W), bbox[3] + pad_c),
            max(0.0,      bbox[4] - pad_z), min(float(Z - 1), bbox[5] + pad_z)]


def compute_bbox_column(metric: str, pred_boxes: dict, conf_thresh: float,
                        gt_bbox: list, gt_boxes: dict, H: int, W: int, meta: dict) -> float:
    """Compute one bbox-only metric at one conf threshold (no slices_df needed).

    Metrics with suffix _reg30mm apply a spatial outlier filter (30mm) before computing
    the base metric: predictions isolated > 30mm from all others are removed (requires >= 3 preds).
    """
    trim_m = re.search(r"_trim(\d+)$", metric)
    if trim_m:
        trim_mm  = float(trim_m.group(1))
        active   = {z: b for z, b in pred_boxes.items() if b[4] >= conf_thresh}
        filtered = trim_z_boundary(active, trim_mm, meta["si_res_mm"], H, W,
                                   meta.get("rl_res_mm", 1.0), meta.get("ap_res_mm", 1.0))
        return compute_bbox_column(metric[:trim_m.start()], filtered, 0.0,
                                   gt_bbox, gt_boxes, H, W, meta)
    if metric.endswith(REG_SUFFIX):
        active = {z: b for z, b in pred_boxes.items() if b[4] >= conf_thresh}
        filtered = reg_dist_filter_slices(active, H, W,
                                          meta.get("rl_res_mm", 1.0), meta.get("ap_res_mm", 1.0),
                                          REG_DIST_MM)
        return compute_bbox_column(metric[:-len(REG_SUFFIX)], filtered, 0.0,
                                   gt_bbox, gt_boxes, H, W, meta)
    active = {z: b for z, b in pred_boxes.items() if b[4] >= conf_thresh}
    if metric == "iou_3d":
        if not active or gt_bbox is None:
            return float("nan")
        return round(iou_3d(reconstruct_bbox3d(active, H, W), gt_bbox), 4)
    if metric == "iou_3d_mm":
        if not active or gt_bbox is None:
            return float("nan")
        return round(iou_3d_mm(reconstruct_bbox3d(active, H, W), gt_bbox, *plane_res(meta)), 4)
    if metric == "iou_3d_mm_filt":
        if not active or gt_bbox is None:
            return float("nan")
        return round(iou_3d_mm(reconstruct_bbox3d(filter_outlier_slices(active), H, W), gt_bbox,
                               *plane_res(meta)), 4)
    if metric == "iou_3d_mm_ransac":
        if not active or gt_bbox is None:
            return float("nan")
        return round(iou_3d_mm(reconstruct_bbox3d(ransac_filter_slices(active), H, W), gt_bbox,
                               *plane_res(meta)), 4)
    if metric in ("iou_3d_mm_pad10", "gt_in_pad10", "iou_3d_mm_padz20", "gt_in_padz20"):
        if not active or gt_bbox is None:
            return float("nan")
        _, _, Z  = get_slice_dims(meta)
        row_res, col_res, z_res = plane_res(meta)
        pad_xy   = 10.0
        pad_z    = 20.0 if "padz20" in metric else 10.0
        padded   = pad_bbox3d(reconstruct_bbox3d(active, H, W), pad_xy, pad_z, H, W, Z,
                              meta["si_res_mm"], meta.get("rl_res_mm", 1.0), meta.get("ap_res_mm", 1.0))
        if "iou_3d_mm" in metric:
            return round(iou_3d_mm(padded, gt_bbox, row_res, col_res, z_res), 4)
        return float(padded[0] <= gt_bbox[0] and gt_bbox[1] <= padded[1] and
                     padded[2] <= gt_bbox[2] and gt_bbox[3] <= padded[3] and
                     padded[4] <= gt_bbox[4] and gt_bbox[5] <= padded[5])
    if metric == "pred_vol_ratio":
        if not active:
            return float("nan")
        _, _, Z = get_slice_dims(meta)
        si_res = meta["si_res_mm"]
        rl_res = meta.get("rl_res_mm", 1.0)
        ap_res = meta.get("ap_res_mm", 1.0)
        b      = reconstruct_bbox3d(active, H, W)
        pred_mm3  = (b[1]-b[0])*rl_res * (b[3]-b[2])*ap_res * (b[5]-b[4]+1)*si_res
        total_mm3 = H*rl_res * W*ap_res * Z*si_res
        return round(pred_mm3 / total_mm3, 6)
    if metric in ("gap_mm_R", "gap_mm_L", "gap_mm_P", "gap_mm_A", "gap_mm_I", "gap_mm_S"):
        if not active or gt_bbox is None:
            return float("nan")
        si_res = meta["si_res_mm"]
        rl_res = meta.get("rl_res_mm", 1.0)
        ap_res = meta.get("ap_res_mm", 1.0)
        b = reconstruct_bbox3d(active, H, W)
        if meta.get("plane", "axial") == "sagittal":
            # Sagittal: row=SI (flipped, 0=Superior), col=AP, z=RL
            # row_min=Superior, row_max=Inferior, col_min=Posterior, col_max=Anterior, z_min=Right, z_max=Left
            gaps = {
                "gap_mm_R": (b[4] - gt_bbox[4]) * rl_res,
                "gap_mm_L": (gt_bbox[5] - b[5]) * rl_res,
                "gap_mm_P": (b[2] - gt_bbox[2]) * ap_res,
                "gap_mm_A": (gt_bbox[3] - b[3]) * ap_res,
                "gap_mm_S": (b[0] - gt_bbox[0]) * si_res,
                "gap_mm_I": (gt_bbox[1] - b[1]) * si_res,
            }
        else:
            # Axial: row=RL (0=Right), col=AP (0=Posterior), z=SI (0=Inferior)
            gaps = {
                "gap_mm_R": (b[0] - gt_bbox[0]) * rl_res,
                "gap_mm_L": (gt_bbox[1] - b[1]) * rl_res,
                "gap_mm_P": (b[2] - gt_bbox[2]) * ap_res,
                "gap_mm_A": (gt_bbox[3] - b[3]) * ap_res,
                "gap_mm_I": (b[4] - gt_bbox[4]) * si_res,
                "gap_mm_S": (gt_bbox[5] - b[5]) * si_res,
            }
        return round(gaps[metric], 2)
    if metric == "iou_sc_mid_box":
        return sc_mid_box_iou(pred_boxes, conf_thresh, gt_boxes, H, W)
    return float("nan")


def build_patient_csv_rows(slices_df: pd.DataFrame, pred_boxes: dict, H: int, W: int,
                            gt_bbox: list, gt_boxes: dict, fp_iou_thresh: float,
                            meta: dict) -> pd.DataFrame:
    """One row per conf threshold: aggregated metrics + iou_3d / iou_3d_mm reconstructed at each threshold."""
    gt_zs        = slices_df.loc[slices_df["has_gt"].astype(bool), "slice_idx"]
    is_inner_gt  = (slices_df["has_gt"].astype(bool)
                    & (slices_df["slice_idx"] > gt_zs.min())
                    & (slices_df["slice_idx"] < gt_zs.max())) if len(gt_zs) > 2 else pd.Series(False, index=slices_df.index)

    rows = []
    for conf_thresh in CONF_STEPS:
        has_gt   = slices_df["has_gt"].astype(bool)
        has_pred = slices_df["has_pred"].astype(bool) & (slices_df["pred_conf"] >= conf_thresh)
        iou      = slices_df["iou"].where(has_pred, 0.0)

        gt_count                = int(has_gt.sum())
        pred_count              = int(has_pred.sum())
        total                   = len(slices_df)
        gt_with_pred_count      = int((has_gt & has_pred).sum())
        inner_gt_with_pred_count = int((is_inner_gt & has_pred).sum())

        active = {z: b for z, b in pred_boxes.items() if b[4] >= conf_thresh}
        if active and gt_bbox is not None:
            pred_3d_box = reconstruct_bbox3d(active, H, W)
            iou3d    = round(iou_3d(pred_3d_box, gt_bbox), 4)
            iou3d_mm = round(iou_3d_mm(pred_3d_box, gt_bbox, *plane_res(meta)), 4)
        else:
            iou3d    = float("nan")
            iou3d_mm = float("nan")

        rows.append({
            "conf_thresh":    round(float(conf_thresh), 3),
            "iou_gt_mean":    round(float(iou[has_gt].mean()), 4)                                    if gt_count > 0          else float("nan"),
            "iou_all_mean":   round(float(iou.mean()), 4)                                            if total > 0             else float("nan"),
            "fp_rate":        round(float((has_pred & ~has_gt).sum()) / total, 4)                    if total > 0             else float("nan"),
            "fn_rate":        round(float((has_gt & ~has_pred).sum()) / gt_count, 4)                 if gt_count > 0          else float("nan"),
            "fp_iou_rate":    round(float((has_pred & (iou < fp_iou_thresh)).sum()) / pred_count, 4) if pred_count > 0        else float("nan"),
            "fn_iou_rate":    round(float((has_gt & (iou < fp_iou_thresh)).sum()) / gt_count, 4)     if gt_count > 0          else float("nan"),
            "fp_on_gt_rate":       round(float((has_gt & has_pred & (slices_df["iou"] == 0)).sum()) / gt_with_pred_count, 4)
                                   if gt_with_pred_count > 0 else float("nan"),
            "fp_on_gt_inner_rate": round(float((is_inner_gt & has_pred & (slices_df["iou"] == 0)).sum()) / inner_gt_with_pred_count, 4)
                                   if inner_gt_with_pred_count > 0 else float("nan"),
            "iou_3d":           iou3d,
            "iou_3d_mm":        iou3d_mm,
            "iou_sc_mid_box":   sc_mid_box_iou(pred_boxes, conf_thresh, gt_boxes, H, W),
        })
    return pd.DataFrame(rows)


def ap_at_iou(df: pd.DataFrame, iou_col: str, iou_thresh: float) -> float:
    n_gt = int(df["has_gt"].sum())
    if n_gt == 0:
        return float("nan")
    preds = df[df["has_pred"]].sort_values("pred_conf", ascending=False)
    if len(preds) == 0:
        return 0.0
    tp        = (preds["has_gt"] & (preds[iou_col] >= iou_thresh)).astype(int).values
    cum_tp    = np.cumsum(tp)
    cum_fp    = np.cumsum(1 - tp)
    precision = cum_tp / (cum_tp + cum_fp)
    recall    = cum_tp / n_gt
    precision = np.concatenate([[1.0], precision])
    recall    = np.concatenate([[0.0], recall])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    return float(np.trapz(precision, recall))


def summarise_group(df: pd.DataFrame, conf_thresh: float) -> dict:
    n_slices    = len(df)
    n_gt        = int(df["has_gt"].sum())
    n_pred      = int(df["has_pred"].sum())
    n_fp        = int(df["is_fp"].sum())
    n_fn        = int(df["is_fn"].sum())
    fp_rate     = round(n_fp / n_pred, 4) if n_pred else float("nan")
    fn_rate     = round(n_fn / n_gt,   4) if n_gt   else float("nan")
    has_thresh  = df["has_pred"] & (df["pred_conf"] >= conf_thresh)
    n_thresh    = int(has_thresh.sum())
    matched     = df[df["has_gt"] & df["has_pred"]]
    iou_mean    = round(float(matched["iou"].mean()), 4) if len(matched) else float("nan")
    dice_mean   = round(float((2*matched["iou"]/(1+matched["iou"])).mean()), 4) if len(matched) else float("nan")
    tp50        = int((df["has_gt"] & has_thresh & (df["iou"] >= 0.5)).sum())
    recall50    = tp50 / n_gt     if n_gt     else float("nan")
    precision50 = tp50 / n_thresh if n_thresh else float("nan")
    denom       = (precision50 + recall50) if not (np.isnan(precision50) or np.isnan(recall50)) else 0.
    f1_50       = 2 * precision50 * recall50 / denom if denom > 0 else float("nan")
    ap50        = ap_at_iou(df, "iou", 0.50)
    ap50_95     = float(np.nanmean([ap_at_iou(df, "iou", t) for t in np.arange(0.50, 1.00, 0.05)]))
    return {
        "n_slices": n_slices, "n_gt": n_gt, "n_pred": n_pred,
        "n_fp": n_fp, "n_fn": n_fn, "fp_rate": fp_rate, "fn_rate": fn_rate,
        "iou_mean": iou_mean, "dice_mean": dice_mean,
        "recall50":    round(recall50,    4) if not np.isnan(recall50)    else float("nan"),
        "precision50": round(precision50, 4) if not np.isnan(precision50) else float("nan"),
        "f1_50":       round(f1_50,       4) if not np.isnan(f1_50)       else float("nan"),
        "ap50": round(ap50, 4), "ap50_95": round(ap50_95, 4),
    }


def build_report(df: pd.DataFrame, conf_thresh: float) -> pd.DataFrame:
    rows = []

    def add(g, split="ALL", dataset="ALL", contrast="ALL"):
        rows.append({"split": split, "dataset": dataset, "contrast": contrast,
                     **summarise_group(g, conf_thresh)})

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


def iou_class(v):
    if pd.isna(v): return "na"
    if v >= 0.8:   return "good"
    if v >= 0.6:   return "ok"
    return "bad"


def fmt_cell(iou, n_acq, n_slices):
    if pd.isna(iou) or n_slices == 0:
        return '<span class="na">—</span>'
    cls = iou_class(iou)
    return (f'<span class="{cls}">{iou:.3f}</span>'
            f'<br><span class="meta">{n_acq}acq / {n_slices}sl</span>')


def count_acq_per_split(inference_dir: Path, splits_map: dict) -> dict:
    by_dataset  = {}
    by_contrast = {}
    for slices_csv in inference_dir.rglob("metrics/slices.csv"):
        patient_dir = slices_csv.parent.parent
        dataset     = patient_dir.parent.name
        stem        = patient_dir.name
        m           = re.match(r"(sub-[^_]+)_?(.*)", stem)
        subject     = m.group(1) if m else stem
        contrast    = m.group(2) if m and m.group(2) else "default"
        split       = splits_map.get((dataset, subject), "unknown")
        by_dataset[(split, dataset)]            = by_dataset.get((split, dataset), 0) + 1
        by_contrast[(split, dataset, contrast)] = by_contrast.get((split, dataset, contrast), 0) + 1
    return by_dataset, by_contrast


def make_table_by_dataset(report: pd.DataFrame, acq_counts: dict, splits: list) -> str:
    sub      = report[(report["contrast"] == "ALL") & (report["dataset"] != "ALL") & (report["split"].isin(splits))]
    datasets = sorted(sub["dataset"].unique())
    headers  = '<th class="left">Dataset</th>' + "".join(
        f'<th>{s}<br><span style="font-size:10px;font-weight:normal">(acq / slices)</span></th>'
        for s in splits)
    html = f"<table><thead><tr>{headers}</tr></thead><tbody>\n"
    for dataset in datasets:
        row = f'<td class="left">{dataset}</td>'
        for split in splits:
            r = sub[(sub["dataset"] == dataset) & (sub["split"] == split)]
            if r.empty:
                row += '<td><span class="na">—</span></td>'
            else:
                row += f"<td>{fmt_cell(r.iloc[0]['iou_mean'], acq_counts.get((split, dataset), 0), int(r.iloc[0]['n_slices']))}</td>"
        html += f"<tr>{row}</tr>\n"
    html += "</tbody></table>"
    return html


def make_table_by_contrast(report: pd.DataFrame, acq_by_contrast: dict, dataset: str, splits: list) -> str:
    sub       = report[(report["dataset"] == dataset) & (report["contrast"] != "ALL") & (report["split"].isin(splits))]
    contrasts = sorted(sub["contrast"].unique())
    if not contrasts:
        return "<p><em>Aucune donnée par contraste.</em></p>"
    headers = '<th class="left">Contrast</th>' + "".join(
        f'<th>{s}<br><span style="font-size:10px;font-weight:normal">(acq / slices)</span></th>'
        for s in splits)
    html = f"<table><thead><tr>{headers}</tr></thead><tbody>\n"
    for contrast in contrasts:
        row = f'<td class="left">{contrast}</td>'
        for split in splits:
            r = sub[(sub["contrast"] == contrast) & (sub["split"] == split)]
            if r.empty:
                row += '<td><span class="na">—</span></td>'
            else:
                n_acq = acq_by_contrast.get((split, dataset, contrast), 0)
                row += f"<td>{fmt_cell(r.iloc[0]['iou_mean'], n_acq, int(r.iloc[0]['n_slices']))}</td>"
        html += f"<tr>{row}</tr>\n"
    html += "</tbody></table>"
    return html


def build_html(report: pd.DataFrame, acq_by_dataset: dict, acq_by_contrast: dict,
               run_id: str, conf: float, splits: list) -> str:
    body  = f"<h1>Metrics report — {run_id}</h1>\n"
    body += f"<p><em>Report threshold: conf ≥ {conf}</em></p>\n"
    body += "<h2>IoU mean par dataset</h2>\n"
    body += make_table_by_dataset(report, acq_by_dataset, splits)
    body += "<h2>IoU mean par dataset × contraste</h2>\n"
    datasets = sorted(report[(report["dataset"] != "ALL") & (report["contrast"] != "ALL")]["dataset"].unique())
    for dataset in datasets:
        body += f"<h3>{dataset}</h3>\n"
        body += make_table_by_contrast(report, acq_by_contrast, dataset, splits)
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Metrics {run_id}</title>{CSS}</head><body>{body}</body></html>"


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-slice and per-patient metrics at all confidence thresholds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference",    required=True,
                        help="Path to inference run directory (predictions/<run-id>/)")
    parser.add_argument("--processed",    default=None,
                        help="processed/<variant> dir (GT source). If omitted, reads from gt/ symlink "
                             "in each patient pred dir (created by evaluate.py).")
    parser.add_argument("--splits-dir",   default="data/datasplits/from_raw",
                        help="Directory with datasplit_*.yaml (used for report.html only)")
    parser.add_argument("--conf",         type=float, default=CONF_REPORT,
                        help="Confidence threshold used for report.html summary")
    parser.add_argument("--fp-iou-thresh", type=float, default=0.1,
                        help="IoU threshold for fp_iou_rate / fn_iou_rate in patient.csv")
    parser.add_argument("--split",         default=None, choices=["train", "val", "test", "unknown"],
                        help="Restrict computation to subjects in this split (default: all)")
    parser.add_argument("--metrics",       nargs="+",
                        default=["iou_3d_mm",
                                 "gap_mm_R", "gap_mm_L", "gap_mm_P", "gap_mm_A", "gap_mm_I", "gap_mm_S"],
                        choices=sorted(BBOX_ONLY_METRICS),
                        help="Patch only these bbox metrics in existing patient.csv (skips slices.csv)")
    parser.add_argument("--datasets",      nargs="+", default=None,
                        help="Restrict to these dataset names (default: all)")
    parser.add_argument("--no-html",       action="store_true",
                        help="Skip generating report.html")
    args = parser.parse_args()

    pred_root     = Path(args.inference)
    run_id        = pred_root.name
    splits_map    = load_splits(Path(args.splits_dir))
    processed_dir = Path(args.processed) if args.processed else None

    def gt_dirs(pred_patient_dir: Path):
        """Return (meta_path, gt_dir): from --processed if given, else from gt/ symlink."""
        if processed_dir is not None:
            proc = processed_dir / pred_patient_dir.parent.name / pred_patient_dir.name
            return proc / "meta.yaml", proc
        return pred_patient_dir / "meta.yaml", pred_patient_dir / "gt"

    # patients: driven by pred_root (what has predictions); processed/ only for GT lookup
    patients = [
        (d.name, p.name)
        for d in sorted(pred_root.iterdir()) if d.is_dir()
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

    # --- bbox-only mode: compute only the requested metrics, no slices.csv needed ---
    if args.metrics:
        for dataset, stem in tqdm(patients, desc="Metrics", unit="pat"):
            pred_txt_dir         = pred_root / dataset / stem / "txt"
            meta_path, proc_dir  = gt_dirs(pred_root / dataset / stem)
            if not meta_path.exists():
                continue
            meta         = yaml.safe_load(meta_path.read_text())
            H, W, _      = get_slice_dims(meta)
            gt_bbox_path = proc_dir / "volume" / "bbox_3d.txt"
            gt_bbox      = list(map(int, gt_bbox_path.read_text().split())) if gt_bbox_path.exists() else None
            pred_boxes   = read_pred_boxes(pred_txt_dir)
            gt_boxes     = read_gt_boxes(proc_dir / "txt")
            metrics_dir  = pred_root / dataset / stem / "metrics"
            patient_csv  = metrics_dir / "patient.csv"
            df = pd.read_csv(patient_csv) if patient_csv.exists() \
                 else pd.DataFrame({"conf_thresh": [round(float(c), 3) for c in CONF_STEPS]})
            for metric in args.metrics:
                df[metric] = [
                    compute_bbox_column(metric, pred_boxes, conf, gt_bbox, gt_boxes, H, W, meta)
                    for conf in CONF_STEPS
                ]
            metrics_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(patient_csv, index=False)
        print(f"Done — {args.metrics}")
        return

    # --- full mode ---
    all_records = []
    for dataset, stem in tqdm(patients, desc="Patients", unit="pat"):
        m        = re.match(r"(sub-[^_]+)_?(.*)", stem)
        subject  = m.group(1)
        contrast = m.group(2) or "default"
        split    = splits_map.get((dataset, subject), "unknown")

        meta_path, proc_dir = gt_dirs(pred_root / dataset / stem)
        pred_txt_dir        = pred_root / dataset / stem / "txt"
        slices_df           = patient_slices(proc_dir / "txt", pred_txt_dir)
        if slices_df.empty:
            continue

        metrics_dir = pred_root / dataset / stem / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        slices_df.to_csv(metrics_dir / "slices.csv", index=False)

        meta    = yaml.safe_load(meta_path.read_text())
        H, W, _ = get_slice_dims(meta)
        gt_bbox_path = proc_dir / "volume" / "bbox_3d.txt"
        gt_bbox = list(map(int, gt_bbox_path.read_text().split())) if gt_bbox_path.exists() else None

        pred_boxes = read_pred_boxes(pred_txt_dir)
        gt_boxes   = read_gt_boxes(proc_dir / "txt")
        build_patient_csv_rows(slices_df, pred_boxes, H, W, gt_bbox, gt_boxes, args.fp_iou_thresh,
                               meta).to_csv(
            metrics_dir / "patient.csv", index=False
        )

        slices_df["dataset"]  = dataset
        slices_df["subject"]  = subject
        slices_df["contrast"] = contrast
        slices_df["split"]    = split
        slices_df["stem"]     = stem
        all_records.append(slices_df)

    if not all_records:
        print("No data found.")
        return

    full_df = pd.concat(all_records, ignore_index=True)
    print(f"Processed {full_df.groupby(['dataset','subject']).ngroups} patients — {len(full_df)} slices")

    report = build_report(full_df, args.conf)
    print(report[report["dataset"] == "ALL"].to_string(index=False))

    if not args.no_html:
        acq_by_dataset, acq_by_contrast = count_acq_per_split(pred_root, splits_map)
        html = build_html(report, acq_by_dataset, acq_by_contrast, run_id, args.conf, SPLITS)
        out_html = pred_root / "report.html"
        out_html.write_text(html)
        print(f"HTML   → {out_html}")


if __name__ == "__main__":
    main()
