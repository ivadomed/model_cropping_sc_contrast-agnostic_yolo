"""
Core logic for spinal cord detection and volume cropping.

Takes a NIfTI volume, runs ONNX YOLO inference slice-by-slice (axial),
aggregates detections into a 3D bounding box, and crops the original volume.

The output is in the same orientation, space, and resolution as the input.
No ultralytics dependency — only nibabel, numpy, onnxruntime, pillow.

Usage:
    from sc_crop.crop import run, load_config
    config = load_config()
    out = run("t2.nii.gz", config=config)
    out = run("t2.nii.gz", config=config, debug=True)
    # debug=True → also saves <stem>_debug.png: panel of all slices
    # with max-confidence bbox overlaid (no threshold), green if conf≥0.1 else orange
"""

from __future__ import annotations

import math
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, ornt_transform
from nibabel.processing import resample_to_output
from PIL import Image as PILImage
from PIL import ImageDraw


def load_config(model_dir: str | Path) -> dict:
    """Load config.yaml from the directory containing model.onnx."""
    import yaml
    return yaml.safe_load((Path(model_dir) / "config.yaml").read_text())


# ─── Orientation helpers ───────────────────────────────────────────────────────

def reorient_to_las(img: nib.Nifti1Image) -> nib.Nifti1Image:
    current = nib.io_orientation(img.affine)
    target  = axcodes2ornt(("L", "A", "S"))
    return img.as_reoriented(ornt_transform(current, target))


def reorient_to_original(img_las: nib.Nifti1Image,
                          original_ornt: np.ndarray) -> nib.Nifti1Image:
    las_ornt  = axcodes2ornt(("L", "A", "S"))
    transform = ornt_transform(las_ornt, original_ornt)
    return img_las.as_reoriented(transform)


# ─── Resampling ───────────────────────────────────────────────────────────────

def resample_for_inference(img_las: nib.Nifti1Image,
                            si_res: float,
                            inplane_res: float | None) -> nib.Nifti1Image:
    """Resample LAS image to match training preprocessing resolution.

    Uses nibabel.processing.resample_to_output (order=1) — identical to training.
    """
    rl_mm, ap_mm, si_mm = [float(v) for v in img_las.header.get_zooms()[:3]]
    target_rl = inplane_res if inplane_res is not None else rl_mm
    target_ap = inplane_res if inplane_res is not None else ap_mm
    if (abs(target_rl - rl_mm) < 0.01
            and abs(target_ap - ap_mm) < 0.01
            and abs(si_res - si_mm) < 0.01):
        return img_las
    return resample_to_output(img_las, voxel_sizes=(target_rl, target_ap, si_res), order=1)


# ─── Normalisation ────────────────────────────────────────────────────────────

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    flat = arr.ravel()
    nz   = flat[flat > 0]
    if not len(nz):
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(nz, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((np.clip(arr, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


# ─── Letterbox ────────────────────────────────────────────────────────────────

def letterbox(img_hwc: np.ndarray, imgsz: int):
    """Resize to imgsz×imgsz preserving aspect ratio, pad with 0.

    Returns (padded, scale, pad_top, pad_left).
    """
    H, W   = img_hwc.shape[:2]
    scale  = imgsz / max(H, W)
    new_h  = int(H * scale)
    new_w  = int(W * scale)
    resized  = np.array(PILImage.fromarray(img_hwc).resize((new_w, new_h), PILImage.BILINEAR))
    pad_top  = (imgsz - new_h) // 2
    pad_left = (imgsz - new_w) // 2
    padded   = np.zeros((imgsz, imgsz, img_hwc.shape[2]), dtype=np.uint8)
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return padded, scale, pad_top, pad_left


# ─── ONNX inference ───────────────────────────────────────────────────────────

def _nms(boxes_x1y1x2y2: np.ndarray, scores: np.ndarray,
         iou_thresh: float = 0.45) -> list:
    x1, y1, x2, y2 = boxes_x1y1x2y2.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep


def infer_slices(session, data: np.ndarray, imgsz: int,
                 conf_thresh: float, channels: int = 3) -> dict:
    """Run ONNX inference on each axial slice.

    data : (RL, AP, Z) float32 LAS volume at inference resolution
    Returns {z_inf: (cx_norm, cy_norm, w_norm, h_norm)} normalised to [0,1]
    in the inference (RL, AP) space — maps directly to native space via native dims.
    """
    RL, AP, Z  = data.shape
    input_name = session.get_inputs()[0].name
    preds      = {}

    for z in range(Z):
        # z=0 is Superior in LAS after preprocessing (slice_000 convention).
        # 3ch: R=Superior neighbor (z-1), G=current, B=Inferior neighbor (z+1).
        if channels == 3:
            sup_z = max(0, z - 1)
            inf_z = min(Z - 1, z + 1)
            img_hwc = np.stack([
                normalize_to_uint8(data[:, :, sup_z]),
                normalize_to_uint8(data[:, :, z]),
                normalize_to_uint8(data[:, :, inf_z]),
            ], axis=-1)
        else:
            img_hwc = normalize_to_uint8(data[:, :, z])[:, :, None]

        padded, scale, pad_top, pad_left = letterbox(img_hwc, imgsz)

        inp = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = session.run(None, {input_name: inp})[0][0]  # [5+, num_anchors]

        scores = out[4]
        mask   = scores > conf_thresh
        if not mask.any():
            continue

        cxcywh = out[:4, mask].T
        sc     = scores[mask]

        x1 = cxcywh[:, 0] - cxcywh[:, 2] / 2
        y1 = cxcywh[:, 1] - cxcywh[:, 3] / 2
        x2 = cxcywh[:, 0] + cxcywh[:, 2] / 2
        y2 = cxcywh[:, 1] + cxcywh[:, 3] / 2
        keep     = _nms(np.stack([x1, y1, x2, y2], axis=1), sc)
        best_idx = keep[int(np.argmax(sc[keep]))]

        cx_lb, cy_lb, w_lb, h_lb = cxcywh[best_idx]

        # Unletterbox → normalised to inference (RL, AP).
        # Normalised coords map 1-to-1 to native space regardless of in-plane resampling.
        cx_norm = (cx_lb - pad_left) / scale / AP
        cy_norm = (cy_lb - pad_top)  / scale / RL
        w_norm  = w_lb / scale / AP
        h_norm  = h_lb / scale / RL

        preds[z] = (cx_norm, cy_norm, w_norm, h_norm)

    return preds


# ─── 3D bbox aggregation ──────────────────────────────────────────────────────

def aggregate_bbox_3d(preds: dict,
                      RL_nat: int, AP_nat: int, Z_nat: int,
                      si_zoom: float) -> tuple:
    """Aggregate per-slice detections → (rl1, rl2, ap1, ap2, z1, z2) in native LAS voxels.

    si_zoom = si_mm_nat / si_res  (Z_inf = Z_nat * si_zoom).
    Normalised in-plane coords scale directly to native dims (FOV-preserving resampling).
    """
    rl1s, rl2s, ap1s, ap2s, zs = [], [], [], [], []
    for z_inf, (cx, cy, w, h) in preds.items():
        z_nat = min(Z_nat - 1, round(z_inf / si_zoom))
        rl1s.append(max(0,      int((cy - h / 2) * RL_nat)))
        rl2s.append(min(RL_nat, int((cy + h / 2) * RL_nat)))
        ap1s.append(max(0,      int((cx - w / 2) * AP_nat)))
        ap2s.append(min(AP_nat, int((cx + w / 2) * AP_nat)))
        zs.append(z_nat)
    return min(rl1s), max(rl2s), min(ap1s), max(ap2s), min(zs), max(zs)


# ─── Crop ─────────────────────────────────────────────────────────────────────

def crop_volume(img_las: nib.Nifti1Image, bbox: tuple,
                padding_mm: float) -> tuple:
    """Crop LAS NIfTI around bbox with padding_mm on all sides.

    Returns (cropped_las_nifti, padded_bbox).
    Affine is updated so the crop sits at the correct world position.
    """
    RL, AP, Z  = img_las.shape
    rl_mm, ap_mm, si_mm = [float(v) for v in img_las.header.get_zooms()[:3]]
    rl1, rl2, ap1, ap2, z1, z2 = bbox

    pad_rl = int(np.ceil(padding_mm / rl_mm))
    pad_ap = int(np.ceil(padding_mm / ap_mm))
    pad_z  = int(np.ceil(padding_mm / si_mm))

    rl1p = max(0,  rl1 - pad_rl); rl2p = min(RL, rl2 + pad_rl)
    ap1p = max(0,  ap1 - pad_ap); ap2p = min(AP, ap2 + pad_ap)
    z1p  = max(0,  z1  - pad_z);  z2p  = min(Z,  z2  + pad_z)

    data    = img_las.get_fdata(dtype=np.float32)
    cropped = data[rl1p:rl2p, ap1p:ap2p, z1p:z2p]

    affine        = img_las.affine.copy()
    affine[:3, 3] = img_las.affine[:3, :3] @ np.array([rl1p, ap1p, z1p]) \
                    + img_las.affine[:3, 3]

    return nib.Nifti1Image(cropped, affine), (rl1p, rl2p, ap1p, ap2p, z1p, z2p)


# ─── Debug panel ──────────────────────────────────────────────────────────────

def save_debug_panel(session, data: np.ndarray, imgsz: int,
                     channels: int, conf_thresh: float, out_path: str) -> None:
    """Save a near-square panel of all axial slices with max-confidence bbox.

    Each cell shows the max-confidence detection regardless of threshold.
    bbox color: green if conf ≥ conf_thresh, orange otherwise.
    """
    CELL = 128
    RL, AP, Z  = data.shape
    input_name = session.get_inputs()[0].name
    cells      = []

    for z in range(Z):
        if channels == 3:
            sup_z = max(0, z - 1)
            inf_z = min(Z - 1, z + 1)
            img_hwc = np.stack([
                normalize_to_uint8(data[:, :, sup_z]),
                normalize_to_uint8(data[:, :, z]),
                normalize_to_uint8(data[:, :, inf_z]),
            ], axis=-1)
        else:
            gray    = normalize_to_uint8(data[:, :, z])[:, :, None]
            img_hwc = np.repeat(gray, 3, axis=-1)

        padded, _scale, _pt, _pl = letterbox(img_hwc, imgsz)
        inp = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = session.run(None, {input_name: inp})[0][0]  # [5+, anchors]

        scores   = out[4]
        best_idx = int(np.argmax(scores))
        conf     = float(scores[best_idx])

        cell_img = PILImage.fromarray(padded).resize((CELL, CELL), PILImage.BILINEAR)
        draw     = ImageDraw.Draw(cell_img)

        k         = CELL / imgsz
        cx, cy, w, h = out[:4, best_idx]
        x1, y1    = (cx - w / 2) * k, (cy - h / 2) * k
        x2, y2    = (cx + w / 2) * k, (cy + h / 2) * k
        color     = (0, 220, 0) if conf >= conf_thresh else (255, 140, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=1)

        draw.text((2, 1),          f"z{z}",        fill=(200, 200, 200))
        draw.text((2, CELL - 11),  f"{conf:.2f}",  fill=color)

        cells.append(cell_img)

    n    = len(cells)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    canvas = PILImage.new("RGB", (cols * CELL, rows * CELL), (20, 20, 20))
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        canvas.paste(cell, (c * CELL, r * CELL))

    canvas.save(out_path)
    print(f"Debug   : {out_path}")


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(input_path: str,
        config: dict | None = None,
        output_path: str | None = None,
        model_path: str | None = None,
        padding_mm: float = 10.0,
        conf: float | None = None,
        device: str = "auto",
        debug: bool = False) -> str:
    """Full pipeline: load → LAS → resample → infer → bbox 3D → crop → reorient → save.

    model_path  : chemin vers model.onnx. Par défaut : ~/.sc_crop/sc_crop_models/model.onnx
                  (téléchargé automatiquement si absent via ensure_model()).
    config      : dict (si_res, inplane_res, channels, conf). Par défaut : lu depuis
                  config.yaml au même endroit que model_path.
    output_path : fichier de sortie. Par défaut : <input>_crop.nii.gz au même endroit.
    Returns     : chemin vers le volume croppé sauvegardé.
    """
    from .download import ensure_model
    import onnxruntime as ort

    model_path = Path(model_path) if model_path else ensure_model()
    config     = config if config is not None else load_config(model_path.parent)

    si_res      = config["si_res"]
    inplane_res = config.get("inplane_res")
    channels    = config.get("channels", 3)
    conf        = conf if conf is not None else config.get("conf", 0.1)

    providers = (["CPUExecutionProvider"] if device == "cpu"
                 else ["CUDAExecutionProvider", "CPUExecutionProvider"])

    img           = nib.load(input_path)
    original_ornt = nib.io_orientation(img.affine)
    img_las       = reorient_to_las(img)
    RL_nat, AP_nat, Z_nat = img_las.shape
    si_mm_nat     = float(img_las.header.get_zooms()[2])
    rl_mm_nat, ap_mm_nat  = [float(v) for v in img_las.header.get_zooms()[:2]]
    si_zoom       = si_mm_nat / si_res

    print(f"Input   : {Path(input_path).name}  shape={img.shape}  zooms={img.header.get_zooms()[:3]}")
    print(f"LAS     : shape=({RL_nat},{AP_nat},{Z_nat})  "
          f"res=({rl_mm_nat:.2f},{ap_mm_nat:.2f},{si_mm_nat:.2f})mm")

    img_inf  = resample_for_inference(img_las, si_res, inplane_res)
    data_inf = img_inf.get_fdata(dtype=np.float32)
    print(f"Infer   : shape={data_inf.shape}  si_zoom={si_zoom:.3f}")

    session = ort.InferenceSession(str(model_path), providers=providers)
    print(f"Device  : {session.get_providers()[0]}")
    imgsz = session.get_inputs()[0].shape[2]
    print(f"Model   : imgsz={imgsz}  channels={channels}  conf={conf}")

    if debug:
        inp_p      = Path(input_path)
        stem       = inp_p.name.replace(".nii.gz", "").replace(".nii", "")
        debug_path = str(inp_p.parent / f"{stem}_debug.png")
        save_debug_panel(session, data_inf, imgsz, channels, conf, debug_path)

    preds = infer_slices(session, data_inf, imgsz, conf, channels)
    print(f"Detected: {len(preds)}/{data_inf.shape[2]} slices")

    if not preds:
        raise RuntimeError(
            "No spinal cord detected — check the volume or lower --conf"
        )

    bbox = aggregate_bbox_3d(preds, RL_nat, AP_nat, Z_nat, si_zoom)
    rl1, rl2, ap1, ap2, z1, z2 = bbox
    print(f"SC bbox : RL [{rl1}:{rl2}]  AP [{ap1}:{ap2}]  SI [{z1}:{z2}]  (before padding)")

    cropped_las, bbox_pad = crop_volume(img_las, bbox, padding_mm)
    rl1p, rl2p, ap1p, ap2p, z1p, z2p = bbox_pad
    print(f"Padded  : RL [{rl1p}:{rl2p}]  AP [{ap1p}:{ap2p}]  SI [{z1p}:{z2p}]  "
          f"→ shape=({rl2p-rl1p},{ap2p-ap1p},{z2p-z1p})")

    cropped = reorient_to_original(cropped_las, original_ornt)

    if output_path is None:
        inp    = Path(input_path)
        stem   = inp.name.replace(".nii.gz", "").replace(".nii", "")
        output_path = str(inp.parent / f"{stem}_crop.nii.gz")

    nib.save(cropped, output_path)
    print(f"Saved   : {output_path}  shape={cropped.shape}")
    return output_path
