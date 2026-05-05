"""
Core logic for spinal cord detection and volume cropping.

Uses ultralytics YOLO (best.pt) for inference — identical preprocessing to the
training pipeline (preprocess.py): LAS reorientation, nibabel resample_to_output,
axial slices as data[:, :, las_idx].T[::-1, ::-1] → (AP, RL, C) uint8.

The output is saved in LAS orientation for visualization (e.g., fsleyes).

Usage:
    from sc_crop.crop import run, load_config
    
    # Symmetric padding (same on both faces)
    result = run("t2.nii.gz", padding_rl_mm=10.0, padding_ap_mm=15.0, padding_si_mm=20.0)
    
    # Asymmetric padding per face: (left/ant/sup, right/post/inf)
    result = run("t2.nii.gz", 
                 padding_rl_mm=(5.0, 15.0),   # 5mm Left, 15mm Right
                 padding_ap_mm=(10.0, 20.0),  # 10mm Anterior, 20mm Posterior
                 padding_si_mm=(15.0, 25.0))  # 15mm Superior, 25mm Inferior
    
    output_path = result["output"]
    corner_mm, sizes_mm = result["bbox_mm"]
    bbox_before_file = result["bbox_before_file"]
    bbox_after_file = result["bbox_after_file"]
    result = run("t2.nii.gz", debug=True)
    # debug=True → also saves <stem>_debug.png: panel of all slices with
    # max-confidence bbox overlaid (no threshold), green if conf ≥ threshold, orange otherwise.
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
    """Load config.yaml from the directory containing model.pt."""
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

    Uses nibabel.processing.resample_to_output (order=1).
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


# ─── Slice extraction ─────────────────────────────────────────────────────────

def _get_slice(data: np.ndarray, las_idx: int,
               black: np.ndarray) -> np.ndarray:
    """Extract one axial slice in (AP, RL) uint8, identical to preprocess.py.

    Convention: data[:, :, las_idx].T[::-1, ::-1]
      rows = AP (row 0 = Anterior), cols = RL (col 0 = Left).
    Out-of-bounds las_idx returns a black frame.
    """
    Z = data.shape[2]
    if las_idx < 0 or las_idx >= Z:
        return black
    return normalize_to_uint8(data[:, :, las_idx]).T[::-1, ::-1]


def build_slices(data: np.ndarray, channels: int) -> tuple:
    """Build all axial slices Superior→Inferior, matching preprocess.py convention.

    3ch: R=Superior neighbour (las_idx+1), G=current, B=Inferior neighbour (las_idx-1).
    Border channels are black (zeros).

    Returns (slices_list, las_idxs_list) where las_idx is the LAS SI index in
    the resampled volume (0 = Inferior, Z-1 = Superior).
    """
    RL, AP, Z = data.shape
    black     = np.zeros((AP, RL), dtype=np.uint8)
    slices, las_idxs = [], []

    for las_idx in range(Z - 1, -1, -1):   # Superior → Inferior
        cur = _get_slice(data, las_idx, black)
        if channels == 3:
            # R=Superior neighbour, G=current, B=Inferior neighbour (preprocess.py axial convention)
            sup = _get_slice(data, las_idx + 1, black)
            inf = _get_slice(data, las_idx - 1, black)
            slices.append(np.stack([sup, cur, inf], axis=2))
        else:
            slices.append(cur)
        las_idxs.append(las_idx)

    return slices, las_idxs


# ─── YOLO inference ───────────────────────────────────────────────────────────

def infer_slices(model, slices: list, las_idxs: list,
                 conf_thresh: float) -> dict:
    """Run YOLO inference on pre-built slices.

    Returns {las_idx: (cx, cy, w, h)} where cx/cy/w/h are normalised [0,1]
    in the slice image space (AP, RL).
    """
    results = model.predict(slices, conf=conf_thresh, verbose=False)
    preds   = {}
    for las_idx, res in zip(las_idxs, results):
        if res.boxes is None or len(res.boxes) == 0:
            continue
        best         = int(res.boxes.conf.argmax())
        cx, cy, w, h = res.boxes.xywhn[best].tolist()
        preds[las_idx] = (cx, cy, w, h)
    return preds


# ─── 3D bbox computation ───────────────────────────────────────────────────

def compute_padded_bbox_3d(bbox: tuple,
                           RL: int, AP: int, Z: int,
                           rl_mm: float, ap_mm: float, si_mm: float,
                           padding_rl_mm: float | tuple = 10.0,
                           padding_ap_mm: float | tuple = 15.0,
                           padding_si_mm: float | tuple = 20.0) -> tuple:
    """Compute final 3D bbox after applying padding to detected bbox.
    
    Returns: (rl1p, rl2p, ap1p, ap2p, z1p, z2p) clamped to image bounds.
    
    Padding parameters:
      - float: symmetric (both sides)
      - tuple (side1, side2): asymmetric
    
    LAS convention: 
      - RL: Left (0) to Right (max)
      - AP: Anterior (0) to Posterior (max)
      - SI: Superior (0) to Inferior (max)
    """
    rl1, rl2, ap1, ap2, z1, z2 = bbox
    
    # Parse padding tuples
    pad_rl_left, pad_rl_right = (padding_rl_mm, padding_rl_mm) if isinstance(padding_rl_mm, (int, float)) else padding_rl_mm
    pad_ap_ant, pad_ap_post = (padding_ap_mm, padding_ap_mm) if isinstance(padding_ap_mm, (int, float)) else padding_ap_mm
    pad_si_sup, pad_si_inf = (padding_si_mm, padding_si_mm) if isinstance(padding_si_mm, (int, float)) else padding_si_mm

    # Convert mm to voxels
    pad_rl_left_vox = int(np.ceil(pad_rl_left / rl_mm))
    pad_rl_right_vox = int(np.ceil(pad_rl_right / rl_mm))
    pad_ap_ant_vox = int(np.ceil(pad_ap_ant / ap_mm))
    pad_ap_post_vox = int(np.ceil(pad_ap_post / ap_mm))
    pad_si_sup_vox = int(np.ceil(pad_si_sup / si_mm))
    pad_si_inf_vox = int(np.ceil(pad_si_inf / si_mm))

    # Apply padding, clamped to bounds (LAS convention)
    rl1p = max(0,  rl1 - pad_rl_left_vox)
    rl2p = min(RL, rl2 + pad_rl_right_vox)
    ap1p = max(0,  ap1 - pad_ap_ant_vox)
    ap2p = min(AP, ap2 + pad_ap_post_vox)
    z1p  = max(0,  z1  - pad_si_sup_vox)
    z2p  = min(Z,  z2  + pad_si_inf_vox)
    
    return rl1p, rl2p, ap1p, ap2p, z1p, z2p


# ─── 3D bbox aggregation ──────────────────────────────────────────────────────

def aggregate_bbox_3d(preds: dict,
                      RL_nat: int, AP_nat: int, Z_nat: int,
                      si_zoom: float) -> tuple:
    """Aggregate per-slice detections → (rl1, rl2, ap1, ap2, z1, z2) in native LAS voxels.

    Slices were presented as (AP, RL) with T[::-1, ::-1], so to map back to native LAS indices:
    rl_c = (1-cx)*RL_nat, ap_c = (1-cy)*AP_nat  (col 0 = Left = LAS max, row 0 = Anterior = LAS max).
    Normalised in-plane coords are FOV-preserving → valid in native space directly.

    si_zoom = si_mm_nat / si_res; z_nat = round(las_idx_inf / si_zoom).
    """
    rl1s, rl2s, ap1s, ap2s, zs = [], [], [], [], []
    for las_idx_inf, (cx, cy, w, h) in preds.items():
        z_nat   = min(Z_nat - 1, round(las_idx_inf / si_zoom))
        rl_c    = cx * RL_nat
        ap_c    = (1.0 - cy) * AP_nat
        rl_half = w / 2 * RL_nat
        ap_half = h / 2 * AP_nat
        rl1s.append(max(0,      int(rl_c - rl_half)))
        rl2s.append(min(RL_nat, int(rl_c + rl_half)))
        ap1s.append(max(0,      int(ap_c - ap_half)))
        ap2s.append(min(AP_nat, int(ap_c + ap_half)))
        zs.append(z_nat)
    return min(rl1s), max(rl2s), min(ap1s), max(ap2s), min(zs), max(zs)


def bbox_vox_to_mm(img_las: nib.Nifti1Image,
                   bbox: tuple,
                   rl_mm: float,
                   ap_mm: float,
                   si_mm: float) -> tuple:
    """Convert a voxel bbox in LAS index space to LAS mm coordinates."""
    rl1, rl2, ap1, ap2, z1, z2 = bbox
    corner_las_mm = img_las.affine[:3, :3] @ np.array([rl1, ap1, z1]) + img_las.affine[:3, 3]
    sizes_mm = np.array([(rl2 - rl1) * rl_mm,
                         (ap2 - ap1) * ap_mm,
                         (z2 - z1) * si_mm])
    return corner_las_mm, sizes_mm


# ─── Crop ─────────────────────────────────────────────────────────────────────

def crop_volume(img_las: nib.Nifti1Image, padded_bbox: tuple,
                rl_mm: float | None = None,
                ap_mm: float | None = None,
                si_mm: float | None = None) -> tuple:
    """Crop LAS NIfTI using a pre-computed padded bbox.

    Returns (cropped_las_nifti, padded_bbox, bbox_mm).
    Affine is updated so the crop sits at the correct LAS world position.
    
    padded_bbox: (rl1p, rl2p, ap1p, ap2p, z1p, z2p) already padded and clamped.
    bbox_mm: (corner_mm, sizes_mm) in LAS mm space.
    """
    rl1p, rl2p, ap1p, ap2p, z1p, z2p = padded_bbox
    
    # Use provided resolutions or get from image
    if rl_mm is None or ap_mm is None or si_mm is None:
        rl_mm_use, ap_mm_use, si_mm_use = [float(v) for v in img_las.header.get_zooms()[:3]]
        rl_mm = rl_mm or rl_mm_use
        ap_mm = ap_mm or ap_mm_use
        si_mm = si_mm or si_mm_use

    data    = img_las.get_fdata(dtype=np.float32)
    cropped = data[rl1p:rl2p, ap1p:ap2p, z1p:z2p]

    affine        = img_las.affine.copy()
    affine[:3, 3] = img_las.affine[:3, :3] @ np.array([rl1p, ap1p, z1p]) \
                    + img_las.affine[:3, 3]

    bbox_mm = bbox_vox_to_mm(
        img_las, padded_bbox, rl_mm, ap_mm, si_mm
    )
    return nib.Nifti1Image(cropped, affine), padded_bbox, bbox_mm


# ─── Debug panel ──────────────────────────────────────────────────────────────

def save_debug_panel(model, slices: list, las_idxs: list,
                     conf_thresh: float, out_path: str,
                     padded_bbox: tuple | None = None,
                     H: int | None = None, W: int | None = None) -> None:
    """Save a near-square panel of all axial slices with max-confidence bbox.
        Runs inference at conf=0.001 so every slice shows its best prediction.
        bbox colors:
            - Green/Orange: YOLO detection (green if conf ≥ conf_thresh, orange otherwise)
            - Red: 3D crop region boundaries (if padded_bbox provided)
    
        If padded_bbox provided, also shows the 3D crop region boundaries:
            - Red rectangle: boundaries of 3D crop region (rl1p:rl2p, ap1p:ap2p)
            - Shown on slices within crop z-range (z1p:z2p)
    
        Slice convention: data[:, :, z].T gives (AP, RL) with row 0 = Anterior and col 0 = Left.
    """
    CELL    = 128
    results = model.predict(slices, conf=0.001, verbose=False)
    cells   = []

    # Unpack padded bbox if provided
    padded_rl1, padded_rl2, padded_ap1, padded_ap2, padded_z1, padded_z2 = padded_bbox if padded_bbox else (None,) * 6
    has_padded = padded_bbox is not None and H is not None and W is not None

    for las_idx, res, sl in zip(las_idxs, results, slices):
        rgb  = sl if sl.ndim == 3 else np.stack([sl] * 3, axis=-1)
        cell = PILImage.fromarray(rgb).resize((CELL, CELL), PILImage.BILINEAR)
        draw = ImageDraw.Draw(cell)

        # Draw YOLO detections
        if res.boxes is not None and len(res.boxes) > 0:
            best         = int(res.boxes.conf.argmax())
            conf         = float(res.boxes.conf[best])
            cx, cy, w, h = res.boxes.xywhn[best].tolist()
            x1, y1 = (cx - w / 2) * CELL, (cy - h / 2) * CELL
            x2, y2 = (cx + w / 2) * CELL, (cy + h / 2) * CELL
            color = (0, 220, 0) if conf >= conf_thresh else (255, 140, 0)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
            draw.text((2, CELL - 11), f"{conf:.2f}", fill=color)

        # Draw padded bbox 3D region (red)
        # Slice convention T[::-1,::-1]: col 0=Left (LAS RL max), row 0=Anterior (LAS AP max).
        # LAS rl_vox → image_x = (1 - rl_vox/W)*CELL ; LAS ap_vox → image_y = (1 - ap_vox/H)*CELL
        if has_padded and padded_z1 <= las_idx <= padded_z2:
            # RL (pas de flip)
            x1_pad = (padded_rl1 / W) * CELL
            x2_pad = (padded_rl2 / W) * CELL

            # AP (flip vertical)
            y1_pad = (1 - padded_ap2 / H) * CELL
            y2_pad = (1 - padded_ap1 / H) * CELL

            # sécurisation PIL
            x0 = min(x1_pad, x2_pad)
            x1 = max(x1_pad, x2_pad)
            y0 = min(y1_pad, y2_pad)
            y1 = max(y1_pad, y2_pad)

            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)

        draw.text((2, 1), f"z{las_idx}", fill=(200, 200, 200))
        cells.append(cell)

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
        padding_rl_mm: float | tuple = 10.0,
        padding_ap_mm: float | tuple = 15.0,
        padding_si_mm: float | tuple = 20.0,
        conf: float | None = None,
        debug: bool = False) -> dict:
    """Full pipeline: load → LAS → resample → infer → bbox 3D → crop → reorient → save.

    model_path    : path to model.pt. Default: ~/.sc_crop/sc_crop_models/model.pt
                    (auto-downloaded if absent via ensure_model()).
    config        : dict (si_res, inplane_res, channels, conf). Default: loaded from
                    config.yaml next to model.pt.
    output_path   : output file. Default: <input>_crop_las.nii.gz next to input.
    padding_rl_mm : padding in Right-Left direction. float or (left_mm, right_mm).
    padding_ap_mm : padding in Anterior-Posterior direction. float or (anterior_mm, posterior_mm).
    padding_si_mm : padding in Superior-Inferior direction. float or (superior_mm, inferior_mm).
        Returns       : dict with:
            - output: path to the saved cropped volume after padding
            - output_before_file: path to the saved cropped volume before padding
            - input_las_file: path to the input volume reoriented to LAS
            - bbox_file: path to the bbox txt file (after padding, backward-compatible)
            - bbox_before_file: path to bbox txt file before padding
            - bbox_after_file: path to bbox txt file after padding
            - bbox_mm: (corner_mm, sizes_mm) in LAS mm space
        * corner_mm: (x, y, z) minimum corner in mm
        * sizes_mm: [RL_size, AP_size, SI_size] dimensions in mm
    """
    from .download import ensure_model
    from ultralytics import YOLO

    model_path = Path(model_path) if model_path else ensure_model()
    config     = config if config is not None else load_config(model_path.parent)

    si_res      = config["si_res"]
    inplane_res = config.get("inplane_res")
    channels    = config.get("channels", 3)
    conf        = conf if conf is not None else config.get("conf", 0.1)

    img           = nib.load(input_path)
    img_las       = reorient_to_las(img)
    RL_nat, AP_nat, Z_nat = img_las.shape
    si_mm_nat     = float(img_las.header.get_zooms()[2])
    rl_mm_nat, ap_mm_nat = [float(v) for v in img_las.header.get_zooms()[:2]]
    si_zoom       = si_mm_nat / si_res

    print(f"Input   : {Path(input_path).name}  shape={img.shape}  zooms={img.header.get_zooms()[:3]}")
    print(f"LAS     : shape=({RL_nat},{AP_nat},{Z_nat})  "
          f"res=({rl_mm_nat:.2f},{ap_mm_nat:.2f},{si_mm_nat:.2f})mm")

    img_inf  = resample_for_inference(img_las, si_res, inplane_res)
    data_inf = img_inf.get_fdata(dtype=np.float32)
    print(f"Infer   : shape={data_inf.shape}  si_zoom={si_zoom:.3f}")

    model = YOLO(str(model_path))
    print(f"Model   : {model_path.name}  channels={channels}  conf={conf}")

    slices, las_idxs = build_slices(data_inf, channels)

    preds = infer_slices(model, slices, las_idxs, conf)
    print(f"Detected: {len(preds)}/{data_inf.shape[2]} slices")

    if not preds:
        raise RuntimeError(
            "No spinal cord detected — check the volume or lower --conf"
        )

    bbox = aggregate_bbox_3d(preds, RL_nat, AP_nat, Z_nat, si_zoom)
    rl1, rl2, ap1, ap2, z1, z2 = bbox
    print(f"SC bbox : RL [{rl1}:{rl2}]  AP [{ap1}:{ap2}]  SI [{z1}:{z2}]  (before padding)")

    # Compute padded bbox ONCE (single source of truth)
    padded_bbox = compute_padded_bbox_3d(
        bbox, RL_nat, AP_nat, Z_nat, rl_mm_nat, ap_mm_nat, si_mm_nat,
        padding_rl_mm, padding_ap_mm, padding_si_mm
    )
    rl1p, rl2p, ap1p, ap2p, z1p, z2p = padded_bbox

    if debug:
        inp_p      = Path(input_path)
        stem       = inp_p.name.replace(".nii.gz", "").replace(".nii", "")
        debug_path = str(inp_p.parent / f"{stem}_debug.png")
        save_debug_panel(model, slices, las_idxs, conf, debug_path,
                        padded_bbox=padded_bbox, H=AP_nat, W=RL_nat)

    cropped_before_las, _, bbox_before_mm = crop_volume(
        img_las, bbox,
        rl_mm=rl_mm_nat, ap_mm=ap_mm_nat, si_mm=si_mm_nat
    )
    cropped_las, bbox_pad, bbox_mm = crop_volume(
        img_las, padded_bbox,
        rl_mm=rl_mm_nat, ap_mm=ap_mm_nat, si_mm=si_mm_nat
    )
    rl1p, rl2p, ap1p, ap2p, z1p, z2p = bbox_pad
    
    # Format padding info for display
    pad_rl_str = f"{padding_rl_mm}mm" if isinstance(padding_rl_mm, (int, float)) else f"L={padding_rl_mm[0]}mm R={padding_rl_mm[1]}mm"
    pad_ap_str = f"{padding_ap_mm}mm" if isinstance(padding_ap_mm, (int, float)) else f"A={padding_ap_mm[0]}mm P={padding_ap_mm[1]}mm"
    pad_si_str = f"{padding_si_mm}mm" if isinstance(padding_si_mm, (int, float)) else f"S={padding_si_mm[0]}mm I={padding_si_mm[1]}mm"
    print(f"Padding : RL={pad_rl_str}  AP={pad_ap_str}  SI={pad_si_str}")
    print(f"Padded  : RL [{rl1p}:{rl2p}]  AP [{ap1p}:{ap2p}]  SI [{z1p}:{z2p}]  "
          f"→ shape=({rl2p-rl1p},{ap2p-ap1p},{z2p-z1p})")

    if output_path is None:
        inp        = Path(input_path)
        stem       = inp.name.replace(".nii.gz", "").replace(".nii", "")
        output_path = str(inp.parent / f"{stem}_crop_las.nii.gz")

    out_path = Path(output_path)
    output_before_path = str(out_path.parent / out_path.name.replace("_crop_las.nii.gz", "_crop_before_padding_las.nii.gz").replace("_crop_las.nii", "_crop_before_padding_las.nii"))
    input_las_path = str(out_path.parent / out_path.name.replace("_crop_las.nii.gz", "_input_las.nii.gz").replace("_crop_las.nii", "_input_las.nii"))

    nib.save(img_las, input_las_path)
    print(f"Saved   : {input_las_path}  shape={img_las.shape}")

    nib.save(cropped_before_las, output_before_path)
    print(f"Saved   : {output_before_path}  shape={cropped_before_las.shape}")

    nib.save(cropped_las, output_path)
    print(f"Saved   : {output_path}  shape={cropped_las.shape}")

    corner_mm, sizes_mm = bbox_mm
    print(f"BBox MM : corner=({corner_mm[0]:.1f}, {corner_mm[1]:.1f}, {corner_mm[2]:.1f}) mm  "
          f"sizes=({sizes_mm[0]:.1f}, {sizes_mm[1]:.1f}, {sizes_mm[2]:.1f}) mm")

    base = Path(output_path).parent / Path(output_path).stem
    bbox_before_path = Path(str(base) + "_bbox_before_padding_las.txt")
    bbox_after_path = Path(str(base) + "_bbox_after_padding_las.txt")
    bbox_path = Path(str(base) + "_bbox_las.txt")

    corner_before_mm, sizes_before_mm = bbox_before_mm
    with open(bbox_before_path, "w") as f:
        f.write("# Bounding box BEFORE padding\n")
        f.write("# Vox format: rl1 rl2 ap1 ap2 z1 z2\n")
        f.write(f"{rl1} {rl2} {ap1} {ap2} {z1} {z2}\n")
        f.write("# MM format: corner_x corner_y corner_z size_rl size_ap size_si\n")
        f.write(f"{corner_before_mm[0]:.2f} {corner_before_mm[1]:.2f} {corner_before_mm[2]:.2f} "
                f"{sizes_before_mm[0]:.2f} {sizes_before_mm[1]:.2f} {sizes_before_mm[2]:.2f}\n")

    with open(bbox_after_path, "w") as f:
        f.write("# Bounding box AFTER padding\n")
        f.write("# Vox format: rl1 rl2 ap1 ap2 z1 z2\n")
        f.write(f"{rl1p} {rl2p} {ap1p} {ap2p} {z1p} {z2p}\n")
        f.write("# MM format: corner_x corner_y corner_z size_rl size_ap size_si\n")
        f.write(f"{corner_mm[0]:.2f} {corner_mm[1]:.2f} {corner_mm[2]:.2f} "
                f"{sizes_mm[0]:.2f} {sizes_mm[1]:.2f} {sizes_mm[2]:.2f}\n")

    # Backward-compatible alias to after-padding bbox
    with open(bbox_path, "w") as f:
        f.write("# Bounding box AFTER padding\n")
        f.write("# MM format: corner_x corner_y corner_z size_rl size_ap size_si\n")
        f.write(f"{corner_mm[0]:.2f} {corner_mm[1]:.2f} {corner_mm[2]:.2f} "
                f"{sizes_mm[0]:.2f} {sizes_mm[1]:.2f} {sizes_mm[2]:.2f}\n")

    print(f"BBox pre: {bbox_before_path}")
    print(f"BBox post: {bbox_after_path}")

    return {
        "input_las_file": input_las_path,
        "output_before_file": output_before_path,
        "output": output_path,
        "bbox_file": str(bbox_path),
        "bbox_before_file": str(bbox_before_path),
        "bbox_after_file": str(bbox_after_path),
        "bbox_mm": bbox_mm,
        "corner_mm": corner_mm,
        "sizes_mm": sizes_mm
    }
