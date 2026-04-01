"""Shared helpers: normalisation, bbox, LAS reorientation, SI resampling."""
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, ornt_transform
from scipy import ndimage


def nifti_stem(path: Path) -> str:
    for ext in (".nii.gz", ".nii"):
        if path.name.endswith(ext):
            return path.name[: -len(ext)]
    return path.stem


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    flat = arr.ravel()
    nz = flat[flat > 0]
    if not len(nz):
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(nz, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((np.clip(arr, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def seg_to_yolo_bbox(seg: np.ndarray):
    """(cx, cy, w, h) normalised to [0,1] from a binary mask slice. None if empty.
    cx = col_center / W  (axis-1)
    cy = row_center / H  (axis-0)
    """
    rows, cols = np.any(seg, axis=1), np.any(seg, axis=0)
    if not rows.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    H, W = seg.shape
    return (c0 + c1) / 2 / W, (r0 + r1) / 2 / H, (c1 - c0 + 1) / W, (r1 - r0 + 1) / H


def resample_z(img: nib.Nifti1Image, target_z_mm: float, order: int) -> nib.Nifti1Image:
    """Resample NIfTI along axis 2 (S in LAS) to target_z_mm voxel spacing.
    order=3 for images, order=0 for binary masks.
    No-op if current spacing already matches target (within 0.01 mm).
    """
    current_z = float(img.header.get_zooms()[2])
    if abs(current_z - target_z_mm) < 0.01:
        return img
    zoom_factor = current_z / target_z_mm
    data = ndimage.zoom(img.get_fdata(dtype=np.float32), (1.0, 1.0, zoom_factor), order=order)
    affine = img.affine.copy()
    affine[:3, 2] *= target_z_mm / current_z
    return nib.Nifti1Image(data, affine)


def reorient_to_las(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient NIfTI to LAS (Left × Anterior × Superior) using nibabel orientation transforms."""
    current = nib.io_orientation(img.affine)
    target = axcodes2ornt(("L", "A", "S"))
    return img.as_reoriented(ornt_transform(current, target))


def read_bbox_3d(path: Path) -> dict:
    """Read bbox_3d.txt → {row1, row2, col1, col2, z1, z2} in voxel coords.
    Convention: row=axis-0, col=axis-1, z=axis-2 (in LAS space).
    """
    r1, r2, c1, c2, z1, z2 = map(int, path.read_text().split())
    return {"row1": r1, "row2": r2, "col1": c1, "col2": c2, "z1": z1, "z2": z2}


def write_bbox_3d(path: Path, row1, row2, col1, col2, z1, z2) -> None:
    """Write bbox_3d.txt: 6 space-separated ints  row1 row2 col1 col2 z1 z2."""
    path.write_text(f"{row1} {row2} {col1} {col2} {z1} {z2}\n")


def bbox_3d_from_txts(txt_dir: Path, H: int, W: int):
    """Aggregate YOLO slice txts into a 3D bbox union.
    YOLO format: class cx cy w h  (cx=col/W, cy=row/H, all in [0,1]).
    Returns {row1,row2,col1,col2,z1,z2} in voxel coords, or None if no detections.
    """
    row1s, row2s, col1s, col2s, zs = [], [], [], [], []
    for txt in sorted(txt_dir.glob("slice_*.txt")):
        content = txt.read_text().strip()
        if not content:
            continue
        z = int(txt.stem.split("_")[1])
        _, cx, cy, w, h = map(float, content.split())
        col1s.append(int((cx - w / 2) * W))
        col2s.append(int((cx + w / 2) * W))
        row1s.append(int((cy - h / 2) * H))
        row2s.append(int((cy + h / 2) * H))
        zs.append(z)
    if not zs:
        return None
    return {"row1": min(row1s), "row2": max(row2s), "col1": min(col1s), "col2": max(col2s), "z1": min(zs), "z2": max(zs)}


def stack_bbox_volume(txt_dir: Path, H: int, W: int, Z: int) -> np.ndarray:
    """Create a binary 3D volume (H×W×Z) where each slice is a filled bbox rectangle.
    Bboxes are in normalised coords relative to an H×W image (used for 320×320 slices).
    """
    vol = np.zeros((H, W, Z), dtype=np.uint8)
    for txt in txt_dir.glob("slice_*.txt"):
        content = txt.read_text().strip()
        if not content:
            continue
        z = int(txt.stem.split("_")[1])
        if z >= Z:
            continue
        _, cx, cy, w, h = map(float, content.split())
        r1 = max(0, int((cy - h / 2) * H))
        r2 = min(H, int((cy + h / 2) * H))
        c1 = max(0, int((cx - w / 2) * W))
        c2 = min(W, int((cx + w / 2) * W))
        vol[r1:r2, c1:c2, z] = 1
    return vol


