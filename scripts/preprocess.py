#!/usr/bin/env python3
"""
Preprocess BIDS datasets: data/raw/ → processed/<variant>/<dataset>/<stem>/

For each (image, mask) pair in each BIDS dataset:
  1. Reorient to LAS (nibabel axis permutation, no interpolation)
  2. Resample all axes to target resolution via resample_to_output:
       SI  (axis 2) → --si-res mm
       RL  (axis 0) → --axial-res mm (if provided, else native)
       AP  (axis 1) → --axial-res mm (if provided, else native)
  3. Export 2D slices → png/slice_NNN.png
       --plane axial    : iterate SI axis Superior→Inferior (slice_000 = Superior end)
                          slice = img[:, :, z].T[::-1, ::-1]  shape (AP × RL)
                          row 0 = Anterior, col 0 = Left
       --plane sagittal : iterate axis 0 (RL),  slice = img[r, :, ::-1].T shape (SI × AP) — Superior at top
       2D mode  : grayscale uint8, normalised per slice
       2.5D mode: pseudo-RGB uint8 (R=prev, G=current, B=next) along the slice axis
                  border slices use a black frame for missing neighbours
  4. Compute YOLO GT bbox per slice → txt/slice_NNN.txt
       without --with-canal: "0 cx cy w h" (SC only, class 0)
       with --with-canal:    up to 2 lines — "0 cx cy w h" (SC) and/or "1 cx cy w h" (canal)
  5. Compute 3D GT bbox → volume/bbox_3d.txt  ("row1 row2 col1 col2 z1 z2", voxels)
     with --with-canal: also volume/bbox_3d_canal.txt (same format, canal class)
  6. Write meta.yaml (raw_image, raw_mask, shape_las [H,W,Z], si_res_mm, rl_res_mm, ap_res_mm,
                      axial_res_mm if --axial-res, channels=3 if --3ch, plane,
                      raw_canal_mask if --with-canal and canal mask found)
  7. Write processed/<variant>/skipped.log (TSV: dataset, subject, reason) for any subject
     skipped due to missing_nifti (git annex not downloaded) or no_sc_voxels (empty mask).

Mask discovery: per-dataset explicit suffix tables DATASET_MASK_SUFFIX (SC) and
DATASET_CANAL_SUFFIX (canal, only for datasets that have it) — crashes on unknown dataset.
Output dir named automatically:
  axial   : processed/<si_res>mm_SI[_<axial_res>mm_axial][_3ch][_sc_and_canal]
  sagittal: processed/<si_res>mm_SI[_<axial_res>mm_axial][_3ch]_sagittal[_sc_and_canal][_sc<N>mm]

Sagittal --sc-pad N:
  Only RL slices within [rl_min − N mm, rl_max + N mm] are saved, where rl_min/rl_max are the first
  and last RL slices containing SC voxels. Empty slices (no SC) within that window are kept 1 in 2
  (even indices only) to halve the background count. Slice filenames preserve original RL indices.

Usage:
    # Axial 10mm SI + 1mm isotropic, pseudo-RGB (current default)
    python scripts/preprocess.py --si-res 10.0 --axial-res 1.0 --3ch

    # Sagittal: 1mm isotropic, pseudo-RGB, restricted to SC region ±10mm
    python scripts/preprocess.py --si-res 1.0 --axial-res 1.0 --rl-res 1.0 --3ch --plane sagittal --sc-pad 10

    # SC + canal dual-class axial
    python scripts/preprocess.py --si-res 10.0 --axial-res 1.0 --3ch --with-canal

    # Patch existing meta.yaml with rl_res_mm/ap_res_mm without re-preprocessing
    python scripts/preprocess.py --update-meta --out processed/10mm_SI

    # Process a single dataset only
    python scripts/preprocess.py --si-res 10.0 --axial-res 1.0 --datasets spider-challenge-2023
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import yaml
from PIL import Image as PILImage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from nibabel.processing import resample_to_output

from utils import (
    bbox_3d_from_txts, nifti_stem, normalize_to_uint8,
    reorient_to_las, seg_to_yolo_bbox, write_bbox_3d,
)

# Dataset registry — single source of truth is data/datasets.yaml
_REGISTRY = yaml.safe_load(
    (Path(__file__).parent.parent / "data" / "datasets.yaml").read_text()
)["datasets"]
DATASET_MASK_SUFFIX  = {d["name"]: d["mask_suffix"] for d in _REGISTRY}
DATASET_LABELS_DIR   = {d["name"]: d["labels_dir"] for d in _REGISTRY if "labels_dir" in d}
DATASET_CANAL_SUFFIX = {d["name"]: d["canal_suffix"] for d in _REGISTRY if d.get("canal_suffix")}


_GZIP_MAGIC = b"\x1f\x8b"


def _is_nifti_downloaded(path: Path) -> bool:
    """Return False if path is a git-annex pointer (not actual gzip data)."""
    with open(path, "rb") as f:
        return f.read(2) == _GZIP_MAGIC


def find_pairs(dataset_root: Path, with_canal: bool = False):
    """Return (valid_pairs, missing_pairs) for a BIDS dataset.

    valid_pairs   : list of (image_path, sc_mask_path, canal_mask_path_or_None)
    missing_pairs : list of image Path objects absent or not yet downloaded via git-annex
    Crashes if the dataset name is not in DATASET_MASK_SUFFIX.
    canal_mask_path is None when with_canal=False or no canal mask exists for the pair.
    """
    mask_suffix = DATASET_MASK_SUFFIX[dataset_root.name]
    canal_suffix = DATASET_CANAL_SUFFIX.get(dataset_root.name) if with_canal else None
    labels_dir = DATASET_LABELS_DIR.get(dataset_root.name, "labels")
    labels_root = dataset_root / "derivatives" / labels_dir

    pairs, missing = [], []
    for sub_dir in sorted(labels_root.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        for anat_seg in list(sub_dir.glob("ses-*/anat")) + list(sub_dir.glob("anat")):
            for mask in sorted(anat_seg.glob(f"*{mask_suffix}")):
                img_path = dataset_root / sub_dir.name / anat_seg.relative_to(sub_dir) / (mask.name[: -len(mask_suffix)] + ".nii.gz")
                if not img_path.exists() or not _is_nifti_downloaded(img_path) or not _is_nifti_downloaded(mask):
                    missing.append(img_path)
                    continue
                canal_mask = None
                if canal_suffix:
                    canal_path = anat_seg / (mask.name[: -len(mask_suffix)] + canal_suffix)
                    if canal_path.exists() and _is_nifti_downloaded(canal_path):
                        canal_mask = canal_path
                pairs.append((img_path, mask, canal_mask))
    return pairs, missing


def _rescale_bbox(bbox, H, W, H_out, W_out):
    """Rescale normalised bbox coords when image was padded to H_out×W_out."""
    if bbox is None or (H_out == H and W_out == W):
        return bbox
    cx, cy, w, h = bbox
    return (cx * W / W_out, cy * H / H_out, w * W / W_out, h * H / H_out)


def _bbox3d_from_mask(mask_data: np.ndarray, H: int, W: int, Z: int,
                      H_out: int, W_out: int, plane: str = "axial") -> Optional[dict]:
    """Compute 3D bbox directly from a 3D mask array. Returns {row1,row2,col1,col2,z1,z2} or None.

    Axial  : mask_data shape (RL,AP,SI); iterates Superior→Inferior; applies .T[::-1,::-1] per slice.
    Sagittal: mask_data pre-transposed to (SI,AP,RL); iterates RL axis normally.
    z in output is the slice counter (0 = first in iteration order).
    """
    rows1, rows2, cols1, cols2, zs = [], [], [], [], []
    for counter in range(Z):
        if plane == "axial":
            las_z = Z - 1 - counter
            sl = mask_data[:, :, las_z].T[::-1, ::-1]
        else:
            sl = mask_data[:, :, counter]
        bbox = seg_to_yolo_bbox(sl)
        bbox = _rescale_bbox(bbox, H, W, H_out, W_out)
        if bbox is None:
            continue
        cx, cy, w, h = bbox
        rows1.append(max(0, int((cy - h / 2) * H_out)))
        rows2.append(min(H_out, int((cy + h / 2) * H_out)))
        cols1.append(max(0, int((cx - w / 2) * W_out)))
        cols2.append(min(W_out, int((cx + w / 2) * W_out)))
        zs.append(counter)
    if not zs:
        return None
    return {"row1": min(rows1), "row2": max(rows2),
            "col1": min(cols1), "col2": max(cols2),
            "z1":   min(zs),    "z2":   max(zs)}


def process_pair(args: tuple):
    img_path_str, mask_path_str, canal_mask_path_str, dataset_name, processed_str, si_res_mm, axial_res_mm, rl_res_mm, three_ch, plane, sc_pad_mm = args
    img_path    = Path(img_path_str)
    patient_dir = Path(processed_str) / dataset_name / nifti_stem(img_path)

    if (patient_dir / "meta.yaml").exists():
        return "already_processed"

    for d in ("png", "txt", "volume"):
        (patient_dir / d).mkdir(parents=True, exist_ok=True)

    img_las  = reorient_to_las(nib.load(str(img_path)))
    mask_las = reorient_to_las(nib.load(str(mask_path_str)))
    rl_native_mm, ap_native_mm = float(img_las.header.get_zooms()[0]), float(img_las.header.get_zooms()[1])

    # RL axis: --rl-res takes priority, then --axial-res, then native
    # AP axis: --axial-res takes priority, then native
    eff_rl = rl_res_mm or axial_res_mm or rl_native_mm
    eff_ap = axial_res_mm or ap_native_mm
    voxel_sizes = (eff_rl, eff_ap, si_res_mm)
    img_r  = resample_to_output(img_las,  voxel_sizes=voxel_sizes, order=1)
    mask_r = resample_to_output(mask_las, voxel_sizes=voxel_sizes, order=0)

    img_data  = img_r.get_fdata(dtype=np.float32)
    mask_data = np.round(mask_r.get_fdata()).astype(np.uint8)

    canal_data = None
    if canal_mask_path_str is not None:
        canal_r    = resample_to_output(reorient_to_las(nib.load(str(canal_mask_path_str))),
                                        voxel_sizes=voxel_sizes, order=0)
        canal_data = np.round(canal_r.get_fdata()).astype(np.uint8)

    # Determine slice axis and extraction function
    # axial   : iterate SI axis Superior→Inferior; slice = data[:,:,z].T[::-1,::-1] shape (AP, RL)
    #           H=AP, W=RL; row 0=Anterior, col 0=Left; slice_000 = Superior
    # sagittal: iterate axis 0 (RL), raw shape = (AP, SI) → transposed to (SI, AP) so SC is vertical
    if plane == "axial":
        N      = min(img_data.shape[2], mask_data.shape[2])
        H, W   = img_data.shape[1], img_data.shape[0]  # H=AP, W=RL
        def get_img_slice(data, i):  return data[:, :, i].T[::-1, ::-1]
        def get_mask_slice(data, i): return data[:, :, i].T[::-1, ::-1]
        def canal_valid(i):          return canal_data is not None and i < canal_data.shape[2]
    else:  # sagittal — flip SI axis then transpose: (AP,SI)→flip→(AP,SI↑)→T→(SI↑,AP)
        N      = min(img_data.shape[0], mask_data.shape[0])
        H, W   = img_data.shape[2], img_data.shape[1]
        def get_img_slice(data, i):  return data[i, :, ::-1].T
        def get_mask_slice(data, i): return data[i, :, ::-1].T
        def canal_valid(i):          return canal_data is not None and i < canal_data.shape[0]

    # YOLO rejects images with any dimension < 10px — pad if needed
    H_out, W_out = max(H, 10), max(W, 10)
    blank = np.zeros((H_out, W_out), dtype=np.uint8)

    def get_slice(i: int) -> np.ndarray:
        if i < 0 or i >= N:
            return blank
        arr = normalize_to_uint8(get_img_slice(img_data, i))
        if H_out != H or W_out != W:
            padded = blank.copy()
            padded[:H, :W] = arr
            return padded
        return arr

    # Sagittal SC-region window: only keep slices within [rl_min−pad, rl_max+pad]
    # and subsample empty slices 1-in-2 within that window.
    if plane == "sagittal" and sc_pad_mm is not None:
        sc_rl_slices = [i for i in range(N) if mask_data[i, :, :].any()]
        if not sc_rl_slices:
            print(f"[SKIP] No SC voxels found in {mask_path_str}")
            return "no_sc_voxels"
        pad_slices   = round(sc_pad_mm / eff_rl)
        win_start    = max(0,     min(sc_rl_slices) - pad_slices)
        win_end      = min(N - 1, max(sc_rl_slices) + pad_slices)
        sc_rl_set    = set(sc_rl_slices)
    else:
        win_start, win_end, sc_rl_set = 0, N - 1, None

    # Axial: iterate Superior→Inferior (slice_idx=0 = Superior, las_idx=N-1 in LAS).
    # Sagittal: iterate RL axis directly (slice_idx = las_idx = RL index).
    if plane == "axial":
        slice_iter = [(c, N - 1 - c) for c in range(N)]
    else:
        slice_iter = [(i, i) for i in range(N)]

    for slice_idx, las_idx in slice_iter:
        if sc_rl_set is not None:
            if las_idx < win_start or las_idx > win_end:
                continue
            if las_idx not in sc_rl_set and las_idx % 2 != 0:
                continue
        arr = get_slice(las_idx)
        if three_ch:
            if plane == "axial":
                # R=Superior neighbour, G=current, B=Inferior neighbour
                rgb = np.stack([get_slice(las_idx + 1), arr, get_slice(las_idx - 1)], axis=2)
            else:
                rgb = np.stack([get_slice(las_idx - 1), arr, get_slice(las_idx + 1)], axis=2)
            PILImage.fromarray(rgb).save(str(patient_dir / "png" / f"slice_{slice_idx:03d}.png"))
        else:
            PILImage.fromarray(arr).save(str(patient_dir / "png" / f"slice_{slice_idx:03d}.png"))

        sc_slice   = get_mask_slice(mask_data, las_idx)
        sc_bbox    = _rescale_bbox(seg_to_yolo_bbox(sc_slice), H, W, H_out, W_out)
        canal_bbox = None
        if canal_valid(las_idx):
            canal_bbox = _rescale_bbox(seg_to_yolo_bbox(get_mask_slice(canal_data, las_idx)), H, W, H_out, W_out)

        lines = ""
        if sc_bbox is not None:
            lines += f"0 {sc_bbox[0]:.6f} {sc_bbox[1]:.6f} {sc_bbox[2]:.6f} {sc_bbox[3]:.6f}\n"
        if canal_bbox is not None:
            lines += f"1 {canal_bbox[0]:.6f} {canal_bbox[1]:.6f} {canal_bbox[2]:.6f} {canal_bbox[3]:.6f}\n"
        (patient_dir / "txt" / f"slice_{slice_idx:03d}.txt").write_text(lines)

    box = bbox_3d_from_txts(patient_dir / "txt", H_out, W_out)
    if box is not None:
        write_bbox_3d(patient_dir / "volume" / "bbox_3d.txt", **box)

    if canal_data is not None:
        n_canal = canal_data.shape[2] if plane == "axial" else canal_data.shape[0]
        canal_box = _bbox3d_from_mask(
            canal_data if plane == "axial" else np.transpose(canal_data, (2, 1, 0)),
            H, W, min(N, n_canal), H_out, W_out, plane
        )
        if canal_box is not None:
            write_bbox_3d(patient_dir / "volume" / "bbox_3d_canal.txt", **canal_box)

    shape_las = list(img_data.shape[:3])
    meta = {
        "raw_image": str(img_path),
        "raw_mask":  str(mask_path_str),
        "shape_las": shape_las,
        "si_res_mm": si_res_mm,
        "rl_res_mm": round(eff_rl, 4),
        "ap_res_mm": round(eff_ap, 4),
        "plane":     plane,
    }
    if axial_res_mm is not None:
        meta["axial_res_mm"] = axial_res_mm
    if rl_res_mm is not None:
        meta["rl_res_mm_explicit"] = rl_res_mm
    if three_ch:
        meta["channels"] = 3
    if canal_mask_path_str is not None:
        meta["raw_canal_mask"] = str(canal_mask_path_str)
    (patient_dir / "meta.yaml").write_text(yaml.dump(meta))

    return "processed"


def update_meta_one(meta_path_str: str) -> str:
    """Add rl_res_mm/ap_res_mm to one meta.yaml. Returns 'updated' or 'skipped'."""
    meta_path = Path(meta_path_str)
    meta = yaml.safe_load(meta_path.read_text())
    if "rl_res_mm" in meta:
        return "skipped"
    img_las = reorient_to_las(nib.load(meta["raw_image"]))
    rl_res_mm, ap_res_mm = float(img_las.header.get_zooms()[0]), float(img_las.header.get_zooms()[1])
    meta["rl_res_mm"] = round(rl_res_mm, 4)
    meta["ap_res_mm"] = round(ap_res_mm, 4)
    meta_path.write_text(yaml.dump(meta))
    return "updated"


def update_meta_resolutions(processed_dir: Path) -> None:
    """Patch all existing meta.yaml in processed_dir to add rl_res_mm and ap_res_mm."""
    metas = sorted(processed_dir.rglob("meta.yaml"))
    print(f"Found {len(metas)} meta.yaml files in {processed_dir}")
    counts = {"updated": 0, "skipped": 0}
    for meta_path in tqdm(metas, desc="meta.yaml", unit="file"):
        counts[update_meta_one(str(meta_path))] += 1
    print(f"Done — updated: {counts['updated']}  skipped (already had fields): {counts['skipped']}")


def main():
    parser = argparse.ArgumentParser(
        description="BIDS raw → processed_{res}mm_SI/ (LAS slices resampled on SI + YOLO labels)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",      default=None, help="YAML config file (configs/preprocess.yaml). CLI flags override config values.")
    parser.add_argument("--plane",       default=None, choices=["axial", "sagittal"])
    parser.add_argument("--si-res",      type=float, default=None)
    parser.add_argument("--axial-res",   type=float, default=None)
    parser.add_argument("--rl-res",      type=float, default=None)
    parser.add_argument("--raw",         default="data/raw")
    parser.add_argument("--out",         default=None)
    parser.add_argument("--datasets",    nargs="+", default=None)
    parser.add_argument("--3ch",         action="store_true", dest="three_ch")
    parser.add_argument("--with-canal",  action="store_true", dest="with_canal")
    parser.add_argument("--sc-pad",      type=float, default=None, dest="sc_pad_mm")
    parser.add_argument("--update-meta", action="store_true")
    args = parser.parse_args()

    # Load config file and apply as defaults (CLI flags take priority)
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        plane = args.plane or cfg.get("plane", "axial")
        plane_cfg = cfg.get(plane, {})
        if args.si_res    is None: args.si_res    = plane_cfg.get("si_res")
        if args.axial_res is None: args.axial_res = plane_cfg.get("inplane_res")
        if args.rl_res    is None: args.rl_res    = plane_cfg.get("rl_res")
        if not args.three_ch:      args.three_ch  = cfg.get("three_ch", False)
        if not args.with_canal:    args.with_canal = cfg.get("with_canal", False)
        if args.sc_pad_mm is None: args.sc_pad_mm = plane_cfg.get("sc_pad")
        args.plane = plane
    else:
        if args.plane is None:
            args.plane = "axial"

    if args.update_meta:
        assert args.out, "--out is required with --update-meta"
        update_meta_resolutions(Path(args.out))
        return

    assert args.si_res is not None, "--si-res is required"
    si_res     = args.si_res
    axial_res  = args.axial_res
    rl_res     = args.rl_res
    three_ch   = args.three_ch
    with_canal = args.with_canal
    plane      = args.plane
    sc_pad_mm  = args.sc_pad_mm
    assert sc_pad_mm is None or plane == "sagittal", "--sc-pad is only valid with --plane sagittal"
    if args.out:
        processed_dir = str(Path(args.out))
    else:
        name = f"{si_res:g}mm_SI"
        if axial_res is not None:
            name += f"_{axial_res:g}mm_axial"
        if rl_res is not None:
            name += f"_{rl_res:g}mm_RL"
        if three_ch:
            name += "_3ch"
        if plane == "sagittal":
            name += "_sagittal"
        if sc_pad_mm is not None:
            name += f"_sc{sc_pad_mm:g}mm"
        if with_canal:
            name += "_sc_and_canal"
        processed_dir = str(Path("processed") / name)

    worker_args = []
    skip_log_entries = []   # {"dataset", "subject", "reason"} — written to skipped.log at the end
    for dataset_dir in sorted(Path(args.raw).iterdir()):
        if not dataset_dir.is_dir():
            continue
        if args.datasets and dataset_dir.name not in args.datasets:
            continue
        pairs, missing = find_pairs(dataset_dir, with_canal=with_canal)
        print(f"{dataset_dir.name}: {len(pairs)} pairs  ({len(missing)} missing NIfTI)")
        for img_path in missing:
            skip_log_entries.append({"dataset": dataset_dir.name,
                                     "subject": nifti_stem(img_path),
                                     "reason":  "missing_nifti"})
        worker_args.extend(
            (str(img), str(mask), str(canal) if canal else None,
             dataset_dir.name, processed_dir, si_res, axial_res, rl_res, three_ch, plane, sc_pad_mm)
            for img, mask, canal in pairs
        )

    if not worker_args:
        print("No pairs found. Check --raw path.")
        return

    counts = {"processed": 0, "already_processed": 0, "no_sc_voxels": 0}
    for args_tuple in tqdm(worker_args, desc="Volumes", unit="vol"):
        result = process_pair(args_tuple)
        counts[result] += 1
        if result == "no_sc_voxels":
            img_path_str, _, _, dataset_name, *_ = args_tuple
            skip_log_entries.append({"dataset": dataset_name,
                                     "subject": nifti_stem(Path(img_path_str)),
                                     "reason":  "no_sc_voxels"})

    print(f"\nDone — processed: {counts['processed']}  "
          f"already_processed: {counts['already_processed']}  "
          f"no_sc_voxels: {counts['no_sc_voxels']}")

    if skip_log_entries:
        log_path = Path(processed_dir) / "skipped.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            f.write("dataset\tsubject\treason\n")
            for e in skip_log_entries:
                f.write(f"{e['dataset']}\t{e['subject']}\t{e['reason']}\n")
        print(f"Skipped subjects logged → {log_path}  ({len(skip_log_entries)} entries)")


if __name__ == "__main__":
    main()
