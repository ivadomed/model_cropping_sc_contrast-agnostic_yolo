#!/usr/bin/env python3
"""
Run YOLO on a single NIfTI volume and output a 3D bbox volume for FSLeyes overlay.

Reorients input to LAS, resamples SI axis to --si-res for inference (model resolution),
then reprojects predictions onto the original (native) SI resolution.
Output volume has the same dimensions and affine as the LAS-reoriented input
→ directly loadable as overlay in FSLeyes alongside the original image.

Only the SI axis is resampled for inference. In-plane dimensions (RL, AP) are unchanged,
so normalised bbox coordinates are valid in both the resampled and native spaces.

Usage:
    python scripts/predict_volume.py \
        --input sub-01_T2w.nii.gz \
        --checkpoint checkpoints/yolo_spine_v1/weights/best.pt \
        --out sub-01_pred_bbox.nii.gz \
        [--si-res 10.0] [--conf 0.25]
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from utils import normalize_to_uint8, reorient_to_las, resample_z

CONF_THRESH = 0.25  # default confidence threshold


def main():
    parser = argparse.ArgumentParser(
        description="YOLO inference on a NIfTI volume → 3D bbox NIfTI for FSLeyes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",      required=True, help="Input NIfTI (.nii.gz)")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--out",        required=True, help="Output bbox NIfTI path")
    parser.add_argument("--si-res",     type=float, default=10.0,
                        help="SI resolution in mm for inference (must match training resolution)")
    parser.add_argument("--conf",       type=float, default=CONF_THRESH,
                        help="Confidence threshold for detections")
    args = parser.parse_args()

    # Load and reorient — keep native LAS image as reference for output
    img_las = reorient_to_las(nib.load(args.input))
    H, W, Z_orig = img_las.shape
    orig_si_mm = float(img_las.header.get_zooms()[2])

    # Resample SI for inference
    img_inf = resample_z(img_las, args.si_res, order=3)
    data_inf = img_inf.get_fdata(dtype=np.float32)
    Z_inf = data_inf.shape[2]

    # zoom_factor: multiply original z index to get resampled z index
    zoom_factor = orig_si_mm / args.si_res  # e.g. 1mm/10mm = 0.1

    # Run inference on resampled slices
    slices_rgb = [
        np.stack([normalize_to_uint8(data_inf[:, :, z])] * 3, axis=-1)
        for z in range(Z_inf)
    ]
    model = YOLO(args.checkpoint)
    results = model.predict(slices_rgb, conf=args.conf, verbose=False)

    # Build predictions dict keyed by resampled slice index
    preds = {}
    for z_inf, res in enumerate(results):
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            best = int(boxes.conf.argmax())
            preds[z_inf] = boxes.xywhn[best].tolist()  # cx, cy, w, h

    # Fill bbox volume at native (original) resolution
    bbox_vol = np.zeros((H, W, Z_orig), dtype=np.uint8)
    for z_orig in range(Z_orig):
        z_inf = round(z_orig * zoom_factor)
        if z_inf not in preds:
            continue
        cx, cy, w, h = preds[z_inf]
        r1 = max(0, int((cy - h / 2) * H))
        r2 = min(H, int((cy + h / 2) * H))
        c1 = max(0, int((cx - w / 2) * W))
        c2 = min(W, int((cx + w / 2) * W))
        bbox_vol[r1:r2, c1:c2, z_orig] = 1

    # Save with native LAS affine → same world space as input
    nib.save(nib.Nifti1Image(bbox_vol, img_las.affine), args.out)
    n_detected = int(bbox_vol.any(axis=(0, 1)).sum())
    print(f"Inference at {args.si_res}mm SI ({Z_inf} slices) → "
          f"reprojected on {Z_orig} native slices, detected on {n_detected} → {args.out}")


if __name__ == "__main__":
    main()
