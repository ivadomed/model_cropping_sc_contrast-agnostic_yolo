"""
Command-line interface for sc-crop.

Usage:
    sc-crop t2.nii.gz                                    # crops, télécharge le modèle si absent
    sc-crop t2.nii.gz -o t2_crop.nii.gz
    sc-crop t2.nii.gz --debug                            # produit aussi <stem>_debug.png
    sc-crop t2.nii.gz --padding-rl 10 15                 # asymmetric: (left, right)
    sc-crop download                                      # téléchargement explicite du modèle
"""

import argparse
import sys

from .crop import run
from .download import download


def _parse_padding(value):
    """Parse padding argument: either single float or 'left right' tuple."""
    if isinstance(value, str) and " " in value:
        parts = value.split()
        if len(parts) == 2:
            return tuple(float(p) for p in parts)
    return float(value)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download()
        return

    parser = argparse.ArgumentParser(
        description="Crop a NIfTI volume around the spinal cord using YOLO detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",
                        help="Input NIfTI volume (.nii or .nii.gz)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file. Default: <input>_crop.nii.gz next to input")
    parser.add_argument("--model", default=None,
                        help="Path to model.pt (override sc_crop/models/)")
    parser.add_argument("--padding-rl", type=str, default="10.0",
                        help="Padding in Right-Left direction (mm). Single value or 'left right' for asymmetric")
    parser.add_argument("--padding-ap", type=str, default="15.0",
                        help="Padding in Anterior-Posterior direction (mm). Single value or 'ant post' for asymmetric")
    parser.add_argument("--padding-si", type=str, default="20.0",
                        help="Padding in Superior-Inferior direction (mm). Single value or 'sup inf' for asymmetric")
    parser.add_argument("--conf", type=float, default=None,
                        help="Detection confidence threshold (default: from config.yaml)")
    parser.add_argument("--debug", action="store_true",
                        help="Save <stem>_debug.png: all slices with max-confidence bbox")
    args = parser.parse_args()

    result = run(
        input_path    = args.input,
        output_path   = args.output,
        model_path    = args.model,
        padding_rl_mm = _parse_padding(args.padding_rl),
        padding_ap_mm = _parse_padding(args.padding_ap),
        padding_si_mm = _parse_padding(args.padding_si),
        conf          = args.conf,
        debug         = args.debug,
    )

    # Print result summary
    print(f"\n✓ Crop saved to: {result['output']}")
    print(f"  BBox file: {result['bbox_file']}")



if __name__ == "__main__":
    main()
