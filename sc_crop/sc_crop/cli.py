"""
Command-line interface for sc-crop.

Usage:
    sc-crop t2.nii.gz                        # default: writes <stem>_bbox.txt (native, inclusive)
    sc-crop t2.nii.gz --crop                 # also saves <stem>_crop.nii.gz (native orientation)
    sc-crop t2.nii.gz --crop --las           # save crop in LAS orientation
    sc-crop t2.nii.gz --crop --translate     # update affine for correct FSLeyes overlay
    sc-crop t2.nii.gz --crop -o output.nii.gz
    sc-crop t2.nii.gz --debug                # also saves <stem>_debug.png
    sc-crop download                         # download the model
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
        description="Detect spinal cord and output crop indices. Optionally crop the volume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input",
                        help="Input NIfTI volume (.nii or .nii.gz)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path: crop volume if --crop, else bbox txt")
    parser.add_argument("--model", default=None,
                        help="Path to model.pt (override sc_crop/models/)")
    parser.add_argument("--crop", action="store_true",
                        help="Save the cropped volume (default: bbox txt only)")
    parser.add_argument("--las", action="store_true",
                        help="Output cropped volume in LAS orientation (requires --crop)")
    parser.add_argument("--translate", action="store_true",
                        help="Update affine so crop overlays correctly in FSLeyes (requires --crop)")
    parser.add_argument("--padding-rl", type=str, default="10.0",
                        help="Padding in Right-Left direction (mm). Single value or 'left right'")
    parser.add_argument("--padding-ap", type=str, default="15.0",
                        help="Padding in Anterior-Posterior direction (mm). Single value or 'ant post'")
    parser.add_argument("--padding-si", type=str, default="20.0",
                        help="Padding in Superior-Inferior direction (mm). Single value or 'sup inf'")
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
        crop          = args.crop,
        las           = args.las,
        translate     = args.translate,
    )

    print(f"\nBBox file : {result['bbox_file']}")
    if "output" in result:
        print(f"Crop file : {result['output']}")


if __name__ == "__main__":
    main()
