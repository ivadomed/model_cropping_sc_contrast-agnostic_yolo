"""
Command-line interface for sc-crop.

Usage:
    sc-crop t2.nii.gz                     # crops, télécharge le modèle si absent
    sc-crop t2.nii.gz -o t2_crop.nii.gz
    sc-crop download                      # téléchargement explicite du modèle
"""

import argparse
import sys

from .crop import run
from .download import download, ensure_model


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
                        help="Path to model.onnx (override ~/.sc_crop/sc_crop_models/)")
    parser.add_argument("--padding", type=float, default=10.0,
                        help="Padding around the 3D bbox (mm)")
    parser.add_argument("--conf", type=float, default=None,
                        help="Detection confidence threshold (default: from config.yaml)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Inference device (GPU requires onnxruntime-gpu)")
    args = parser.parse_args()

    run(
        input_path  = args.input,
        output_path = args.output,
        model_path  = args.model,
        padding_mm  = args.padding,
        conf        = args.conf,
        device      = args.device,
    )


if __name__ == "__main__":
    main()
