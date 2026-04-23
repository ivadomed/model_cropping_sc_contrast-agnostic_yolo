#!/usr/bin/env python3
"""
Display the R, G, B channels of a pseudo-RGB MRI slice separately.

In 2.5D preprocessing: R = slice z-1 (red), G = slice z (green), B = slice z+1 (blue).

Usage:
    python scripts/show_rgb_channels.py processed/10mm_SI_1mm_axial_3ch/dataset/patient/png/slice_008.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CHANNEL_LABELS = {0: "R — slice z-1", 1: "G — slice z (current)", 2: "B — slice z+1"}
CHANNEL_CMAPS  = {0: "Reds_r",        1: "Greens_r",              2: "Blues_r"}


def main():
    parser = argparse.ArgumentParser(description="Show R/G/B channels of a pseudo-RGB MRI slice")
    parser.add_argument("image", help="Path to the PNG slice")
    parser.add_argument("--out", default=None, help="Save figure to this path instead of displaying")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    img  = np.array(Image.open(args.image).convert("RGB"))
    path = Path(args.image)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(path.name, fontsize=12, fontweight="bold")

    axes[0].imshow(img)
    axes[0].set_title("Original (RGB)", fontsize=10)
    axes[0].axis("off")

    for ch in range(3):
        plane = np.zeros_like(img)
        plane[:, :, ch] = img[:, :, ch]
        axes[ch + 1].imshow(plane)
        axes[ch + 1].set_title(CHANNEL_LABELS[ch], fontsize=10)
        axes[ch + 1].axis("off")

    plt.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved → {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
