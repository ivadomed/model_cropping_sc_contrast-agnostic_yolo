#!/usr/bin/env python3
"""
Benchmark YOLO inference speed on CPU for different image sizes.

Runs on dummy images (zeros) — content does not affect inference time.
Repeats each size multiple times and reports the median to reduce noise
from concurrent processes. Saves a bar chart for presentation.

Usage:
    python scripts/benchmark_cpu.py --checkpoint checkpoints/yolo26_1mm_axial_v2/weights/best.pt
    python scripts/benchmark_cpu.py --checkpoint checkpoints/yolo26_1mm_axial_v2/weights/best.pt \\
        --sizes 160 320 640 --n-slices 30 --n-runs 5 --out cpu_benchmark.png
"""

import argparse
import platform
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "Unknown CPU"


def get_cpu_info() -> str:
    name = get_cpu_name()
    try:
        import psutil
        phys  = psutil.cpu_count(logical=False)
        logic = psutil.cpu_count(logical=True)
        freq  = psutil.cpu_freq()
        freq_str = f"{freq.max / 1000:.2f} GHz" if freq else ""
        return f"{name}\n{phys} cores / {logic} threads  {freq_str}"
    except ImportError:
        return name


def benchmark_size(model, size: int, n_slices: int, n_runs: int) -> tuple[float, float]:
    """Returns (median_ms_per_slice, std_ms_per_slice) over n_runs."""
    images = [np.zeros((size, size, 3), dtype=np.uint8)] * n_slices

    print(f"  warmup...", end="\r")
    model.predict(images[:2], device="cpu", verbose=False)
    model.predict(images[:2], device="cpu", verbose=False)

    times = []
    for i in range(n_runs):
        print(f"  run {i+1}/{n_runs}...", end="\r")
        t0 = time.perf_counter()
        model.predict(images, device="cpu", verbose=False)
        times.append((time.perf_counter() - t0) * 1000 / n_slices)
    print()

    return float(np.median(times)), float(np.std(times))


def make_plot(sizes: list, medians: list, stds: list, cpu_info: str,
              model_name: str, n_slices: int, n_runs: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor("white")

    labels = [f"{s}×{s}" for s in sizes]
    x      = np.arange(len(sizes))
    bars   = ax.bar(x, medians, yerr=stds, capsize=5,
                    color="#4C72B0", edgecolor="#2a4a80", width=0.5,
                    error_kw={"elinewidth": 1.5, "ecolor": "#888"})

    for bar, med, std in zip(bars, medians, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.15,
                f"{med:.1f} ms", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Inference time per slice (ms)", fontsize=11)
    ax.set_title(f"CPU inference speed — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, max(medians) * 1.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    note = f"{cpu_info}\n{n_slices} slices/run · median over {n_runs} runs · error bars = std"
    fig.text(0.5, -0.04, note, ha="center", fontsize=8, color="#555",
             wrap=True, transform=fig.transFigure)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO CPU inference speed per image size",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--sizes",      nargs="+", type=int, default=[160, 320, 640],
                        help="Image sizes to benchmark (square)")
    parser.add_argument("--n-slices",   type=int, default=30,
                        help="Number of slices per run")
    parser.add_argument("--n-runs",     type=int, default=5,
                        help="Number of repeat runs per size (median taken)")
    parser.add_argument("--out",        default="cpu_benchmark.png",
                        help="Output plot path")
    args = parser.parse_args()

    from ultralytics import YOLO

    cpu_info   = get_cpu_info()
    model_name = Path(args.checkpoint).parent.parent.name
    print(f"CPU: {cpu_info}")
    print(f"Model: {model_name}")
    print(f"Sizes: {args.sizes}  n_slices={args.n_slices}  n_runs={args.n_runs}\n")

    model = YOLO(args.checkpoint)

    medians, stds = [], []
    for size in args.sizes:
        med, std = benchmark_size(model, size, args.n_slices, args.n_runs)
        medians.append(med)
        stds.append(std)
        print(f"  {size}×{size}: {med:.1f} ± {std:.1f} ms/slice  ({1000/med:.1f} slices/s)")

    make_plot(args.sizes, medians, stds, cpu_info, model_name,
              args.n_slices, args.n_runs, Path(args.out))


if __name__ == "__main__":
    main()
