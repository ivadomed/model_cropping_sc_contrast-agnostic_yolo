#!/usr/bin/env python3
"""
Full training pipeline: download → preprocess → splits → build dataset → train → evaluate → metrics → plots → failures.

Reads configs/preprocess.yaml, configs/training_dataset.yaml, configs/training.yaml.
All outputs go into runs/<TS>/ (datasplits/, dataset/, checkpoints/, predictions/).
The processed/ directory is shared across runs (path specified via 'out:' in preprocess.yaml).

Steps:
  1 = download datasets
  2 = preprocess
  3 = make splits  (always regenerated for reproducibility)
  4 = build YOLO dataset
  5 = train
  6 = evaluate
  7 = metrics
  8 = plot metrics
  9 = find failures

Usage:
    python scripts/run_pipeline.py --run-dir runs/20260101_120000
    python scripts/run_pipeline.py --run-dir runs/20260101_120000 --start 5 --end 7
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
import make_splits
import preprocess
import build_dataset
import train
import evaluate
import metrics as metrics_mod
import plot_metrics
import find_failures


def step_active(n: int, start: int, end: int) -> bool:
    return start <= n <= end


def banner(n: int, end: int, label: str) -> None:
    print("")
    print("══════════════════════════════════════════════════════════════")
    print(f"  STEP {n}/{end} : {label}")
    print("══════════════════════════════════════════════════════════════")


def main():
    parser = argparse.ArgumentParser(
        description="Full spine detection pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir",  default=None, help="Run directory (default: runs/<TS>)")
    parser.add_argument("--start",    type=int, default=1, help="First step to run (1-9)")
    parser.add_argument("--end",      type=int, default=9, help="Last step to run (1-9)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging during training")
    args = parser.parse_args()

    from datetime import datetime
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    preprocess_cfg = yaml.safe_load(Path("configs/preprocess.yaml").read_text())
    dataset_cfg    = yaml.safe_load(Path("configs/training_dataset.yaml").read_text())

    processed_dir   = Path(preprocess_cfg["out"])
    seed            = dataset_cfg["seed"]
    split_train     = dataset_cfg.get("train", 0.5)
    split_val       = dataset_cfg.get("val",   0.2)
    split_test      = dataset_cfg.get("test",  0.3)
    test_datasets   = dataset_cfg.get("test_datasets") or []
    splits_dir      = run_dir / "datasplits"
    dataset_dir     = run_dir / "dataset"
    checkpoint      = run_dir / "checkpoints" / "weights" / "best.pt"
    predictions_dir = run_dir / "predictions"

    shutil.copytree("configs", run_dir / "configs")

    print("══════════════════════════════════════════════════════════════")
    print("  Spine detection pipeline")
    print(f"  Run dir     : {run_dir}")
    print(f"  Steps       : {args.start} → {args.end}")
    print(f"  Processed   : {processed_dir}")
    print(f"  Dataset     : {dataset_dir}")
    print(f"  Checkpoint  : {checkpoint}")
    print(f"  Predictions : {predictions_dir}")
    print("══════════════════════════════════════════════════════════════")

    S, E = args.start, args.end

    if step_active(1, S, E):
        banner(1, E, "Download datasets")
        subprocess.run(["bash", "scripts/download_all_datasets.sh"], check=True)

    if step_active(2, S, E):
        banner(2, E, "Preprocess")
        preprocess.run(config="configs/preprocess.yaml")

    if step_active(3, S, E):
        banner(3, E, f"Make splits (seed={seed})")
        make_splits.run(raw="data/raw", out=splits_dir, seed=seed,
                        train=split_train, val=split_val, test=split_test)

    if step_active(4, S, E):
        banner(4, E, "Build YOLO dataset")
        build_dataset.run(
            config="configs/training_dataset.yaml",
            processed=processed_dir,
            splits_dir=splits_dir,
            out=dataset_dir,
            test_datasets=test_datasets,
        )

    if step_active(5, S, E):
        banner(5, E, "Train")
        train.run(
            config="configs/training.yaml",
            dataset_yaml=dataset_dir / "dataset.yaml",
            run_dir=run_dir,
            no_wandb=args.no_wandb,
        )

    if step_active(6, S, E):
        banner(6, E, "Evaluate")
        if checkpoint.exists():
            evaluate.run(
                checkpoint=checkpoint,
                processed=processed_dir,
                out=predictions_dir,
                splits_dir=splits_dir,
            )
        else:
            print(f"  WARNING: checkpoint not found at {checkpoint} — skipping evaluate")

    if step_active(7, S, E):
        banner(7, E, "Compute metrics")
        metrics_mod.run(inference=predictions_dir, splits_dir=splits_dir)

    if step_active(8, S, E):
        banner(8, E, "Plot metrics")
        plot_metrics.run(inference=predictions_dir, splits_dir=splits_dir)

    if step_active(9, S, E):
        banner(9, E, "Find failures")
        find_failures.run(inference=predictions_dir, splits_dir=splits_dir)

    print("")
    print("══════════════════════════════════════════════════════════════")
    print("  Pipeline complete")
    print(f"  Run dir     : {run_dir}/")
    print(f"  Predictions : {predictions_dir}/")
    print("══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
