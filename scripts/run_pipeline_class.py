#!/usr/bin/env python3
"""
Classification pipeline for spinal cord SC/no-SC axial slice classification.

Trains a binary slice classifier (yolo26n-cls) to determine whether a slice
contains spinal cord. Evaluated standalone via gap_mm_S and gap_mm_I metrics
(SC-positive slices produce a dummy bbox encoding the predicted SI z-range).

Reads configs/preprocess.yaml, configs/training_dataset.yaml, configs/training_cls.yaml.

Steps:
  1 = download datasets
  2 = preprocess
  3 = make splits
  4 = build classification dataset (sc / no_sc folders from GT)
  5 = train classifier (yolo26n-cls)
  6 = evaluate classifier (dummy bbox → gap_mm_S, gap_mm_I)
  7 = metrics (gap_mm_S, gap_mm_I only)
  8 = find failures (gap_mm_S, gap_mm_S_neg, gap_mm_I, gap_mm_I_neg)
  9 = plot metrics (gap_mm_S, gap_mm_I only)

Usage:
    python scripts/run_pipeline_class.py
    python scripts/run_pipeline_class.py --run-dir runs/20260601_120000 --start 4 --end 6
    python scripts/run_pipeline_class.py --no-wandb
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
import make_splits
import preprocess
import build_class_dataset
import train_cls
import evaluate_cls
import metrics as metrics_mod
import plot_metrics as plot_metrics_mod
import find_failures as find_failures_mod

CLS_METRICS          = ["gap_mm_S", "gap_mm_I"]
CLS_FAILURE_METRICS  = ["gap_mm_S", "gap_mm_S_neg", "gap_mm_I", "gap_mm_I_neg"]


def step_active(n: int, start: int, end: int) -> bool:
    return start <= n <= end


def banner(n: int, end: int, label: str) -> None:
    print("")
    print("══════════════════════════════════════════════════════════════")
    print(f"  STEP {n}/{end} : {label}")
    print("══════════════════════════════════════════════════════════════")


def main():
    parser = argparse.ArgumentParser(
        description="SC/no-SC slice classifier pipeline (gap_mm_S / gap_mm_I)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir",  default=None,
                        help="Run directory (default: runs/<TS>)")
    parser.add_argument("--start",    type=int, default=1, help="First step (1-9)")
    parser.add_argument("--end",      type=int, default=9, help="Last step (1-9)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    from datetime import datetime
    run_dir = (Path(args.run_dir) if args.run_dir
               else Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    preprocess_cfg = yaml.safe_load(Path("configs/preprocess.yaml").read_text())
    dataset_cfg    = yaml.safe_load(Path("configs/training_dataset.yaml").read_text())
    cls_cfg        = yaml.safe_load(Path("configs/training_cls.yaml").read_text())

    processed_dir   = Path(preprocess_cfg["out"])
    seed            = dataset_cfg["seed"]
    split_train     = dataset_cfg.get("train", 0.5)
    split_val       = dataset_cfg.get("val",   0.2)
    split_test      = dataset_cfg.get("test",  0.3)
    test_datasets   = dataset_cfg.get("test_datasets") or []
    splits_dir      = run_dir / "datasplits"
    dataset_cls_dir = run_dir / "dataset_cls"
    cls_checkpoint  = run_dir / "checkpoints_cls" / "weights" / "best.pt"
    predictions_dir = run_dir / "predictions"
    cls_conf        = float(cls_cfg.get("cls_conf", 0.5))
    superior_only   = bool(cls_cfg.get("superior_only", False))

    shutil.copytree("configs", run_dir / "configs", dirs_exist_ok=True)

    print("══════════════════════════════════════════════════════════════")
    print("  Spine SC/no-SC classifier pipeline")
    print(f"  Run dir       : {run_dir}")
    print(f"  Steps         : {args.start} → {args.end}")
    print(f"  Processed     : {processed_dir}")
    print(f"  Cls dataset   : {dataset_cls_dir}")
    print(f"  Cls checkpoint: {cls_checkpoint}")
    print(f"  cls_conf      : {cls_conf}")
    print(f"  superior_only : {superior_only}")
    print(f"  Predictions   : {predictions_dir}")
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
        banner(4, E, "Build classification dataset (sc / no_sc)")
        build_class_dataset.run(
            processed=processed_dir,
            splits_dir=splits_dir,
            out=dataset_cls_dir,
            test_datasets=test_datasets,
            superior_only=superior_only,
        )

    if step_active(5, S, E):
        banner(5, E, "Train classifier (yolo26n-cls)")
        train_cls.run(
            config="configs/training_cls.yaml",
            dataset_dir=dataset_cls_dir,
            run_dir=run_dir,
            no_wandb=args.no_wandb,
        )

    if step_active(6, S, E):
        banner(6, E, "Evaluate classifier")
        if cls_checkpoint.exists():
            evaluate_cls.run(
                cls_checkpoint=cls_checkpoint,
                processed=processed_dir,
                inference=predictions_dir,
                cls_conf=cls_conf,
                superior_only=superior_only,
            )
        else:
            print(f"  WARNING: cls checkpoint not found at {cls_checkpoint} — skipping")

    if step_active(7, S, E):
        banner(7, E, f"Compute metrics ({', '.join(CLS_METRICS)})")
        metrics_mod.main([
            "--inference", str(predictions_dir),
            "--splits-dir", str(splits_dir),
            "--metrics",   *CLS_METRICS,
        ])

    if step_active(8, S, E):
        banner(8, E, f"Find failures ({', '.join(CLS_FAILURE_METRICS)})")
        find_failures_mod.main([
            "--inference",  str(predictions_dir),
            "--splits-dir", str(splits_dir),
            "--metrics",    *CLS_FAILURE_METRICS,
            "--conf",       str(cls_conf),
        ])

    if step_active(9, S, E):
        banner(9, E, f"Plot metrics ({', '.join(CLS_METRICS)})")
        plot_metrics_mod.main([
            "--inference", str(predictions_dir),
            "--splits-dir", str(splits_dir),
            "--metrics",   *CLS_METRICS,
            "--conf",      str(cls_conf),
        ])

    print("")
    print("══════════════════════════════════════════════════════════════")
    print("  Pipeline complete")
    print(f"  Run dir     : {run_dir}/")
    print(f"  Predictions : {predictions_dir}/")
    print("══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
