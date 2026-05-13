#!/usr/bin/env python3
"""
Full training pipeline: download → preprocess → splits → build dataset → train → evaluate → metrics → plots → failures.

Reads configs/preprocess.yaml, configs/training_dataset.yaml, configs/training.yaml, configs/evaluation.yaml.
The mode (detection|classification) is read from configs/training.yaml and controls steps 4-9.
Configs are snapshotted into runs/<TS>/configs/ at startup — configs/ can be modified for the next run immediately.

Steps:
  1 = download datasets
  2 = preprocess
  3 = make splits
  4 = build dataset  (detection: YOLO detection format / classification: sc|no_sc folders)
  5 = train          (detection: yolo26n / classification: yolo26n-cls)
  6 = evaluate       (detection: bbox IoU / classification: gap_mm_S, gap_mm_I)
  7 = metrics
  8 = plot metrics
  9 = find failures

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --run-dir runs/20260101_120000 --start 5
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def _run_info() -> dict:
    try:
        hash_ = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip())
    except Exception:
        return {"git_hash": "unknown", "git_dirty": False}
    return {"git_hash": hash_, "git_dirty": dirty}

sys.path.insert(0, str(Path(__file__).parent))
import make_splits
import preprocess
import build_dataset
import build_class_dataset
import train
import evaluate
import evaluate_cls
import metrics as metrics_mod
import plot_metrics
import find_failures


CLS_METRICS         = ["gap_mm_S", "gap_mm_I"]
CLS_FAILURE_METRICS = ["gap_mm_S", "gap_mm_S_neg", "gap_mm_I", "gap_mm_I_neg"]


def step_active(n: int, start: int, end: int) -> bool:
    return start <= n <= end


def banner(n: int, end: int, label: str) -> None:
    print("")
    print("══════════════════════════════════════════════════════════════")
    print(f"  STEP {n}/{end} : {label}")
    print("══════════════════════════════════════════════════════════════")


def main():
    parser = argparse.ArgumentParser(
        description="Spine detection/classification pipeline — mode read from configs/training.yaml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir",  default=None, help="Run directory (default: runs/<TS>)")
    parser.add_argument("--start",    type=int, default=1, help="First step to run (1-9)")
    parser.add_argument("--end",      type=int, default=9, help="Last step to run (1-9)")
    parser.add_argument("--no-wandb",      action="store_true")
    parser.add_argument("--require-clean", action="store_true",
                        help="Abort if the git repo has uncommitted changes")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Check git state and save run_info.yaml
    info = _run_info()
    if args.require_clean and info.get("git_dirty"):
        print("ERROR: repo has uncommitted changes — commit or stash before running (or drop --require-clean)")
        sys.exit(1)

    (run_dir / "run_info.yaml").write_text(yaml.dump({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **info,
    }, sort_keys=False))

    if info.get("git_dirty"):
        print("  WARNING: repo has uncommitted changes — run may not be fully reproducible")

    # Snapshot configs immediately — user can edit configs/ for the next run right away.
    shutil.copytree("configs", run_dir / "configs", dirs_exist_ok=True)
    cfg_dir = run_dir / "configs"

    preprocess_cfg = yaml.safe_load((cfg_dir / "preprocess.yaml").read_text())
    dataset_cfg    = yaml.safe_load((cfg_dir / "training_dataset.yaml").read_text())
    training_cfg   = yaml.safe_load((cfg_dir / "training.yaml").read_text())
    eval_cfg       = yaml.safe_load((cfg_dir / "evaluation.yaml").read_text())

    mode = training_cfg["mode"]
    assert mode in ("detection", "classification"), \
        f"training.yaml: mode must be detection|classification, got {mode!r}"

    processed_dir   = Path(preprocess_cfg["out"])
    seed            = dataset_cfg["seed"]
    split_train     = dataset_cfg.get("train", 0.5)
    split_val       = dataset_cfg.get("val",   0.2)
    split_test      = dataset_cfg.get("test",  0.3)
    test_datasets   = dataset_cfg.get("test_datasets") or []
    splits_dir      = run_dir / "datasplits"
    predictions_dir = run_dir / "predictions"

    # Mode-specific paths
    if mode == "detection":
        dataset_dir    = run_dir / "dataset"
        checkpoint     = run_dir / "checkpoints" / "weights" / "best.pt"
    else:
        dataset_dir    = run_dir / "dataset_cls"
        checkpoint     = run_dir / "checkpoints_cls" / "weights" / "best.pt"
        cls_conf       = float(eval_cfg.get("cls_conf", 0.5))
        superior_only  = bool(training_cfg.get("classification", {}).get("superior_only", True))
        eval_sup_only  = bool(eval_cfg.get("eval_superior_only", False))

    print("══════════════════════════════════════════════════════════════")
    print(f"  Spine {mode} pipeline")
    print(f"  Run dir     : {run_dir}")
    print(f"  Steps       : {args.start} → {args.end}")
    print(f"  Mode        : {mode}")
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
        preprocess.run(config=cfg_dir / "preprocess.yaml")

    if step_active(3, S, E):
        banner(3, E, f"Make splits (seed={seed})")
        make_splits.run(raw="data/raw", out=splits_dir, seed=seed,
                        train=split_train, val=split_val, test=split_test)

    if step_active(4, S, E):
        if mode == "detection":
            banner(4, E, "Build YOLO detection dataset")
            build_dataset.run(
                config=cfg_dir / "training_dataset.yaml",
                processed=processed_dir,
                splits_dir=splits_dir,
                out=dataset_dir,
                test_datasets=test_datasets,
            )
        else:
            banner(4, E, "Build classification dataset (sc / no_sc)")
            build_class_dataset.run(
                processed=processed_dir,
                splits_dir=splits_dir,
                out=dataset_dir,
                test_datasets=test_datasets,
                superior_only=superior_only,
            )

    if step_active(5, S, E):
        banner(5, E, f"Train ({mode})")
        dataset_path = dataset_dir / "dataset.yaml" if mode == "detection" else dataset_dir
        train.run(
            config=cfg_dir / "training.yaml",
            dataset=dataset_path,
            run_dir=run_dir,
            no_wandb=args.no_wandb,
        )

    if step_active(6, S, E):
        if mode == "detection":
            banner(6, E, "Evaluate (bbox IoU)")
            if checkpoint.exists():
                evaluate.run(
                    checkpoint=checkpoint,
                    processed=processed_dir,
                    out=predictions_dir,
                    splits_dir=splits_dir,
                )
            else:
                print(f"  WARNING: checkpoint not found at {checkpoint} — skipping")
        else:
            banner(6, E, "Evaluate classifier (gap_mm_S / gap_mm_I)")
            if checkpoint.exists():
                evaluate_cls.run(
                    cls_checkpoint=checkpoint,
                    processed=processed_dir,
                    inference=predictions_dir,
                    cls_conf=cls_conf,
                    superior_only=eval_sup_only,
                )
            else:
                print(f"  WARNING: checkpoint not found at {checkpoint} — skipping")

    if step_active(7, S, E):
        banner(7, E, "Compute metrics")
        if mode == "detection":
            metrics_mod.run(inference=predictions_dir, splits_dir=splits_dir)
        else:
            metrics_mod.main([
                "--inference", str(predictions_dir),
                "--splits-dir", str(splits_dir),
                "--metrics", *CLS_METRICS,
            ])

    if step_active(8, S, E):
        banner(8, E, "Plot metrics")
        if mode == "detection":
            plot_metrics.run(inference=predictions_dir, splits_dir=splits_dir)
        else:
            plot_metrics.main([
                "--inference", str(predictions_dir),
                "--splits-dir", str(splits_dir),
                "--metrics", *CLS_METRICS,
                "--conf", str(cls_conf),
            ])

    if step_active(9, S, E):
        banner(9, E, "Find failures")
        if mode == "detection":
            find_failures.run(inference=predictions_dir, splits_dir=splits_dir)
        else:
            find_failures.main([
                "--inference",  str(predictions_dir),
                "--splits-dir", str(splits_dir),
                "--metrics",    *CLS_FAILURE_METRICS,
                "--conf",       str(cls_conf),
            ])

    print("")
    print("══════════════════════════════════════════════════════════════")
    print("  Pipeline complete")
    print(f"  Run dir     : {run_dir}/")
    print(f"  Predictions : {predictions_dir}/")
    print("══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
