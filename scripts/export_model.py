#!/usr/bin/env python3
"""
Produce a release-ready zip for sc_crop from a trained YOLO checkpoint.

Reads preprocess.yaml and run_info.yaml from the run snapshot, copies best.pt
directly (no ONNX conversion), and writes a config.yaml with all inference
parameters needed to exactly reproduce the preprocessing used at training time.

Output:
  sc_crop_models_v<version>.zip
  ├── model.pt
  └── config.yaml   ← si_res, inplane_res, channels, norm_scope, conf,
                       training_run, training_git_hash, training_git_dirty,
                       training_repo_commit (current repo HEAD)

This script writes nothing to the sc_crop package itself.

Requires: conda activate contrast_agnostic

Usage:
    python scripts/export_model.py --run-dir runs/20260524_224406 --version 0.3.0
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml


def _git_head_commit() -> str:
    """Return the current HEAD commit SHA, or 'unknown' if not in a git repo."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def load_preprocess_cfg(run_dir: Path) -> dict:
    for candidate in [
        run_dir / "configs" / "preprocess.yaml",
        run_dir / "preprocess.yaml",
        Path("configs") / "preprocess.yaml",
    ]:
        if candidate.exists():
            return yaml.safe_load(candidate.read_text())
    raise FileNotFoundError(f"preprocess.yaml not found in {run_dir} or configs/")


def load_run_info(run_dir: Path) -> dict:
    path = run_dir / "run_info.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(
        description="Produce a sc_crop release zip from a training run directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True,
                        help="Run directory (runs/<timestamp>/) containing checkpoints/ and configs/")
    parser.add_argument("--version", default="0.3.0",
                        help="Model version string (used in the zip filename)")
    parser.add_argument("--out-dir", default=".",
                        help="Directory to write the release zip")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_path = run_dir / "checkpoints" / "weights" / "best.pt"
    assert pt_path.exists(), f"Checkpoint not found: {pt_path}"

    pre_cfg  = load_preprocess_cfg(run_dir)
    run_info = load_run_info(run_dir)

    assert pre_cfg.get("plane", "axial") == "axial", "Only axial plane is supported."

    config = {
        # ── Preprocessing parameters (must match training exactly) ──────────
        "si_res":       pre_cfg["axial"]["si_res"],
        "inplane_res":  pre_cfg["axial"].get("inplane_res"),
        "channels":     3 if pre_cfg.get("three_ch", False) else 1,
        "norm_scope":   pre_cfg.get("norm_scope", "slice"),
        # ── Inference parameters ────────────────────────────────────────────
        "conf":         0.1,
        # ── Traceability ─────────────────────────────────────────────────────
        # training_run      : run directory name (timestamp) in this repo
        # training_git_hash : git commit of this repo when the run was launched
        # training_git_dirty: whether the repo had uncommitted changes at launch
        # training_repo_commit: current HEAD of this repo at export time
        "training_run":           run_dir.name,
        "training_git_hash":      run_info.get("git_hash", "unknown"),
        "training_git_dirty":     run_info.get("git_dirty", False),
        "training_repo_commit":   _git_head_commit(),
    }
    config_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)

    # Write config.yaml next to best.pt (useful for local testing).
    (pt_path.parent / "config.yaml").write_text(config_yaml)

    zip_name = f"sc_crop_models_v{args.version}.zip"
    zip_path = out_dir / zip_name

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy2(pt_path, tmp / "model.pt")
        (tmp / "config.yaml").write_text(config_yaml)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(tmp / "model.pt",    "model.pt")
            zf.write(tmp / "config.yaml", "config.yaml")

    print(f"Release zip : {zip_path.resolve()}")
    print(f"Config      :")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"\nNext steps:")
    print(f"  1. gh release create v{args.version} {zip_path} "
          f"--repo ivadomed/sc-crop --title 'sc-crop model v{args.version}'")
    print(f"  2. Update _MODEL_TAG + SHA256 in sc_crop/download.py")
    print(f"  3. Bump version in pyproject.toml")
    print(f"  4. Add row in VERSIONS.md: "
          f"version={args.version}  run={run_dir.name}  "
          f"git_hash={run_info.get('git_hash', 'unknown')[:12]}")


if __name__ == "__main__":
    main()
