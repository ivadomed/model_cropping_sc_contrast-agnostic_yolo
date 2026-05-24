#!/usr/bin/env python3
"""
Produce a release-ready zip for sc_crop from a trained YOLO checkpoint.

Reads preprocess.yaml from the run snapshot, copies best.pt directly (no ONNX
conversion), and writes a config.yaml with the inference parameters.
The training run name and repo commit are embedded in config.yaml for traceability
(see VERSIONS.md in the sc-crop package for the version linkage table).

Output:
  sc_crop_models_v<version>.zip
  ├── model.pt
  └── config.yaml   ← includes training_run and training_repo_commit fields

This script writes nothing to the sc_crop package itself.

Requires: conda activate contrast_agnostic

Usage:
    python scripts/export_model.py --run-dir checkpoints/pipeline_yolo26n_axial_200ep_20260430_2319582 --version 0.0.4
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
    """Return the current HEAD commit SHA (short), or 'unknown' if not in a git repo."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def load_preprocess_cfg(run_dir: Path) -> dict:
    for candidate in [
        run_dir / "configs" / "preprocess.yaml",
        run_dir / "preprocess.yaml",
        Path("configs") / "preprocess.yaml",
    ]:
        if candidate.exists():
            return yaml.safe_load(candidate.read_text())
    raise FileNotFoundError(f"preprocess.yaml introuvable dans {run_dir} ou configs/")


def main():
    parser = argparse.ArgumentParser(
        description="Produit un zip de release sc_crop depuis un checkpoint YOLO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True,
                        help="Run directory contenant checkpoints/weights/best.pt")
    parser.add_argument("--version", default="0.2.0",
                        help="Version du modèle (utilisée dans le nom du zip)")
    parser.add_argument("--out-dir", default=".",
                        help="Dossier où écrire le zip de release")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_path = run_dir / "checkpoints" / "weights" / "best.pt"
    if not pt_path.exists():
        print(f"Checkpoint introuvable : {pt_path}", file=sys.stderr)
        sys.exit(1)

    pre_cfg = load_preprocess_cfg(run_dir)
    if pre_cfg.get("plane", "axial") != "axial":
        print("Seul le mode axial est supporté.", file=sys.stderr)
        sys.exit(1)

    commit = _git_head_commit()

    config = {
        "si_res":                 pre_cfg["axial"]["si_res"],
        "inplane_res":            pre_cfg["axial"].get("inplane_res"),
        "channels":               pre_cfg["channels"],
        "conf":                   0.1,
        # Traceability: link this model back to its training run and repo commit.
        # See VERSIONS.md in ivadomed/sc-crop for the full version linkage table.
        "training_run":           run_dir.name,
        "training_repo_commit":   commit,
    }
    config_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)

    # Write config.yaml next to best.pt so --model path/to/best.pt works directly.
    (pt_path.parent / "config.yaml").write_text(config_yaml)
    print(f"Config      : {pt_path.parent / 'config.yaml'}")

    zip_name = f"sc_crop_models_v{args.version}.zip"
    zip_path = out_dir / zip_name

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy2(pt_path, tmp / "model.pt")
        (tmp / "config.yaml").write_text(config_yaml)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(tmp / "model.pt",     "model.pt")
            zf.write(tmp / "config.yaml",  "config.yaml")

    print(f"Release zip : {zip_path.resolve()}")
    print(f"Config      :")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"\nProchaines étapes :")
    print(f"  1. gh release create v{args.version} model.onnx model.pt cls_model.onnx cls_model.pt "
          f"--repo ivadomed/sc-crop --title 'sc-crop model v{args.version}'")
    print(f"  2. Mettre à jour _MODEL_TAG + SHA256 dans sc_crop/download.py")
    print(f"  3. Bumper version dans pyproject.toml")
    print(f"  4. Ajouter une ligne dans VERSIONS.md (run: {run_dir.name}, commit: {commit[:12]})")


if __name__ == "__main__":
    main()
