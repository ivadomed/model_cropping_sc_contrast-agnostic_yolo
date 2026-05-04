#!/usr/bin/env python3
"""
Export a trained YOLO checkpoint to a release-ready zip for sc_crop.

Lit preprocess.yaml depuis le snapshot du run, exporte best.pt → model.onnx
via ultralytics, puis produit un zip prêt à attacher à une GitHub release :

  sc_crop_models_v<version>.zip
  ├── model.onnx
  └── config.yaml

Les utilisateurs téléchargent ce zip via `sc-crop download` ou `sct_download_data`.
Ce script n'écrit RIEN dans le package sc_crop lui-même.

Requires: conda activate contrast_agnostic

Usage:
    python scripts/export_model.py --run-dir runs/20260504_134652 --version 0.1.0
"""

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml


def load_preprocess_cfg(run_dir: Path) -> dict:
    for candidate in [
        run_dir / "configs" / "preprocess.yaml",
        run_dir / "preprocess.yaml",
        Path("configs") / "preprocess.yaml",
    ]:
        if candidate.exists():
            return yaml.safe_load(candidate.read_text())
    raise FileNotFoundError(f"preprocess.yaml introuvable dans {run_dir} ou configs/")


def export_onnx(pt_path: Path) -> Path:
    from ultralytics import YOLO
    model = YOLO(str(pt_path))
    model.export(format="onnx", simplify=True, dynamic=False, opset=11)
    onnx_path = pt_path.with_suffix(".onnx")
    assert onnx_path.exists(), f"Export n'a produit aucun fichier à {onnx_path}"
    return onnx_path


def main():
    parser = argparse.ArgumentParser(
        description="Exporte un checkpoint YOLO en zip de release pour sc_crop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True,
                        help="Run directory contenant checkpoints/weights/best.pt")
    parser.add_argument("--version", default="0.1.0",
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
        print(f"Seul le mode axial est supporté.", file=sys.stderr)
        sys.exit(1)

    config = {
        "si_res":      pre_cfg["axial"]["si_res"],
        "inplane_res": pre_cfg["axial"].get("inplane_res"),
        "channels":    pre_cfg["channels"],
        "conf":        0.1,
    }

    print(f"Export {pt_path} …")
    onnx_path = export_onnx(pt_path)

    zip_name = f"sc_crop_models_v{args.version}.zip"
    zip_path = out_dir / zip_name

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy2(onnx_path, tmp / "model.onnx")
        (tmp / "config.yaml").write_text(
            yaml.dump(config, default_flow_style=False, sort_keys=False)
        )
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(tmp / "model.onnx", "model.onnx")
            zf.write(tmp / "config.yaml", "config.yaml")

    print(f"Release zip : {zip_path.resolve()}")
    print(f"Config      : {config}")
    print(f"\nAttacher {zip_name} à la GitHub release v{args.version}.")
    print(f"Puis mettre à jour _RELEASE_URL dans sc_crop/sc_crop/download.py.")


if __name__ == "__main__":
    main()
