"""
Produce a release-ready bundle for sc_crop from a detector run + a classifier run.

Exports both best.pt checkpoints to ONNX and assembles 4 release files:
  model.pt        ← detector checkpoint
  model.onnx      ← detector, ONNX format
  cls_model.pt    ← classifier checkpoint
  cls_model.onnx  ← classifier, ONNX format

Also writes config.yaml with full provenance (preprocessing params, git hashes,
wandb run ids) and prints SHA256 hashes ready to paste into sc_crop/download.py.

Requires: conda activate contrast_agnostic

Usage:
    python scripts/export_model.py \\
        --run-dir     runs/20260524_224406 \\
        --cls-run-dir runs/20260525_150625 \\
        --version     0.0.5
"""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

import yaml
from ultralytics import YOLO


# ── helpers ──────────────────────────────────────────────────────────────────

def _git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def _export_onnx(pt_path: Path, imgsz: int) -> Path:
    """Export a YOLO .pt to ONNX in-place and return the .onnx path."""
    model = YOLO(str(pt_path))
    model.export(format="onnx", imgsz=imgsz)
    onnx_path = pt_path.with_suffix(".onnx")
    assert onnx_path.exists(), f"ONNX export failed — {onnx_path} not found"
    return onnx_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export detector + classifier to a sc_crop release bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir",     required=True,
                        help="Detector run directory (runs/<timestamp>/)")
    parser.add_argument("--cls-run-dir", required=True,
                        help="Classifier run directory (runs/<timestamp>/)")
    parser.add_argument("--version",     required=True,
                        help="Model version string, e.g. 0.0.5")
    parser.add_argument("--out-dir",     default="release_export",
                        help="Directory to write the 4 release files")
    args = parser.parse_args()

    run_dir     = Path(args.run_dir)
    cls_run_dir = Path(args.cls_run_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    version     = args.version

    det_pt  = run_dir     / "checkpoints"     / "weights" / "best.pt"
    cls_pt  = cls_run_dir / "checkpoints_cls" / "weights" / "best.pt"
    assert det_pt.exists(),  f"Detector checkpoint not found: {det_pt}"
    assert cls_pt.exists(),  f"Classifier checkpoint not found: {cls_pt}"

    pre_cfg  = _load_yaml(run_dir / "configs" / "preprocess.yaml")
    run_info = _load_yaml(run_dir / "run_info.yaml")
    cls_info = _load_yaml(cls_run_dir / "run_info.yaml")
    det_args = _load_yaml(run_dir / "checkpoints" / "args.yaml")
    cls_args = _load_yaml(cls_run_dir / "checkpoints_cls" / "args.yaml")
    imgsz    = int(det_args.get("imgsz", 320))

    wandb_id_file = run_dir / "wandb_run_id.txt"
    wandb_id      = wandb_id_file.read_text().strip() if wandb_id_file.exists() else None

    # ── ONNX export ──────────────────────────────────────────────────────────
    print("Exporting detector to ONNX …")
    det_onnx = _export_onnx(det_pt, imgsz)
    print("Exporting classifier to ONNX …")
    cls_onnx = _export_onnx(cls_pt, imgsz)

    # ── Copy the 4 files to out_dir ──────────────────────────────────────────
    files = {
        "model.pt":        det_pt,
        "model.onnx":      det_onnx,
        "cls_model.pt":    cls_pt,
        "cls_model.onnx":  cls_onnx,
    }
    for name, src in files.items():
        shutil.copy2(src, out_dir / name)

    # ── config.yaml — full provenance ─────────────────────────────────────────
    config = {
        "version":       version,
        # preprocessing — must match inference exactly
        "si_res":        pre_cfg.get("axial", {}).get("si_res", 10.0),
        "inplane_res":   pre_cfg.get("axial", {}).get("inplane_res", 1.0),
        "channels":      3 if pre_cfg.get("three_ch", False) else 1,
        "norm_scope":    pre_cfg.get("norm_scope", "slice"),
        # inference
        "conf":          0.1,
        # traceability — detector
        "det_run":             run_dir.name,
        "det_git_hash":        run_info.get("git_hash", "unknown"),
        "det_git_dirty":       run_info.get("git_dirty", False),
        "det_wandb_run_id":    wandb_id,
        # traceability — classifier
        "cls_run":             cls_run_dir.name,
        "cls_git_hash":        cls_info.get("git_hash", "unknown"),
        "cls_git_dirty":       cls_info.get("git_dirty", False),
        # traceability — export
        "export_git_hash":     _git_head(),
    }
    config_path = out_dir / "config.yaml"
    config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))

    # ── SHA256 ────────────────────────────────────────────────────────────────
    shas = {name: _sha256(out_dir / name) for name in files}

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Release bundle v{version} → {out_dir.resolve()}/")
    for name in files:
        print(f"  {name}")
    print(f"  config.yaml")

    print(f"\n{'─'*60}")
    print("SHA256 hashes — paste into sc_crop/download.py :")
    print(f'_MODEL_TAG = "v{version}"')
    print(f'_ASSETS = {{')
    for name, sha in shas.items():
        key = name.replace("-", "_")  # cls_model.onnx → already correct
        print(f'    "{name}": {{"url": f"{{_BASE_URL}}/{name}", "sha256": "{sha}"}},')
    print(f'}}')

    print(f"\n{'─'*60}")
    print("Next steps :")
    print(f"  1. gh release create v{version} {out_dir}/model.pt {out_dir}/model.onnx "
          f"{out_dir}/cls_model.pt {out_dir}/cls_model.onnx \\")
    print(f"       --repo ivadomed/sc-crop \\")
    print(f"       --title 'sc-crop model v{version}' \\")
    print(f"       --notes 'det={run_dir.name}  cls={cls_run_dir.name}'")
    print(f"  2. Update _MODEL_TAG + SHA256 in sc_crop/download.py")
    print(f"  3. Update VERSIONS.md")
    print(f"  4. Bump version in pyproject.toml + sc_crop/__init__.py")
    print(f"  5. git commit + tag v{version} + push")


if __name__ == "__main__":
    main()
