"""
Produce a release-ready bundle for sc_crop from a detector run + a classifier run.

Exports both best.pt checkpoints to ONNX and assembles 4 release files:
  model.pt        ← detector checkpoint
  model.onnx      ← detector, ONNX format
  cls_model.pt    ← classifier checkpoint
  cls_model.onnx  ← classifier, ONNX format

Also writes config.yaml with full provenance (preprocessing params, git hashes,
wandb run ids) and a sha256.yaml with SHA256 hashes of the 4 files. Tags this repo
model-v{version} at the exported commit.

config.yaml includes all inference parameters (si_res, inplane_res, channels,
norm_scope, imgsz, conf, regularization, cls_conf). The sc-crop repo's
scripts/publish_release.sh reads it and deploys the inference-relevant subset
to sc_crop/config.yaml — see that script, or MIGRATION.md, for the full release flow.

Requires: conda activate sc_crop_training

Usage:
    python scripts/export_model.py \\
        --det-run-dir runs/20260524_224406 \\
        --cls-run-dir runs/20260525_150625 \\
        --version     0.0.6
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


def _export_onnx(pt_path: Path, imgsz: int, dynamic: bool) -> Path:
    """Export a YOLO .pt to ONNX in-place and return the .onnx path.

    dynamic=True exports with variable height/width axes — required for the
    detector, which is fed rectangular letterboxed inputs (320×W, W a multiple
    of 32) to match YOLO's predict() exactly. The classifier uses a fixed
    320×320 square input, so dynamic=False.
    """
    model = YOLO(str(pt_path))
    model.export(format="onnx", imgsz=imgsz, opset=19, dynamic=dynamic)
    onnx_path = pt_path.with_suffix(".onnx")
    assert onnx_path.exists(), f"ONNX export failed — {onnx_path} not found"
    return onnx_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export detector + classifier to a sc_crop release bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--det-run-dir",     required=True,
                        help="Detector run directory (runs/<timestamp>/)")
    parser.add_argument("--cls-run-dir",     required=True,
                        help="Classifier run directory (runs/<timestamp>/)")
    parser.add_argument("--version",         required=True,
                        help="Model version string, e.g. 0.0.6")
    parser.add_argument("--out-dir",         default="release_export",
                        help="Directory to write the 4 release files")
    parser.add_argument("--det-checkpoint",  default="best.pt",
                        help="Detector weight file in checkpoints/weights/ (default: best.pt)")
    parser.add_argument("--cls-checkpoint",  default="best.pt",
                        help="Classifier weight file in checkpoints/weights/ "
                             "(default: best.pt — use loss_best.pt for min-loss checkpoint)")
    args = parser.parse_args()

    det_run_dir = Path(args.det_run_dir)
    cls_run_dir = Path(args.cls_run_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    version     = args.version

    det_pt  = det_run_dir / "checkpoints" / "weights" / args.det_checkpoint
    cls_pt  = cls_run_dir / "checkpoints" / "weights" / args.cls_checkpoint
    assert det_pt.exists(),  f"Detector checkpoint not found: {det_pt}"
    assert cls_pt.exists(),  f"Classifier checkpoint not found: {cls_pt}"

    pre_cfg    = _load_yaml(det_run_dir / "configs" / "preprocess.yaml")
    norm_scope = pre_cfg["norm_scope"]
    assert norm_scope in ("slice", "slice_all", "volume"), (
        f"preprocess.yaml has norm_scope={norm_scope!r} — sc_crop only implements "
        f"'slice', 'slice_all', and 'volume'. Fix preprocess.yaml or add support in sc_crop first."
    )
    # Same thresholds used to evaluate/select each checkpoint (see run_pipeline.py step 6),
    # from each run's own configs/ snapshot -- not the repo's current configs/evaluation.yaml,
    # which may have moved on since these runs were trained.
    det_eval_cfg = _load_yaml(det_run_dir / "configs" / "evaluation.yaml")
    cls_eval_cfg = _load_yaml(cls_run_dir / "configs" / "evaluation.yaml")
    det_conf     = float(det_eval_cfg.get("det_conf", 0.1))
    cls_conf     = float(cls_eval_cfg.get("cls_conf", 0.5))
    det_info = _load_yaml(det_run_dir / "run_info.yaml")
    cls_info = _load_yaml(cls_run_dir / "run_info.yaml")
    det_args = _load_yaml(det_run_dir / "checkpoints" / "args.yaml")
    cls_args = _load_yaml(cls_run_dir / "checkpoints" / "args.yaml")
    imgsz    = int(det_args.get("imgsz", 320))

    wandb_id_file = det_run_dir / "wandb_run_id.txt"
    wandb_id      = wandb_id_file.read_text().strip() if wandb_id_file.exists() else None

    # ── ONNX export ──────────────────────────────────────────────────────────
    # Detector: dynamic axes (rectangular letterbox, like YOLO predict()).
    # Classifier: fixed 320×320 square input.
    print("Exporting detector to ONNX (dynamic) …")
    det_onnx = _export_onnx(det_pt, imgsz, dynamic=True)
    print("Exporting classifier to ONNX (fixed) …")
    cls_onnx = _export_onnx(cls_pt, imgsz, dynamic=False)

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
        "norm_scope":    norm_scope,
        "imgsz":         imgsz,
        # inference thresholds -- from each run's own configs/evaluation.yaml, see above
        "conf":          det_conf,
        "regularization": "cls",   # classifier run always provided → cls regularization
        "cls_conf":      cls_conf,
        # traceability — detector
        "det_run":             det_run_dir.name,
        "det_git_hash":        det_info.get("git_hash", "unknown"),
        "det_git_dirty":       det_info.get("git_dirty", False),
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

    # Write sha256.yaml — read by sc-crop's scripts/publish_release.sh to avoid parsing stdout
    sha256_data = {
        "version":    version,
        "det_run":    det_run_dir.name,
        "cls_run":    cls_run_dir.name,
        "export_git_hash": _git_head(),
        "assets":     shas,
    }
    (out_dir / "sha256.yaml").write_text(yaml.dump(sha256_data, default_flow_style=False, sort_keys=False))

    # ── Tag this repo at the commit that produced the export ───────────────────
    # Marks which training-repo commit v{version} was exported from — read by
    # anyone tracing a published model back to the code that trained it.
    #
    # release_export/ is gitignored (ephemeral) — regenerating it for an
    # already-tagged version (e.g. to redo a package-only sc-crop release that
    # doesn't touch the model) must stay possible. Only refuse if the existing
    # tag points to a *different* commit than this one, which would mean the
    # version number is being reused for genuinely different code.
    tag = f"model-v{version}"
    tag_commit = subprocess.run(["git", "rev-list", "-n", "1", tag],
                                capture_output=True, text=True).stdout.strip()
    head_commit = _git_head()
    if tag_commit:
        assert tag_commit == head_commit, (
            f"tag {tag} already exists at commit {tag_commit[:12]}, but HEAD is "
            f"{head_commit[:12]} — bump --version instead of reusing one already "
            f"exported from different code."
        )
        print(f"Tag {tag} already exists at this exact commit — not re-tagging (regenerating release_export/ only).")
    else:
        subprocess.run(["git", "tag", tag], check=True)
        subprocess.run(["git", "push", "origin", tag], check=True)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Release bundle v{version} → {out_dir.resolve()}/")
    for name in list(files) + ["config.yaml", "sha256.yaml"]:
        print(f"  {name}")
    print(f"Tagged this repo: {tag}")
    print(f"\n{'─'*60}")
    print(f"Done. Now, in the sc-crop repo:")
    print(f"  bash scripts/publish_release.sh --export-dir {out_dir.resolve()} --package-version <PACKAGE_VERSION>")


if __name__ == "__main__":
    main()
