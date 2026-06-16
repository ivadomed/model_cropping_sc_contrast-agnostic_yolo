#!/usr/bin/env python3
"""
Detector experiment — one inference pass over the held-out test split.

Runs the sc-crop detector (package `sc_crop`, ONNX/CPU, cls regularisation = the
shipped default) once per test volume and derives THREE paper results from the
single bbox it returns:

  E1  FOV reduction eta = original_voxels / cropped_box_voxels   (efficiency)
  E2  cord coverage      = does the padded box contain 100% of the cord GT?
                           (sc_crop.check_label_crop; the GT label is first
                            cleaned to its largest 3D connected component so a
                            stray annotation voxel cannot create a false miss)
  E3  detector latency   = wall-clock of detect() per volume on CPU

It is also a usage proof: the only sc-crop calls are `detect()` and
`check_label_crop()`. No dependency on this repo's scripts/ (metrics.py,
evaluate.py) — those are slated for refactor.

LATENCY NOTE: sc_crop.detect() reloads the ONNX detector+classifier sessions on
every call (no session cache). So the raw per-volume time includes model load.
We measure model-load ONCE at startup (t_model_load_s) and report both the raw
per-call time (t_detect_full_s) and the amortised steady-state time
(t_detect_steady_s = full - load) — the latter is the throughput-relevant number.

Plumbing the package cannot know (kept explicit, no fallback):
  - the test-subject list      -> data/datasplits_seed50/*.yaml (`test` key)
  - the (raw image, raw mask)  -> processed/<variant>/<dataset>/<patient>/meta.yaml
    pairs                         (records the exact pairs used for training)

Outputs (all under one folder, paper-ready):
  experiments/out/detector_<MODEL_VERSION>/
    results.csv             — one row per image/contrast (the crop report + eta + latency)
    summary.json            — global + per-contrast + per-dataset aggregates (paper tables)
    missing_from_processed.txt — test subjects absent from processed/ (known cases)

Usage (run from any directory; long CPU job -> launch via set_slot):
    python experiments/experiment_detector.py
    python experiments/experiment_detector.py --limit 20        # quick smoke test
    python experiments/experiment_detector.py --repeat-timing 5 # stable latency (median)

Author: Quentin Revillon
"""

import argparse
import csv
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from scipy.ndimage import label as cc_label

from sc_crop import detect, check_label_crop, ensure_model, ensure_cls_model, MODEL_VERSION
from sc_crop.infer_onnx import load_session

REPO    = Path(__file__).resolve().parents[1]
SPLITS  = REPO / "data" / "datasplits_seed50"
VARIANT = "10mm_SI_1mm_axial_3ch_normslice_all"   # processed variant = shipped detector preprocessing
FACES   = ["superior", "inferior", "left", "right", "anterior", "posterior"]
# Datasets excluded from the test: beijing-tumor has faulty SC labels (the GT does
# not delineate the cord reliably), so its coverage figures are not meaningful.
EXCLUDE_DATASETS = {"beijing-tumor"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default=VARIANT, help="processed/ variant used only to locate raw image/mask pairs")
    p.add_argument("--out", default=None, help="output dir (default experiments/out/detector_<MODEL_VERSION>)")
    p.add_argument("--limit", type=int, default=0, help="process only the first N volumes (smoke test)")
    p.add_argument("--repeat-timing", type=int, default=1, help="repeat detect() N times per volume, keep median")
    return p.parse_args()


def test_subjects() -> dict:
    """dataset -> list of test subject ids, from the seed50 split YAMLs."""
    out = {}
    for f in sorted(SPLITS.glob("datasplit_*_seed50.yaml")):
        d = yaml.safe_load(f.read_text())
        name = d["meta"]["name"]
        if name in EXCLUDE_DATASETS:
            continue
        out[name] = d["test"]
    return out


def raw_pairs(variant: str, dataset: str, subject: str) -> list:
    """[(case_id, raw_image, raw_mask)] for every processed contrast dir of this subject."""
    base = REPO / "processed" / variant / dataset
    dirs = sorted(base.glob(f"{subject}_*"))
    if (base / subject).is_dir():
        dirs.append(base / subject)
    pairs = []
    for patient_dir in dirs:
        meta = yaml.safe_load((patient_dir / "meta.yaml").read_text())
        pairs.append((patient_dir.name, REPO / meta["raw_image"], REPO / meta["raw_mask"]))
    return pairs


def contrast_of(img_path: Path) -> str:
    """BIDS modality suffix = last '_' token of the filename stem (e.g. T2w, T2star, dwi, UNIT1, MTS)."""
    stem = img_path.name.replace(".nii.gz", "").replace(".nii", "")
    return stem.split("_")[-1]


def largest_component(mask: nib.Nifti1Image):
    """Return (cleaned NIfTI keeping only the largest 3D CC, n_components, stray_voxel_count)."""
    data = np.asarray(mask.dataobj) > 0
    cc, n = cc_label(data)
    if n == 0:
        return mask, 0, 0
    sizes = np.bincount(cc.ravel())[1:]           # skip background label 0
    keep  = int(np.argmax(sizes)) + 1
    clean = (cc == keep).astype(np.uint8)
    stray = int(data.sum() - sizes.max())
    return nib.Nifti1Image(clean, mask.affine, mask.header), int(n), stray


def model_load_seconds() -> float:
    """One-time cost detect() pays per call: load detector + classifier ONNX sessions."""
    t0 = time.perf_counter()
    load_session(ensure_model())
    load_session(ensure_cls_model())
    return time.perf_counter() - t0


def run_one(dataset, case_id, img_path, mask_path, repeat, t_load):
    times = []
    for _ in range(repeat):
        t0   = time.perf_counter()
        bbox = detect(img_path)                       # ← the single inference (ONNX/CPU, cls)
        times.append(time.perf_counter() - t0)
    t_full = float(np.median(times))

    img         = nib.load(str(img_path))
    voxels_box  = ((bbox["xmax"] - bbox["xmin"] + 1)
                   * (bbox["ymax"] - bbox["ymin"] + 1)
                   * (bbox["zmax"] - bbox["zmin"] + 1))
    voxels_orig = int(np.prod(img.shape[:3]))

    clean, n_comp, stray = largest_component(nib.load(str(mask_path)))
    qc = check_label_crop(clean, bbox)                # ← coverage: ok + extra_pad_*_mm per face

    row = {
        "dataset": dataset, "case_id": case_id, "contrast": contrast_of(img_path),
        "eta": round(voxels_orig / voxels_box, 3),
        "voxels_orig": voxels_orig, "voxels_box": int(voxels_box),
        "t_detect_full_s": round(t_full, 4),
        "t_detect_steady_s": round(max(0.0, t_full - t_load), 4),
        "cov_ok": qc["ok"], "stray_voxels": stray, "n_components": n_comp,
    }
    for face in FACES:
        row[f"extra_{face}_mm"] = qc[f"extra_pad_{face}_mm"]
    return row


def group_stats(rows, key) -> dict:
    out = {}
    for g in sorted({r[key] for r in rows}):
        sub = [r for r in rows if r[key] == g]
        eta = np.array([r["eta"] for r in sub])
        ok  = np.array([r["cov_ok"] for r in sub])
        out[g] = {
            "n": len(sub),
            "eta_mean": round(float(eta.mean()), 3),
            "eta_median": round(float(np.median(eta)), 3),
            "eta_min": round(float(eta.min()), 3),
            "eta_max": round(float(eta.max()), 3),
            "coverage_ok": int(ok.sum()),
            "coverage_pct": round(100 * float(ok.mean()), 2),
        }
    return out


def main():
    args    = parse_args()
    out_dir = Path(args.out) if args.out else REPO / "experiments" / "out" / f"detector_{MODEL_VERSION}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_load = model_load_seconds()
    print(f"model load (det+cls) once: {t_load:.3f}s  | version {MODEL_VERSION}")

    fields = (["dataset", "case_id", "contrast", "eta", "voxels_orig", "voxels_box",
               "t_detect_full_s", "t_detect_steady_s", "cov_ok", "stray_voxels", "n_components"]
              + [f"extra_{f}_mm" for f in FACES])

    # Rows are written incrementally so progress is never lost on a long run.
    rows, missing = [], []
    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()
        done = False
        for dataset, subjects in test_subjects().items():
            for subject in subjects:
                pairs = raw_pairs(args.variant, dataset, subject)
                if not pairs:
                    missing.append(f"{dataset}/{subject}")   # absent from processed/ (known cases, see CLAUDE.md)
                    continue
                for case_id, img_path, mask_path in pairs:
                    row = run_one(dataset, case_id, img_path, mask_path, args.repeat_timing, t_load)
                    rows.append(row)
                    writer.writerow(row)
                    fcsv.flush()
                    print(f"[{len(rows):4d}] {dataset}/{case_id}  eta={row['eta']}  ok={row['cov_ok']}  "
                          f"t_steady={row['t_detect_steady_s']}s")
                    if args.limit and len(rows) >= args.limit:
                        done = True; break
                if done:
                    break
            if done:
                break

    eta = np.array([r["eta"] for r in rows])
    ok  = np.array([r["cov_ok"] for r in rows])
    ts  = np.array([r["t_detect_steady_s"] for r in rows])
    summary = {
        "model_version": MODEL_VERSION,
        "n_volumes": len(rows),
        "n_subjects_missing_from_processed": len(missing),
        "t_model_load_s": round(t_load, 4),
        "global": {
            "eta_mean": round(float(eta.mean()), 3), "eta_median": round(float(np.median(eta)), 3),
            "eta_min": round(float(eta.min()), 3), "eta_max": round(float(eta.max()), 3),
            "coverage_ok": int(ok.sum()), "coverage_total": len(ok),
            "coverage_pct": round(100 * float(ok.mean()), 2),
            "latency_steady_median_s": round(float(np.median(ts)), 4),
            "latency_steady_mean_s": round(float(ts.mean()), 4),
            "latency_steady_max_s": round(float(ts.max()), 4),
        },
        "by_contrast": group_stats(rows, "contrast"),
        "by_dataset":  group_stats(rows, "dataset"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if missing:
        (out_dir / "missing_from_processed.txt").write_text("\n".join(missing) + "\n")

    g = summary["global"]
    print("\n==================== SUMMARY ====================")
    print(f"volumes processed   : {g['coverage_total']}   (subjects missing from processed/: {len(missing)})")
    print(f"E1 eta (FOV)        : mean {g['eta_mean']}  median {g['eta_median']}  range [{g['eta_min']}, {g['eta_max']}]")
    print(f"E2 coverage (cls)   : {g['coverage_ok']}/{g['coverage_total']} keep 100% of cord GT ({g['coverage_pct']}%)")
    print(f"E3 latency steady   : median {g['latency_steady_median_s']}s  (model load once: {t_load:.3f}s)")
    print(f"-> {out_dir/'results.csv'}\n-> {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
