#!/usr/bin/env python3
"""
Find worst-performing volumes per dataset for iou_gt_mean and iou_all_mean.

Requires patients.csv (index) and per-patient metrics/patient.csv from metrics.py.
Split assignment resolved at runtime from --splits-dir YAMLs.

Output structure:
  predictions/<run_id>/<dataset>/failures/<split>/<metric>/
    ranking.csv              ← top-K worst volumes with metric value
    001_<stem>/
      data                   ← symlink → ../../../../<stem>  (pngs, txts, volume/)
      overview.png           ← composite of all overlay slices

Usage:
    python scripts/find_failures.py --inference predictions/yolo26_1mm_axial_v2
    python scripts/find_failures.py --inference predictions/yolo26_1mm_axial_v2 --conf 0.1
    python scripts/find_failures.py --inference predictions/yolo26_1mm_axial_v2 \\
        --splits val train test --top-k 10
    python scripts/find_failures.py --inference predictions/yolo26_1mm_axial_v2 \\
        --exclude-csv bad_gt.csv
"""

import argparse
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

METRICS = {
    "iou_gt_mean":         True,    # ascending=True  → lowest (worst) first
    "iou_all_mean":        True,
    "iou_3d":              True,
    "iou_3d_mm":           True,
    "iou_3d_mm_filt":      True,
    "iou_3d_mm_ransac":    True,
    "iou_3d_mm_pad10":     True,
    "gt_in_pad10":         True,
    "iou_3d_mm_padz20":    True,
    "gt_in_padz20":        True,
    "iou_sc_mid_box":      True,
    "fp_on_gt_rate":       False,   # ascending=False → highest (worst) first
    "fp_on_gt_inner_rate": False,
    "gap_mm_R":            False,   # highest positive gap = pred misses GT on that face
    "gap_mm_L":            False,
    "gap_mm_P":            False,
    "gap_mm_A":            False,
    "gap_mm_I":            False,
    "gap_mm_S":            False,
    "gap_mm_R_neg":        True,    # most negative gap = pred over-extends on that face
    "gap_mm_L_neg":        True,
    "gap_mm_P_neg":        True,
    "gap_mm_A_neg":        True,
    "gap_mm_I_neg":        True,
    "gap_mm_S_neg":        True,
}

# IoU threshold used in metric definition (None = no IoU threshold)
METRIC_IOU_THRESH = {
    "iou_gt_mean":         None,
    "iou_all_mean":        None,
    "iou_3d":              None,
    "iou_3d_mm":           None,
    "iou_3d_mm_filt":      None,
    "iou_3d_mm_ransac":    None,
    "iou_3d_mm_pad10":     None,
    "gt_in_pad10":         None,
    "iou_3d_mm_padz20":    None,
    "gt_in_padz20":        None,
    "iou_sc_mid_box":      None,
    "fp_on_gt_rate":       "FP if IoU=0",
    "fp_on_gt_inner_rate": "FP if IoU=0  inner GT only",
    "gap_mm_R":            None,
    "gap_mm_L":            None,
    "gap_mm_P":            None,
    "gap_mm_A":            None,
    "gap_mm_I":            None,
    "gap_mm_S":            None,
    "gap_mm_R_neg":        None,
    "gap_mm_L_neg":        None,
    "gap_mm_P_neg":        None,
    "gap_mm_A_neg":        None,
    "gap_mm_I_neg":        None,
    "gap_mm_S_neg":        None,
}

# Maps metric name → actual column in patient.csv (for _neg variants)
METRIC_COLUMN = {m: m for m in METRIC_IOU_THRESH}
METRIC_COLUMN.update({
    "gap_mm_R_neg": "gap_mm_R",
    "gap_mm_L_neg": "gap_mm_L",
    "gap_mm_P_neg": "gap_mm_P",
    "gap_mm_A_neg": "gap_mm_A",
    "gap_mm_I_neg": "gap_mm_I",
    "gap_mm_S_neg": "gap_mm_S",
})


def load_splits(splits_dir: Path) -> dict:
    """Returns {(dataset, subject): split_name}."""
    mapping = {}
    for f in sorted(Path(splits_dir).glob("datasplit_*.yaml")):
        dataset = re.sub(r"_seed\d+$", "", f.stem[len("datasplit_"):])
        for split_name, subjects in yaml.safe_load(f.read_text()).items():
            for subj in (subjects or []):
                mapping[(dataset, subj)] = split_name
    return mapping


def load_patients_at_conf(pred_root: Path, patients_idx: pd.DataFrame, splits_map: dict,
                           conf_thresh: float, split: str) -> pd.DataFrame:
    """Build per-patient DataFrame by reading patient.csv at the given conf threshold."""
    rows = []
    for _, row in patients_idx.iterrows():
        dataset, stem = row["dataset"], row["stem"]
        m       = re.match(r"(sub-[^_]+)", stem)
        subject = m.group(1) if m else stem
        if splits_map.get((dataset, subject), "unknown") != split:
            continue
        patient_csv = pred_root / dataset / stem / "metrics" / "patient.csv"
        if not patient_csv.exists():
            continue
        df      = pd.read_csv(patient_csv)
        matched = df[np.isclose(df["conf_thresh"], conf_thresh, atol=0.0005)]
        if matched.empty:
            continue
        rows.append({"dataset": dataset, "stem": stem, **matched.iloc[0].to_dict()})
    return pd.DataFrame(rows)


def ordinal(n: int) -> str:
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th")
    return f"{n}{suffix}"


def make_overview(pred_root: Path, dataset: str, stem: str, out_path: Path,
                  metric_name: str, metric_val: float, conf_thresh: float, rank: int) -> None:
    """Tile all overlay slice PNGs into a near-square composite image."""
    pngs = sorted((pred_root / dataset / stem / "png").glob("slice_*.png"))
    if not pngs:
        return

    n    = len(pngs)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    W, H = imgs[0].size

    try:
        font_large  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=20)
        font_small  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      size=11)
    except OSError:
        font_large = font_small = ImageFont.load_default()

    header_h = 48
    canvas = Image.new("RGB", (cols * W, rows * H + header_h), (20, 20, 20))
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas.paste(img, (c * W, header_h + r * H))

    draw    = ImageDraw.Draw(canvas)
    val_str = f"{metric_val:.4f}" if not pd.isna(metric_val) else "NaN"

    iou_label  = METRIC_IOU_THRESH.get(metric_name)
    thresholds = f"conf≥{conf_thresh}"
    if iou_label is not None:
        thresholds += f"  {iou_label}"

    # Large: rank + dataset name + metric score + thresholds
    draw.text((8, 4),  f"{ordinal(rank)} worst — {dataset}   {metric_name} = {val_str}   ({thresholds})", fill=(255, 255, 255), font=font_large)
    # Small: patient stem, dimmed
    draw.text((8, 30), stem, fill=(160, 160, 160), font=font_small)

    canvas.save(out_path)


def write_failures(out_dir: Path, top: pd.DataFrame, metric: str, col: str,
                   pred_root: Path, dataset: str, conf_thresh: float) -> None:
    """Create ranking.csv + per-patient folders with data symlink and overview image."""
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    out_dir.mkdir(parents=True, exist_ok=True)
    top[["stem", col]].to_csv(out_dir / "ranking.csv", index=False)

    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        patient_dir = out_dir / f"{rank:03d}_{row['stem']}"
        patient_dir.mkdir(exist_ok=True)

        # symlink → pred_root/<dataset>/<stem>
        # out_dir = pred_root/<dataset>/failures/<split>/<metric>/
        # patient_dir/data is 5 levels deep → ../../../../<stem>
        data_link = patient_dir / "data"
        if data_link.exists() or data_link.is_symlink():
            data_link.unlink()
        data_link.symlink_to(Path("../../../../") / row["stem"])

        make_overview(pred_root, dataset, row["stem"],
                      patient_dir / "overview.png", metric, row[col], conf_thresh, rank)


def main():
    parser = argparse.ArgumentParser(
        description="Top-K worst volumes per dataset for iou_gt_mean and iou_all_mean",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference",  required=True, help="Path to predictions/<run-id>/")
    parser.add_argument("--splits-dir", default="data/datasplits/from_raw")
    parser.add_argument("--splits",     nargs="+", default=["val", "train", "test"],
                        choices=["train", "val", "test", "unknown"])
    parser.add_argument("--conf",       type=float, default=0.001,
                        help="Confidence threshold")
    parser.add_argument("--top-k",      type=int, default=10)
    parser.add_argument("--metrics",     nargs="+", default=None,
                        choices=list(METRICS), metavar="METRIC",
                        help="Metrics to compute (default: all)")
    parser.add_argument("--exclude-csv", default=None,
                        help="CSV with columns 'dataset' and 'stem' — matching pairs are excluded")
    args = parser.parse_args()

    pred_root    = Path(args.inference)
    splits_map   = load_splits(Path(args.splits_dir))
    patients_idx = pd.read_csv(pred_root / "patients.csv")
    if args.exclude_csv:
        excl     = pd.read_csv(args.exclude_csv)
        excl_set = set(zip(excl["dataset"], excl["stem"]))
        mask     = patients_idx.apply(lambda r: (r["dataset"], r["stem"]) not in excl_set, axis=1)
        n_excl   = (~mask).sum()
        patients_idx = patients_idx[mask]
        print(f"Excluded {n_excl} patient(s) from {args.exclude_csv}")

    for split in args.splits:
        df = load_patients_at_conf(pred_root, patients_idx, splits_map, args.conf, split)
        if df.empty:
            print(f"  [{split}] no data at conf={args.conf}")
            continue
        metrics_to_run = {m: v for m, v in METRICS.items()
                          if args.metrics is None or m in args.metrics}
        jobs = []
        for metric, ascending in metrics_to_run.items():
            col = METRIC_COLUMN.get(metric, metric)
            if col not in df.columns:
                continue
            for dataset, group in df.groupby("dataset"):
                top = group.dropna(subset=[col]).sort_values(col, ascending=ascending).head(args.top_k)
                if not top.empty:
                    jobs.append((dataset, metric, col, top))

        for dataset, metric, col, top in tqdm(jobs, desc=split, unit="dataset×metric"):
            out_dir = pred_root / dataset / "failures" / split / metric
            write_failures(out_dir, top, metric, col, pred_root, dataset, args.conf)
        print(f"  [{split}] conf={args.conf} → done")


if __name__ == "__main__":
    main()
