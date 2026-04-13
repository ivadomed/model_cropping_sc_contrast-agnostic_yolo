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
    "iou_gt_mean":  True,   # ascending=True → lowest (worst) first
    "iou_all_mean": True,
}


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


def make_overview(pred_root: Path, dataset: str, stem: str, out_path: Path,
                  metric_name: str, metric_val: float) -> None:
    """Tile all overlay slice PNGs into a near-square composite image."""
    pngs = sorted((pred_root / dataset / stem / "png").glob("slice_*.png"))
    if not pngs:
        return

    n    = len(pngs)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    W, H = imgs[0].size

    header_h = 24
    canvas = Image.new("RGB", (cols * W, rows * H + header_h), (20, 20, 20))
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas.paste(img, (c * W, header_h + r * H))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=14)
    except OSError:
        font = ImageFont.load_default()

    val_str = f"{metric_val:.4f}" if not pd.isna(metric_val) else "NaN"
    draw.text((4, 5), f"{dataset}/{stem}   {metric_name} = {val_str}",
              fill=(255, 255, 255), font=font)
    canvas.save(out_path)


def write_failures(out_dir: Path, top: pd.DataFrame, metric: str, pred_root: Path, dataset: str) -> None:
    """Create ranking.csv + per-patient folders with data symlink and overview image."""
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    out_dir.mkdir(parents=True, exist_ok=True)
    top[["stem", metric]].to_csv(out_dir / "ranking.csv", index=False)

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
                      patient_dir / "overview.png", metric, row[metric])


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
    args = parser.parse_args()

    pred_root    = Path(args.inference)
    splits_map   = load_splits(Path(args.splits_dir))
    patients_idx = pd.read_csv(pred_root / "patients.csv")

    for split in args.splits:
        df = load_patients_at_conf(pred_root, patients_idx, splits_map, args.conf, split)
        if df.empty:
            print(f"  [{split}] no data at conf={args.conf}")
            continue
        jobs = []
        for metric, ascending in METRICS.items():
            for dataset, group in df.groupby("dataset"):
                top = group.dropna(subset=[metric]).sort_values(metric, ascending=ascending).head(args.top_k)
                if not top.empty:
                    jobs.append((dataset, metric, top))

        for dataset, metric, top in tqdm(jobs, desc=split, unit="dataset×metric"):
            out_dir = pred_root / dataset / "failures" / split / metric
            write_failures(out_dir, top, metric, pred_root, dataset)
        print(f"  [{split}] conf={args.conf} → done")


if __name__ == "__main__":
    main()
