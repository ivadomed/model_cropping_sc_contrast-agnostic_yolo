#!/usr/bin/env python3
"""
Compute 2D bbox metrics from saved predictions: IoU, Dice, AP50, AP50:95.

For every slice of every patient:
  - iou            : IoU vs same-slice GT (0 if either is absent)
  - iou_nearest_gt : IoU vs nearest GT slice in z (same as iou when GT present on slice;
                     uses neighbour GT when no GT on slice; 0 if no pred)
  - is_fp          : pred present but no GT on this slice
  - is_fn          : GT present but no pred on this slice

Per-patient slice table saved to:
  predictions/<run_id>/<dataset>/<patient>/metrics/slices.csv

Per-patient aggregated table saved to:
  predictions/<run_id>/patients.csv

Aggregated metrics (cross-dataset/contrast/split) saved to:
  predictions/<run_id>/metrics.csv

HTML report saved to:
  predictions/<run_id>/report.html

Usage:
    python scripts/metrics.py \
        --inference predictions/yolo_spine_v1 \
        --processed processed_10mm_SI \
        [--conf 0.5] [--split val] [--splits-dir data/datasplits]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

CONF_THRESH = 0.5  # default confidence threshold for fixed-threshold metrics

SPLITS = ["test", "val", "train", "unknown"]

CSS = """
<style>
  body { font-family: Arial, sans-serif; font-size: 13px; margin: 30px; background: #f9f9f9; }
  h1   { color: #222; }
  h2   { color: #444; margin-top: 40px; border-bottom: 2px solid #ccc; padding-bottom: 4px; }
  h3   { color: #555; margin-top: 30px; }
  table { border-collapse: collapse; margin-bottom: 20px; background: white;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  th   { background: #3a3a3a; color: white; padding: 7px 12px; text-align: center; }
  th.left { text-align: left; }
  td   { padding: 5px 12px; border-bottom: 1px solid #e0e0e0; text-align: center; }
  td.left { text-align: left; }
  tr:hover td { background: #f0f4ff; }
  .good  { color: #1a7a1a; font-weight: bold; }
  .ok    { color: #7a6a00; }
  .bad   { color: #aa1a1a; }
  .na    { color: #aaa; font-style: italic; }
  .meta  { font-size: 11px; color: #888; }
</style>
"""


def load_splits(splits_dir: Path) -> dict:
    """Returns {(dataset, subject): split_name} from all datasplit_*.yaml."""
    mapping = {}
    for f in sorted(splits_dir.glob("datasplit_*.yaml")):
        dataset = re.sub(r"_seed\d+$", "", f.stem[len("datasplit_"):])
        for split_name, subjects in yaml.safe_load(f.read_text()).items():
            for subj in (subjects or []):
                mapping[(dataset, subj)] = split_name
    return mapping


def read_gt_box(txt_path: Path):
    """Returns (cx,cy,w,h) or None if file empty/missing."""
    if not txt_path.exists():
        return None
    content = txt_path.read_text().strip()
    if not content:
        return None
    p = content.split()
    return (float(p[1]), float(p[2]), float(p[3]), float(p[4]))


def read_pred(txt_path: Path):
    """Returns ((cx,cy,w,h), conf) or (None, 0.0) if empty/missing."""
    if not txt_path.exists():
        return None, 0.0
    content = txt_path.read_text().strip()
    if not content:
        return None, 0.0
    p = content.split()
    return (float(p[1]), float(p[2]), float(p[3]), float(p[4])), (float(p[5]) if len(p) > 5 else 1.0)


def bbox_iou(a, b) -> float:
    """IoU between two (cx,cy,w,h) normalised bboxes."""
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    inter = max(0., min(ax2,bx2)-max(ax1,bx1)) * max(0., min(ay2,by2)-max(ay1,by1))
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.


def patient_slices(gt_txt_dir: Path, pred_txt_dir: Path) -> pd.DataFrame:
    """Build one row per slice for a patient."""
    gt_txts   = {int(p.stem.split("_")[1]): p for p in sorted(gt_txt_dir.glob("slice_*.txt"))}
    pred_txts = {int(p.stem.split("_")[1]): p
                 for p in sorted(pred_txt_dir.glob("slice_*.txt"))} if pred_txt_dir.is_dir() else {}

    all_z = sorted(set(gt_txts) | set(pred_txts))
    if not all_z:
        return pd.DataFrame()

    gt_by_z   = {z: read_gt_box(p) for z, p in gt_txts.items()}
    gt_by_z   = {z: b for z, b in gt_by_z.items() if b is not None}
    gt_slices = np.array(sorted(gt_by_z)) if gt_by_z else np.array([], dtype=int)

    rows = []
    for z in all_z:
        gt_box              = gt_by_z.get(z)
        pred_box, pred_conf = read_pred(pred_txts[z]) if z in pred_txts else (None, 0.0)
        has_gt              = gt_box is not None
        has_pred            = pred_box is not None
        iou                 = bbox_iou(gt_box, pred_box) if (has_gt and has_pred) else 0.0

        if not has_pred or len(gt_slices) == 0:
            iou_nearest_gt, z_dist_to_ref, ref_gt_slice = 0.0, None, None
        else:
            dists          = np.abs(gt_slices - z)
            nearest_z      = int(gt_slices[dists.argmin()])
            z_dist_to_ref  = int(dists.min())
            ref_gt_slice   = nearest_z
            iou_nearest_gt = bbox_iou(gt_by_z[nearest_z], pred_box)

        rows.append({
            "slice_idx":        z,
            "has_gt":           has_gt,
            "has_pred":         has_pred,
            "pred_conf":        round(pred_conf, 4),
            "iou":              round(iou, 4),
            "iou_nearest_gt":   round(iou_nearest_gt, 4),
            "z_dist_to_ref_gt": z_dist_to_ref,
            "ref_gt_slice":     ref_gt_slice,
            "is_fp":            has_pred and not has_gt,
            "is_fn":            has_gt and not has_pred,
        })
    return pd.DataFrame(rows)


def ap_at_iou(df: pd.DataFrame, iou_col: str, iou_thresh: float) -> float:
    n_gt = int(df["has_gt"].sum())
    if n_gt == 0:
        return float("nan")
    preds = df[df["has_pred"]].sort_values("pred_conf", ascending=False)
    if len(preds) == 0:
        return 0.0
    tp        = (preds["has_gt"] & (preds[iou_col] >= iou_thresh)).astype(int).values
    cum_tp    = np.cumsum(tp)
    cum_fp    = np.cumsum(1 - tp)
    precision = cum_tp / (cum_tp + cum_fp)
    recall    = cum_tp / n_gt
    precision = np.concatenate([[1.0], precision])
    recall    = np.concatenate([[0.0], recall])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    return float(np.trapezoid(precision, recall))


def summarise_group(df: pd.DataFrame, conf_thresh: float) -> dict:
    n_slices    = len(df)
    n_gt        = int(df["has_gt"].sum())
    n_pred      = int(df["has_pred"].sum())
    n_fp        = int(df["is_fp"].sum())
    n_fn        = int(df["is_fn"].sum())
    fp_rate     = round(n_fp / n_pred, 4) if n_pred else float("nan")
    fn_rate     = round(n_fn / n_gt,   4) if n_gt   else float("nan")
    has_thresh  = df["has_pred"] & (df["pred_conf"] >= conf_thresh)
    n_thresh    = int(has_thresh.sum())
    matched     = df[df["has_gt"] & df["has_pred"]]
    iou_mean    = round(float(matched["iou"].mean()), 4) if len(matched) else float("nan")
    dice_mean   = round(float((2*matched["iou"]/(1+matched["iou"])).mean()), 4) if len(matched) else float("nan")
    tp50        = int((df["has_gt"] & has_thresh & (df["iou"] >= 0.5)).sum())
    recall50    = tp50 / n_gt     if n_gt     else float("nan")
    precision50 = tp50 / n_thresh if n_thresh else float("nan")
    denom       = (precision50 + recall50) if not (np.isnan(precision50) or np.isnan(recall50)) else 0.
    f1_50       = 2 * precision50 * recall50 / denom if denom > 0 else float("nan")
    ap50        = ap_at_iou(df, "iou", 0.50)
    ap50_95     = float(np.nanmean([ap_at_iou(df, "iou", t) for t in np.arange(0.50, 1.00, 0.05)]))
    return {
        "n_slices": n_slices, "n_gt": n_gt, "n_pred": n_pred,
        "n_fp": n_fp, "n_fn": n_fn, "fp_rate": fp_rate, "fn_rate": fn_rate,
        "iou_mean": iou_mean, "dice_mean": dice_mean,
        "recall50":    round(recall50,    4) if not np.isnan(recall50)    else float("nan"),
        "precision50": round(precision50, 4) if not np.isnan(precision50) else float("nan"),
        "f1_50":       round(f1_50,       4) if not np.isnan(f1_50)       else float("nan"),
        "ap50": round(ap50, 4), "ap50_95": round(ap50_95, 4),
    }


def build_patients(full_df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient volume with aggregated FP/FN/IoU and a failure score."""
    rows = []
    for (dataset, subject, contrast, split, stem), g in full_df.groupby(
        ["dataset", "subject", "contrast", "split", "stem"], sort=False
    ):
        n_slices = len(g)
        n_gt     = int(g["has_gt"].sum())
        n_pred   = int(g["has_pred"].sum())
        n_fp     = int(g["is_fp"].sum())
        n_fn     = int(g["is_fn"].sum())
        matched      = g[g["has_gt"] & g["has_pred"]]
        iou_mean     = float(matched["iou"].mean()) if len(matched) else float("nan")
        # iou_gt_mean: IoU averaged over ALL GT slices (FN counted as 0)
        gt_slices    = g[g["has_gt"]]
        iou_gt_mean  = float(gt_slices["iou"].mean()) if len(gt_slices) else float("nan")
        fp_rate  = n_fp / n_slices if n_slices else float("nan")
        fn_rate  = n_fn / n_gt     if n_gt     else float("nan")
        rows.append({
            "dataset": dataset, "subject": subject, "contrast": contrast,
            "split": split, "stem": stem,
            "n_slices": n_slices, "n_gt_slices": n_gt, "n_pred_slices": n_pred,
            "n_fp": n_fp, "n_fn": n_fn,
            "fp_rate":     round(fp_rate,     4) if not np.isnan(fp_rate)     else float("nan"),
            "fn_rate":     round(fn_rate,     4) if not np.isnan(fn_rate)     else float("nan"),
            "iou_mean":    round(iou_mean,    4) if not np.isnan(iou_mean)    else float("nan"),
            "iou_gt_mean": round(iou_gt_mean, 4) if not np.isnan(iou_gt_mean) else float("nan"),
        })
    return pd.DataFrame(rows)


def build_report(df: pd.DataFrame, conf_thresh: float) -> pd.DataFrame:
    rows = []

    def add(g, split="ALL", dataset="ALL", contrast="ALL"):
        rows.append({"split": split, "dataset": dataset, "contrast": contrast,
                     **summarise_group(g, conf_thresh)})

    add(df)
    for split, g in df.groupby("split"):
        add(g, split=split)
    for dataset, g in df.groupby("dataset"):
        add(g, dataset=dataset)
        for split, gg in g.groupby("split"):
            add(gg, split=split, dataset=dataset)
        for contrast, gg in g.groupby("contrast"):
            add(gg, dataset=dataset, contrast=contrast)
            for split, ggg in gg.groupby("split"):
                add(ggg, split=split, dataset=dataset, contrast=contrast)

    return pd.DataFrame(rows)


# ── HTML report ───────────────────────────────────────────────────────────────

def iou_class(v):
    if pd.isna(v): return "na"
    if v >= 0.8:   return "good"
    if v >= 0.6:   return "ok"
    return "bad"


def fmt_cell(iou, n_acq, n_slices):
    if pd.isna(iou) or n_slices == 0:
        return '<span class="na">—</span>'
    cls = iou_class(iou)
    return (f'<span class="{cls}">{iou:.3f}</span>'
            f'<br><span class="meta">{n_acq}acq / {n_slices}sl</span>')


def count_acq_per_split(inference_dir: Path, splits_map: dict) -> dict:
    """Returns {(split, dataset): n_acq} and {(split, dataset, contrast): n_acq}."""
    by_dataset  = {}
    by_contrast = {}
    for slices_csv in inference_dir.rglob("metrics/slices.csv"):
        patient_dir = slices_csv.parent.parent
        dataset     = patient_dir.parent.name
        stem        = patient_dir.name
        m           = re.match(r"(sub-[^_]+)_?(.*)", stem)
        subject     = m.group(1) if m else stem
        contrast    = m.group(2) if m and m.group(2) else "default"
        split       = splits_map.get((dataset, subject), "unknown")
        by_dataset[(split, dataset)]             = by_dataset.get((split, dataset), 0) + 1
        by_contrast[(split, dataset, contrast)]  = by_contrast.get((split, dataset, contrast), 0) + 1
    return by_dataset, by_contrast


def make_table_by_dataset(report: pd.DataFrame, acq_counts: dict, splits: list) -> str:
    sub      = report[(report["contrast"] == "ALL") & (report["dataset"] != "ALL") & (report["split"].isin(splits))]
    datasets = sorted(sub["dataset"].unique())
    headers  = '<th class="left">Dataset</th>' + "".join(
        f'<th>{s}<br><span style="font-size:10px;font-weight:normal">(acq / slices)</span></th>'
        for s in splits)
    html = f"<table><thead><tr>{headers}</tr></thead><tbody>\n"
    for dataset in datasets:
        row = f'<td class="left">{dataset}</td>'
        for split in splits:
            r = sub[(sub["dataset"] == dataset) & (sub["split"] == split)]
            if r.empty:
                row += '<td><span class="na">—</span></td>'
            else:
                row += f"<td>{fmt_cell(r.iloc[0]['iou_mean'], acq_counts.get((split, dataset), 0), int(r.iloc[0]['n_slices']))}</td>"
        html += f"<tr>{row}</tr>\n"
    html += "</tbody></table>"
    return html


def make_table_by_contrast(report: pd.DataFrame, acq_by_contrast: dict, dataset: str, splits: list) -> str:
    sub       = report[(report["dataset"] == dataset) & (report["contrast"] != "ALL") & (report["split"].isin(splits))]
    contrasts = sorted(sub["contrast"].unique())
    if not contrasts:
        return "<p><em>Aucune donnée par contraste.</em></p>"
    headers = '<th class="left">Contrast</th>' + "".join(
        f'<th>{s}<br><span style="font-size:10px;font-weight:normal">(acq / slices)</span></th>'
        for s in splits)
    html = f"<table><thead><tr>{headers}</tr></thead><tbody>\n"
    for contrast in contrasts:
        row = f'<td class="left">{contrast}</td>'
        for split in splits:
            r = sub[(sub["contrast"] == contrast) & (sub["split"] == split)]
            if r.empty:
                row += '<td><span class="na">—</span></td>'
            else:
                n_acq = acq_by_contrast.get((split, dataset, contrast), 0)
                row += f"<td>{fmt_cell(r.iloc[0]['iou_mean'], n_acq, int(r.iloc[0]['n_slices']))}</td>"
        html += f"<tr>{row}</tr>\n"
    html += "</tbody></table>"
    return html


def build_html(report: pd.DataFrame, acq_by_dataset: dict, acq_by_contrast: dict,
               run_id: str, metrics_csv: Path, splits: list) -> str:
    body  = f"<h1>Metrics report — {run_id}</h1>\n"
    body += f"<p><em>Source: {metrics_csv}</em></p>\n"
    body += "<h2>IoU mean par dataset</h2>\n"
    body += make_table_by_dataset(report, acq_by_dataset, splits)
    body += "<h2>IoU mean par dataset × contraste</h2>\n"
    datasets = sorted(report[(report["dataset"] != "ALL") & (report["contrast"] != "ALL")]["dataset"].unique())
    for dataset in datasets:
        body += f"<h3>{dataset}</h3>\n"
        body += make_table_by_contrast(report, acq_by_contrast, dataset, splits)
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Metrics {run_id}</title>{CSS}</head><body>{body}</body></html>"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute 2D bbox metrics — per-patient slices.csv + global metrics.csv + report.html",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference",  required=True,
                        help="Path to inference run directory (predictions/<run-id>/)")
    parser.add_argument("--processed",  default="processed", help="processed/ root dir (GT source)")
    parser.add_argument("--splits-dir", default="data/datasplits")
    parser.add_argument("--conf",       type=float, default=CONF_THRESH,
                        help="Confidence threshold for precision/recall/f1 fixed-threshold metrics")
    parser.add_argument("--split",      default=None, choices=["train", "val", "test"],
                        help="Restrict to a single split (default: all)")
    args = parser.parse_args()

    pred_root  = Path(args.inference)
    run_id     = pred_root.name
    splits_map = load_splits(Path(args.splits_dir))
    all_records = []

    for dataset_dir in sorted(Path(args.processed).iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for patient_dir in sorted(dataset_dir.iterdir()):
            if not (patient_dir / "txt").is_dir():
                continue
            stem     = patient_dir.name
            m        = re.match(r"(sub-[^_]+)_?(.*)", stem)
            subject  = m.group(1)
            contrast = m.group(2) or "default"
            split    = splits_map.get((dataset, subject), "unknown")

            if args.split and split != args.split:
                continue

            pred_txt_dir = pred_root / dataset / stem / "txt"
            df = patient_slices(patient_dir / "txt", pred_txt_dir)
            if df.empty:
                continue

            metrics_dir = pred_root / dataset / stem / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(metrics_dir / "slices.csv", index=False)

            df["dataset"]  = dataset
            df["subject"]  = subject
            df["contrast"] = contrast
            df["split"]    = split
            df["stem"]     = stem
            all_records.append(df)

    if not all_records:
        print("No data found.")
        return

    full_df    = pd.concat(all_records, ignore_index=True)
    n_patients = full_df.groupby(["dataset", "subject"]).ngroups
    print(f"Processed {n_patients} patients — {len(full_df)} slices total"
          + (f" [{args.split}]" if args.split else ""))

    patients_df  = build_patients(full_df)
    patients_csv = pred_root / "patients.csv"
    patients_df.to_csv(patients_csv, index=False)
    print(f"Patients → {patients_csv}")

    report    = build_report(full_df, args.conf)
    out_csv   = pred_root / "metrics.csv"
    report.to_csv(out_csv, index=False)
    print(f"CSV    → {out_csv}")
    print(report[report["dataset"] == "ALL"].to_string(index=False))

    acq_by_dataset, acq_by_contrast = count_acq_per_split(pred_root, splits_map)
    html = build_html(report, acq_by_dataset, acq_by_contrast, run_id, out_csv, SPLITS)
    out_html   = pred_root / "report.html"
    out_html.write_text(html)
    print(f"HTML   → {out_html}")


if __name__ == "__main__":
    main()
