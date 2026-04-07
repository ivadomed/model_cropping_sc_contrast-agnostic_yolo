#!/usr/bin/env python3
"""
Generate an HTML report from a metrics CSV and per-patient slices.csv files.

Produces two tables per split combination (test / val / train / unknown):
  1. IoU mean per dataset (contrast=ALL)
  2. IoU mean per dataset × contrast

Each cell shows: iou_mean (n_patients / n_slices)
n_patients is counted from predictions/<run_id>/<dataset>/<patient>/metrics/slices.csv

Usage:
    python scripts/visualize_metrics.py \
        --metrics metrics_yolo26_10mm.csv \
        --inference predictions/yolo26_10mm \
        --out report_yolo26_10mm.html
"""

import argparse
import re
from pathlib import Path

import pandas as pd

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


def iou_class(v):
    if pd.isna(v):
        return "na"
    if v >= 0.8:
        return "good"
    if v >= 0.6:
        return "ok"
    return "bad"


def fmt_cell(iou, n_patients, n_slices):
    if pd.isna(iou) or n_slices == 0:
        return '<span class="na">—</span>'
    cls = iou_class(iou)
    return (f'<span class="{cls}">{iou:.3f}</span>'
            f'<br><span class="meta">{n_patients}p / {n_slices}sl</span>')


def count_patients(inference_dir: Path) -> dict:
    """Returns {(dataset, split): n_patients} by scanning slices.csv files."""
    counts = {}
    if not inference_dir.is_dir():
        return counts
    for slices_csv in inference_dir.rglob("metrics/slices.csv"):
        # path: <inference_dir>/<dataset>/<patient>/metrics/slices.csv
        patient_dir = slices_csv.parent.parent
        dataset     = patient_dir.parent.name
        # read split from slices.csv — all rows have same split
        df = pd.read_csv(slices_csv, usecols=["has_gt"], nrows=1)
        # split is not in slices.csv — we infer it from the global metrics later
        # just count presence here; split attribution done via metrics CSV
        key = (dataset, patient_dir.name)
        counts[key] = counts.get(key, 0) + 1  # always 1 patient per dir
    return counts


def build_patient_counts_from_metrics(metrics_df: pd.DataFrame, inference_dir: Path) -> dict:
    """
    Returns {(dataset, subject): split} by reading the splits from the global metrics CSV
    via the slices.csv directory structure.
    We cross-reference: metrics CSV tells us n_slices per (split,dataset),
    slices.csv files tell us which patients exist.
    We count patients per (split, dataset) by using the splits_map embedded in slices.csv
    directory names and the splits known from metrics.
    """
    # Build splits map from metrics: {dataset: {split: n_slices}}
    # Then count patient dirs per dataset and assign split from metrics' split column
    # Since slices.csv don't store split, we scan all patient dirs and count them
    # per dataset, then match with splits from the global metrics row ordering.
    #
    # Simpler approach: for each (split, dataset) row in metrics,
    # count patient dirs that have slices.csv AND whose slices match that split's n_slices.
    # This is fragile. Better: re-read splits from datasplits YAMLs.

    # Read splits from datasplits YAMLs co-located with inference_dir
    project_root = inference_dir.parent.parent
    splits_dir   = project_root / "data" / "datasplits"
    subject_split = {}  # {(dataset, subject): split}
    if splits_dir.is_dir():
        for f in sorted(splits_dir.glob("datasplit_*.yaml")):
            import yaml
            dataset = re.sub(r"_seed\d+$", "", f.stem[len("datasplit_"):])
            for split_name, subjects in yaml.safe_load(f.read_text()).items():
                for subj in (subjects or []):
                    subject_split[(dataset, subj)] = split_name

    # Count patients per (split, dataset) from slices.csv dirs
    patient_counts = {}  # {(split, dataset): n_patients}
    for slices_csv in inference_dir.rglob("metrics/slices.csv"):
        patient_dir = slices_csv.parent.parent
        dataset     = patient_dir.parent.name
        stem        = patient_dir.name
        m           = re.match(r"(sub-[^_]+)", stem)
        subject     = m.group(1) if m else stem
        split       = subject_split.get((dataset, subject), "unknown")
        key         = (split, dataset)
        patient_counts[key] = patient_counts.get(key, 0) + 1

    return patient_counts


def make_table_by_dataset(metrics_df: pd.DataFrame, patient_counts: dict, splits: list) -> str:
    """Table 1: one row per dataset, columns = splits."""
    sub = metrics_df[(metrics_df["contrast"] == "ALL") &
                     (metrics_df["dataset"] != "ALL") &
                     (metrics_df["split"].isin(splits))]

    datasets = sorted(sub["dataset"].unique())

    headers = '<th class="left">Dataset</th>' + "".join(
        f'<th>{s}<br><span style="font-size:10px;font-weight:normal">(acq / slices)</span></th>'
        for s in splits
    )
    html = f"<table><thead><tr>{headers}</tr></thead><tbody>\n"

    for dataset in datasets:
        row = f'<td class="left">{dataset}</td>'
        for split in splits:
            r = sub[(sub["dataset"] == dataset) & (sub["split"] == split)]
            if r.empty:
                row += '<td><span class="na">—</span></td>'
            else:
                iou      = r.iloc[0]["iou_mean"]
                n_slices = int(r.iloc[0]["n_slices"])
                n_pat    = patient_counts.get((split, dataset), 0)
                row += f"<td>{fmt_cell(iou, n_pat, n_slices)}</td>"
        html += f"<tr>{row}</tr>\n"

    html += "</tbody></table>"
    return html


def make_table_by_contrast(metrics_df: pd.DataFrame, patient_counts: dict,
                           dataset: str, splits: list) -> str:
    """Table 2: one row per contrast for a given dataset, columns = splits."""
    sub = metrics_df[(metrics_df["dataset"] == dataset) &
                     (metrics_df["contrast"] != "ALL") &
                     (metrics_df["split"].isin(splits))]

    contrasts = sorted(sub["contrast"].unique())
    if not contrasts:
        return "<p><em>Aucune donnée par contraste.</em></p>"

    headers = '<th class="left">Contrast</th>' + "".join(
        f'<th>{s}</th>' for s in splits
    )
    html = f"<table><thead><tr>{headers}</tr></thead><tbody>\n"

    for contrast in contrasts:
        row = f'<td class="left">{contrast}</td>'
        for split in splits:
            r = sub[(sub["contrast"] == contrast) & (sub["split"] == split)]
            if r.empty:
                row += '<td><span class="na">—</span></td>'
            else:
                iou      = r.iloc[0]["iou_mean"]
                n_slices = int(r.iloc[0]["n_slices"])
                # patient count at contrast level not tracked — use slice count only
                row += f"<td>{fmt_cell(iou, '?', n_slices)}</td>"
        html += f"<tr>{row}</tr>\n"

    html += "</tbody></table>"
    return html


def main():
    parser = argparse.ArgumentParser(
        description="HTML report from metrics CSV + per-patient slices.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metrics",   required=True, help="Path to metrics_<run_id>.csv")
    parser.add_argument("--inference", required=True, help="Path to inference run dir (predictions/<run_id>/)")
    parser.add_argument("--out",       default=None,  help="Output HTML (default: <inference_dir>/report.html)")
    parser.add_argument("--splits",    nargs="+", default=SPLITS,
                        choices=SPLITS, help="Splits to include (default: all)")
    args = parser.parse_args()

    metrics_df     = pd.read_csv(args.metrics)
    inference_dir  = Path(args.inference)
    out_path       = Path(args.out) if args.out else inference_dir / "report.html"
    splits         = args.splits

    patient_counts = build_patient_counts_from_metrics(metrics_df, inference_dir)

    run_id = inference_dir.name
    body   = f"<h1>Metrics report — {run_id}</h1>\n"
    body  += f"<p><em>Source: {args.metrics}</em></p>\n"

    # Table 1 — by dataset
    body += "<h2>IoU mean par dataset</h2>\n"
    body += make_table_by_dataset(metrics_df, patient_counts, splits)

    # Table 2 — by dataset × contrast
    body += "<h2>IoU mean par dataset × contraste</h2>\n"
    datasets = sorted(metrics_df[(metrics_df["dataset"] != "ALL") &
                                  (metrics_df["contrast"] != "ALL")]["dataset"].unique())
    for dataset in datasets:
        body += f"<h3>{dataset}</h3>\n"
        body += make_table_by_contrast(metrics_df, patient_counts, dataset, splits)

    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Metrics {run_id}</title>{CSS}</head><body>{body}</body></html>"
    out_path.write_text(html)
    print(f"Report → {out_path}")


if __name__ == "__main__":
    main()
