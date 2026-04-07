#!/usr/bin/env python3
"""
Audit a YOLO dataset directory and report exactly which patients and slices
were used per split and per dataset.

Reads the flat symlinks in datasets/{images,labels}/{train,val,test}/ as produced
by build_dataset.py — this is the ground truth of what the model actually saw.

Symlink naming convention: <dataset>_<subject>[_<contrast>]_slice_NNN.png
Subject always starts with "sub-", used as delimiter between dataset name and subject.

Usage:
    python scripts/audit_dataset.py --dataset-dir datasets_10mm_SI
    python scripts/audit_dataset.py --dataset-dir datasets_10mm_SI --csv audit.csv
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_symlink_name(name: str):
    """
    Parse <dataset>_<subject>[_<contrast>]_slice_NNN.png
    Returns (dataset, subject, contrast) or None if unparseable.
    """
    m = re.match(r"^(.+?)_(sub-[^_]+(?:_.*?)?)_slice_\d+\.png$", name)
    if not m:
        return None
    dataset = m.group(1)
    rest    = m.group(2)
    # subject is first token, contrast is everything after
    parts   = rest.split("_", 1)
    subject = parts[0]
    contrast = parts[1] if len(parts) > 1 else "default"
    return dataset, subject, contrast


def main():
    parser = argparse.ArgumentParser(
        description="Audit YOLO dataset — patients and slices per split/dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", required=True, help="Path to datasets/ directory")
    parser.add_argument("--csv", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    splits = [d.name for d in sorted((dataset_dir / "images").iterdir()) if d.is_dir()]

    # Count slices and collect patients per (split, dataset, subject)
    slice_counts  = defaultdict(int)   # (split, dataset) → n_slices
    patient_sets  = defaultdict(set)   # (split, dataset) → {subject}

    for split in splits:
        images_dir = dataset_dir / "images" / split
        for link in images_dir.iterdir():
            parsed = parse_symlink_name(link.name)
            if parsed is None:
                continue
            dataset, subject, _ = parsed
            slice_counts[(split, dataset)] += 1
            patient_sets[(split, dataset)].add(subject)

    # Build report
    all_datasets = sorted({d for _, d in slice_counts})
    rows = []
    for dataset in all_datasets:
        for split in splits:
            n_slices   = slice_counts.get((split, dataset), 0)
            n_patients = len(patient_sets.get((split, dataset), set()))
            rows.append({"dataset": dataset, "split": split,
                         "n_patients": n_patients, "n_slices": n_slices})
        # Total row per dataset
        total_patients = len(set.union(*[patient_sets.get((s, dataset), set()) for s in splits]))
        total_slices   = sum(slice_counts.get((s, dataset), 0) for s in splits)
        rows.append({"dataset": dataset, "split": "TOTAL",
                     "n_patients": total_patients, "n_slices": total_slices})

    # Grand total
    grand_patients = len(set.union(*patient_sets.values()) if patient_sets else [set()])
    grand_slices   = sum(slice_counts.values())
    rows.append({"dataset": "TOTAL", "split": "ALL",
                 "n_patients": grand_patients, "n_slices": grand_slices})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n→ {args.csv}")


if __name__ == "__main__":
    main()
