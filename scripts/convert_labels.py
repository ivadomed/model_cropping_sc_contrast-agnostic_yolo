#!/usr/bin/env python3
"""
Convert single-class SC labels to 2-class schema in-place.

Reads processed/<variant>/<dataset>/<stem>/txt/slice_NNN.txt and rewrites:
  class 0 (sc_mid) : SC present on this slice AND on z-1 AND z+1
  class 1 (sc_tip) : SC present on this slice but absent on z-1 OR z+1

Empty slices (no SC) are unchanged.
Idempotent: running twice gives the same result.

Prints per-dataset and global stats (n_mid, n_tip, tip%).

Usage:
    python scripts/convert_labels.py --processed processed/10mm_SI_1mm_axial_3ch
    python scripts/convert_labels.py \\
        --processed processed/10mm_SI_1mm_axial processed/10mm_SI_1mm_axial_3ch
"""

import argparse
from pathlib import Path

from tqdm import tqdm


def classify_slice(z: int, gt_set: set) -> int:
    """Return 0 (sc_mid) or 1 (sc_tip) for a slice known to have SC."""
    return 1 if (z - 1) not in gt_set or (z + 1) not in gt_set else 0


def convert_stem(stem_dir: Path) -> tuple[int, int]:
    """Relabel all SC slices in one processed stem. Returns (n_mid, n_tip)."""
    txt_dir = stem_dir / "txt"
    txts    = sorted(txt_dir.glob("slice_*.txt"))

    gt_set  = {int(t.stem.split("_")[1]) for t in txts if t.read_text().strip()}
    n_mid = n_tip = 0

    for txt in txts:
        content = txt.read_text().strip()
        if not content:
            continue
        z        = int(txt.stem.split("_")[1])
        class_id = classify_slice(z, gt_set)
        parts    = content.split()
        parts[0] = str(class_id)
        txt.write_text(" ".join(parts) + "\n")
        if class_id == 0:
            n_mid += 1
        else:
            n_tip += 1

    return n_mid, n_tip


def main():
    parser = argparse.ArgumentParser(
        description="Convert single-class SC labels to 2-class (sc_mid / sc_tip) in-place",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--processed", nargs="+", required=True,
                        help="One or more processed/<variant> directories to convert")
    args = parser.parse_args()

    for processed_root in args.processed:
        processed_dir = Path(processed_root)
        print(f"\n=== {processed_dir} ===")

        stems = [
            stem_dir
            for dataset_dir in sorted(processed_dir.iterdir()) if dataset_dir.is_dir()
            for stem_dir    in sorted(dataset_dir.iterdir())   if (stem_dir / "txt").is_dir()
        ]

        total_mid = total_tip = 0
        dataset_stats: dict[str, tuple[int, int]] = {}

        for stem_dir in tqdm(stems, desc="Stems", unit="stem"):
            dataset = stem_dir.parent.name
            n_mid, n_tip = convert_stem(stem_dir)
            total_mid += n_mid
            total_tip += n_tip
            prev = dataset_stats.get(dataset, (0, 0))
            dataset_stats[dataset] = (prev[0] + n_mid, prev[1] + n_tip)

        print(f"\nPer-dataset breakdown:")
        for dataset, (n_mid, n_tip) in sorted(dataset_stats.items()):
            total = n_mid + n_tip
            tip_pct = 100 * n_tip / total if total else 0
            print(f"  {dataset:40s}  mid={n_mid:6d}  tip={n_tip:4d}  tip%={tip_pct:.1f}")

        grand = total_mid + total_tip
        print(f"\nTotal: mid={total_mid}  tip={total_tip}  "
              f"tip%={100*total_tip/grand:.1f}  ratio mid:tip = {total_mid//max(total_tip,1):.0f}:1")


if __name__ == "__main__":
    main()
