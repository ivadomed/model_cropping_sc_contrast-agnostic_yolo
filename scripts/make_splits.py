#!/usr/bin/env python3
"""
Generate train/val/test datasplit YAMLs from actual subject folders in data/raw/.

Reads sub-* directories from data/raw/<dataset>/ to get exact subject IDs,
shuffles them with a fixed seed, and writes one YAML per dataset.

Output format (same as existing contrast_agnostic splits):
  train: [sub-xxx, ...]
  val:   [sub-yyy, ...]
  test:  [sub-zzz, ...]

Usage:
    python scripts/make_splits.py
    python scripts/make_splits.py --raw data/raw --out data/datasplits/from_raw
    python scripts/make_splits.py --train 0.7 --val 0.15 --test 0.15
    python scripts/make_splits.py --seed 42 --out data/datasplits/from_raw_seed42
"""

import argparse
import random
from pathlib import Path

import yaml


def split_subjects(subjects: list, train: float, val: float, seed: int) -> dict:
    """Shuffle and split subjects into train/val/test. test = remainder."""
    rng = random.Random(seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * train)
    n_val   = round(n * val)
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate datasplit YAMLs from actual sub-* folders in data/raw/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw",   default="data/raw",                 help="BIDS raw root directory")
    parser.add_argument("--out",   default="data/datasplits/from_raw", help="Output directory for YAML files")
    parser.add_argument("--train", type=float, default=0.5,            help="Fraction for train split")
    parser.add_argument("--val",   type=float, default=0.2,            help="Fraction for val split")
    parser.add_argument("--test",  type=float, default=0.3,            help="Fraction for test split (remainder)")
    parser.add_argument("--seed",  type=int,   default=50,             help="Random seed")
    args = parser.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, \
        "--train + --val + --test must sum to 1.0"

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset_dir in sorted(raw_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue
        subjects = sorted(d.name for d in dataset_dir.iterdir()
                          if d.is_dir() and d.name.startswith("sub-"))
        if not subjects:
            print(f"  skip {dataset_dir.name} (no sub-* folders)")
            continue

        splits = split_subjects(subjects, args.train, args.val, args.seed)
        out_path = out_dir / f"datasplit_{dataset_dir.name}_seed{args.seed}.yaml"
        out_path.write_text(yaml.dump(splits, default_flow_style=False, sort_keys=True))
        print(f"{dataset_dir.name}: {len(subjects)} subjects → "
              f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
              f"  → {out_path}")


if __name__ == "__main__":
    main()
