#!/usr/bin/env python3
"""
Generate train/val/test datasplit YAMLs from actual subject folders in data/raw/.

Reads sub-* directories from data/raw/<dataset>/ to get exact subject IDs,
shuffles them with a fixed seed, and writes one YAML per dataset.

Each output file starts with a `meta:` block containing the dataset info from
configs/datasets.yaml (name, host, url_https, url_ssh, commit) and the seed used.
Parsers must skip non-list values (i.e. the `meta` key) when iterating splits.

Output format:
  meta:
    name:      <dataset>
    host:      <host>
    url_https: <url>
    url_ssh:   <url>
    commit:    <sha>
    seed:      <int>
  train: [sub-xxx, ...]
  val:   [sub-yyy, ...]
  test:  [sub-zzz, ...]

Usage:
    python scripts/make_splits.py
    python scripts/make_splits.py --raw data/raw
    python scripts/make_splits.py --train 0.7 --val 0.15 --test 0.15
    python scripts/make_splits.py --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml


def load_datasets_registry(datasets_yaml: Path) -> dict:
    """Return {name: entry_dict} from configs/datasets.yaml."""
    entries = yaml.safe_load(datasets_yaml.read_text())["datasets"]
    return {e["name"]: e for e in entries}


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


def run(raw: str | Path, out: str | Path, seed: int,
        train: float = 0.5, val: float = 0.2, test: float = 0.3,
        datasets: list | None = None,
        datasets_yaml: str | Path = "configs/datasets.yaml") -> None:
    """Generate datasplit YAMLs from sub-* folders in raw_dir."""
    raw_dir = Path(raw)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = load_datasets_registry(Path(datasets_yaml))

    for dataset_dir in sorted(raw_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue
        if datasets and dataset_dir.name not in datasets:
            continue
        subjects = sorted(d.name for d in dataset_dir.iterdir()
                          if d.is_dir() and d.name.startswith("sub-"))
        if not subjects:
            print(f"  skip {dataset_dir.name} (no sub-* folders)")
            continue

        info   = registry.get(dataset_dir.name, {})
        splits = split_subjects(subjects, train, val, seed)
        doc    = {
            "meta": {
                "name":      dataset_dir.name,
                "host":      info.get("host"),
                "url_https": info.get("url_https"),
                "url_ssh":   info.get("url_ssh"),
                "commit":    info.get("commit"),
                "seed":      seed,
            },
            **splits,
        }
        out_path = out_dir / f"datasplit_{dataset_dir.name}_seed{seed}.yaml"
        out_path.write_text(yaml.dump(doc, default_flow_style=False, sort_keys=False))
        print(f"{dataset_dir.name}: {len(subjects)} subjects → "
              f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
              f"  → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate datasplit YAMLs from actual sub-* folders in data/raw/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw",           default="data/raw",       help="BIDS raw root directory")
    parser.add_argument("--out",           default=None,             help="Output directory (default: data/datasplits_seed<seed>)")
    parser.add_argument("--datasets-yaml", default="configs/datasets.yaml", help="Registry with dataset metadata")
    parser.add_argument("--train", type=float, default=0.5,  help="Fraction for train split")
    parser.add_argument("--val",   type=float, default=0.2,  help="Fraction for val split")
    parser.add_argument("--test",  type=float, default=0.3,  help="Fraction for test split (remainder)")
    parser.add_argument("--seed",     type=int,  default=50,   help="Random seed")
    parser.add_argument("--datasets", nargs="+", default=None, help="Restrict to these dataset names (default: all)")
    args = parser.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, \
        "--train + --val + --test must sum to 1.0"

    out_dir = Path(args.out) if args.out else Path(f"data/datasplits_seed{args.seed}")
    run(raw=args.raw, out=out_dir, seed=args.seed,
        train=args.train, val=args.val, test=args.test,
        datasets=args.datasets, datasets_yaml=args.datasets_yaml)


if __name__ == "__main__":
    main()
