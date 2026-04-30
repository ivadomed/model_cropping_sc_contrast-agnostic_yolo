"""
Clone one dataset and set it up for git-annex, using metadata from data/datasets.yaml.

All datasets: git clone + git annex dead here (client-only, no local annex storage).
If a commit is pinned in the YAML, the repo is checked out at that commit.
The cloned commit is appended to <ofolder>/git_branch_commit.log.

Usage:
    python 01_clone_dataset.py --ofolder <dir> --dataset <name>
"""

import argparse
import subprocess
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_YAML = PROJECT_ROOT / "data" / "datasets.yaml"


def load_dataset_config(name: str) -> dict:
    with open(DATASETS_YAML) as f:
        registry = yaml.safe_load(f)
    for entry in registry["datasets"]:
        if entry["name"] == name:
            return entry
    raise ValueError(f"Dataset '{name}' not found in {DATASETS_YAML}")


def run(cmd: list[str], **kwargs):
    subprocess.run(cmd, check=True, **kwargs)


def clone_dataset(name: str, ofolder: Path) -> None:
    cfg = load_dataset_config(name)
    url = cfg["url_ssh"]
    commit = cfg.get("commit")
    dest = ofolder / name

    run(["git", "clone", url, str(dest)])

    run(["git", "annex", "dead", "here"], cwd=dest)

    if commit:
        run(["git", "checkout", commit], cwd=dest)

    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=dest, text=True
    ).strip()
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=dest, text=True
    ).strip()

    log_path = ofolder / "git_branch_commit.log"
    with open(log_path, "a") as f:
        f.write(f"{name}: git-{branch}-{actual_commit}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ofolder", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=str)
    args = parser.parse_args()

    clone_dataset(args.dataset, args.ofolder.resolve())
