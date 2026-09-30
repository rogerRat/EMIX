"""Prune obsolete V0 binary artifacts while retaining text and 10% of media.

Dry run by default. Only the explicitly listed historical roots are visited;
new V1 collections and outputs are outside this script's scope.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOTS = (
    ROOT / "archive",
    ROOT / "data" / "emix_index_archive_root",
    *sorted((ROOT / "outputs").glob("run_v0*")),
    ROOT / "outputs" / "gaussian_noise_smoke",
)
MODEL_SUFFIXES = {".safetensors", ".pth", ".pt", ".ckpt"}
BINARY_SUFFIXES = {".mp4", ".avi", ".mov", ".hdf5", ".h5", ".parquet",
                   ".arrow", ".npy", ".npz", ".bin", ".pkl"}


def deletion_plan():
    groups = defaultdict(list)
    checkpoints = []
    for root in OLD_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in MODEL_SUFFIXES or "checkpoints" in path.parts:
                checkpoints.append(path)
            elif suffix in BINARY_SUFFIXES and path.stat().st_size >= 1024 * 1024:
                # Sample independently within each directory/type so every
                # task or evaluation condition retains inspection examples.
                groups[(path.parent, suffix)].append(path)
    delete = list(checkpoints)
    keep = []
    for paths in groups.values():
        for i, path in enumerate(sorted(paths)):
            (keep if i % 10 == 0 else delete).append(path)
    return keep, delete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually unlink selected files")
    args = parser.parse_args()
    keep, delete = deletion_plan()
    bytes_deleted = sum(path.stat().st_size for path in delete)
    print(f"retain {len(keep)} sampled binaries; remove {len(delete)} old binaries "
          f"({bytes_deleted / 1e9:.2f} GB); text files are preserved")
    if args.apply:
        for path in delete:
            path.unlink()
        print("pruning complete")


if __name__ == "__main__":
    main()
