#!/usr/bin/env python3
"""Upload an archive folder to ModelScope as a dataset repo, then trim large
files locally (keeping only configs + eval JSON metrics).

Usage:
    python scripts/archive_upload_cleanup.py <archive_folder> --repo-name EmixArchive --repo-type dataset

Steps:
  1. Read token from ms_token.txt (repo root) or MS_TOKEN/MODELSCOPE_API_TOKEN env.
  2. Create repo (if missing) and upload top-level entries via hub/ms.py.
  3. On success, delete large artifacts from the archive:
       - data/lerobot/..., data/robotwin/..., data/emix_index/... (large binaries)
       - models/train/... (checkpoints, ~53G)
       - evaluation/eval/.../videos/  (keep metrics.json, delete videos dirs)
     Keeping: collection_config/, training_configs/, evaluation_configs/,
              and all *.json / *.yml / *.yaml files (including metrics.json).

IMPORTANT: This script deliberately uses the `emix` conda environment
(/home/rogerspyke/miniforge3/envs/emix) for all subprocess calls. It never
relies on global/system packages. If EMIX_CONDA_BIN is not found, it refuses
to run rather than falling back to whatever python/ms is on PATH.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

EMIX_CONDA_BIN = Path("/home/rogerspyke/miniforge3/envs/emix/bin")


def env_path() -> str:
    """Return PATH with the emix conda bin first, so subprocess 'ms' resolves
    to the emix environment and never to a global install."""
    if not EMIX_CONDA_BIN.exists():
        raise SystemExit(f"emix conda env not found: {EMIX_CONDA_BIN}")
    return str(EMIX_CONDA_BIN) + os.pathsep + os.environ.get("PATH", "")


def check_emix_runtime() -> str:
    """Verify the emix env has a python and the ms CLI; return the ms path."""
    if not EMIX_CONDA_BIN.exists():
        raise SystemExit(f"emix conda env not found: {EMIX_CONDA_BIN}")
    py = EMIX_CONDA_BIN / "python"
    ms = EMIX_CONDA_BIN / "ms"
    if not py.exists():
        raise SystemExit(f"emix env missing python: {py}")
    if not ms.exists():
        raise SystemExit(f"emix env missing ms CLI: {ms}")
    return str(ms)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("archive_upload")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def read_token(root: Path, cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    for env in ("MODELSCOPE_API_TOKEN", "MS_TOKEN"):
        if os.environ.get(env):
            return os.environ[env]
    tok_file = root / "ms_token.txt"
    if tok_file.exists():
        return tok_file.read_text(encoding="utf-8").strip()
    return None


def run(logger: logging.Logger, argv: list[str]) -> None:
    logger.info("RUN: %s", " ".join(argv))
    r = subprocess.run(argv, env={**os.environ, "PATH": env_path()})
    if r.returncode != 0:
        raise SystemExit(f"Command failed (exit={r.returncode}): {' '.join(argv)}")


def upload(logger: logging.Logger, folder: Path, token: str, repo_name: str) -> None:
    """Upload the archive folder directly to the (existing) ModelScope repo.

    The repo already exists, so we skip `ms create` entirely and only run
    `ms upload`. The token is passed via MODELSCOPE_API_TOKEN env var (the
    `ms upload` subcommand has no --token flag; the CLI reads it from env).
    """
    ms_bin = check_emix_runtime()
    argv = [
        ms_bin, "upload",
        f"rogerspyke/{repo_name}",
        str(folder.resolve()),
        "--repo-type", "dataset",
        "--disable-tqdm",
    ]
    env = {**os.environ, "PATH": env_path(), "MODELSCOPE_API_TOKEN": token}
    logger.info("RUN: %s", " ".join(argv))
    r = subprocess.run(argv, env=env)
    if r.returncode != 0:
        raise SystemExit(f"Upload failed (exit={r.returncode}): {' '.join(argv)}")


def trim_archive(logger: logging.Logger, folder: Path) -> None:
    """Delete large binary artifacts; keep configs + JSON results."""
    removed = []

    # 1. Raw datasets (large binaries / hdf5 / videos)
    data_root = folder / "data"
    for sub in ("lerobot", "robotwin", "emix_index"):
        p = data_root / sub
        if p.exists():
            shutil.rmtree(p)
            removed.append(str(p))

    # 2. Training outputs (checkpoints ~53G)
    models_root = folder / "models"
    if models_root.exists():
        shutil.rmtree(models_root)
        removed.append(str(models_root))

    # 3. Eval videos: keep metrics.json, drop videos dirs
    eval_root = folder / "evaluation" / "eval"
    if eval_root.exists():
        for run_dir in eval_root.iterdir():
            if not run_dir.is_dir():
                continue
            for result_dir in run_dir.iterdir():
                if not result_dir.is_dir():
                    continue
                vids = result_dir / "videos"
                if vids.exists():
                    shutil.rmtree(vids)
                    removed.append(str(vids))

    # 4. Prune empty directories
    pruned = 0
    for p in sorted(folder.rglob("*"), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()
            pruned += 1

    logger.info("Removed %d large artifacts; pruned %d empty dirs.", len(removed), pruned)
    for r in removed:
        logger.info("  removed: %s", r)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=str, help="Archive folder to upload + trim.")
    parser.add_argument("--repo-name", default="EmixArchive", help="ModelScope dataset repo name.")
    parser.add_argument("--token", default=None, help="ModelScope token (else ms_token.txt / env).")
    parser.add_argument("--skip-trim", action="store_true", help="Only upload, do not trim.")
    args = parser.parse_args()

    logger = setup_logging()
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        logger.error("Not a directory: %s", folder)
        return 1

    # This script lives in scripts/temp/, while ms_token.txt is at the
    # repository root.
    root = Path(__file__).resolve().parent.parent.parent
    token = read_token(root, args.token)
    if not token:
        logger.error("No ModelScope token found (ms_token.txt or MODELSCOPE_API_TOKEN/MS_TOKEN).")
        return 1

    upload(logger, folder, token, args.repo_name)

    if not args.skip_trim:
        trim_archive(logger, folder)

    logger.info("Done. Uploaded %s, trimmed=%s", folder, not args.skip_trim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
