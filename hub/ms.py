#!/usr/bin/env python3
# Purpose:
#   Upload/download ModelScope Hub repos through the `ms` or `modelscope` CLI.
#
# Dependencies:
#   pip install -U modelscope-hub
#   # or, if you still use old ModelScope package:
#   pip install -U modelscope
#
# Usage:
#   export MODELSCOPE_API_TOKEN=<your_token>
#   # or export MS_TOKEN=<your_token>
#
#   Upload:
#     python ms.py upload /path/to/folder \
#       --repo-name my_model \
#       --namespace rogerspyke \
#       --repo-type model
#
#   Download:
#     python ms.py download rogerspyke/my_model \
#       --repo-type model \
#       --local-dir ./my_model

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, TypeVar


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

DEFAULT_NAMESPACE = "rogerspyke"

# New official env name in modelscope-hub is MODELSCOPE_API_TOKEN.
# Keep MS_TOKEN for compatibility with your old script.
ENV_TOKEN_NAMES = ("MODELSCOPE_API_TOKEN", "MS_TOKEN")


def _utc8_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=8)))


def _setup_logging(task: str = "ms") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = _utc8_now().strftime("%Y%m%d%H%M%S")
    log_path = LOG_DIR / f"{task}_{ts}.log"

    logger = logging.getLogger("ms_tool")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(name)s] [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.debug("Log file: %s", log_path)

    return logger


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token and cli_token.strip():
        return cli_token.strip()

    for name in ENV_TOKEN_NAMES:
        t = os.environ.get(name, "").strip()
        if t:
            return t

    return None


def sanitize_repo_name(name: str) -> str:
    s = name.strip().strip("/")
    if not s:
        return "upload-repo"

    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-._")

    return s or "upload-repo"


def make_repo_id(namespace: str, repo_name: str) -> str:
    ns = sanitize_repo_name(namespace)
    name = sanitize_repo_name(repo_name)
    return f"{ns}/{name}"


def find_modelscope_cli(logger: logging.Logger) -> Optional[str]:
    # Prefer the CLI shipped in the same env as the running interpreter, so
    # `python hub/ms.py` under e.g. the emix env doesn't pick up a stale
    # base-env CLI off PATH (which may lack newer flags like --exist-ok).
    interpreter_dir = Path(sys.executable).resolve().parent
    for cmd in ("ms", "modelscope"):
        p = interpreter_dir / cmd
        if p.is_file() and os.access(p, os.X_OK):
            logger.debug("Found CLI (interpreter env): %s", p)
            return str(p)

    for cmd in ("ms", "modelscope"):
        p = shutil.which(cmd)
        if p:
            logger.debug("Found CLI: %s", p)
            return p

    logger.error(
        "'ms' or 'modelscope' CLI not found. "
        "Install: pip install -U modelscope-hub"
    )
    return None


def _argv_for_log(argv: List[str]) -> str:
    out: List[str] = []
    skip_next = False

    for a in argv:
        if skip_next:
            out.append("<redacted>")
            skip_next = False
            continue

        if a in ("--token", "-t"):
            out.append(a)
            skip_next = True
        elif a.startswith("--token=") or a.startswith("-t="):
            out.append(a.split("=", 1)[0] + "=<redacted>")
        else:
            out.append(a)

    return " ".join(out)


_T = TypeVar("_T")


def _chunks(items: List[_T], size: int) -> Iterable[List[_T]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_cmd(
    logger: logging.Logger,
    argv: List[str],
    cwd: Optional[str] = None,
) -> None:
    logger.info("RUN: %s", _argv_for_log(argv))
    r = subprocess.run(argv, cwd=cwd)

    if r.returncode != 0:
        logger.error("Command failed, exit=%s", r.returncode)
        r.check_returncode()


def run_cmd_with_retry(
    logger: logging.Logger,
    argv: List[str],
    cwd: Optional[str] = None,
    *,
    max_retries: int = -1,
    retry_base_seconds: float = 5.0,
    retry_max_seconds: float = 300.0,
) -> None:
    failures = 0

    while True:
        try:
            run_cmd(logger, argv, cwd)
            return
        except subprocess.CalledProcessError:
            if max_retries >= 0 and failures >= max_retries:
                logger.error("Giving up after %s failed attempt(s)", failures + 1)
                raise

            delay = min(retry_base_seconds * (2**failures), retry_max_seconds)
            logger.warning(
                "Command failed, attempt=%s; retrying in %.1fs",
                failures + 1,
                delay,
            )
            time.sleep(delay)
            failures += 1


def add_token_args(argv: List[str], token: Optional[str]) -> List[str]:
    if token:
        return argv + ["--token", token]
    return argv


def ensure_repo_and_upload(
    *,
    cli: str,
    token: Optional[str],
    repo_id: str,
    folder: Path,
    repo_type: str,
    visibility: Optional[str],
    logger: logging.Logger,
    commit_message: Optional[str],
    batch_size: int,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
) -> None:
    create_argv = [
        cli,
        "create",
        repo_id,
        "--repo_type",
        repo_type,
        "--exist-ok",
    ]

    if visibility:
        create_argv.extend(["--visibility", visibility])

    create_argv = add_token_args(create_argv, token)

    run_cmd_with_retry(
        logger,
        create_argv,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
    )

    if batch_size <= 0:
        upload_argv = [
            cli,
            "upload",
            repo_id,
            str(folder.resolve()),
            "--repo-type",
            repo_type,
        ]

        if commit_message:
            upload_argv.extend(["--commit-message", commit_message])

        upload_argv = add_token_args(upload_argv, token)

        run_cmd_with_retry(
            logger,
            upload_argv,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
        )

        logger.info("Done: uploaded %s -> %s", folder, repo_id)
        return

    entries = sorted(folder.iterdir(), key=lambda p: p.name)

    if not entries:
        logger.warning("No top-level files or directories under %s; repo created only", folder)
        return

    total_batches = (len(entries) + batch_size - 1) // batch_size

    for batch_idx, batch in enumerate(_chunks(entries, batch_size), start=1):
        logger.info(
            "Upload batch %s/%s, item_count=%s",
            batch_idx,
            total_batches,
            len(batch),
        )

        for entry in batch:
            upload_argv = [
                cli,
                "upload",
                repo_id,
                str(entry.resolve()),
                entry.name,
                "--repo-type",
                repo_type,
            ]

            if commit_message:
                upload_argv.extend(["--commit-message", commit_message])

            upload_argv = add_token_args(upload_argv, token)

            run_cmd_with_retry(
                logger,
                upload_argv,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
            )

    logger.info("Done: uploaded %s -> %s", folder, repo_id)


def download_repo(
    *,
    cli: str,
    token: Optional[str],
    repo_id: str,
    repo_type: str,
    local_dir: Optional[Path],
    revision: Optional[str],
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    files: Optional[List[str]],
    max_workers: Optional[int],
    dry_run: bool,
    logger: logging.Logger,
    max_retries: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
) -> None:
    argv = [
        cli,
        "download",
        repo_id,
    ]

    if files:
        argv.extend(files)

    argv.extend(["--repo-type", repo_type])

    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
        argv.extend(["--local-dir", str(local_dir.resolve())])

    if revision:
        argv.extend(["--revision", revision])

    if include:
        for pat in include:
            argv.extend(["--include", pat])

    if exclude:
        for pat in exclude:
            argv.extend(["--exclude", pat])

    if max_workers is not None:
        argv.extend(["--max-workers", str(max_workers)])

    if dry_run:
        argv.append("--dry-run")

    argv = add_token_args(argv, token)

    run_cmd_with_retry(
        logger,
        argv,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
    )

    if local_dir:
        logger.info("Done: downloaded %s -> %s", repo_id, local_dir)
    else:
        logger.info("Done: downloaded %s to ModelScope cache", repo_id)


def add_common_retry_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--max-retries",
        type=int,
        default=-1,
        help="Max retries per CLI failure (-1=until success, 0=fail fast).",
    )
    p.add_argument(
        "--retry-base-seconds",
        type=float,
        default=5.0,
        help="Initial retry backoff seconds. Default: 5.",
    )
    p.add_argument(
        "--retry-max-seconds",
        type=float,
        default=300.0,
        help="Max retry backoff seconds. Default: 300.",
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload/download ModelScope Hub repos using ms/modelscope CLI.",
    )

    sub = p.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="Upload a local folder to ModelScope Hub.")
    up.add_argument(
        "folder",
        type=str,
        help="Local directory to upload. Repo name defaults to folder basename.",
    )
    up.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help=f"ModelScope namespace/user. Default: {DEFAULT_NAMESPACE}.",
    )
    up.add_argument(
        "--repo-name",
        default=None,
        help="Remote repo name. Default: sanitized basename of folder.",
    )
    up.add_argument(
        "--token",
        default=None,
        help=(
            "ModelScope API token. Else read from MODELSCOPE_API_TOKEN or MS_TOKEN; "
            "else use local ms login state."
        ),
    )
    up.add_argument(
        "--repo-type",
        choices=("model", "dataset"),
        default="model",
        help="Repository type. Default: model.",
    )
    up.add_argument(
        "--visibility",
        choices=("public", "private", "internal"),
        default=None,
        help="Optional repo visibility.",
    )
    up.add_argument(
        "--commit-message",
        default=None,
        help="Optional commit message.",
    )
    up.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help=(
            "Top-level files/dirs per upload pass. "
            "Use 0 for one upload of the whole folder. Default: 20."
        ),
    )
    add_common_retry_args(up)

    down = sub.add_parser("download", help="Download a ModelScope Hub repo.")
    down.add_argument(
        "repo_id",
        type=str,
        help="Remote repo id, e.g. rogerspyke/my_model.",
    )
    down.add_argument(
        "files",
        nargs="*",
        help="Optional specific file paths inside repo. Omit for full snapshot.",
    )
    down.add_argument(
        "--repo-type",
        choices=("model", "dataset"),
        default="model",
        help="Repository type. Default: model.",
    )
    down.add_argument(
        "--local-dir",
        default=None,
        help="Local output directory. If omitted, downloads to ModelScope cache.",
    )
    down.add_argument(
        "--revision",
        default=None,
        help="Branch, tag, or commit hash. Default is decided by ModelScope CLI.",
    )
    down.add_argument(
        "--include",
        nargs="+",
        default=None,
        help='Allow patterns, e.g. --include "*.safetensors" "*.json".',
    )
    down.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        help='Ignore patterns, e.g. --exclude "*.bin".',
    )
    down.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel download workers.",
    )
    down.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading, if supported by your CLI version.",
    )
    down.add_argument(
        "--token",
        default=None,
        help=(
            "ModelScope API token. Else read from MODELSCOPE_API_TOKEN or MS_TOKEN; "
            "else use local ms login state."
        ),
    )
    add_common_retry_args(down)

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logger = _setup_logging(f"ms_{args.command}")

    cli = find_modelscope_cli(logger)
    if not cli:
        return 127

    token = resolve_token(getattr(args, "token", None))

    try:
        if args.command == "upload":
            folder = Path(args.folder).expanduser().resolve()

            if not folder.is_dir():
                logger.error("Not a directory: %s", folder)
                return 1

            base = args.repo_name or folder.name
            repo_id = make_repo_id(args.namespace, base)

            logger.info(
                "Upload config: repo_id=%s folder=%s repo_type=%s",
                repo_id,
                folder,
                args.repo_type,
            )

            ensure_repo_and_upload(
                cli=cli,
                token=token,
                repo_id=repo_id,
                folder=folder,
                repo_type=args.repo_type,
                visibility=args.visibility,
                logger=logger,
                commit_message=args.commit_message,
                batch_size=args.batch_size,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                retry_max_seconds=args.retry_max_seconds,
            )
            return 0

        if args.command == "download":
            repo_id = args.repo_id.strip()

            if not repo_id or "/" not in repo_id:
                logger.error("Invalid repo_id: %s. Expected format: namespace/name", repo_id)
                return 1

            local_dir = Path(args.local_dir).expanduser().resolve() if args.local_dir else None

            logger.info(
                "Download config: repo_id=%s local_dir=%s repo_type=%s revision=%s",
                repo_id,
                local_dir,
                args.repo_type,
                args.revision,
            )

            download_repo(
                cli=cli,
                token=token,
                repo_id=repo_id,
                repo_type=args.repo_type,
                local_dir=local_dir,
                revision=args.revision,
                include=args.include,
                exclude=args.exclude,
                files=args.files,
                max_workers=args.max_workers,
                dry_run=args.dry_run,
                logger=logger,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                retry_max_seconds=args.retry_max_seconds,
            )
            return 0

        logger.error("Unknown command: %s", args.command)
        return 2

    except subprocess.CalledProcessError:
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())