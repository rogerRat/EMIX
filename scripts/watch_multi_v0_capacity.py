"""Expand the parallel evaluation pool after the existing owned task exits."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-pid", type=int, required=True)
    parser.add_argument("--resume-file", type=Path, required=True)
    parser.add_argument("--max-jobs-file", type=Path, required=True)
    parser.add_argument("--new-max-jobs", type=int, required=True)
    args = parser.parse_args()
    while os.path.exists(f"/proc/{args.external_pid}"):
        time.sleep(20)
    if not args.resume_file.is_file():
        print("External evaluation exited without a resume file; capacity unchanged", flush=True)
        return
    state = json.loads(args.resume_file.read_text())
    metrics = Path(state["output"]) / "metrics.json"
    if not metrics.is_file() or json.loads(metrics.read_text()).get("episodes") != 100:
        print("External evaluation did not finish; capacity unchanged", flush=True)
        return
    args.max_jobs_file.write_text(f"{args.new_max_jobs}\n")
    print(f"External evaluation complete; parallel pool limit now {args.new_max_jobs}", flush=True)


if __name__ == "__main__":
    main()
