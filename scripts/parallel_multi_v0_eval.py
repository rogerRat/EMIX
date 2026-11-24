"""Fill available GPU memory with independent multi-v0 task evaluations."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_multi_v0 import GENERATED, ROOT, completed_eval, load_spec

ARMS = ("emix-mix", "emix-patch")
ENV_PYTHON = "/root/autodl-tmp/conda-envs/emix-twin-reinstall/bin/python"
LOG_ROOT = ROOT / "outputs/multi-v0/parallel"


def checkpoint_for(arm: str, steps: int) -> Path:
    model = ROOT / "outputs/multi-v0/train" / arm / "checkpoints" / f"{steps:06d}" / "pretrained_model"
    if not (model / "model.safetensors").is_file():
        raise FileNotFoundError(f"Final checkpoint is missing: {model}")
    return model


def gpu_free_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(output.splitlines()[0].strip())


def desired_jobs(default: int) -> int:
    control = LOG_ROOT / "max_jobs.txt"
    try:
        return max(1, int(control.read_text().strip()))
    except (OSError, ValueError):
        return default


def run(args: argparse.Namespace) -> int:
    spec = load_spec()
    checkpoints = {arm: checkpoint_for(arm, spec["training"]["steps"]) for arm in ARMS}
    excluded = set(args.exclude)
    tasks = deque(
        (arm, task)
        for task in spec["sources"]["tasks"]
        for arm in ARMS
        if f"{arm}/{task}" not in excluded
        and not completed_eval(
            ROOT / "outputs/multi-v0/eval" / arm / task,
            checkpoints[arm],
            spec["evaluation"]["episodes"],
        )
    )
    print(f"Pending evaluations: {len(tasks)}; excluded: {sorted(excluded)}", flush=True)
    if args.dry_run:
        for arm, task in tasks:
            print(f"{arm}/{task}", flush=True)
        return 0

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    control = LOG_ROOT / "max_jobs.txt"
    if not control.exists():
        control.write_text(f"{args.max_jobs}\n")
    env = os.environ.copy()
    env["HF_HOME"] = "/root/autodl-tmp/hf-cache"
    env["HF_HUB_OFFLINE"] = "1"
    env["EMIX_POLICY_WORKER_ENV"] = "/root/autodl-tmp/conda-envs/emix-reinstall"

    active: dict[tuple[str, str], tuple[subprocess.Popen, object]] = {}
    attempts: dict[tuple[str, str], int] = {}
    failed: list[tuple[str, str]] = []
    while tasks or active:
        for key, (proc, handle) in list(active.items()):
            status = proc.poll()
            if status is None:
                continue
            handle.close()
            del active[key]
            arm, task = key
            output = ROOT / "outputs/multi-v0/eval" / arm / task
            if status == 0 and completed_eval(output, checkpoints[arm], spec["evaluation"]["episodes"]):
                print(f"COMPLETE {arm}/{task} pid={proc.pid}", flush=True)
            elif attempts[key] < 2:
                print(f"RETRY {arm}/{task} exit={status}", flush=True)
                tasks.appendleft(key)
            else:
                print(f"FAILED {arm}/{task} exit={status}; sequential runner will retry", flush=True)
                failed.append(key)

        if tasks and len(active) < desired_jobs(args.max_jobs):
            free = gpu_free_mib()
            if free >= args.min_free_mib:
                arm, task = tasks.popleft()
                key = (arm, task)
                attempts[key] = attempts.get(key, 0) + 1
                output = ROOT / "outputs/multi-v0/eval" / arm / task
                if completed_eval(output, checkpoints[arm], spec["evaluation"]["episodes"]):
                    print(f"ALREADY_COMPLETE {arm}/{task}", flush=True)
                    continue
                config = GENERATED / arm / f"eval_{task}.yaml"
                log_path = LOG_ROOT / f"{arm}_{task}.log"
                handle = log_path.open("a", encoding="utf-8")
                handle.write(f"\n=== parallel launch {time.strftime('%Y-%m-%d %H:%M:%S')} attempt={attempts[key]} ===\n")
                handle.flush()
                proc = subprocess.Popen(
                    [ENV_PYTHON, "-u", "evaluation/evaluate.py", "--config", str(config)],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=handle, stderr=subprocess.STDOUT,
                )
                active[key] = (proc, handle)
                (LOG_ROOT / f"{arm}_{task}.pid").write_text(f"{proc.pid}\n")
                print(f"START {arm}/{task} pid={proc.pid} free_before={free}MiB active={len(active)}", flush=True)
                time.sleep(args.launch_stagger)
                continue
        time.sleep(5)

    print(f"Parallel queue drained; failed={len(failed)}", flush=True)
    if args.runner_pid:
        os.kill(args.runner_pid, signal.SIGCONT)
        print(f"Resumed sequential runner pid={args.runner_pid}", flush=True)
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--min-free-mib", type=int, default=8000)
    parser.add_argument("--launch-stagger", type=int, default=30)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--runner-pid", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_jobs < 1 or args.min_free_mib < 0 or args.launch_stagger < 0:
        parser.error("invalid concurrency limits")
    if args.dry_run:
        raise SystemExit(run(args))
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with (LOG_ROOT / "scheduler.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("a multi-v0 parallel scheduler is already running")
        (LOG_ROOT / "scheduler.pid").write_text(f"{os.getpid()}\n")
        raise SystemExit(run(args))


if __name__ == "__main__":
    main()
