# Run the configured segment experiment arms task by task.

import argparse
import copy
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("seg_mix", "seg_patch", "seg_mix_overlap033", "seg_patch_overlap033")


def _stop_active(active):
    for process in active:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    for process in active:
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def _run_task(task, configs, directory, args):
    pending = []
    for arm, template in zip(ARMS, configs):
        config = copy.deepcopy(template)
        config["gpu_batches"] = {"0": [task]}
        config["environments"].update(
            training=args.training_env, preprocessing=args.training_env,
            robotwin=args.evaluation_env, evaluation=args.evaluation_env,
        )
        path = Path(directory) / f"run_sin_v0_seg_formal_{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = [sys.executable, str(ROOT / "scripts/run.py"), "--phase", "train-eval",
                   "--config", str(path), "--cuda", "0"]
        pending.append((arm, command))

    if args.dry_run:
        for arm, _ in pending:
            print(f"[seg-task-first] task={task} arm={arm}", flush=True)
        return

    environment = os.environ.copy()
    environment["EMIX_POLICY_WORKER_ENV"] = args.training_env
    active = {}
    failure = None
    try:
        while pending or active:
            while pending and len(active) < args.parallel_arms and failure is None:
                arm, command = pending.pop(0)
                print(f"[seg-task-first] START task={task} arm={arm}", flush=True)
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                           start_new_session=True)
                active[process] = arm
            for process, arm in tuple(active.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                del active[process]
                print(f"[seg-task-first] END task={task} arm={arm} exit={return_code}", flush=True)
                if return_code and failure is None:
                    failure = subprocess.CalledProcessError(return_code, process.args)
            if failure is not None and not active:
                break
            if active:
                time.sleep(2)
        if failure is not None:
            raise failure
    except BaseException:
        _stop_active(active)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-env", default="emix")
    parser.add_argument("--evaluation-env", default="emix-twin")
    parser.add_argument("--parallel-arms", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configs = []
    for arm in ARMS:
        path = ROOT / "configs/scripts" / f"run_sin_v0_seg_formal_{arm}.yaml"
        with path.open(encoding="utf-8") as handle:
            configs.append(yaml.safe_load(handle))
    tasks = configs[0]["gpu_batches"]["0"][:5]
    if len(tasks) != 5 or len(set(tasks)) != 5:
        raise ValueError("Expected five distinct configured tasks")
    if any(config["gpu_batches"]["0"][:5] != tasks for config in configs):
        raise ValueError("Arm task lists do not agree")
    with tempfile.TemporaryDirectory(prefix="emix-seg-task-first-") as directory:
        for task in tasks:
            _run_task(task, configs, directory, args)
    print("[seg-task-first] complete", flush=True)


if __name__ == "__main__":
    main()
