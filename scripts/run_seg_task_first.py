# Run the configured segment experiment arms task by task.

import argparse
import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("seg_mix", "seg_patch", "seg_mix_overlap033", "seg_patch_overlap033")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-env", default="emix")
    parser.add_argument("--evaluation-env", default="emix-twin")
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
                print(f"[seg-task-first] task={task} arm={arm}", flush=True)
                if not args.dry_run:
                    environment = os.environ.copy()
                    environment["EMIX_POLICY_WORKER_ENV"] = args.training_env
                    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    print("[seg-task-first] complete", flush=True)


if __name__ == "__main__":
    main()
