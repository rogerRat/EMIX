"""Continue the multi-v0 SmolVLA baseline to 400k and evaluate each 100k checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from run_multi_v0 import completed_eval, make_eval_config, make_training_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/scripts/multi_v0_searchparam.yaml"
GENERATED = ROOT / "configs/scripts/multi_v0_searchparam"
OUTPUT = ROOT / "outputs/multi-v0-searchparam"
LOG_DIR = ROOT / "logs/multi-v0-searchparam"
TRAIN_DIR = OUTPUT / "train/smolvla-baseline"
OLD_CKPT = ROOT / "outputs/multi-v0/train/smolvla-baseline/checkpoints/100000"
MILESTONES = (100000, 200000, 300000, 400000)


def load_spec():
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    assert spec["milestones"] == list(MILESTONES)
    assert len(spec["sources"]["tasks"]) == 10
    assert spec["training"]["pretrained_path"] == "lerobot/smolvla_base"
    assert spec["training"]["batch_size"] == 8
    assert spec["training"]["steps"] == 400000
    assert spec["training"]["save_freq"] == 100000
    return spec


def prepare(spec):
    GENERATED.mkdir(parents=True, exist_ok=True)
    train = make_training_config(spec, "smolvla-baseline")
    train["output_dir"] = str(TRAIN_DIR)
    train["training_debug"] = {
        "enabled": True,
        "every_n_steps": 10000,
        "max_samples": 32,
        "output_dir": str(TRAIN_DIR / "checkpoints/debug"),
    }
    (GENERATED / "train.yaml").write_text(yaml.safe_dump(train, sort_keys=False), encoding="utf-8")
    for step in MILESTONES:
        for task, prompt in spec["sources"]["tasks"].items():
            payload = make_eval_config(spec, "smolvla-baseline", task, prompt)
            payload["evaluation"]["checkpoint"] = str(TRAIN_DIR / "checkpoints" / f"{step:06d}" / "pretrained_model")
            payload["evaluation"]["output_dir"] = str(OUTPUT / "eval" / f"{step:06d}" / task)
            payload["protocol"]["milestone_step"] = step
            (GENERATED / f"eval_{step:06d}_{task}.yaml").write_text(
                yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print("Prepared 1 training config and 40 checkpoint-task evaluation configs", flush=True)


def mock(spec):
    from embodied_mixup.policies.train import _build_train_config

    path = GENERATED / "train.yaml"
    current = _build_train_config(yaml.safe_load(path.read_text()), path)
    old_path = ROOT / "configs/scripts/multi_v0/smolvla-baseline/train.yaml"
    old = _build_train_config(yaml.safe_load(old_path.read_text()), old_path)
    assert current.policy_cfg == old.policy_cfg
    assert current.batch_size == old.batch_size == 8
    assert current.num_workers == old.num_workers
    assert current.dataset_repo_ids == old.dataset_repo_ids == list(spec["sources"]["tasks"])
    assert current.steps == 400000 and current.save_freq == 100000
    assert current.training_mode == old.training_mode == "default"
    assert not current.emix
    assert (OLD_CKPT / "pretrained_model/model.safetensors").is_file()
    assert (OLD_CKPT / "training_state/optimizer_state.safetensors").is_file()
    assert json.loads((OLD_CKPT / "training_state/training_step.json").read_text())["step"] == 100000
    for step in MILESTONES:
        for task in spec["sources"]["tasks"]:
            payload = yaml.safe_load((GENERATED / f"eval_{step:06d}_{task}.yaml").read_text())
            assert payload["evaluation"]["task_prompt"] == spec["sources"]["tasks"][task]
            assert payload["evaluation"]["episodes"] == 100
            assert payload["evaluation"]["checkpoint"].endswith(f"{step:06d}/pretrained_model")
    print("Mock: original fine-tuning policy/batch/data match; 100k state complete; 40 eval configs valid", flush=True)


def seed_checkpoint():
    target = TRAIN_DIR / "checkpoints/100000"
    last = TRAIN_DIR / "checkpoints/last"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(OLD_CKPT, target)
    for relative in ("pretrained_model/model.safetensors", "training_state/training_step.json"):
        assert (target / relative).is_file(), relative
    if not last.exists() and not last.is_symlink():
        last.symlink_to("100000")
    current = last.resolve()
    assert current.parent.resolve() == target.parent.resolve()
    assert int(current.name) in MILESTONES
    for relative in ("pretrained_model/model.safetensors", "training_state/optimizer_state.safetensors",
                     "training_state/rng_state.safetensors", "training_state/training_step.json"):
        assert (current / relative).is_file(), relative
    return current


def evaluate_one(step, task, env):
    checkpoint = TRAIN_DIR / "checkpoints" / f"{step:06d}" / "pretrained_model"
    output = OUTPUT / "eval" / f"{step:06d}" / task
    if completed_eval(output, checkpoint, 100):
        return task, "already complete"
    cmd = [
        "/root/miniconda3/bin/conda", "run", "--no-capture-output",
        "-p", "/root/autodl-tmp/conda-envs/emix-twin-reinstall",
        "python", "-u", "evaluation/evaluate.py", "--config",
        str(GENERATED / f"eval_{step:06d}_{task}.yaml"),
    ]
    with (LOG_DIR / f"eval_{step:06d}_{task}.log").open("a") as log:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True,
                       stdout=log, stderr=subprocess.STDOUT)
    if not completed_eval(output, checkpoint, 100):
        raise RuntimeError(f"Evaluation lacks complete metrics: {step}/{task}")
    return task, "complete"


def run(spec):
    seed_checkpoint()
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["EMIX_POLICY_WORKER_ENV"] = "/root/autodl-tmp/conda-envs/emix-reinstall"
    cmd = [
        "/root/miniconda3/bin/conda", "run", "--no-capture-output",
        "-p", "/root/autodl-tmp/conda-envs/emix-reinstall",
        "python", "-u", "train/train.py", "--config", str(GENERATED / "train.yaml"),
        "--seed", str(spec["protocol"]["training_seed"]),
        "--resume_from", str((TRAIN_DIR / "checkpoints/last").resolve()),
    ]
    training_log = (LOG_DIR / "train.log").open("a")
    final_model = TRAIN_DIR / "checkpoints/400000/pretrained_model/model.safetensors"
    training = None
    if not final_model.is_file():
        training = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=training_log,
                                    stderr=subprocess.STDOUT)
        print(f"Training PID {training.pid}; resuming to 400k", flush=True)
    else:
        print("Final 400k checkpoint already present; verifying evaluations", flush=True)
    try:
        for step in MILESTONES:
            checkpoint = TRAIN_DIR / "checkpoints" / f"{step:06d}"
            while not all((checkpoint / name).is_file() for name in (
                "pretrained_model/model.safetensors", "training_state/training_step.json")):
                if training is None or training.poll() is not None:
                    raise RuntimeError(f"Training stopped before checkpoint {step}")
                time.sleep(20)
            print(f"Checkpoint {step} ready; evaluating ten tasks", flush=True)
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(evaluate_one, step, task, env): task
                           for task in spec["sources"]["tasks"]}
                for future in as_completed(futures):
                    task, status = future.result()
                    print(f"Evaluation {step}/{task}: {status}", flush=True)
        if training is not None and training.wait() != 0:
            raise RuntimeError(f"Training exited {training.returncode}")
        final = TRAIN_DIR / "checkpoints/400000"
        assert (final / "pretrained_model/model.safetensors").is_file()
        assert (final / "training_state/optimizer_state.safetensors").is_file()
        (OUTPUT / ".complete").write_text("Baseline 400k trained; four checkpoint evaluations complete; 400k state retained\n")
        print("multi-v0-searchparam complete; 400k checkpoint retained", flush=True)
    finally:
        training_log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "mock", "run"), required=True)
    args = parser.parse_args()
    spec = load_spec()
    prepare(spec)
    if args.mode != "prepare":
        mock(spec)
    if args.mode == "run":
        OUTPUT.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        run(spec)


if __name__ == "__main__":
    main()
