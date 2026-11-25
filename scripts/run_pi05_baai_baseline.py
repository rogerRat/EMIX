#!/usr/bin/env python3
"""Prepare and launch the training-only BAAI PI0.5 baseline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from embodied_mixup.policies.utils.multi_dataset import load_multi_dataset, validate_sources

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/scripts/pi05_baai_baseline.yaml"
GENERATED = ROOT / "configs/scripts/pi05_baai_baseline/train.yaml"


def _load_spec() -> dict:
    return yaml.safe_load(SPEC.read_text(encoding="utf-8"))


def _check_sources(dataset_root: Path, tasks: list[str]) -> None:
    validate_sources(dataset_root, tasks)
    for task in tasks:
        source = dataset_root / task
        for relative in (
            "data/chunk-000/file-000.parquet",
            "videos/observation.images.head_camera/chunk-000/file-000.mp4",
        ):
            path = source / relative
            if not path.is_file() or path.stat().st_size < 1024:
                raise RuntimeError(f"Dataset file is missing or still an LFS pointer: {path}")


def _make_train_config(spec: dict) -> dict:
    training = spec["training"]
    tasks = list(spec["sources"]["tasks"])
    dataset_root = Path(spec["sources"]["dataset_root"]).expanduser().resolve()
    _check_sources(dataset_root, tasks)
    output_dir = ROOT / "outputs" / spec["protocol"]["name"]
    policy_keys = (
        "pretrained_path", "device", "dtype", "compile_model",
        "gradient_checkpointing", "freeze_vision_encoder", "train_expert_only",
        "optimizer_lr", "scheduler_warmup_steps", "scheduler_decay_steps",
        "scheduler_decay_lr", "n_obs_steps", "chunk_size", "n_action_steps",
        "push_to_hub",
    )
    return {
        "batch_size": training["batch_size"],
        "grad_accum_steps": training["grad_accum_steps"],
        "num_workers": training["num_workers"],
        "steps": training["steps"],
        "save_freq": training["save_freq"],
        "log_freq": training["log_freq"],
        "output_dir": str(output_dir),
        "policy": {"type": "pi05", **{key: training[key] for key in policy_keys}},
        "dataset": {
            "root": str(dataset_root),
            "repo_id": tasks[0],
            "repo_ids": tasks,
            "video_backend": training["video_backend"],
        },
        "protocol": spec["protocol"],
    }


def _prepare(spec: dict) -> dict:
    config = _make_train_config(spec)
    GENERATED.parent.mkdir(parents=True, exist_ok=True)
    GENERATED.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Prepared training config: {GENERATED}", flush=True)
    return config


def _mock(config: dict) -> None:
    from embodied_mixup.policies.train import _build_train_config

    parsed = _build_train_config(config, GENERATED)
    if parsed.policy_cfg.type != "pi05" or parsed.peft is not None:
        raise RuntimeError("Expected full-parameter PI0.5 training without PEFT")
    if parsed.batch_size != 32 or parsed.grad_accum_steps != 1 or parsed.steps != 300000:
        raise RuntimeError("Training batch or step settings differ from the approved protocol")
    if parsed.policy_cfg.freeze_vision_encoder or parsed.policy_cfg.train_expert_only:
        raise RuntimeError("Expected unfrozen vision encoder and VLM")
    _, dataset = load_multi_dataset(
        parsed.dataset_root, parsed.dataset_repo_ids,
        parsed.policy_cfg, parsed.video_backend,
    )
    sample = dataset[0]
    print(
        f"Validated {len(parsed.dataset_repo_ids)} tasks, {len(dataset)} frames, "
        f"first action shape={tuple(sample['action'].shape)}",
        flush=True,
    )


def _run(config: dict, spec: dict) -> None:
    import torch
    from huggingface_hub import hf_hub_download

    if not torch.cuda.is_available():
        raise RuntimeError("The submitted job has no CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < 70 * 1024**3:
        raise RuntimeError(f"Expected an 80 GB GPU, got {properties.name}")
    print(f"GPU: {properties.name}, {properties.total_memory / 1024**3:.1f} GiB", flush=True)
    os.environ.setdefault("HF_HOME", "/home/zylong/hf-cache")
    hf_hub_download(repo_id=spec["training"]["pretrained_path"], filename="model.safetensors")
    hf_hub_download(repo_id="google/paligemma-3b-pt-224", filename="config.json")
    output_dir = Path(config["output_dir"])
    last = output_dir / "checkpoints/last"
    args = [
        sys.executable, "-u", str(ROOT / "train/train.py"),
        "--config", str(GENERATED),
        "--seed", str(spec["protocol"]["training_seed"]),
    ]
    if last.is_symlink():
        args.extend(["--resume_from", str(last.resolve())])
        print(f"Resuming from {last.resolve()}", flush=True)
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Nonempty output without a resumable checkpoint: {output_dir}")
    print("Launching training without evaluation", flush=True)
    os.chdir(ROOT)
    os.execv(sys.executable, args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare", "mock", "run"))
    args = parser.parse_args()
    spec = _load_spec()
    config = _prepare(spec)
    if args.mode in ("mock", "run"):
        _mock(config)
    if args.mode == "run":
        _run(config, spec)


if __name__ == "__main__":
    main()
