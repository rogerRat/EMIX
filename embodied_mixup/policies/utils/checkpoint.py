"""
Checkpoint helpers for local resume-capable training.

ResumeConfig
        │
resolve_checkpoint_dir()
        │
load_train_state()
        ├── load policy weights
        ├── load optimizer state
        ├── load scheduler state
        └── load RNG state
        │
infer_start_step() (optional)
        │
Determine total_steps
        │
Continue training:
    for step in range(start_step + 1, total_steps + 1)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, load_model

from .bootstrap import bootstrap_lerobot_runtime

LOGGER = logging.getLogger(__name__)

CHECKPOINTS_DIR_NAME = "checkpoints"
PRETRAINED_MODEL_DIR_NAME = "pretrained_model"
TRAINING_STATE_DIR_NAME = "training_state"
LAST_CHECKPOINT_NAME = "last"
TRAINING_STEP_FILE_NAME = "training_step.json"


@dataclass(frozen=True)
class ResumeConfig:
    """Resume controls for a training run.

    @input: [ResumeConfig, resume and output path controls, CLI-derived]
    @output: [ResumeConfig, validated resume metadata, success/fail]
    @scenario: [Carry training forward from a prior checkpoint without writing into the source tree]
    """

    resume_from: Path | None
    output_dir_override: Path | None
    seed: int
    start_step: int | None
    total_steps_override: int | None


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch RNG state."""

    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint_dir(path: str | Path) -> Path:
    """Resolve supported checkpoint path forms to the step directory."""

    candidate = Path(path).expanduser()
    resolved = candidate.resolve()

    if resolved.name in {PRETRAINED_MODEL_DIR_NAME, TRAINING_STATE_DIR_NAME}:
        resolved = resolved.parent
    elif resolved.name == LAST_CHECKPOINT_NAME and resolved.is_symlink():
        resolved = resolved.resolve()

    if resolved.is_dir() and (resolved / PRETRAINED_MODEL_DIR_NAME).is_dir():
        return resolved

    checkpoints_root = resolved if resolved.name == CHECKPOINTS_DIR_NAME else resolved.parent
    if checkpoints_root.name == CHECKPOINTS_DIR_NAME and resolved.name != CHECKPOINTS_DIR_NAME:
        step_dir = resolved
        if step_dir.is_dir() and (step_dir / PRETRAINED_MODEL_DIR_NAME).is_dir():
            return step_dir

    if candidate.is_symlink():
        link_target = candidate.resolve()
        if link_target.is_dir() and (link_target / PRETRAINED_MODEL_DIR_NAME).is_dir():
            return link_target

    raise FileNotFoundError(f"Unable to resolve checkpoint directory from: {path}")


def infer_start_step(checkpoint_dir: Path, fallback: int | None = None) -> int:
    """Infer the next training step from a checkpoint directory."""

    state_file = checkpoint_dir / TRAINING_STATE_DIR_NAME / TRAINING_STEP_FILE_NAME
    if state_file.is_file():
        with open(state_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return int(payload["step"])

    if fallback is not None:
        return int(fallback)

    raise FileNotFoundError(f"Missing training step metadata in {checkpoint_dir}")


def _load_policy_weights(policy: torch.nn.Module, pretrained_dir: Path, device: torch.device) -> dict[str, Any]:
    model_file = pretrained_dir / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(model_file)

    load_model(policy, str(model_file), strict=False, device=str(device))
    return {
        "policy_weights": True,
        "model_file": str(model_file),
    }


def _load_training_state(
    checkpoint_dir: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, bool]:
    bootstrap_lerobot_runtime()

    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR_NAME
    loaded: dict[str, bool] = {
        "optimizer_state": False,
        "scheduler_state": False,
        "rng_state": False,
    }

    if training_state_dir.is_dir():
        rng_file = training_state_dir / "rng_state.safetensors"
        if rng_file.is_file():
            from lerobot.utils.random_utils import load_rng_state

            load_rng_state(training_state_dir)
            loaded["rng_state"] = True

        opt_file = training_state_dir / "optimizer_state.safetensors"
        if opt_file.is_file():
            from lerobot.optim.optimizers import load_optimizer_state

            load_optimizer_state(optimizer, training_state_dir)
            loaded["optimizer_state"] = True

        if scheduler is not None and (training_state_dir / "scheduler_state.json").is_file():
            from lerobot.optim.schedulers import load_scheduler_state

            load_scheduler_state(scheduler, training_state_dir)
            loaded["scheduler_state"] = True

    return loaded


def load_train_state(
    checkpoint_dir: Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Load a model and optional optimizer/scheduler state from a checkpoint."""

    checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR_NAME
    loaded = _load_policy_weights(policy, pretrained_dir, device)
    loaded.update(_load_training_state(checkpoint_dir, optimizer, scheduler))
    loaded["checkpoint_dir"] = str(checkpoint_dir)
    loaded["pretrained_dir"] = str(pretrained_dir)
    return loaded


def save_config_snapshot(output_dir: Path, payload: dict[str, Any]) -> Path:
    """Persist a JSON config snapshot for resumable runs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "train_config_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return snapshot_path


def log_resume_metadata(metadata: dict[str, Any]) -> None:
    """Emit a compact resume metadata log line."""

    LOGGER.info("[resume] metadata=%s", metadata)


def describe_resume_config(resume_cfg: ResumeConfig) -> dict[str, Any]:
    """Serialize resume config for logging and snapshots."""

    data = asdict(resume_cfg)
    for key in ("resume_from", "output_dir_override"):
        value = data[key]
        data[key] = str(value) if value is not None else None
    return data
