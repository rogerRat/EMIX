"""Local training entrypoint with resume support."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field  # noqa: F401
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from .utils.bootstrap import bootstrap_lerobot_runtime
from .utils.debug_encoder import set_step as _debug_set_step
from .utils.registry import is_emix_policy_type
from .utils.batch import assert_single_observation_batch, assert_single_observation_config
from .utils.image_noise import ImageNoiseConfig, add_image_noise
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from embodied_mixup.configs.config import resolve_project_output_path
from embodied_mixup.datasets import require_lerobot_dataset_root
from .utils.checkpoint import (
    ResumeConfig,
    describe_resume_config,
    infer_start_step,
    load_train_state,
    log_resume_metadata,
    resolve_checkpoint_dir,
    save_config_snapshot,
    set_global_seed,
)

LOGGER = logging.getLogger(__name__)
CODEBASE_VERSION = "v3.0"
EMIX_TRAINING_MODES = frozenset({"default", "mixed", "emix"})
EMIX_MIXED_POLICY_TYPES = frozenset({"emix_diffusion", "emix_smolvla"})


# --- Config models ---
@dataclass
class EmixTrainConfig:
    """Minimal local train config parsed from repo YAML."""

    policy_cfg: Any
    dataset_root: str
    dataset_repo_id: str
    output_dir: Path
    batch_size: int
    num_workers: int
    steps: int
    save_freq: int
    log_freq: int
    use_policy_training_preset: bool = True
    training_mode: str = "default"
    emix: dict[str, Any] = field(default_factory=dict)
    # Optional CLI-only cap for the number of observation frames rendered by
    # the EMIX debug exporter. None means use policy_cfg.n_obs_steps.
    vis_n_obs_steps: int | None = None
    # Encoder debug config for DiffusionRgbEncoder inspection (read-only).
    # Populated from the top-level ``encoder_debug`` YAML section.
    encoder_debug: dict[str, Any] | None = None
    image_noise: ImageNoiseConfig = field(default_factory=ImageNoiseConfig)
    protocol: dict[str, Any] = field(default_factory=dict)
    # All emix related configs are stored in the emix field to keep the training config clean.


# --- Checkpoint config wrapper ---
class SaveableTrainConfig:
    """
    Tiny checkpoint-compatible config wrapper.
    Used to save the training config to a checkpoint.
    Since lerobot ask for a certain format of the config,
    we need to wrap the config into a class to make it compatible.
    """

    def __init__(
        self,
        policy_cfg: Any,
        dataset_root: str,
        dataset_repo_id: str,
        steps: int,
        optimizer: Any,
        scheduler: Any,
        training_mode: str,
        image_noise: ImageNoiseConfig | None = None,
        protocol: dict[str, Any] | None = None,
    ):
        self.policy = policy_cfg
        self.dataset = type("DatasetCfg", (), {"root": dataset_root, "repo_id": dataset_repo_id})()
        self.steps = steps
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_mode = training_mode
        self.image_noise = image_noise or ImageNoiseConfig()
        self.protocol = dict(protocol or {})
        self.peft = None
        self.use_policy_training_preset = True

    def save_pretrained(self, save_directory: Path) -> None:
        payload = {
            "policy": {"type": self.policy.type},
            "dataset": {"root": self.dataset.root, "repo_id": self.dataset.repo_id},
            "steps": self.steps,
            "training_mode": self.training_mode,
            "image_noise": self.image_noise.to_dict(),
            "protocol": self.protocol,
        }
        with open(save_directory / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


# --- Logging and CLI helpers ---
def _configure_logging() -> Path:
    """Configure console and file logging."""

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d%H%M%S")
    log_path = logs_dir / f"train_{stamp}.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(name)s] [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return log_path


def _make_output_dir(policy_cfg: Any, yaml_path: Path) -> Path:
    return resolve_project_output_path(
        Path(__file__).resolve().parents[2] / "outputs" / policy_cfg.type,
        field="train.output_dir",
    )


def _load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Config normalization and validation ---
def _reject_legacy_policy_emix_keys(policy_payload: dict[str, Any]) -> None:
    """Reject EMIX parameters placed under policy in YAML.

    @input: policy_payload dict loaded from the YAML policy section.
    @output: None; raises ValueError when legacy EMIX keys are present.
    @scenario: Enforce that YAML EMIX parameters live under top-level emix.
    """

    legacy_policy_keys = {
        "emix_files_root",
        "mixup_alpha_mode",
        "mixup_beta",
        "mixup_alpha_value",
        "mixup_loss_weight",
        "mixup_visual_keys",
        "mixup_state_mode",
    }
    legacy_keys_present = sorted(legacy_policy_keys.intersection(policy_payload))
    if legacy_keys_present:
        raise ValueError(
            "Legacy EMIX keys under policy are no longer supported. "
            f"Move these keys under top-level emix: {legacy_keys_present}"
        )


def _get_emix_training_cfg(cfg: EmixTrainConfig) -> dict[str, Any]:
    """Return top-level emix.training config.

    @input: EmixTrainConfig with optional emix mapping.
    @output: emix.training dict or empty dict when EMIX is not configured.
    @scenario: Centralize EMIX training config access under the new schema.
    """

    if not cfg.emix:
        return {}
    training = cfg.emix.get("training", {})
    if training is None:
        return {}
    if not isinstance(training, dict):
        raise ValueError("emix.training must be a mapping")
    return dict(training)


def _validate_training_mode(policy_type: str, training_mode: str) -> None:
    """Validate the explicit EMIX training mode against the policy type."""

    is_emix_policy = is_emix_policy_type(policy_type)
    if training_mode == "default":
        if is_emix_policy:
            raise ValueError(
                f"policy.type='{policy_type}' cannot use training_mode='default'. "
                "Use training_mode='mixed' or training_mode='emix'."
            )
        return
    if training_mode == "mixed":
        if policy_type not in EMIX_MIXED_POLICY_TYPES:
            raise ValueError(
                f"training_mode='mixed' requires policy type in {sorted(EMIX_MIXED_POLICY_TYPES)}, "
                f"got '{policy_type}'"
            )
        return
    if training_mode == "emix":
        if not is_emix_policy:
            raise ValueError(
                f"training_mode='emix' requires an EMIX policy type, got '{policy_type}'"
            )
        return
    raise ValueError(
        f"Unknown emix.training.training_mode: '{training_mode}'. "
        f"Available: {sorted(EMIX_TRAINING_MODES)}"
    )


def _resolve_training_mode(policy_type: str, emix_payload: dict[str, Any]) -> str:
    """Resolve and validate the explicit training mode from the EMIX config."""

    if not emix_payload:
        if is_emix_policy_type(policy_type):
            raise ValueError(f"policy.type='{policy_type}' requires top-level emix config")
        return "default"

    if not is_emix_policy_type(policy_type):
        raise ValueError(
            f"Top-level emix config requires an EMIX policy type, got '{policy_type}'"
        )

    training = emix_payload.get("training") or {}
    if not isinstance(training, dict):
        raise ValueError("emix.training must be a mapping")

    training_mode_str = str(training.get("training_mode", "default"))
    _validate_training_mode(policy_type, training_mode_str)
    return training_mode_str


def _load_dataset_metadata(dataset_root: str | Path, repo_id: str) -> Any:
    """Load LeRobot dataset metadata using the library's own resolution logic.

    The LeRobot metadata loader already prefers local files when they exist and
    only falls back to Hub syncing when the local cache is missing. That keeps
    standard datasets on the official path instead of duplicating metadata
    parsing in this project.
    """

    bootstrap_lerobot_runtime()
    root = require_lerobot_dataset_root(dataset_root)
    try:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    except Exception:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(repo_id=repo_id, root=root)


def _required_stat_names_for_norm_mode(norm_mode: Any) -> tuple[str, ...]:
    """Return the stat field names required by one normalization mode.

    @input: norm_mode [Any], value from policy_cfg.normalization_mapping.
    @output: tuple[str, ...] required stat names for that mode.
    @scenario: Build explicit dataset metadata validation before training starts.
    """

    mode_value = str(getattr(norm_mode, "value", norm_mode))
    if mode_value == "IDENTITY":
        return ()
    if mode_value == "MEAN_STD":
        return ("mean", "std")
    if mode_value == "MIN_MAX":
        return ("min", "max")
    if mode_value == "QUANTILES":
        return ("q01", "q99")
    if mode_value == "QUANTILE10":
        return ("q10", "q90")
    raise ValueError(f"Unsupported normalization mode for strict stats validation: {mode_value}")


def _validate_dataset_stats(policy_cfg: Any, dataset_stats: dict[str, Any]) -> None:
    """Validate that dataset stats fully cover the policy normalization contract.

    @input: policy_cfg [Any] loaded policy config with input/output features and
        normalization_mapping; dataset_stats [dict[str, Any]] from dataset meta.
    @output: None on success; raises ValueError on missing keys or stat fields.
    @scenario: Fail before training if the dataset metadata cannot support the
        configured LeRobot normalizer/unnormalizer behavior.
    """

    required_features = {}
    required_features.update(getattr(policy_cfg, "input_features", {}) or {})
    required_features.update(getattr(policy_cfg, "output_features", {}) or {})

    missing_features = sorted(key for key in required_features if key not in dataset_stats)
    if missing_features:
        raise ValueError(
            "Dataset stats are missing policy feature(s): "
            f"{missing_features}. Available stats keys: {sorted(dataset_stats.keys())}."
        )

    normalization_mapping = getattr(policy_cfg, "normalization_mapping", {}) or {}
    missing_fields: list[str] = []
    for key, feature in required_features.items():
        stat_names = _required_stat_names_for_norm_mode(normalization_mapping.get(feature.type))
        stats_for_key = dataset_stats.get(key, {}) or {}
        for stat_name in stat_names:
            if stat_name not in stats_for_key:
                missing_fields.append(f"{key}.{stat_name}")

    if missing_fields:
        raise ValueError(
            "Dataset stats are incomplete for policy normalization: "
            f"{missing_fields}."
        )


def _build_train_processors(policy_cfg: Any, dataset_stats: dict[str, Any]) -> tuple[Any, Any]:
    """Build the training preprocessor and an action-only normalizer.

    @input: policy_cfg [Any] with populated features; dataset_stats [dict[str, Any]].
    @output: tuple[preprocessor, action_normalizer].
    @scenario: Match LeRobot official training preprocessing and normalize
        EMIX-only extra action tensors that are outside the standard batch schema.
    """

    from lerobot.processor import NormalizerProcessorStep
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        dataset_stats=dataset_stats,
    )
    action_normalizer = NormalizerProcessorStep(
        features=getattr(policy_cfg, "output_features", {}) or {},
        norm_map=getattr(policy_cfg, "normalization_mapping", {}) or {},
        stats=dataset_stats,
        device=policy_cfg.device,
    )
    return preprocessor, action_normalizer


def _normalize_emix_action_tensor(action_normalizer: Any, action_tensor: torch.Tensor) -> torch.Tensor:
    """Normalize one EMIX action tensor with the same contract as LeRobot.

    @input: action_normalizer [NormalizerProcessorStep], action_tensor
        [torch.Tensor] shaped like policy action batches.
    @output: normalized action tensor on the same device as action_tensor.
    @scenario: Keep EMIX custom action branches aligned with the normal action
        field normalized by the standard LeRobot preprocessor.
    """

    from lerobot.processor.converters import create_transition
    from lerobot.processor.core import TransitionKey

    transition = create_transition(action=action_tensor)
    normalized = action_normalizer(transition)
    normalized_action = normalized[TransitionKey.ACTION]
    if not isinstance(normalized_action, torch.Tensor):
        raise TypeError(f"Expected normalized action tensor, got {type(normalized_action)}")
    return normalized_action


def _preprocess_training_batch(
    batch: dict[str, Any],
    preprocessor: Any,
    action_normalizer: Any,
) -> dict[str, Any]:
    """Normalize a training batch while preserving EMIX-only extra fields.

    @input: batch [dict[str, Any]] raw dataloader batch; preprocessor [Any]
        from make_pre_post_processors; action_normalizer [Any].
    @output: normalized batch dict ready for policy.forward().
    @scenario: Apply standard LeRobot preprocessing and extend it to EMIX fields
        that are not part of the canonical batch-to-transition conversion.
    """

    raw_batch = dict(batch)
    # Source states for the two EMIX branches are debug-only metadata used to
    # express absolute qpos targets relative to each branch's current state.
    # They must not be forwarded to the policy as an unknown feature.
    raw_batch.pop("emix_states", None)
    raw_batch.pop("emix_source_images", None)
    raw_batch.pop("emix_eef_poses", None)
    processed_batch = preprocessor(raw_batch)

    preserved_keys = (
        "emix_alpha",
        "emix_mask",
        "sample_id",
        "group_id",
        "member_ids",
    )
    for key in preserved_keys:
        if key in raw_batch:
            processed_batch[key] = raw_batch[key]

    if "emix_actions" in raw_batch:
        processed_batch["emix_actions"] = [
            _normalize_emix_action_tensor(action_normalizer, action_tensor)
            for action_tensor in raw_batch["emix_actions"]
        ]

    return processed_batch


# --- Dataset and policy assembly ---
def _build_train_config(
    payload: dict[str, Any],
    config_path: Path,
    output_dir_override: Path | None = None,
    *,
    vis_n_obs_steps: int | None = None,
) -> EmixTrainConfig:
    bootstrap_lerobot_runtime()
    from lerobot.policies.factory import make_policy_config

    if "mix_training" in payload:
        raise ValueError("Legacy top-level mix_training is no longer supported. Use emix.training instead.")

    from .utils.emix_config import apply_emix_schema

    train_payload = payload.get("train") or {}
    if not isinstance(train_payload, dict):
        raise ValueError("train must be a mapping when present")
    policy_payload = dict(payload["policy"])
    emix_payload = dict(payload.get("emix") or {})
    _reject_legacy_policy_emix_keys(policy_payload)
    policy_type = str(policy_payload.get("type", ""))
    training_mode = _resolve_training_mode(policy_type, emix_payload)
    if vis_n_obs_steps is not None and int(vis_n_obs_steps) <= 0:
        raise ValueError("--vis_n_obs_steps must be positive")
    policy_cfg = make_policy_config(policy_payload.pop("type"), **policy_payload)
    assert_single_observation_config(policy_cfg, context="train config")
    if str(policy_cfg.type).startswith("emix_") and emix_payload:
        apply_emix_schema(policy_cfg, emix_payload)
    configured_output = train_payload.get("output_dir", payload.get("output_dir"))
    if output_dir_override is None and not configured_output:
        raise ValueError("train.output_dir is required; refusing to use an implicit output directory")
    output_dir = resolve_project_output_path(
        output_dir_override or configured_output,
        field="train.output_dir",
    )
    def train_value(name: str, default: Any) -> Any:
        return train_payload.get(name, payload.get(name, default))
    # Encoder debug config (top-level, separate from emix.debug).
    encoder_debug = payload.get("encoder_debug", None)
    if encoder_debug is not None and not isinstance(encoder_debug, dict):
        raise ValueError("encoder_debug must be a mapping when present")
    protocol = payload.get("protocol") or {}
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a mapping when present")

    return EmixTrainConfig(
        policy_cfg=policy_cfg,
        dataset_root=str(payload["dataset"]["root"]),
        dataset_repo_id=str(payload["dataset"]["repo_id"]),
        output_dir=output_dir,
        batch_size=int(train_value("batch_size", 8)),
        num_workers=int(train_value("num_workers", 0)),
        steps=int(train_value("steps", 1000)),
        save_freq=int(train_value("save_freq", 1000)),
        log_freq=int(train_value("log_freq", 100)),
        training_mode=training_mode,
        emix=emix_payload,
        vis_n_obs_steps=(None if vis_n_obs_steps is None else int(vis_n_obs_steps)),
        encoder_debug=encoder_debug,
        image_noise=ImageNoiseConfig.from_payload(train_value("image_noise", None)),
        protocol=dict(protocol),
    )


def _create_dataset_and_policy(cfg: EmixTrainConfig):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy

    from .utils.batch import get_policy_temporal_spec
    from .utils.emix_loader_registry import make_emix_loader
    from .utils.emix_loader_registry_utils import EmixLoaderSettings

    dataset_root = str(require_lerobot_dataset_root(cfg.dataset_root))
    ds_meta = _load_dataset_metadata(dataset_root, cfg.dataset_repo_id)
    policy = make_policy(cfg.policy_cfg, ds_meta=ds_meta)
    _validate_dataset_stats(policy.config, ds_meta.stats)

    if cfg.training_mode == "emix":
        temporal_spec = get_policy_temporal_spec(cfg.policy_cfg)
        loader_cfg = EmixLoaderSettings(
            policy_type=cfg.policy_cfg.type,
            mixup_alpha_mode=str(cfg.policy_cfg.mixup_alpha_mode),
            mixup_alpha_value=cfg.policy_cfg.mixup_alpha_value,
            mixup_beta=float(cfg.policy_cfg.mixup_beta),
            mixup_visual_keys=cfg.policy_cfg.mixup_visual_keys,
            mixup_state_mode=str(cfg.policy_cfg.mixup_state_mode),
            mixup_image_mode=str(getattr(cfg.policy_cfg, "mixup_image_mode", "linear")),
            annotation_root=str(getattr(cfg.policy_cfg, "annotation_root", "")),
            detect_overlap_enable=bool(getattr(cfg.policy_cfg, "detect_overlap_enable", False)),
            detect_overlap_refuse_ratio=float(getattr(cfg.policy_cfg, "detect_overlap_refuse_ratio", 0.5)),
            seed=0,
        )
        dataset = make_emix_loader(
            policy_type=cfg.policy_cfg.type,
            source_root=dataset_root,
            source_repo_id=cfg.dataset_repo_id,
            emix_index_path=cfg.policy_cfg.emix_index_path,
            policy_temporal_spec=temporal_spec,
            loader_cfg=loader_cfg,
        )
    else:
        LOGGER.info(
            "[train] Loading LeRobot dataset: repo_id=%s root=%s",
            cfg.dataset_repo_id,
            dataset_root,
        )
        dataset = LeRobotDataset(
            repo_id=cfg.dataset_repo_id,
            root=dataset_root,
            delta_timestamps=_build_delta_timestamps(
                cfg.policy_cfg,
                ds_meta.features,
                float(ds_meta.info["fps"]),
            ),
        )
    return ds_meta, dataset, policy


def _build_emix_dataset(cfg: EmixTrainConfig):
    """Build the EMIX dataset for mixed training mode."""

    from .utils.batch import get_policy_temporal_spec
    from .utils.emix_loader_registry import make_emix_loader
    from .utils.emix_loader_registry_utils import EmixLoaderSettings

    temporal_spec = get_policy_temporal_spec(cfg.policy_cfg)
    loader_cfg = EmixLoaderSettings(
        policy_type=cfg.policy_cfg.type,
        mixup_alpha_mode=str(cfg.policy_cfg.mixup_alpha_mode),
        mixup_alpha_value=cfg.policy_cfg.mixup_alpha_value,
        mixup_beta=float(cfg.policy_cfg.mixup_beta),
        mixup_visual_keys=cfg.policy_cfg.mixup_visual_keys,
        mixup_state_mode=str(cfg.policy_cfg.mixup_state_mode),
        mixup_image_mode=str(getattr(cfg.policy_cfg, "mixup_image_mode", "linear")),
        annotation_root=str(getattr(cfg.policy_cfg, "annotation_root", "")),
        detect_overlap_enable=bool(getattr(cfg.policy_cfg, "detect_overlap_enable", False)),
        detect_overlap_refuse_ratio=float(getattr(cfg.policy_cfg, "detect_overlap_refuse_ratio", 0.5)),
        seed=0,
    )
    dataset_root = str(require_lerobot_dataset_root(cfg.dataset_root))
    return make_emix_loader(
        policy_type=cfg.policy_cfg.type,
        source_root=dataset_root,
        source_repo_id=cfg.dataset_repo_id,
        emix_index_path=cfg.policy_cfg.emix_index_path,
        policy_temporal_spec=temporal_spec,
        loader_cfg=loader_cfg,
    )


def _build_delta_timestamps(
    policy_cfg: Any,
    dataset_features: dict[str, Any],
    dataset_fps: float,
) -> dict[str, list[float]]:
    """Build frame-aligned timestamp queries for non-EMIX LeRobot loaders.

    @input: policy_cfg policy configuration with observation and action delta indices;
        dataset_features LeRobot feature mapping; dataset_fps positive source FPS.
    @output: dict mapping dataset feature names to frame-aligned timestamp offsets.
    @scenario: Keep policy temporal windows valid for datasets that are not 30 FPS.
    """

    if dataset_fps <= 0:
        raise ValueError(f"Dataset FPS must be positive, got {dataset_fps}.")
    fps = float(dataset_fps)
    observation_deltas = list(getattr(policy_cfg, "observation_delta_indices", []) or [])
    action_deltas = list(getattr(policy_cfg, "action_delta_indices", []) or [])
    delta_timestamps: dict[str, list[float]] = {
        "observation.state": [float(idx) / fps for idx in observation_deltas],
        "action": [float(idx) / fps for idx in action_deltas],
    }

    input_features = getattr(policy_cfg, "input_features", {}) or {}
    image_keys = [key for key in input_features if key.startswith("observation.images.")]
    for key in image_keys:
        if key in dataset_features:
            delta_timestamps[key] = [float(idx) / fps for idx in observation_deltas]
    if (
        "observation.eef_pose" in dataset_features
        and (
            "observation.eef_pose" in getattr(policy_cfg, "output_features", {})
            or "observation.eef_pose" in input_features
        )
    ):
        delta_timestamps["observation.eef_pose"] = [float(idx) / fps for idx in observation_deltas]
    return delta_timestamps


# --- Batch adaptation ---
def _prepare_policy_batch(policy_cfg: Any, policy: Any, batch: dict[str, Any]) -> dict[str, Any]:
    """Apply lightweight batch compatibility fixes before policy.forward()."""

    batch = dict(batch)
    if is_emix_policy_type(str(policy_cfg.type)):
        assert_single_observation_batch(
            batch,
            context=f"train._prepare_policy_batch[{policy_cfg.type}]",
        )
    if policy_cfg.type == "smolvla":
        if "action_is_pad" in batch and "actions_id_pad" not in batch:
            batch["actions_id_pad"] = batch["action_is_pad"]
        if "task" in batch and OBS_LANGUAGE_TOKENS not in batch:
            tokenizer = policy.model.vlm_with_expert.processor.tokenizer
            tasks = batch["task"]
            if isinstance(tasks, str):
                task_list = [tasks]
            else:
                task_list = list(tasks)
            task_list = [text if text.endswith("\n") else f"{text}\n" for text in task_list]
            encoded = tokenizer(
                task_list,
                padding=policy_cfg.pad_language_to,
                truncation=True,
                max_length=policy_cfg.tokenizer_max_length,
                return_tensors="pt",
            )
            device = batch["observation.state"].device
            batch[OBS_LANGUAGE_TOKENS] = encoded["input_ids"].to(device)
            batch[OBS_LANGUAGE_ATTENTION_MASK] = encoded["attention_mask"].to(device).bool()
    return batch


def _move_batch_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors inside nested batch containers onto one device.

    @input: value [Any], device [torch.device].
    @output: value with all nested torch.Tensor instances moved to device.
    @scenario: EMIX batches contain tensor lists such as emix_actions that are
        missed by top-level-only device transfer.
    """

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_batch_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_batch_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_batch_to_device(item, device) for key, item in value.items()}
    return value


# --- Snapshot and resume helpers ---
def _write_train_config_snapshot(cfg: EmixTrainConfig) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_type": cfg.policy_cfg.type,
        "dataset_root": cfg.dataset_root,
        "dataset_repo_id": cfg.dataset_repo_id,
        "steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "image_noise": cfg.image_noise.to_dict(),
        "protocol": cfg.protocol,
    }
    snapshot_path = cfg.output_dir / "emix_train_config.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return snapshot_path


def _prepare_output_dir(output_dir: Path, *, allow_existing: bool = False) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _record_detect_overlap_fallbacks(batch: dict[str, Any], output_dir: Path, step: int) -> None:
    events = [event for event in batch.get("detect_overlap_fallback", []) if event]
    if not events:
        return
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_root / "detect_overlap_fallback.txt"
    with log_path.open("a", encoding="utf-8") as handle:
        for event in events:
            entry = {"train_step": int(step), **event}
            handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
            LOGGER.warning(
                "DETECT_OVERLAP FALLBACK: step=%d refused=%s overlap=%.6f "
                "selected=%s overlap=%.6f refuse_ratio=%.6f reason=%s log=%s",
                step,
                event.get("refused_sample_id"),
                float(event.get("refused_overlap_ratio", 0.0)),
                event.get("selected_sample_id"),
                float(event.get("selected_overlap_ratio", 0.0)),
                float(event.get("refuse_ratio", 0.0)),
                event.get("reason"),
                log_path,
            )
        handle.flush()
        import os
        os.fsync(handle.fileno())


def _prune_optimizer_states(output_dir: Path, keep_checkpoint: Path) -> int:
    """Keep optimizer state only for the newest complete/resumable checkpoint."""
    checkpoint_root = output_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        return 0
    keep_checkpoint = keep_checkpoint.resolve()
    removed = 0
    for step_dir in checkpoint_root.iterdir():
        if not step_dir.is_dir() or not step_dir.name.isdigit():
            continue
        optimizer_state = step_dir / "training_state" / "optimizer_state.safetensors"
        if step_dir.resolve() != keep_checkpoint and optimizer_state.is_file() and not optimizer_state.is_symlink():
            optimizer_state.unlink()
            removed += 1
    if removed:
        LOGGER.info("[checkpoint] pruned %d stale optimizer state(s); kept=%s", removed, keep_checkpoint)
    return removed


def _build_resume_config(args: argparse.Namespace) -> ResumeConfig:
    return ResumeConfig(
        resume_from=Path(args.resume_from).expanduser().resolve() if args.resume_from else None,
        output_dir_override=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
        seed=int(args.seed),
        start_step=int(args.start_step) if args.start_step is not None else None,
        total_steps_override=int(args.total_steps) if args.total_steps is not None else None,
    )


def _snapshot_payload(cfg: EmixTrainConfig, resume_cfg: ResumeConfig, start_step: int, total_steps: int) -> dict[str, Any]:
    payload = {
        "policy": {
            "type": cfg.policy_cfg.type,
            "device": cfg.policy_cfg.device,
            "push_to_hub": bool(getattr(cfg.policy_cfg, "push_to_hub", False)),
        },
        "dataset": {"root": cfg.dataset_root, "repo_id": cfg.dataset_repo_id},
        "output_dir": str(cfg.output_dir),
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "steps": total_steps,
        "save_freq": cfg.save_freq,
        "log_freq": cfg.log_freq,
        "training_mode": cfg.training_mode,
        "image_noise": cfg.image_noise.to_dict(),
        "protocol": cfg.protocol,
        "seed": resume_cfg.seed,
        "resume": describe_resume_config(resume_cfg),
        "start_step": start_step,
        "total_steps": total_steps,
    }
    emix_training = _get_emix_training_cfg(cfg)
    if cfg.emix:
        payload["emix"] = cfg.emix
    if emix_training:
        payload["mix_scheduler_type"] = emix_training.get("scheduler", {}).get("type", "")
    if cfg.vis_n_obs_steps is not None:
        payload["vis_n_obs_steps_override"] = cfg.vis_n_obs_steps
    return payload


def _make_emix_debug_exporter(cfg: EmixTrainConfig):
    """Create the optional training-input snapshot exporter."""

    from embodied_mixup.visualization.emix_debug import EmixDebugExporter, EmixDebugSettings

    debug_payload = dict((cfg.emix or {}).get("debug") or {})
    settings = EmixDebugSettings.from_payload(
        debug_payload,
        default_output_dir=cfg.output_dir / "emix_debug",
        policy_type=str(cfg.policy_cfg.type),
        default_n_obs_steps=int(getattr(cfg.policy_cfg, "n_obs_steps", 1)),
        vis_n_obs_steps=cfg.vis_n_obs_steps,
    )
    if settings.output_dir is not None:
        resolve_project_output_path(settings.output_dir, field="emix.debug.output_dir")
    if not settings.enabled or settings.max_samples == 0:
        return None
    return EmixDebugExporter(settings)


# --- Training runtime ---
def train(
    config_path: str | Path,
    *,
    resume_cfg: ResumeConfig | None = None,
    vis_n_obs_steps: int | None = None,
) -> Path:
    """Train one policy using the local runtime bridge."""

    bootstrap_lerobot_runtime()
    from torch.utils.data import DataLoader

    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.utils.train_utils import get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint

    from .utils.batch import stack_tensor_list
    from .utils.mix_scheduler import make_mix_scheduler

    if resume_cfg is None:
        resume_cfg = ResumeConfig(
            resume_from=None,
            output_dir_override=None,
            seed=0,
            start_step=None,
            total_steps_override=None,
        )

    raw_cfg = _load_yaml_config(config_path)
    cfg = _build_train_config(
        raw_cfg,
        Path(config_path),
        resume_cfg.output_dir_override,
        vis_n_obs_steps=vis_n_obs_steps,
    )
    cfg.output_dir = _prepare_output_dir(cfg.output_dir, allow_existing=resume_cfg.resume_from is not None)
    snapshot_path = _write_train_config_snapshot(cfg)
    set_global_seed(resume_cfg.seed)

    # Configure encoder debug capture from YAML (before encoder is built).
    # When encoder_debug.enabled is true, forward hooks are attached to every
    # DiffusionRgbEncoder instance to periodically save input images and
    # intermediate feature maps for inspection.
    from .utils.debug_encoder import configure as _debug_configure

    _debug_configure(cfg.encoder_debug)

    ds_meta, dataset, policy = _create_dataset_and_policy(cfg)
    preprocessor, action_normalizer = _build_train_processors(policy.config, ds_meta.stats)
    optimizer_cfg = cfg.policy_cfg.get_optimizer_preset()
    scheduler_cfg = cfg.policy_cfg.get_scheduler_preset()

    class OptimBuildCfg:
        optimizer = optimizer_cfg
        scheduler = scheduler_cfg
        steps = cfg.steps
        use_policy_training_preset = True

    optimizer, lr_scheduler = make_optimizer_and_scheduler(OptimBuildCfg, policy)
    device = torch.device(cfg.policy_cfg.device)
    policy.to(device)
    checkpoint_cfg = SaveableTrainConfig(
        policy_cfg=cfg.policy_cfg,
        dataset_root=cfg.dataset_root,
        dataset_repo_id=cfg.dataset_repo_id,
        steps=cfg.steps,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        training_mode=cfg.training_mode,
        image_noise=cfg.image_noise,
        protocol=cfg.protocol,
    )

    total_steps = int(resume_cfg.total_steps_override) if resume_cfg.total_steps_override is not None else int(cfg.steps)
    if resume_cfg.start_step is not None:
        start_step = int(resume_cfg.start_step)
    elif resume_cfg.resume_from is not None:
        start_step = 0
    else:
        start_step = 0
    if total_steps < start_step:
        raise ValueError(f"total_steps ({total_steps}) must be greater than or equal to start_step ({start_step})")

    resume_metadata = {
        "resume_from": str(resume_cfg.resume_from) if resume_cfg.resume_from is not None else None,
        "output_dir": str(cfg.output_dir),
        "seed": resume_cfg.seed,
        "start_step": start_step,
        "total_steps": total_steps,
        "policy_type": cfg.policy_cfg.type,
        "dataset_repo_id": cfg.dataset_repo_id,
    }

    if resume_cfg.resume_from is not None:
        resolved_checkpoint = resolve_checkpoint_dir(resume_cfg.resume_from)
        latest_link = resolved_checkpoint.parent / "last"
        if not latest_link.is_symlink() or latest_link.resolve() != resolved_checkpoint:
            raise ValueError(
                f"Only the latest checkpoint referenced by {latest_link} can be resumed; "
                f"requested {resolved_checkpoint}"
            )
        required_resume_files = (
            resolved_checkpoint / "pretrained_model" / "model.safetensors",
            resolved_checkpoint / "training_state" / "optimizer_state.safetensors",
            resolved_checkpoint / "training_state" / "rng_state.safetensors",
            resolved_checkpoint / "training_state" / "training_step.json",
        )
        missing_resume_files = [str(path) for path in required_resume_files if not path.is_file()]
        if missing_resume_files:
            raise FileNotFoundError(
                "Latest checkpoint is incomplete and cannot be resumed: " + ", ".join(missing_resume_files)
            )
        loaded_state = load_train_state(resolved_checkpoint, policy, optimizer, lr_scheduler, device)
        _prune_optimizer_states(cfg.output_dir, resolved_checkpoint)
        if resume_cfg.start_step is None:
            start_step = infer_start_step(resolved_checkpoint, fallback=start_step)
        loaded_state["resolved_checkpoint"] = str(resolved_checkpoint)
        log_resume_metadata({**resume_metadata, **loaded_state, "start_step": start_step})
    else:
        log_resume_metadata({**resume_metadata, "loaded_state": None})

    config_snapshot = _snapshot_payload(cfg, resume_cfg, start_step, total_steps)
    snapshot_path = save_config_snapshot(cfg.output_dir, config_snapshot)
    LOGGER.info("[train] config_snapshot=%s", snapshot_path)
    LOGGER.info("[train] output_dir=%s seed=%d total_steps=%d start_step=%d", cfg.output_dir, resume_cfg.seed, total_steps, start_step)
    LOGGER.info("[train] dataset_repo_id=%s policy_type=%s", cfg.dataset_repo_id, cfg.policy_cfg.type)
    LOGGER.info("[train] training_mode=%s", cfg.training_mode)
    LOGGER.info("[train] image_noise=%s (training only, before policy normalization)", cfg.image_noise.to_dict())
    if ds_meta is not None:
        LOGGER.debug("[train] dataset_root=%s", cfg.dataset_root)

    emix_training = _get_emix_training_cfg(cfg)
    emix_debug_exporter = _make_emix_debug_exporter(cfg)

    if cfg.training_mode == "mixed":
        mix_seed = int(emix_training.get("seed", resume_cfg.seed))
        mix_rng = random.Random(mix_seed)
        scheduler_payload = emix_training.get("scheduler", {})
        mix_scheduler = make_mix_scheduler(scheduler_payload)
        LOGGER.info(
            "[train] mixed training enabled scheduler_type=%s mix_seed=%d",
            scheduler_payload.get("type", ""),
            mix_seed,
        )

        emix_dataset = _build_emix_dataset(cfg)
        emix_dataloader = DataLoader(
            emix_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=stack_tensor_list,
            drop_last=False,
        )
        nonemix_dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=stack_tensor_list,
            drop_last=False,
        )
        emix_iter = iter(emix_dataloader)
        nonemix_iter = iter(nonemix_dataloader)

        progress = tqdm(range(start_step + 1, total_steps + 1), desc=f"train:{cfg.policy_cfg.type}")
        for step in progress:
            _debug_set_step(step)
            emix_prob = mix_scheduler.emix_batch_prob(step, total_steps)
            batch_source = "emix" if mix_rng.random() < emix_prob else "non_emix"

            if batch_source == "emix":
                try:
                    batch = next(emix_iter)
                except StopIteration:
                    emix_iter = iter(emix_dataloader)
                    batch = next(emix_iter)
            else:
                try:
                    batch = next(nonemix_iter)
                except StopIteration:
                    nonemix_iter = iter(nonemix_dataloader)
                    batch = next(nonemix_iter)

            _record_detect_overlap_fallbacks(batch, cfg.output_dir, step)
            batch = add_image_noise(batch, cfg.image_noise, seed=resume_cfg.seed, step=step)
            if batch_source == "emix" and emix_debug_exporter is not None:
                emix_debug_exporter.export_batch(batch, step)
            batch = _preprocess_training_batch(batch, preprocessor, action_normalizer)
            batch = _move_batch_to_device(batch, device)
            batch = _prepare_policy_batch(cfg.policy_cfg, policy, batch)

            policy.train()
            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            grad_clip = getattr(optimizer_cfg, "grad_clip_norm", 0.0)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if step % cfg.log_freq == 0 or step == start_step + 1:
                progress.set_postfix(loss=float(loss.item()))
                LOGGER.info(
                    "[train] step=%d loss=%.6f batch_source=%s emix_batch_prob=%.4f extra=%s",
                    step, float(loss.item()), batch_source, emix_prob, loss_dict,
                )

            if step % cfg.save_freq == 0 or step == total_steps:
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, total_steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=checkpoint_cfg,
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                )
                update_last_checkpoint(checkpoint_dir)
                _prune_optimizer_states(cfg.output_dir, checkpoint_dir)
                LOGGER.info("[checkpoint] saved=%s", checkpoint_dir)

    elif cfg.training_mode == "emix":
        LOGGER.info("[train] pure emix training enabled")
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=stack_tensor_list,
            drop_last=False,
        )
        dl_iter = iter(dataloader)

        progress = tqdm(range(start_step + 1, total_steps + 1), desc=f"train:{cfg.policy_cfg.type}")
        for step in progress:
            _debug_set_step(step)
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dataloader)
                batch = next(dl_iter)

            _record_detect_overlap_fallbacks(batch, cfg.output_dir, step)
            batch = add_image_noise(batch, cfg.image_noise, seed=resume_cfg.seed, step=step)
            if emix_debug_exporter is not None:
                emix_debug_exporter.export_batch(batch, step)
            batch = _preprocess_training_batch(batch, preprocessor, action_normalizer)
            batch = _move_batch_to_device(batch, device)
            batch = _prepare_policy_batch(cfg.policy_cfg, policy, batch)

            policy.train()
            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            grad_clip = getattr(optimizer_cfg, "grad_clip_norm", 0.0)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if step % cfg.log_freq == 0 or step == start_step + 1:
                progress.set_postfix(loss=float(loss.item()))
                LOGGER.info("[train] step=%d loss=%.6f extra=%s", step, float(loss.item()), loss_dict)

            if step % cfg.save_freq == 0 or step == total_steps:
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, total_steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=checkpoint_cfg,
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                )
                update_last_checkpoint(checkpoint_dir)
                _prune_optimizer_states(cfg.output_dir, checkpoint_dir)
                LOGGER.info("[checkpoint] saved=%s", checkpoint_dir)

    else:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=stack_tensor_list,
            drop_last=False,
        )
        dl_iter = iter(dataloader)

        progress = tqdm(range(start_step + 1, total_steps + 1), desc=f"train:{cfg.policy_cfg.type}")
        for step in progress:
            _debug_set_step(step)
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dataloader)
                batch = next(dl_iter)

            _record_detect_overlap_fallbacks(batch, cfg.output_dir, step)
            batch = add_image_noise(batch, cfg.image_noise, seed=resume_cfg.seed, step=step)
            batch = _preprocess_training_batch(batch, preprocessor, action_normalizer)
            batch = _move_batch_to_device(batch, device)
            batch = _prepare_policy_batch(cfg.policy_cfg, policy, batch)

            policy.train()
            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            grad_clip = getattr(optimizer_cfg, "grad_clip_norm", 0.0)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if step % cfg.log_freq == 0 or step == start_step + 1:
                progress.set_postfix(loss=float(loss.item()))
                LOGGER.info("[train] step=%d loss=%.6f extra=%s", step, float(loss.item()), loss_dict)

            if step % cfg.save_freq == 0 or step == total_steps:
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, total_steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=checkpoint_cfg,
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                )
                update_last_checkpoint(checkpoint_dir)
                _prune_optimizer_states(cfg.output_dir, checkpoint_dir)
                LOGGER.info("[checkpoint] saved=%s", checkpoint_dir)

    return cfg.output_dir
