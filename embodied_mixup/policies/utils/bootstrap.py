"""
Runtime bootstrap for Emix policies without modifying site-packages.

Is a runtime monkey patch to make the emix policies work in the local lerobot runtime.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Any

from .registry import EMIX_POLICY_REGISTRATIONS, is_emix_policy_type

LOGGER = logging.getLogger(__name__)


def _ensure_package_stub(module_name: str, package_path: Path) -> None:
    """Register a lightweight package module to avoid executing broken __init__.py files."""

    if module_name in sys.modules:
        return
    package = types.ModuleType(module_name)
    package.__path__ = [str(package_path)]
    package.__package__ = module_name
    sys.modules[module_name] = package


def _patch_lerobot_dataset_utils() -> None:
    """Patch missing dataset utility symbols in drifted local lerobot installs."""

    lerobot_spec = importlib.util.find_spec("lerobot")
    if lerobot_spec is None or not lerobot_spec.submodule_search_locations:
        raise ImportError("Unable to locate installed lerobot package for runtime bootstrap.")
    lerobot_root = Path(lerobot_spec.submodule_search_locations[0])

    # Import the real package namespaces when available so we do not shadow
    # exported symbols such as lerobot.optim.OptimizerConfig.
    for module_name in ("lerobot.datasets", "lerobot.policies", "lerobot.configs", "lerobot.optim"):
        try:
            importlib.import_module(module_name)
        except Exception:
            package_path = lerobot_root / module_name.split(".", 1)[1]
            _ensure_package_stub(module_name, package_path)

    if "lerobot.datasets.utils" in sys.modules:
        utils_mod = sys.modules["lerobot.datasets.utils"]
    else:
        utils_path = lerobot_root / "datasets" / "utils.py"
        spec = importlib.util.spec_from_file_location("lerobot.datasets.utils", utils_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load lerobot dataset utils from {utils_path}")
        utils_mod = importlib.util.module_from_spec(spec)
        sys.modules["lerobot.datasets.utils"] = utils_mod
        spec.loader.exec_module(utils_mod)

    defaults = {
        "DEFAULT_DATA_FILE_SIZE_IN_MB": 256,
        "DEFAULT_VIDEO_FILE_SIZE_IN_MB": 256,
        "DEFAULT_DATA_PATH": "data/chunk-{chunk_index:03d}/file_{file_index:03d}.parquet",
        "LEGACY_EPISODES_PATH": "meta/episodes.jsonl",
        "LEGACY_EPISODES_STATS_PATH": "meta/episodes_stats.jsonl",
        "LEGACY_TASKS_PATH": "meta/tasks.jsonl",
    }
    for key, value in defaults.items():
        if not hasattr(utils_mod, key):
            setattr(utils_mod, key, value)
            LOGGER.debug("[bootstrap] Patched lerobot.datasets.utils.%s", key)


def _load_symbol(import_path: str, symbol_name: str) -> Any:
    module = importlib.import_module(import_path)
    return getattr(module, symbol_name)


def _build_make_pre_post_processors():
    def make_pre_post_processors(policy_cfg, pretrained_path=None, dataset_stats=None, **_kwargs):
        del pretrained_path

        # --------------------------------------------------------------
        # BUGFIX: Override VISUAL normalization to IDENTITY for
        # diffusion-based policies.
        #
        # Root cause:
        #   LeRobot's default normalization_mapping sets VISUAL ->
        #   MEAN_STD, which normalizes images using dataset-specific
        #   statistics (e.g. mean~[0.89, 0.89, 0.91] for bright
        #   workshop scenes).  The ResNet18 backbone, however, was
        #   pretrained on ImageNet and has frozen BatchNorm statistics
        #   calibrated for ImageNet-normalized inputs (mean=[0.485,
        #   0.456, 0.406], std=[0.229, 0.224, 0.225]).
        #
        #   The mismatch means the backbone receives values in
        #   approximately [-6.2, +0.7] instead of the expected
        #   [-2.1, +2.6].  The frozen BatchNorm running_mean/var are
        #   wrong for this distribution, producing corrupted feature
        #   maps that cascade through SpatialSoftmax into garbage
        #   action predictions.
        #
        #   This is especially fatal at 224x224 because the 7x7
        #   feature map gives SpatialSoftmax 49 spatial locations
        #   (vs 9 at 84x84).  The finer grid needs accurate features
        #   -- corrupted features cause keypoints to select spurious
        #   locations, leading to oscillating actions near objects.
        #
        # Fix:
        #   Switch VISUAL to IDENTITY (pass-through) so images arrive
        #   at the encoder in raw [0, 1].  ImageNet normalization is
        #   then applied inside DiffusionRgbEncoder.forward() via the
        #   _patch_diffusion_rgb_encoder() monkey-patch below.
        #
        #   pipeline after fix:
        #     [0,1] -> IDENTITY -> ImageNet norm -> ResNet18 
        # --------------------------------------------------------------
        from lerobot.configs.types import NormalizationMode

        is_diffusion_based = (
            policy_cfg.type == "diffusion" or policy_cfg.type in EMIX_POLICY_REGISTRATIONS
        )
        norm_map = getattr(policy_cfg, "normalization_mapping", None)
        if (
            is_diffusion_based
            and norm_map is not None
            and norm_map.get("VISUAL") == NormalizationMode.MEAN_STD
        ):
            norm_map["VISUAL"] = NormalizationMode.IDENTITY
            LOGGER.debug(
                "[bootstrap] Overrode VISUAL normalization to IDENTITY for %s", policy_cfg.type
            )

        reg = EMIX_POLICY_REGISTRATIONS.get(policy_cfg.type)
        if reg is not None:
            processor_fn = _load_symbol(reg.processor_import, reg.processor_name)
            return processor_fn(policy_cfg, dataset_stats=dataset_stats)

        name = policy_cfg.type
        if name == "act":
            from lerobot.policies.act.processor_act import make_act_pre_post_processors

            return make_act_pre_post_processors(policy_cfg, dataset_stats=dataset_stats)
        if name == "diffusion":
            from lerobot.policies.diffusion.processor_diffusion import (
                make_diffusion_pre_post_processors,
            )

            return make_diffusion_pre_post_processors(policy_cfg, dataset_stats=dataset_stats)
        if name == "smolvla":
            from lerobot.policies.smolvla.processor_smolvla import (
                make_smolvla_pre_post_processors,
            )

            return make_smolvla_pre_post_processors(policy_cfg, dataset_stats=dataset_stats)

        raise NotImplementedError(f"Processor factory for policy type '{name}' is not implemented.")

    return make_pre_post_processors


def _patch_policy_factory() -> None:
    from torch import nn

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features

    factory_mod = types.ModuleType("lerobot.policies.factory")

    builtin_configs = {
        "act": ("lerobot.policies.act.configuration_act", "ACTConfig"),
        "diffusion": ("lerobot.policies.diffusion.configuration_diffusion", "DiffusionConfig"),
        "smolvla": ("lerobot.policies.smolvla.configuration_smolvla", "SmolVLAConfig"),
    }
    builtin_models = {
        "act": ("lerobot.policies.act.modeling_act", "ACTPolicy"),
        "diffusion": ("lerobot.policies.diffusion.modeling_diffusion", "DiffusionPolicy"),
        "smolvla": ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
    }

    def get_policy_class(name: str):
        reg = EMIX_POLICY_REGISTRATIONS.get(name)
        if reg is not None:
            return _load_symbol(reg.model_import, reg.model_name)
        if name in builtin_models:
            return _load_symbol(*builtin_models[name])
        raise NotImplementedError(f"Policy with name {name} is not implemented.")

    def make_policy_config(policy_type: str, **kwargs):
        reg = EMIX_POLICY_REGISTRATIONS.get(policy_type)
        if reg is not None:
            config_cls = _load_symbol(reg.config_import, reg.config_name)
            return config_cls(**kwargs)
        if policy_type in builtin_configs:
            config_cls = _load_symbol(*builtin_configs[policy_type])
            return config_cls(**kwargs)
        raise ValueError(f"Policy type '{policy_type}' is not available.")

    def make_policy(cfg: PreTrainedConfig, ds_meta=None, env_cfg=None):
        if bool(ds_meta) == bool(env_cfg):
            raise ValueError("Either one of a dataset metadata or a sim env must be provided.")

        policy_cls = get_policy_class(cfg.type)
        kwargs: dict[str, Any] = {"config": cfg}
        if ds_meta is not None:
            features = dataset_to_policy_features(ds_meta.features)
            kwargs["dataset_stats"] = ds_meta.stats
        else:
            from lerobot.envs.utils import env_to_policy_features

            features = env_to_policy_features(env_cfg)

        cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
        if getattr(cfg, "input_keys", None) is not None:
            cfg.input_features = {key: ft for key, ft in cfg.input_features.items() if key in cfg.input_keys}

        if getattr(cfg, "pretrained_path", None):
            kwargs["pretrained_name_or_path"] = cfg.pretrained_path
            policy = policy_cls.from_pretrained(**kwargs)
        else:
            policy = policy_cls(**kwargs)

        policy.to(cfg.device)
        assert isinstance(policy, nn.Module)
        return policy

    factory_mod.get_policy_class = get_policy_class
    factory_mod.make_policy_config = make_policy_config
    factory_mod.make_pre_post_processors = _build_make_pre_post_processors()
    factory_mod.make_policy = make_policy
    sys.modules["lerobot.policies.factory"] = factory_mod


def _patch_scripts_if_present() -> None:
    for module_name in (
        "lerobot.scripts.lerobot_train",
        "lerobot.scripts.lerobot_eval",
        "lerobot.scripts.lerobot_record",
    ):
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue

        factory_mod = importlib.import_module("lerobot.policies.factory")
        if hasattr(mod, "make_policy"):
            mod.make_policy = factory_mod.make_policy
        if hasattr(mod, "make_pre_post_processors"):
            mod.make_pre_post_processors = factory_mod.make_pre_post_processors


def _patch_diffusion_rgb_encoder() -> None:
    """Fix DiffusionRgbEncoder incompatibilities for Diffusion Policy & EmixDP.

    IMPORTANT: These patches only affect "diffusion" and "emix_*" policy types
    (they patch DiffusionRgbEncoder which is exclusive to those policies).
    ACT and SmolVLA use different encoders and are left untouched.

    Fix 1 (always-on): ImageNet normalization in forward().
      The ResNet18 backbone uses ImageNet-pretrained weights with frozen
      BatchNorm statistics.  LeRobot's default preprocessor applies
      dataset-specific MEAN_STD normalization instead of ImageNet
      normalization, causing a distribution mismatch.  This patch adds
      ImageNet normalization at the start of forward() so the backbone
      receives values in its expected range.

    Fix 2 (conditional): ReLU -> Identity for input resolutions > ~84x84.
      LeRobot's Diffusion Policy supports a range of input resolutions, but
      its defaults are tuned for ~84x84 inputs.  At that resolution the
      backbone produces a 3x3 feature map (9 locations).  SpatialSoftmax
      with temperature=1.0 over 9 positions produces keypoint coordinates
      with large magnitude that survive the downstream ReLU.

      At higher resolutions (e.g. 224x224 -> 7x7 feature map, 49 locations),
      the same temperature=1.0 softmax is much softer, producing keypoint
      coordinates with very small magnitude (std ~ 0.03).  The downstream
      Linear(64,64) + ReLU then produces mostly negative values -> ReLU zeros
      them out -> encoder outputs nearly identical features for any image.

      This patch conditionally replaces ReLU with Identity ONLY when the
      SpatialSoftmax grid exceeds 3x3 (i.e. input > ~84x84).  At 84x84 and
      below, the original ReLU behavior is preserved.
    """
    import torch

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder

    # --- Fix 1 (always-on): ImageNet normalization in forward() ---

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

    _original_forward = DiffusionRgbEncoder.forward

    def _patched_forward(self, x):
        # x is (B, C, H, W) with pixel values in [0, 1].
        mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = IMAGENET_STD.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        x = (x - mean) / std

        # Debug: capture the normalized input image (read-only, no training impact).
        # Controlled by EMIX_DEBUG_ENCODER=1 env var; a no-op when disabled.
        from embodied_mixup.policies.utils import debug_encoder as _dbg

        _dbg.capture_input(id(self), x)

        return _original_forward(self, x)

    DiffusionRgbEncoder.forward = _patched_forward

    # --- Fix 2 (conditional): ReLU -> Identity for > ~84x84 inputs ---

    _original_init = DiffusionRgbEncoder.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)

        # SpatialSoftmax grid dimensions: 3x3 at 84x84, 7x7 at 224x224, etc.
        grid_h = getattr(self.pool, "_in_h", 0)
        grid_w = getattr(self.pool, "_in_w", 0)

        if grid_h > 3 or grid_w > 3:
            # > 84x84 -- ReLU will zero out most of the Linear(64,64) output
            # because SpatialSoftmax keypoint coordinates are too soft.
            self.relu = torch.nn.Identity()

            # Resolve the approximate input image size for the warning.
            # Backbone downsamples by 32x (conv1/2 + maxpool/2 + layer2/2 + layer3/2 + layer4/2).
            approx_h = grid_h * 32
            approx_w = grid_w * 32

            warning = (
                f"\n{'=' * 72}\n"
                f"  [WARNING] LEROBOT RESOLUTION COMPATIBILITY\n"
                f"  {'-' * 60}\n"
                f"  Policy type          : diffusion / emix_diffusion\n"
                f"  Input resolution     : ~{approx_h}x{approx_w}\n"
                f"  SpatialSoftmax grid  : {grid_h}x{grid_w} ({grid_h * grid_w} locations)\n"
                f"  ReLU replaced with   : Identity\n"
                f"  {'-' * 60}\n"
                f"  LeRobot's DiffusionRgbEncoder defaults are tuned for ~84x84\n"
                f"  inputs (SpatialSoftmax on a 3x3 grid).  It does support higher\n"
                f"  resolutions, but at those the softmax over more spatial locations\n"
                f"  produces keypoint coordinates that are too small -- the downstream\n"
                f"  Linear+ReLU zeros out most dimensions, causing the encoder to output\n"
                f"  nearly identical features for any input image.  Higher resolutions\n"
                f"  therefore need the additional adaptation applied below.\n"
                f"  {'-' * 60}\n"
                f"  Fix automatically applied: ReLU -> Identity.\n"
                f"  No action needed.  At 84x84 this patch is a no-op.\n"
                f"{'=' * 72}"
            )
            LOGGER.warning(warning)

        # --- Debug hook attachment (read-only, no training impact) ---
        # Attached unconditionally (all resolutions) so that 84x84 training
        # can also be inspected.  When EMIX_DEBUG_ENCODER=1, forward hooks
        # capture intermediate activations from this encoder instance.  Hooks
        # run with torch.no_grad() + .detach().clone() -- gradients and
        # computation graph are never touched.  File I/O only occurs at the
        # configured interval (default: every 10 000 steps), so the throughput
        # impact is negligible.  When EMIX_DEBUG_ENCODER is not set, attach()
        # is a no-op.
        from embodied_mixup.policies.utils import debug_encoder as _dbg

        _dbg.attach(self)

    DiffusionRgbEncoder.__init__ = _patched_init

    LOGGER.info(
        "[bootstrap] Patched DiffusionRgbEncoder.forward with ImageNet normalization "
        "(diffusion / emix_diffusion policies only)"
    )


_BOOTSTRAPPED: bool = False


def bootstrap_lerobot_runtime() -> None:
    """Patch the local lerobot runtime so Emix policies work in-process.

    Idempotent — repeated calls are no-ops.  This is critical because
    ``_patch_diffusion_rgb_encoder()`` wraps ``DiffusionRgbEncoder``
    methods (e.g.  forward / __init__), and each additional wrap layer
    would compound ImageNet normalization and re-emit warnings.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True

    _patch_lerobot_dataset_utils()
    _patch_policy_factory()
    _patch_scripts_if_present()
    _patch_diffusion_rgb_encoder()

    # Force-import config modules so PreTrainedConfig registry sees emix_* types.
    for reg in EMIX_POLICY_REGISTRATIONS.values():
        importlib.import_module(reg.config_import)


def maybe_bootstrap_from_policy_type(policy_type: str | None) -> None:
    """Bootstrap only when an Emix policy type is requested."""

    if policy_type and is_emix_policy_type(policy_type):
        bootstrap_lerobot_runtime()
