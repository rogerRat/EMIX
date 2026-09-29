"""
Read-only debug capture for DiffusionRgbEncoder internals.

Configuration (priority: env var > training YAML > default):

  # Preferred: training YAML config file
  encoder_debug:
    enabled: true          # enable / disable
    interval: 10000        # capture every N training steps
    max_samples: 4         # max batch samples per capture
    output_dir: ./debug_encoder   # output directory

  # Alternative: environment variables (override YAML)
  EMIX_DEBUG_ENCODER=1                enable debug capture
  EMIX_DEBUG_ENCODER_DIR=...          output directory
  EMIX_DEBUG_ENCODER_INTERVAL=10000   capture interval
  EMIX_DEBUG_ENCODER_MAX_SAMPLES=4    max samples per capture

What is captured at each debug step:
  1. Input images (after ImageNet normalization, denormalized back to [0,1] for viewing)
  2. Backbone feature map (after ResNet18, before SpatialSoftmax)
  3. SpatialSoftmax keypoint coordinates (where the model is "looking")
  4. Linear projection output (before activation)
  5. Final encoder output (after activation: ReLU or Identity)

All captures use torch.no_grad() and .detach().clone() -- gradients are not
touched, tensors are not modified, and the computation graph is unaffected.
File I/O is synchronous but only occurs at the debug interval (e.g. every
10 000 steps), so the impact on training throughput is negligible.

Output structure (one directory per captured step):
  {output_dir}/
    step_{step:08d}/
      metadata.json          # step, timestamp, grid_h, grid_w, has_relu, shapes
      input_sample_{i:02d}.png     # denormalized input images
      backbone_fmap.npy      # (N, 512, H, W) first N samples
      pool_keypoints.npy     # (N, 32, 2)
      linear_out.npy         # (N, 64) before activation
      final_out.npy          # (N, 64) after activation

Design rationale:
  This module is intentionally decoupled from the bootstrap monkey-patches.
  The bootstrap calls attach() once per encoder instance; hooks fire
  automatically during forward(); train.py calls set_step() each iteration
  and configure() once at startup from the training YAML config.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment-variable gating
# ---------------------------------------------------------------------------

_ENABLED: bool | None = None  # cached so we only read the env var once


def is_enabled() -> bool:
    """Return True when debug capture is active (cached).

    Priority: EMIX_DEBUG_ENCODER env var > YAML ``enabled: true`` > default (False).
    """
    global _ENABLED
    if _ENABLED is None:
        env_val = os.environ.get("EMIX_DEBUG_ENCODER")
        if env_val is not None:
            _ENABLED = env_val == "1"
        elif _CONFIG_ENABLED is not None:
            _ENABLED = _CONFIG_ENABLED
        else:
            _ENABLED = False
    return _ENABLED


def get_output_dir() -> Path:
    """Directory that will contain per-step debug sub-directories.

    Priority: EMIX_DEBUG_ENCODER_DIR env var > YAML ``output_dir`` > default.
    """
    env_val = os.environ.get("EMIX_DEBUG_ENCODER_DIR")
    if env_val is not None:
        return Path(env_val)
    if _CONFIG_OUTPUT_DIR is not None:
        return Path(_CONFIG_OUTPUT_DIR)
    return Path("./debug_encoder")


def get_interval() -> int:
    """Number of training steps between captures.

    Priority: EMIX_DEBUG_ENCODER_INTERVAL env var > YAML ``interval`` > default (10000).
    """
    env_val = os.environ.get("EMIX_DEBUG_ENCODER_INTERVAL")
    if env_val is not None:
        return int(env_val)
    if _CONFIG_INTERVAL is not None:
        return _CONFIG_INTERVAL
    return 10000


def get_max_samples() -> int:
    """Maximum number of batch samples to persist per capture.

    Priority: EMIX_DEBUG_ENCODER_MAX_SAMPLES env var > YAML ``max_samples`` > default (4).
    """
    env_val = os.environ.get("EMIX_DEBUG_ENCODER_MAX_SAMPLES")
    if env_val is not None:
        return int(env_val)
    if _CONFIG_MAX_SAMPLES is not None:
        return _CONFIG_MAX_SAMPLES
    return 4


# ---------------------------------------------------------------------------
# Configuration from training YAML (env vars take precedence)
# ---------------------------------------------------------------------------

# Cached values that configure() fills in.  Any env var overrides are
# resolved at access time via get_interval() / get_output_dir() / etc.
_CONFIG_ENABLED: bool | None = None
_CONFIG_OUTPUT_DIR: str | None = None
_CONFIG_INTERVAL: int | None = None
_CONFIG_MAX_SAMPLES: int | None = None


def configure(config: dict | None) -> None:
    """Apply debug settings from the training YAML ``encoder_debug`` section.

    Called once at training startup.  Environment variables always take
    precedence over YAML values so that ad-hoc overrides work without
    editing the config file.  When *config* is None or empty the module
    behaves as if the YAML section was absent (env vars only).

    Example YAML:
        encoder_debug:
          enabled: true
          interval: 10000
          max_samples: 4
          output_dir: outputs/train/my_experiment/encoder_debug
    """
    global _ENABLED, _CONFIG_ENABLED
    global _CONFIG_OUTPUT_DIR, _CONFIG_INTERVAL, _CONFIG_MAX_SAMPLES

    if config is None:
        return

    cfg_enabled = config.get("enabled", None)
    if cfg_enabled is not None:
        _CONFIG_ENABLED = bool(cfg_enabled)

    cfg_dir = config.get("output_dir", None)
    if cfg_dir is not None:
        _CONFIG_OUTPUT_DIR = str(cfg_dir)

    cfg_interval = config.get("interval", None)
    if cfg_interval is not None:
        _CONFIG_INTERVAL = int(cfg_interval)

    cfg_max = config.get("max_samples", None)
    if cfg_max is not None:
        _CONFIG_MAX_SAMPLES = int(cfg_max)

    # Invalidate the enabled cache so it picks up the new config value.
    _ENABLED = None

    LOGGER.info(
        "[debug_encoder] Configured from YAML: enabled=%s interval=%s max_samples=%s output_dir=%s",
        is_enabled(),
        get_interval(),
        get_max_samples(),
        get_output_dir(),
    )


# ---------------------------------------------------------------------------
# Global step counter (set by the training loop)
# ---------------------------------------------------------------------------

_step: int = 0


def set_step(step: int) -> None:
    """Record the current training step.

    Called once per optimizer step from train.py.  When debug is disabled
    this is a near-zero-cost no-op (one integer assignment).
    """
    global _step
    _step = step


def get_step() -> int:
    """Return the last step value set by the training loop."""
    return _step


# ---------------------------------------------------------------------------
# In-progress capture buffers (keyed by encoder id)
# ---------------------------------------------------------------------------

# Each encoder gets one entry in _BUFFERS.  Forward hooks push intermediates
# into the entry; the 'relu' (or 'final') hook triggers flush to disk.
_BUFFERS: dict[int, dict[str, torch.Tensor]] = {}


def _capture(encoder_id: int, name: str, tensor: torch.Tensor) -> None:
    """Record one intermediate tensor for *encoder_id*.

    This is called from forward hooks registered on encoder submodules.
    Using torch.no_grad() + detach().cpu().clone() guarantees zero impact
    on the computation graph.
    """
    if not is_enabled():
        return
    if get_step() % get_interval() != 0:
        return

    if encoder_id not in _BUFFERS:
        _BUFFERS[encoder_id] = {}

    # Only keep the first max_samples along the batch dimension.
    max_n = get_max_samples()
    t = tensor.detach().cpu().clone()[:max_n]
    _BUFFERS[encoder_id][name] = t


def _finalize(encoder_id: int) -> None:
    """Called after the last submodule hook fires; saves everything to disk."""
    if encoder_id not in _BUFFERS:
        return
    data = _BUFFERS.pop(encoder_id)
    if not data:
        return
    _save_capture(get_step(), data)


# ---------------------------------------------------------------------------
# Submodule hook factory
# ---------------------------------------------------------------------------

def make_hook(encoder_id: int, name: str):
    """Return a forward hook that captures *name* output for *encoder_id*."""

    def _hook(_module: torch.nn.Module, _input: Any, output: torch.Tensor) -> None:
        with torch.no_grad():
            _capture(encoder_id, name, output)
        # The "final" submodule can be either relu or Identity.  We detect
        # the last capture by checking if we've already captured "out"
        # (linear) -- if so, this hook corresponds to the activation layer.
        if name in ("relu", "identity"):
            _finalize(encoder_id)

    return _hook


# ---------------------------------------------------------------------------
# Public: attach debug hooks to a DiffusionRgbEncoder instance
# ---------------------------------------------------------------------------

# ImageNet constants used to reverse the normalization for visual inspection.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def attach(encoder: "torch.nn.Module") -> None:
    """Attach read-only forward hooks to *encoder* submodules.

    Must be called after the encoder is fully constructed (i.e. at the end
    of DiffusionRgbEncoder.__init__).  When debug is disabled this function
    is a no-op.

    The hooks fire in the following order during forward():
      backbone -> pool (SpatialSoftmax) -> out (Linear) -> relu (or Identity)

    Hooks are registered with no gradient tracking, and captured tensors are
    immediately detached and copied to CPU so the GPU computation graph is
    never referenced by the debug buffer.
    """
    if not is_enabled():
        return

    enc_id = id(encoder)

    encoder.backbone.register_forward_hook(make_hook(enc_id, "backbone"))
    encoder.pool.register_forward_hook(make_hook(enc_id, "pool"))
    encoder.out.register_forward_hook(make_hook(enc_id, "out"))

    # The activation submodule may be ReLU (default) or Identity (patched).
    encoder.relu.register_forward_hook(make_hook(enc_id, "relu"))

    LOGGER.info(
        "[debug_encoder] Attached read-only hooks to DiffusionRgbEncoder(id=%s). "
        "Captures every %s steps to %s",
        enc_id,
        get_interval(),
        get_output_dir(),
    )


# ---------------------------------------------------------------------------
# Capture the input image (called from the patched forward)
# ---------------------------------------------------------------------------

def capture_input(encoder_id: int, image_tensor: torch.Tensor) -> None:
    """Capture the ImageNet-normalized input image.

    Called from the patched DiffusionRgbEncoder.forward() right after
    ImageNet normalization is applied.  *image_tensor* is (B, 3, H, W)
    with values roughly in [-2.1, +2.6].
    """
    if not is_enabled():
        return
    if get_step() % get_interval() != 0:
        return

    max_n = get_max_samples()
    with torch.no_grad():
        _BUFFERS.setdefault(encoder_id, {})["input_normed"] = (
            image_tensor.detach().cpu().clone()[:max_n]
        )


# ---------------------------------------------------------------------------
# Save captured data to disk
# ---------------------------------------------------------------------------

def _denormalize_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization and convert [0,1] float -> uint8 [0,255].

    Args:
        tensor: (N, 3, H, W) float32 tensor in ImageNet-normalized range.
    Returns:
        (N, H, W, 3) uint8 numpy array in [0, 255].
    """
    # tensor shape: (N, 3, H, W)
    arr = tensor.numpy()  # (N, 3, H, W)
    # Broadcast per-channel reverse normalization.
    mean = _IMAGENET_MEAN.reshape(1, 3, 1, 1)
    std = _IMAGENET_STD.reshape(1, 3, 1, 1)
    arr = arr * std + mean  # back to [0, 1] (approximately)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    # Convert from (N, 3, H, W) to (N, H, W, 3) for PIL.
    arr = arr.transpose(0, 2, 3, 1)
    return arr


def _save_capture(step: int, data: dict[str, torch.Tensor]) -> None:
    """Persist one set of captured intermediates to disk.

    Directory layout:
      {output_dir}/step_{step:08d}/
    """
    out_dir = get_output_dir() / f"step_{step:08d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- input images (denormalized for visual inspection) ---
    if "input_normed" in data:
        imgs = _denormalize_to_uint8(data["input_normed"])  # (N, H, W, 3)
        for i in range(imgs.shape[0]):
            img_path = out_dir / f"input_sample_{i:02d}.png"
            Image.fromarray(imgs[i]).save(str(img_path))
        LOGGER.info("[debug_encoder] Saved %d input images to %s", imgs.shape[0], out_dir)

    # --- backbone feature map ---
    if "backbone" in data:
        np.save(str(out_dir / "backbone_fmap.npy"), data["backbone"].numpy())

    # --- SpatialSoftmax keypoints ---
    if "pool" in data:
        np.save(str(out_dir / "pool_keypoints.npy"), data["pool"].numpy())

    # --- linear output (before activation) ---
    if "out" in data:
        np.save(str(out_dir / "linear_out.npy"), data["out"].numpy())

    # --- final output (after activation) ---
    if "relu" in data:
        np.save(str(out_dir / "final_out.npy"), data["relu"].numpy())

    # --- metadata ---
    metadata = {
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "captured_tensors": sorted(data.keys()),
        "shapes": {k: list(v.shape) for k, v in data.items()},
    }
    # Add per-channel stats for backbone feature map and final output.
    if "backbone" in data:
        b = data["backbone"]  # (N, C, H, W)
        metadata["backbone_stats"] = {
            "channel_mean": float(b.mean().item()),
            "channel_std": float(b.std().item()),
            "channel_min": float(b.min().item()),
            "channel_max": float(b.max().item()),
        }
    if "relu" in data:
        f = data["relu"]  # (N, 64)
        # Count near-zero dimensions (|val| < 1e-5) -- a symptom of ReLU collapse.
        near_zero = (f.abs() < 1e-5).float().mean().item()
        metadata["final_out_stats"] = {
            "mean": float(f.mean().item()),
            "std": float(f.std().item()),
            "min": float(f.min().item()),
            "max": float(f.max().item()),
            "fraction_near_zero": near_zero,
            "num_dims": f.shape[-1],
        }
    if "pool" in data:
        kps = data["pool"]  # (N, 32, 2)
        metadata["pool_keypoints_stats"] = {
            "x_mean": float(kps[:, :, 0].mean().item()),
            "x_std": float(kps[:, :, 0].std().item()),
            "y_mean": float(kps[:, :, 1].mean().item()),
            "y_std": float(kps[:, :, 1].std().item()),
        }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    LOGGER.info(
        "[debug_encoder] Capture complete for step=%d -> %s (metadata.json + %d files)",
        step,
        out_dir,
        len(data),
    )
