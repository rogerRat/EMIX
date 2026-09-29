"""Explicit additive RGB noise in the decoded [0, 1] pixel domain."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch


@dataclass(frozen=True)
class ImageNoiseConfig:
    enabled: bool = False
    variance: float = 0.0025

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("image_noise.enabled must be a boolean")
        if isinstance(self.variance, bool) or not isinstance(self.variance, (int, float)):
            raise ValueError("image_noise.variance must be a finite nonnegative number")
        if not math.isfinite(self.variance) or self.variance < 0:
            raise ValueError("image_noise.variance must be a finite nonnegative number")

    @classmethod
    def from_payload(cls, payload: Any) -> ImageNoiseConfig:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("image_noise must be a mapping")
        unknown = set(payload) - {"enabled", "variance"}
        if unknown:
            raise ValueError(f"Unknown image_noise setting(s): {sorted(unknown)}")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def add_image_noise(
    batch: dict[str, Any],
    config: ImageNoiseConfig,
    *,
    seed: int,
    step: int,
) -> dict[str, Any]:
    """Add independent N(0, variance) samples to every pixel and RGB channel.

    Call after decoding/resizing/EMIX blending and before policy preprocessing.
    Images must be floating point (..., 3, H, W) in [0, 1]. Clipping keeps the
    valid pixel range; variance describes the Gaussian *before* clipping.
    Other fields and source tensors are preserved. A private RNG indexed by
    training seed and absolute step avoids changing model/dataloader RNG state
    and reproduces noise when a step is resumed with the same input batch.
    Training and evaluation enable this independently in their own configs;
    policy checkpoints never implicitly enable observation noise.
    """
    if not config.enabled or config.variance == 0:
        return batch

    result = dict(batch)
    generators: dict[torch.device, torch.Generator] = {}
    std = math.sqrt(config.variance)
    for key, value in batch.items():
        if not key.startswith("observation.images.") or key.endswith("_is_pad"):
            continue
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f"{key}: image noise requires floating point RGB images in [0, 1]")
        if value.ndim < 3 or value.shape[-3] != 3:
            raise ValueError(f"{key}: expected image shape (..., 3, H, W), got {tuple(value.shape)}")
        if value.device not in generators:
            generators[value.device] = torch.Generator(device=value.device).manual_seed(
                (int(seed) + int(step)) % (2**63 - 1)
            )
        noise = torch.randn(
            value.shape, dtype=value.dtype, device=value.device,
            generator=generators[value.device],
        )
        result[key] = (value + std * noise).clamp(0.0, 1.0)
    return result
