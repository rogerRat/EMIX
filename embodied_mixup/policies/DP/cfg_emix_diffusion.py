"""Emix Diffusion config."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@DiffusionConfig.register_subclass("emix_diffusion")
@dataclass
class EmixDiffusionConfig(DiffusionConfig):
    """Diffusion config extended for continuous-action Emix training."""

    emix_index_path: str = ""
    mixup_beta: float = 1.6
    mixup_alpha_mode: str = "beta"
    mixup_alpha_value: float | None = None
    mixup_loss_weight: float = 1.0
    mixup_visual_keys: list[str] | None = None
    mixup_state_mode: str = "anchor"
    mixup_image_mode: str = "linear"
    annotation_root: str = ""
    detect_overlap_enable: bool = False
    detect_overlap_refuse_ratio: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mixup_alpha_mode not in ("beta", "fixed"):
            raise ValueError(f"mixup_alpha_mode must be 'beta' or 'fixed', got {self.mixup_alpha_mode}")
        if self.mixup_beta <= 0.0:
            raise ValueError(f"mixup_beta must be positive, got {self.mixup_beta}")
        if self.mixup_alpha_mode == "fixed":
            if self.mixup_alpha_value is None:
                raise ValueError("mixup_alpha_value is required when mixup_alpha_mode='fixed'")
            if not (0.0 < self.mixup_alpha_value < 1.0):
                raise ValueError(f"mixup_alpha_value must be in (0.0, 1.0), got {self.mixup_alpha_value}")
        if self.mixup_loss_weight <= 0:
            raise ValueError(f"mixup_loss_weight must be positive, got {self.mixup_loss_weight}")
        if self.mixup_state_mode != "anchor":
            raise ValueError(f"Only mixup_state_mode='anchor' is supported, got {self.mixup_state_mode}")
        if self.mixup_image_mode not in ("linear", "patch"):
            raise ValueError("mixup_image_mode must be 'linear' or 'patch'")
        if self.mixup_image_mode == "patch" and not self.annotation_root:
            raise ValueError("annotation_root is required when mixup_image_mode='patch'")
        if not 0.0 <= self.detect_overlap_refuse_ratio <= 1.0:
            raise ValueError("detect_overlap.refuse_ratio must be in [0.0, 1.0]")
        if self.detect_overlap_enable and self.mixup_image_mode != "patch":
            raise ValueError("detect_overlap can only be enabled with patch image mode")

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError("Emix diffusion requires a robot state input.")
