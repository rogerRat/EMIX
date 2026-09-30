"""Emix ACT config."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.policies.act.configuration_act import ACTConfig


@ACTConfig.register_subclass("emix_act")
@dataclass
class EmixACTConfig(ACTConfig):
    """ACT config extended for continuous-action Emix training."""

    emix_index_path: str = ""
    mixup_beta: float = 1.6
    mixup_alpha_mode: str = "beta"
    mixup_alpha_value: float | None = None
    mixup_loss_weight: float = 1.0
    mixup_visual_keys: list[str] | None = None
    mixup_state_mode: str = "anchor"
    mixup_method: str = "overlay"

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
        if self.mixup_method != "overlay":
            raise ValueError("EmixACT currently supports only Emix-overlay")

    def validate_features(self) -> None:
        if not self.image_features and self.env_state_feature is None and self.robot_state_feature is None:
            raise ValueError(
                "Emix ACT requires at least one image, robot state, or environment state input."
            )
