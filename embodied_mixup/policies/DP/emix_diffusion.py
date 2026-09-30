"""Emix Diffusion policy with weighted dual-branch score matching."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.utils.constants import OBS_IMAGES
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from .cfg_emix_diffusion import EmixDiffusionConfig
from ..utils.batch import (
    assert_single_observation_batch,
    assert_single_observation_config,
)


def combine_emix_losses(loss_0: Tensor, loss_1: Tensor, valid_mask: Tensor,
                        alpha: Tensor, method: str) -> Tensor:
    """Average valid dimensions; collage keeps the full sum of both losses."""
    if method == "collage":
        mixed_loss = loss_0 + loss_1
    elif method == "overlay":
        alpha = alpha.to(dtype=loss_0.dtype).view(-1, 1, 1)
        mixed_loss = alpha * loss_0 + (1.0 - alpha) * loss_1
    else:
        raise ValueError(f"Unknown EMIX method: {method}")
    mask = valid_mask.unsqueeze(-1).to(dtype=loss_0.dtype)
    denominator = mask.expand_as(mixed_loss).sum().clamp_min(1.0)
    return (mixed_loss * mask).sum() / denominator


class EmixDiffusionPolicy(DiffusionPolicy):
    """Diffusion policy with shared-condition dual-branch training loss."""

    config_class = EmixDiffusionConfig
    name = "emix_diffusion"

    def __init__(self, config: EmixDiffusionConfig, **kwargs):
        assert_single_observation_config(config, context="EmixDiffusionPolicy.__init__")
        super().__init__(config, **kwargs)

    def _compute_emix_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        assert set(batch).issuperset(
            {"observation.state", "emix_actions", "action_is_pad"}
        )
        global_cond = self.diffusion._prepare_global_conditioning(batch)

        emix_actions = batch["emix_actions"]
        action_0 = emix_actions[0]
        action_1 = emix_actions[1]
        eps = torch.randn(action_0.shape, device=action_0.device)
        timesteps = torch.randint(
            low=0,
            high=self.diffusion.noise_scheduler.config.num_train_timesteps,
            size=(action_0.shape[0],),
            device=action_0.device,
        ).long()

        noisy_0 = self.diffusion.noise_scheduler.add_noise(action_0, eps, timesteps)
        noisy_1 = self.diffusion.noise_scheduler.add_noise(action_1, eps, timesteps)

        pred_0 = self.diffusion.unet(noisy_0, timesteps, global_cond=global_cond)
        pred_1 = self.diffusion.unet(noisy_1, timesteps, global_cond=global_cond)

        if self.config.prediction_type == "epsilon":
            target_0 = eps
            target_1 = eps
        elif self.config.prediction_type == "sample":
            target_0 = action_0
            target_1 = action_1
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss_0 = F.mse_loss(pred_0, target_0, reduction="none")
        loss_1 = F.mse_loss(pred_1, target_1, reduction="none")

        return combine_emix_losses(loss_0, loss_1, ~batch["action_is_pad"],
                                   batch["emix_alpha"], self.config.mixup_method), {
            "emix_loss_weight": float(self.config.mixup_loss_weight),
        }

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        assert_single_observation_batch(batch, context="EmixDiffusionPolicy.forward")
        if self.config.image_features and all(key in batch for key in self.config.image_features):
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        if "emix_actions" not in batch:
            return super().forward(batch)

        loss, loss_dict = self._compute_emix_loss(batch)
        return loss * self.config.mixup_loss_weight, loss_dict

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise=None, **kwargs):
        assert_single_observation_batch(batch, context="EmixDiffusionPolicy.select_action")
        return super().select_action(batch, noise=noise, **kwargs)
