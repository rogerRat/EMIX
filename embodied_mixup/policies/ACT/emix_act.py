"""Emix ACT policy using weighted dual-branch continuous reconstruction loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.act.modeling_act import ACTPolicy

from .cfg_emix_act import EmixACTConfig


class EmixACTPolicy(ACTPolicy):
    """ACT with Emix weighted dual-branch L1 supervision."""

    config_class = EmixACTConfig
    name = "emix_act"

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if OBS_IMAGES not in batch and OBS_ENV_STATE not in batch and OBS_STATE in batch:
            batch = dict(batch)
            batch[OBS_ENV_STATE] = batch[OBS_STATE]

        if self.config.image_features and all(key in batch for key in self.config.image_features):
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
        valid_mask = (~batch["action_is_pad"]).unsqueeze(-1).to(dtype=actions_hat.dtype)
        alpha = batch["emix_alpha"].to(dtype=actions_hat.dtype).view(-1, 1, 1)

        emix_actions = batch["emix_actions"]
        loss_0 = F.l1_loss(emix_actions[0], actions_hat, reduction="none")
        loss_1 = F.l1_loss(emix_actions[1], actions_hat, reduction="none")
        mixed_l1 = ((alpha * loss_0) + ((1.0 - alpha) * loss_1)) * valid_mask
        denom = valid_mask.expand_as(mixed_l1).sum().clamp_min(1.0)
        l1_loss = mixed_l1.sum() / denom

        loss_dict = {
            "mix_l1_loss": float(l1_loss.item()),
            "emix_loss_weight": float(self.config.mixup_loss_weight),
        }
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = float(mean_kld.item())
            loss = (l1_loss + (mean_kld * self.config.kl_weight)) * self.config.mixup_loss_weight
        else:
            loss = l1_loss * self.config.mixup_loss_weight

        return loss, loss_dict
