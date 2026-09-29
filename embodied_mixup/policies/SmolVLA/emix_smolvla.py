"""Emix SmolVLA policy with weighted dual-branch flow matching loss."""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Any

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from .cfg_emix_smolvla import EmixSmolVLAConfig
from ..utils.batch import (
    assert_single_observation_batch,
    assert_single_observation_config,
)


class EmixSmolVLAPolicy(SmolVLAPolicy):
    """SmolVLA with shared-condition dual-branch continuous FM supervision."""

    config_class = EmixSmolVLAConfig
    name = "emix_smolvla"

    def __init__(self, config: EmixSmolVLAConfig, **kwargs):
        assert_single_observation_config(config, context="EmixSmolVLAPolicy.__init__")
        super().__init__(config, **kwargs)

    def _tokenize_task(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        tokens = batch.get(OBS_LANGUAGE_TOKENS)
        masks = batch.get(OBS_LANGUAGE_ATTENTION_MASK)
        if tokens is not None and masks is not None:
            return tokens, masks

        task = batch.get("task")
        if task is None:
            raise KeyError("SmolVLA training requires either language tokens or a task string.")

        if isinstance(task, str):
            tasks = [task]
        else:
            tasks = list(task)

        tasks = [text if text.endswith("\n") else f"{text}\n" for text in tasks]
        tokenizer = self.model.vlm_with_expert.processor.tokenizer
        encoded = tokenizer(
            tasks,
            padding=self.config.pad_language_to,
            truncation=True,
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        device = batch[OBS_STATE].device
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device).bool()

    def prepare_images(self, batch: dict[str, Any]):
        present_img_keys = [key for key in self.config.image_features if key in batch]
        if len(present_img_keys) > 0:
            return super().prepare_images(batch)

        if self.config.empty_cameras <= 0:
            raise ValueError(
                f"All image features are missing from the batch and empty_cameras is disabled. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        device = batch[OBS_STATE].device
        batch_size = batch[OBS_STATE].shape[0]
        images = []
        img_masks = []
        for key in list(self.config.image_features.keys())[: self.config.empty_cameras]:
            image_shape = self.config.image_features[key].shape
            images.append(torch.full((batch_size, *image_shape), -1.0, dtype=torch.float32, device=device))
            img_masks.append(torch.zeros(batch_size, dtype=torch.bool, device=device))
        return images, img_masks

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict]:
        assert_single_observation_batch(batch, context="EmixSmolVLAPolicy.forward")
        batch = dict(batch)
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
            if "emix_actions" in batch:
                emix_actions = batch["emix_actions"]
                batch["emix_actions"] = [
                    self._pi_aloha_encode_actions_inv(emix_actions[0]),
                    self._pi_aloha_encode_actions_inv(emix_actions[1]),
                ]

        if OBS_LANGUAGE_TOKENS not in batch or OBS_LANGUAGE_ATTENTION_MASK not in batch:
            lang_tokens, lang_masks = self._tokenize_task(batch)
            batch[OBS_LANGUAGE_TOKENS] = lang_tokens
            batch[OBS_LANGUAGE_ATTENTION_MASK] = lang_masks

        if "emix_actions" not in batch:
            return super().forward(batch, noise=noise, time=time)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK].bool()
        emix_actions = batch["emix_actions"]
        action_0 = self.prepare_action({ACTION: emix_actions[0]})
        action_1 = self.prepare_action({ACTION: emix_actions[1]})
        actions_is_pad = batch.get("actions_id_pad", batch.get("action_is_pad"))

        if noise is None:
            noise = self.model.sample_noise(action_0.shape, action_0.device)
        if time is None:
            time = self.model.sample_time(action_0.shape[0], action_0.device)

        loss_0 = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, action_0, noise, time
        )
        loss_1 = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, action_1, noise, time
        )

        alpha = batch["emix_alpha"].to(dtype=loss_0.dtype).view(-1, 1, 1)
        losses = (alpha * loss_0) + ((1.0 - alpha) * loss_1)

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

        losses = losses[:, :, : self.config.max_action_dim]
        loss = losses.mean() * self.config.mixup_loss_weight
        return loss, {
            "loss": float(loss.item()),
            "emix_loss_weight": float(self.config.mixup_loss_weight),
        }

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise=None, **kwargs):
        assert_single_observation_batch(batch, context="EmixSmolVLAPolicy.select_action")
        return super().select_action(batch, noise=noise, **kwargs)
