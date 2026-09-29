"""
Batch assembly helpers for Embodied Mixup continuous-action training.

Used to assemble mini-batches for training.

The DataLoader provides a list of samples (each sample is a dictionary).
This module collates them into a batch, stacking tensor fields while
preserving non-tensor fields (e.g. task strings), so that the resulting
batch can be directly consumed by policy.forward().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PolicyTemporalSpec:
    """Time-shape requirements extracted from a policy config."""

    n_obs_steps: int
    action_steps: int
    action_key: str = "action"
    action_pad_key: str = "action_is_pad"


def assert_single_observation_config(policy_cfg: Any, *, context: str) -> None:
    """Enforce the current point-observation contract.

    @input: policy_cfg policy configuration; context identifies the failing
        construction path.
    @output: None on success; raises AssertionError otherwise.
    @scenario: Prevent a temporal observation window from silently changing
        the semantics of point-based EMIX samples.
    """

    n_obs_steps = int(getattr(policy_cfg, "n_obs_steps", 1))
    if n_obs_steps != 1:
        raise AssertionError(
            f"{context}: policies must use n_obs_steps=1 (current frame only), "
            f"got policy.type={getattr(policy_cfg, 'type', '<unknown>')!r}, "
            f"n_obs_steps={n_obs_steps}."
        )


def assert_single_observation_batch(batch: dict[str, Any], *, context: str) -> None:
    """Assert that a policy batch contains at most one observation frame.

    Batched training tensors use ``(B, 1, ...)`` while single-environment
    inference tensors use ``(B, ...)``. Both represent exactly one frame.
    Action tensors are intentionally not checked here because future action
    windows are valid supervision targets, not policy observations.
    """

    state = batch.get("observation.state")
    if not isinstance(state, torch.Tensor):
        raise AssertionError(f"{context}: missing tensor observation.state.")
    if state.ndim not in (2, 3):
        raise AssertionError(
            f"{context}: observation.state must have shape (B,D) or (B,1,D), got {tuple(state.shape)}."
        )
    if state.ndim == 3 and state.shape[1] != 1:
        raise AssertionError(
            f"{context}: observation.state contains {state.shape[1]} frames; expected exactly 1."
        )

    for key, value in batch.items():
        if not isinstance(value, torch.Tensor) or not key.startswith("observation."):
            continue
        if "language" in key:
            continue
        if key == "observation.state":
            continue
        if key == "observation.images":
            # Combined image tensors use (B,T,N,C,H,W); individual camera
            # tensors use (B,T,C,H,W).
            if value.ndim == 6 and value.shape[1] != 1:
                raise AssertionError(
                    f"{context}: {key} contains {value.shape[1]} frames; expected exactly 1."
                )
            continue
        if key.startswith("observation.images."):
            if value.ndim == 5 and value.shape[1] != 1:
                raise AssertionError(
                    f"{context}: {key} contains {value.shape[1]} frames; expected exactly 1."
                )
            continue
        if value.ndim >= 3 and value.shape[1] != 1:
            raise AssertionError(
                f"{context}: {key} contains {value.shape[1]} frames; expected exactly 1."
            )


def get_policy_temporal_spec(policy_cfg: Any) -> PolicyTemporalSpec:
    """Infer observation and action horizon requirements from a policy config."""

    assert_single_observation_config(policy_cfg, context="get_policy_temporal_spec")

    n_obs_steps = int(getattr(policy_cfg, "n_obs_steps", 1))

    if hasattr(policy_cfg, "horizon"):
        action_steps = int(policy_cfg.horizon)
        action_pad_key = "action_is_pad"
    else:
        action_steps = int(getattr(policy_cfg, "chunk_size", getattr(policy_cfg, "n_action_steps", 1)))
        action_pad_key = "actions_id_pad" if policy_cfg.type in {"smolvla", "emix_smolvla"} else "action_is_pad"

    return PolicyTemporalSpec(
        n_obs_steps=n_obs_steps,
        action_steps=action_steps,
        action_key="action",
        action_pad_key=action_pad_key,
    )


def stack_tensor_list(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Default collate that preserves string task values."""

    batch: dict[str, Any] = {}
    keys = samples[0].keys()
    for key in keys:
        values = [sample[key] for sample in samples]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], list) and values[0] and all(isinstance(v, torch.Tensor) for v in values[0]):
            batch[key] = [
                torch.stack([sample_value[idx] for sample_value in values], dim=0)
                for idx in range(len(values[0]))
            ]
        elif isinstance(values[0], list) and values[0] and all(isinstance(v, dict) for v in values[0]):
            # EMIX source-image metadata is represented as a list of branch
            # dictionaries: [{image_key: tensor}, {image_key: tensor}].
            batch[key] = [
                {
                    image_key: torch.stack(
                        [sample_value[branch_index][image_key] for sample_value in values],
                        dim=0,
                    )
                    for image_key in values[0][branch_index]
                }
                for branch_index in range(len(values[0]))
            ]
        else:
            batch[key] = values
    return batch
