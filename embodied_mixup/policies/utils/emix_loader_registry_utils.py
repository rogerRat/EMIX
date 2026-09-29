"""Utilities for direct EmixIndex training loader registration and reading.

@input: source LeRobotDataset roots, EmixIndex paths, and policy type strings.
@output: Index-backed readers and policy configuration records.
@scenario: Resolve policy-specific training batches directly from source dataset
    frames referenced by an EmixIndex.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...candidates.emix_index import EMIX_INDEX_SCHEMA_VERSION, EmixIndex
from ...candidates.emix_index_utils import load_emix_index, validate_source_dataset_fingerprint
from ...datasets import require_lerobot_dataset_root
from .batch import PolicyTemporalSpec

logger = logging.getLogger(__name__)


@dataclass
class EmixLoaderSettings:
    """Policy-independent EMIX loader settings."""

    policy_type: str
    mixup_alpha_mode: str = "beta"
    mixup_alpha_value: float | None = None
    mixup_beta: float = 1.6
    mixup_visual_keys: list[str] | None = None
    mixup_state_mode: str = "anchor"
    seed: int = 0
    include_task: bool = False


@dataclass(frozen=True)
class EmixTrainingSample:
    """One two-member training sample expanded from a group view."""

    group: Any
    member_ids: list[str]
    sample_id: str


class EmixIndexReader:
    """Read EMIX samples directly from a source LeRobotDataset and EmixIndex.

    @input: source_root [str|Path], source_repo_id [str], emix_index_path [str|Path],
        policy_temporal_spec [PolicyTemporalSpec].
    @output: validated reader with access to EmixIndex groups and source frames.
    @scenario: Load EmixIndex and validate source dataset fingerprint before any
        training sample is produced.
    """

    def __init__(
        self,
        source_root: str | Path,
        source_repo_id: str,
        emix_index_path: str | Path,
        policy_temporal_spec: PolicyTemporalSpec,
    ) -> None:
        logger.info("[emix.loader] Loading EmixIndex: %s", emix_index_path)

        # Validate source root
        resolved_root = require_lerobot_dataset_root(source_root)
        source_root_str = str(resolved_root)

        # Load index
        self.index: EmixIndex = load_emix_index(emix_index_path)

        # Validate schema version
        if self.index.public.schema_version != EMIX_INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported EmixIndex schema_version: {self.index.public.schema_version}. "
                f"Expected {EMIX_INDEX_SCHEMA_VERSION}."
            )

        # Validate exactly one source dataset
        source_datasets = self.index.public.source_datasets
        if len(source_datasets) != 1:
            raise ValueError(
                f"EmixIndex must declare exactly one source dataset, got {len(source_datasets)}."
            )
        source_decl = source_datasets[0]

        # Validate repo id
        if source_decl.repo_id != source_repo_id:
            raise ValueError(
                f"EmixIndex source_repo_id mismatch: "
                f"index has '{source_decl.repo_id}', "
                f"loader received '{source_repo_id}'."
            )

        # Validate source root matches declaration
        if source_decl.root != source_root_str:
            raise ValueError(
                f"EmixIndex source root mismatch: "
                f"index has '{source_decl.root}', "
                f"resolved root is '{source_root_str}'. "
                f"Use the exact root that was used when building the index."
            )

        # Validate strict source dataset fingerprint
        fingerprint = self.index.public.metadata.get("source_dataset_fingerprint")
        if fingerprint is None:
            raise ValueError(
                "EmixIndex does not contain source_dataset_fingerprint in metadata. "
                "Rebuild the index with the current version."
            )
        validate_source_dataset_fingerprint(fingerprint, resolved_root)

        # Load source LeRobotDataset
        dataset_cls = _get_lerobot_dataset_class()
        logger.info(
            "[emix.loader] Loading source LeRobotDataset: repo_id=%s root=%s",
            source_repo_id,
            source_root_str,
        )
        self.dataset = dataset_cls(repo_id=source_repo_id, root=source_root_str)

        # Build group map
        all_groups = self.index.content.cross_point_groups
        if not all_groups:
            raise ValueError(
                f"EmixIndex has zero groups: {emix_index_path}"
            )
        self._group_map = {group.group_id: group for group in all_groups}

        # Build episode boundary maps from the source dataset.
        froms, tos = self._resolve_episode_bounds()
        # episode_end_map: frame_index -> exclusive end of its episode
        self._episode_end_map: dict[int, int] = {}
        for ep_start, ep_end in zip(froms, tos):
            for fi in range(int(ep_start), int(ep_end)):
                self._episode_end_map[fi] = int(ep_end)

        self.policy_temporal_spec = policy_temporal_spec
        if self.policy_temporal_spec.n_obs_steps != 1:
            raise AssertionError(
                "EmixIndexReader: EMIX samples must expose exactly one current observation frame; "
                f"got n_obs_steps={self.policy_temporal_spec.n_obs_steps}."
            )
        logger.info(
            "[emix.loader] Loaded direct EMIX samples: groups=%d",
            len(all_groups),
        )

    def _resolve_episode_bounds(self) -> tuple[list[int], list[int]]:
        """Resolve episode [from, to) bounds across LeRobot versions.

        @input: None.
        @output: tuple[list[int], list[int]] containing inclusive starts and exclusive ends.
        @scenario: Support both old LeRobotDataset.episode_data_index and newer
            LeRobotDataset.meta.episodes metadata layouts.
        """

        episode_data_index = getattr(self.dataset, "episode_data_index", None)
        if episode_data_index is not None:
            return (
                [int(value) for value in episode_data_index["from"]],
                [int(value) for value in episode_data_index["to"]],
            )

        meta = getattr(self.dataset, "meta", None)
        episodes = getattr(meta, "episodes", None)
        if episodes is not None:
            froms, tos = self._resolve_episode_bounds_from_meta(episodes)
            if froms and tos:
                return froms, tos

        froms, tos = self._resolve_episode_bounds_from_hf_dataset()
        if froms and tos:
            return froms, tos

        raise ValueError(
            "Unable to resolve episode boundaries from LeRobotDataset. "
            "Expected one of: dataset.episode_data_index, "
            "dataset.meta.episodes[{dataset_from_index,dataset_to_index}], "
            "or hf_dataset['episode_index']."
        )

    def _resolve_episode_bounds_from_meta(self, episodes: Any) -> tuple[list[int], list[int]]:
        """Read episode bounds from dataset.meta.episodes.

        @input: episodes [datasets.Dataset|pandas.DataFrame|sequence-like].
        @output: tuple[list[int], list[int]].
        @scenario: LeRobot v3 stores episode boundaries in metadata rows using
            dataset_from_index and dataset_to_index.
        """

        froms: list[int] = []
        tos: list[int] = []
        for row_idx in range(len(episodes)):
            if hasattr(episodes, "iloc"):
                row = episodes.iloc[row_idx]
                if hasattr(row, "to_dict"):
                    row = row.to_dict()
            else:
                row = episodes[row_idx]
            if not isinstance(row, dict):
                continue
            if "dataset_from_index" not in row or "dataset_to_index" not in row:
                continue
            froms.append(int(row["dataset_from_index"]))
            tos.append(int(row["dataset_to_index"]))
        return froms, tos

    def _resolve_episode_bounds_from_hf_dataset(self) -> tuple[list[int], list[int]]:
        """Infer episode bounds by scanning hf_dataset['episode_index'].

        @input: None.
        @output: tuple[list[int], list[int]].
        @scenario: Final compatibility fallback when explicit episode metadata is
            unavailable but frame-level episode_index is present.
        """

        hf = getattr(self.dataset, "hf_dataset", None)
        if hf is None:
            return [], []
        if "episode_index" not in getattr(hf, "column_names", []):
            return [], []

        episode_indices = hf["episode_index"]
        if len(episode_indices) == 0:
            return [], []

        normalized = [int(value.item()) if hasattr(value, "item") else int(value) for value in episode_indices]
        froms = [0]
        tos: list[int] = []
        current = normalized[0]
        for frame_idx, ep_idx in enumerate(normalized[1:], start=1):
            if ep_idx != current:
                tos.append(frame_idx)
                froms.append(frame_idx)
                current = ep_idx
        tos.append(len(normalized))
        return froms, tos

    def training_samples(self, policy_type: str) -> list[EmixTrainingSample]:
        """Return two-member training samples for trainable groups.

        @input: policy_type [str] for logging only.
        @output: list[EmixTrainingSample] with deterministic member ids.
        @scenario: Enumerate trainable groups and assign positional member ids.
        """

        samples: list[EmixTrainingSample] = []
        for group in self.index.content.cross_point_groups:
            if not group.state.training:
                continue
            if group.member_count != 2:
                raise ValueError(
                    f"Trainable group {group.group_id} has member_count={group.member_count}, "
                    f"expected exactly 2."
                )
            member_0_id = f"{group.group_id}:member_0"
            member_1_id = f"{group.group_id}:member_1"
            sample_id = f"{group.group_id}:{member_0_id}:{member_1_id}"
            samples.append(
                EmixTrainingSample(
                    group=group,
                    member_ids=[member_0_id, member_1_id],
                    sample_id=sample_id,
                )
            )

        if not samples:
            raise ValueError(
                f"EmixIndex has zero trainable groups. "
                f"Total groups: {len(self.index.content.cross_point_groups)}."
            )

        logger.info(
            "[emix.loader] Loaded direct EMIX samples: groups=%d samples=%d",
            len(self.index.content.cross_point_groups),
            len(samples),
        )
        return samples

    def load_member_window(self, group_id: str, member_id: str) -> dict[str, Any]:
        """Load one member window directly from source dataset frames.

        @input: group_id [str], member_id [str] of form '{group_id}:member_0' or ':member_1'.
        @output: dict with 'feature_payload' mapping feature keys to stacked numpy arrays.
        @scenario: Read source frames without crossing episode boundary or using window_len.
        """

        group = self._group_map.get(group_id)
        if group is None:
            raise KeyError(f"Group not found: {group_id}")

        # Resolve member by positional suffix
        if member_id.endswith(":member_0"):
            member = group.members[0]
        elif member_id.endswith(":member_1"):
            member = group.members[1]
        else:
            raise KeyError(
                f"Unknown member_id format: {member_id}. "
                f"Expected suffix ':member_0' or ':member_1'."
            )

        start = int(member.source_frame_index)
        total_frames = len(self.dataset.hf_dataset) if hasattr(self.dataset, "hf_dataset") else len(self.dataset)

        if start < 0 or start >= total_frames:
            raise ValueError(
                f"Member source_frame_index={start} is outside dataset range [0, {total_frames})."
            )

        # Compute span length from policy temporal spec
        span_len = max(
            self.policy_temporal_spec.n_obs_steps,
            self.policy_temporal_spec.action_steps,
        )

        # Episode boundary: do not cross
        episode_end = self._episode_end_map.get(start)
        if episode_end is None:
            # Frame not found in boundary map; use dataset end as fallback
            episode_end = total_frames

        end = min(episode_end, start + span_len)

        if end <= start:
            raise ValueError(
                f"Member window is empty: group_id={group_id} member_id={member_id} "
                f"start={start} end={end}."
            )

        return self._load_frame_range(start, end)

    def _load_frame_range(self, start: int, end: int) -> dict[str, Any]:
        """Load frames [start, end) from the source dataset.

        @input: start [int], end [int] exclusive.
        @output: dict with 'feature_payload'.
        @scenario: Concatenate sequential frame dicts into stacked arrays.
        """
        frames = [self.dataset[i] for i in range(start, end)]
        if not frames:
            raise ValueError(f"Empty frame range requested: start={start} end={end}.")
        payload = {}
        for key in frames[0].keys():
            values = [f[key] for f in frames]
            payload[key] = _stack_column(values)
        return {"feature_payload": payload}


class EmixPolicyDataset(torch.utils.data.Dataset):
    """Dataset backed by an EmixIndex reading directly from source LeRobotDataset.

    @input: source_root [str], source_repo_id [str], emix_index_path [str],
        policy_temporal_spec [PolicyTemporalSpec], loader_cfg [EmixLoaderSettings].
    @output: training batches with EMIX mixup applied.
    @scenario: Mixed and pure EMIX training from source dataset frames.
    """

    state_source_keys: tuple[str, ...] = (
        "observation.state",
        "observation.eef_pose",
        "observation.environment_state",
    )

    def __init__(
        self,
        source_root: str,
        source_repo_id: str,
        emix_index_path: str,
        policy_temporal_spec: PolicyTemporalSpec,
        loader_cfg: EmixLoaderSettings,
    ) -> None:
        self.reader = EmixIndexReader(
            source_root=source_root,
            source_repo_id=source_repo_id,
            emix_index_path=emix_index_path,
            policy_temporal_spec=policy_temporal_spec,
        )
        self.records = self.reader.training_samples(loader_cfg.policy_type)
        if len(self.records) == 0:
            raise ValueError(
                f"No trainable EMIX groups found in index: {emix_index_path}"
            )

        self.policy_temporal_spec = policy_temporal_spec
        self.loader_cfg = loader_cfg
        self._rng = random.Random(loader_cfg.seed)

        alpha_mode = loader_cfg.mixup_alpha_mode
        if alpha_mode not in ("beta", "fixed"):
            raise ValueError(f"mixup_alpha_mode must be 'beta' or 'fixed', got {alpha_mode}")
        if alpha_mode == "beta":
            if loader_cfg.mixup_beta <= 0.0:
                raise ValueError(f"mixup_beta must be positive, got {loader_cfg.mixup_beta}")
            self._beta = torch.distributions.Beta(loader_cfg.mixup_beta, loader_cfg.mixup_beta)
        else:
            self._beta = None
            if loader_cfg.mixup_alpha_value is None:
                raise ValueError("mixup_alpha_value is required when mixup_alpha_mode='fixed'")
            if not (0.0 < loader_cfg.mixup_alpha_value < 1.0):
                raise ValueError(
                    f"mixup_alpha_value must be in (0.0, 1.0), got {loader_cfg.mixup_alpha_value}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def _sample_alpha(self) -> float:
        if self.loader_cfg.mixup_alpha_mode == "fixed":
            return float(self.loader_cfg.mixup_alpha_value)
        return float(self._beta.sample().item())

    def _resolve_state_key(self, example: dict[str, Any]) -> str:
        for key in self.state_source_keys:
            if key in example:
                return key
        raise KeyError(
            "No state-like observation key found. Expected one of "
            f"{self.state_source_keys}, got {sorted(example.keys())}."
        )

    def _stack_items(self, items: list[dict[str, Any]], key: str) -> torch.Tensor:
        return torch.stack([item[key] for item in items], dim=0)

    def _resolve_obs_indices(self, aligned_len: int) -> list[int]:
        n_obs = self.policy_temporal_spec.n_obs_steps
        indices = list(range(min(aligned_len, n_obs)))
        if not indices:
            return [0] * n_obs
        return indices + [indices[-1]] * (n_obs - len(indices))

    def _resolve_action_indices(self, aligned_len: int) -> list[tuple[int, bool]]:
        action_len = self.policy_temporal_spec.action_steps
        if aligned_len == 0:
            return [(0, False)] * action_len
        result: list[tuple[int, bool]] = []
        for step in range(action_len):
            if step < aligned_len:
                result.append((step, True))
            else:
                result.append((aligned_len - 1, False))
        return result

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        group = record.group
        member_0_id, member_1_id = record.member_ids

        window_0 = self.reader.load_member_window(group.group_id, member_0_id)
        window_1 = self.reader.load_member_window(group.group_id, member_1_id)
        payload_0 = window_0["feature_payload"]
        payload_1 = window_1["feature_payload"]

        example = _frame_dict_from_window(window_0, 0)
        obs_len = min(len(payload_0["action"]), len(payload_1["action"]))
        if obs_len <= 0:
            raise ValueError(f"Group {group.group_id} has empty action windows.")
        obs_pairs = self._resolve_obs_indices(obs_len)
        if obs_pairs != [0]:
            raise AssertionError(
                "EmixPolicyDataset: EMIX observation construction must use only the member's "
                f"current frame [0], got {obs_pairs}."
            )
        action_pairs = self._resolve_action_indices(obs_len)
        alpha = self._sample_alpha()

        obs_0 = [_frame_dict_from_window(window_0, i) for i in obs_pairs]
        obs_1 = [_frame_dict_from_window(window_1, i) for i in obs_pairs]
        action_items_0 = [
            _frame_dict_from_window(window_0, i) for i, valid in action_pairs
        ]
        action_items_1 = [
            _frame_dict_from_window(window_1, i) for i, valid in action_pairs
        ]
        valid_mask = torch.tensor([valid for _, valid in action_pairs], dtype=torch.bool)

        batch: dict[str, Any] = {}
        source_images_0: dict[str, torch.Tensor] = {}
        source_images_1: dict[str, torch.Tensor] = {}
        visual_keys = self.loader_cfg.mixup_visual_keys
        if visual_keys is None:
            visual_keys = sorted(k for k in example.keys() if k.startswith("observation.images."))
        state_key = self._resolve_state_key(example)

        for key in example.keys():
            if key.startswith("observation.images."):
                if key not in visual_keys:
                    batch[key] = self._stack_items(obs_0, key)
                    continue
                source_images_0[key] = self._stack_items(obs_0, key)
                source_images_1[key] = self._stack_items(obs_1, key)
                mixed = []
                for item_0, item_1 in zip(obs_0, obs_1):
                    tensor_0 = item_0[key].to(dtype=torch.float32)
                    tensor_1 = item_1[key].to(dtype=torch.float32)
                    mixed.append(alpha * tensor_0 + (1.0 - alpha) * tensor_1)
                batch[key] = torch.stack(mixed, dim=0)
            elif key == state_key:
                state_tensor = self._stack_items(obs_0, key)
                batch["observation.state"] = state_tensor
                if key != "observation.state":
                    batch[key] = state_tensor
            elif key.startswith("observation.") and key != "observation.images":
                batch[key] = self._stack_items(obs_0, key)

        image_keys = sorted(k for k in example.keys() if k.startswith("observation.images."))
        if image_keys:
            batch["observation.images"] = torch.stack([batch[key] for key in image_keys], dim=1)
            if batch["observation.images"].shape[0] != 1:
                raise AssertionError(
                    "EmixPolicyDataset: observation.images must contain exactly one frame, "
                    f"got shape={tuple(batch['observation.images'].shape)}."
                )

        if batch["observation.state"].ndim != 2 or batch["observation.state"].shape[0] != 1:
            raise AssertionError(
                "EmixPolicyDataset: observation.state must have per-sample shape (1,D), "
                f"got shape={tuple(batch['observation.state'].shape)}."
            )
        if source_images_0:
            batch["emix_source_images"] = [source_images_0, source_images_1]

        action_0 = torch.stack([item["action"] for item in action_items_0], dim=0).to(
            dtype=torch.float32
        )
        action_1 = torch.stack([item["action"] for item in action_items_1], dim=0).to(
            dtype=torch.float32
        )
        batch["action"] = action_0.clone()
        batch["emix_actions"] = [action_0, action_1]
        batch["emix_states"] = [
            self._stack_items(obs_0, state_key).to(dtype=torch.float32),
            self._stack_items(obs_1, state_key).to(dtype=torch.float32),
        ]
        if "observation.eef_pose" in example:
            batch["emix_eef_poses"] = [
                torch.stack([item["observation.eef_pose"] for item in action_items_0], dim=0).to(
                    dtype=torch.float32
                ),
                torch.stack([item["observation.eef_pose"] for item in action_items_1], dim=0).to(
                    dtype=torch.float32
                ),
            ]
        batch["emix_alpha"] = torch.tensor(alpha, dtype=torch.float32)
        batch["emix_mask"] = valid_mask
        batch[self.policy_temporal_spec.action_pad_key] = ~valid_mask
        batch["action_is_pad"] = ~valid_mask
        batch["sample_id"] = record.sample_id
        batch["group_id"] = group.group_id
        batch["member_ids"] = list(record.member_ids)
        if self.loader_cfg.include_task:
            batch["task"] = self.reader.load_member_window(
                group.group_id, member_0_id
            )["feature_payload"]["task"][0]
        return batch


def _frame_dict_from_window(window_payload: dict[str, Any], frame_idx: int) -> dict[str, Any]:
    feature_payload = window_payload["feature_payload"]
    result: dict[str, Any] = {}
    for key, value in feature_payload.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            result[key] = torch.tensor(arr.item())
            continue
        frame = arr[frame_idx]
        if key.startswith("observation.images."):
            if frame.ndim == 3 and frame.shape[-1] in (1, 3):
                frame = np.transpose(frame, (2, 0, 1))
            frame = np.asarray(frame)
            # LeRobot may return decoded video frames as float32 [0, 1] or
            # uint8 [0, 255]. Preserve normalized frames instead of casting
            # them to uint8 first, which would make almost all pixels black.
            if np.issubdtype(frame.dtype, np.integer) or (
                frame.size and float(np.nanmax(frame)) > 1.0
            ):
                frame = np.clip(frame, 0.0, 255.0) / 255.0
            else:
                frame = np.clip(frame, 0.0, 1.0)
            result[key] = torch.as_tensor(frame, dtype=torch.float32)
        elif key == "task":
            result[key] = str(frame)
        else:
            result[key] = torch.as_tensor(np.asarray(frame, dtype=np.float32))
    return result


def _stack_column(values: list[Any]) -> Any:
    if not values:
        return []
    first = values[0]
    if isinstance(first, str):
        return list(values)
    arrays = [np.asarray(value) for value in values]
    if arrays[0].ndim == 0:
        return np.asarray([array.item() for array in arrays])
    return np.stack(arrays, axis=0)


def _get_lerobot_dataset_class():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
