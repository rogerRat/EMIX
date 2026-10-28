"""Load per-frame operation segments from instance annotation sidecars."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class FrameSegmentMap:
    """Compact global-frame lookup into unique segment ID/object pairs."""

    frame_codes: np.ndarray
    segments: tuple[dict[str, str], ...]

    def for_frame(self, frame_index: int) -> dict[str, str] | None:
        code = int(self.frame_codes[int(frame_index)])
        return None if code < 0 else self.segments[code]


def load_frame_segment_map(dataset_root: str | Path, episode_indices: np.ndarray) -> FrameSegmentMap | None:
    """Load synchronized segment labels from annotations/episode_N.instances.h5.

    Sidecars must contain one UTF-8 segment_id and segment_object value per
    converted LeRobot frame. Empty strings mark frames excluded from
    segment-wise augmentation, such as the terminal return after final place.
    """
    root = Path(dataset_root).expanduser().resolve() / "annotations"
    episodes = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    if not root.is_dir():
        return None
    unique_episodes = np.unique(episodes)
    sidecars = [root / f"episode_{int(ep)}.instances.h5" for ep in unique_episodes]
    present = [path.is_file() for path in sidecars]
    if not any(present):
        return None
    if not all(present):
        missing = [str(path) for path, exists in zip(sidecars, present) if not exists]
        raise FileNotFoundError("Segment annotations are incomplete; missing sidecars: " + ", ".join(missing[:8]))

    frame_codes = np.full(len(episodes), -1, dtype=np.int32)
    segment_codes: dict[tuple[str, str], int] = {}
    segments: list[dict[str, str]] = []
    schema_presence: list[bool] = []
    for episode_index, path in zip(unique_episodes, sidecars):
        frame_positions = np.flatnonzero(episodes == episode_index)
        with h5py.File(path, "r") as handle:
            has_id = "segment_id" in handle
            has_object = "segment_object" in handle
            if has_id != has_object:
                raise ValueError(f"Sidecar must contain both segment_id and segment_object: {path}")
            schema_presence.append(has_id)
            if not has_id:
                continue
            if int(handle.attrs.get("schema_version", -1)) < 2:
                raise ValueError(f"Segment annotations require instance sidecar schema_version >= 2: {path}")
            try:
                object_table = handle.attrs["objects"]
            except KeyError as exc:
                raise ValueError(f"Segment sidecar is missing its canonical object table: {path}") from exc
            if isinstance(object_table, bytes):
                object_table = object_table.decode("utf-8")
            try:
                object_names = set(json.loads(object_table))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid object table in segment sidecar: {path}") from exc
            ids = _read_strings(handle["segment_id"][:])
            objects = _read_strings(handle["segment_object"][:])
            if len(ids) != len(frame_positions) or len(objects) != len(frame_positions):
                raise ValueError(
                    f"Segment sidecar length does not match episode {episode_index}: "
                    f"sidecar={len(ids)}, dataset={len(frame_positions)}"
                )
            for local_index, (segment_id, object_name) in enumerate(zip(ids, objects)):
                if not segment_id and not object_name:
                    continue
                if not segment_id or not object_name:
                    raise ValueError(f"Partial segment label in {path} at frame {local_index}")
                if object_name == "patch_target":
                    raise ValueError(f"Legacy object name 'patch_target' is unsupported in {path}")
                if object_name == "background" or object_name.startswith("scene_"):
                    raise ValueError(f"Segment object must be a canonical task object, got {object_name!r} in {path}")
                if object_name not in object_names:
                    raise ValueError(
                        f"Segment object {object_name!r} is absent from the sidecar object table in {path}"
                    )
                key = (segment_id, object_name)
                code = segment_codes.get(key)
                if code is None:
                    code = len(segments)
                    segment_codes[key] = code
                    segments.append({"id": segment_id, "object": object_name})
                frame_codes[frame_positions[local_index]] = code
    if any(schema_presence) and not all(schema_presence):
        raise ValueError("Segment labels must be present for every episode or none")
    if not any(schema_presence):
        return None
    return FrameSegmentMap(frame_codes=frame_codes, segments=tuple(segments))


def _read_strings(values) -> list[str]:
    result = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(str(value))
    return result
