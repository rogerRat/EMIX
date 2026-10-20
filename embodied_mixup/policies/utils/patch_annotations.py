"""Read per-episode instance labels aligned with a LeRobot dataset."""
from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path

import h5py
import numpy as np


class PatchAnnotationReader:
    """Resolve one target mask from a global LeRobot frame index."""

    def __init__(self, root: Path, dataset, episode_bounds: tuple[list[int], list[int]]):
        self.root = Path(root).expanduser().resolve()
        self.dataset = dataset
        self.starts, self.ends = episode_bounds
        if not self.root.is_dir():
            raise FileNotFoundError(f"Patch annotation directory does not exist: {self.root}")
        if not self.starts or len(self.starts) != len(self.ends):
            raise ValueError("Dataset episode boundaries are missing or invalid")

    def target_mask(self, global_frame_index: int) -> np.ndarray:
        """Load the patch_target mask for a global source-dataset frame."""
        episode_index = bisect_right(self.starts, global_frame_index) - 1
        if 0 <= episode_index < len(self.starts):
            start, end = self.starts[episode_index], self.ends[episode_index]
            if start <= global_frame_index < end:
                path = self.root / f"episode_{episode_index}.instances.h5"
                if not path.is_file():
                    raise FileNotFoundError(f"Patch annotations are missing for episode {episode_index}: {path}")
                local_index = global_frame_index - start
                with h5py.File(path, "r") as handle:
                    required = {"schema_version", "objects"}
                    if not required.issubset(handle.attrs) or "labels" not in handle:
                        raise ValueError(f"Patch annotation file lacks required fields: {path}")
                    if int(handle.attrs["schema_version"]) != 1:
                        raise ValueError(f"Unsupported patch annotation schema in {path}")
                    objects = json.loads(handle.attrs["objects"])
                    if "patch_target" not in objects:
                        raise ValueError(f"Annotation file has no patch_target role: {path}")
                    labels = handle["labels"]
                    if local_index >= len(labels):
                        raise ValueError(
                            f"Frame {local_index} is outside annotation episode {episode_index}"
                        )
                    return np.asarray(labels[local_index] == objects.index("patch_target"))
        raise IndexError(f"Global frame index is outside the dataset: {global_frame_index}")
