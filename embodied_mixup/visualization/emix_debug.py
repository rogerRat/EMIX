"""Training-time EMIX input snapshots.

The exporter is intentionally called after an EMIX sample has been collated,
but before policy preprocessing.  This makes the saved image the actual
mixed observation selected for the current training step, while keeping the
visualization independent from policy-specific normalization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmixDebugSettings:
    """Configuration for training-time EMIX snapshots."""

    enabled: bool = False
    policy_type: str | None = None
    output_dir: str | None = None
    max_samples: int = 32
    every_n_steps: int = 10_000
    image_keys: tuple[str, ...] | None = None
    vis_n_obs_steps: int = 1
    max_action_steps: int = 1
    action_xyz_indices: tuple[int, int, int] = (0, 1, 2)
    action_is_delta: bool = True
    min_display_fraction: float = 0.12
    dpi: int = 120

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any] | None,
        default_output_dir: str | Path,
        policy_type: str | None = None,
        default_n_obs_steps: int = 1,
        vis_n_obs_steps: int | None = None,
    ) -> "EmixDebugSettings":
        """Parse ``emix.debug`` and resolve the default artifact directory."""

        raw = dict(payload or {})
        if "max_obs_steps" in raw:
            raise ValueError(
                "emix.debug.max_obs_steps was renamed to vis_n_obs_steps; "
                "set the latter with the training CLI --vis_n_obs_steps option"
            )
        if "vis_n_obs_steps" in raw:
            raise ValueError(
                "emix.debug.vis_n_obs_steps is CLI-only; "
                "set it with --vis_n_obs_steps"
            )
        default_n_obs_steps = int(default_n_obs_steps)
        if default_n_obs_steps <= 0:
            raise ValueError("policy n_obs_steps must be positive")
        if vis_n_obs_steps is not None:
            vis_n_obs_steps = int(vis_n_obs_steps)
            if vis_n_obs_steps <= 0:
                raise ValueError("--vis_n_obs_steps must be positive")
        image_keys = raw.get("image_keys")
        if image_keys is not None:
            image_keys = tuple(str(key) for key in image_keys)
        xyz_indices = tuple(int(index) for index in raw.get("action_xyz_indices", (0, 1, 2)))
        if len(xyz_indices) != 3 or min(xyz_indices) < 0:
            raise ValueError("emix.debug.action_xyz_indices must contain three non-negative indices")

        settings = cls(
            enabled=bool(raw.get("enabled", False)),
            policy_type=policy_type,
            output_dir=str(raw["output_dir"]) if raw.get("output_dir") else str(default_output_dir),
            max_samples=int(raw.get("max_samples", 32)),
            every_n_steps=int(raw.get("every_n_steps", 10_000)),
            image_keys=image_keys,
            vis_n_obs_steps=(
                default_n_obs_steps if vis_n_obs_steps is None else vis_n_obs_steps
            ),
            max_action_steps=int(raw.get("max_action_steps", 1)),
            action_xyz_indices=xyz_indices,
            action_is_delta=bool(raw.get("action_is_delta", True)),
            min_display_fraction=float(raw.get("min_display_fraction", 0.12)),
            dpi=int(raw.get("dpi", 120)),
        )
        if settings.max_samples < 0:
            raise ValueError("emix.debug.max_samples must be non-negative")
        if settings.every_n_steps <= 0:
            raise ValueError("emix.debug.every_n_steps must be positive")
        if settings.vis_n_obs_steps <= 0 or settings.max_action_steps < 0:
            raise ValueError("emix.debug vis_n_obs_steps must be positive and max_action_steps must be non-negative")
        if not (0.0 <= settings.min_display_fraction < 1.0):
            raise ValueError("emix.debug.min_display_fraction must be in [0, 1)")
        if settings.dpi <= 0:
            raise ValueError("emix.debug.dpi must be positive")
        return settings


class EmixDebugExporter:
    """Write a bounded number of mixed samples with action direction plots."""

    def __init__(self, settings: EmixDebugSettings) -> None:
        self.settings = settings
        self.exported_samples = 0
        self._next_due_step = settings.every_n_steps
        self.output_dir = Path(settings.output_dir or "outputs/emix_debug").expanduser().resolve()
        self._manifest_path = self.output_dir / "manifest.jsonl"
        if settings.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info("[emix.debug] Exporting snapshots to %s", self.output_dir)

    @property
    def finished(self) -> bool:
        return self.exported_samples >= self.settings.max_samples

    def export_batch(self, batch: dict[str, Any], step: int) -> int:
        """Export samples from one collated EMIX batch and return the count."""

        if not self.settings.enabled or self.finished or step < self._next_due_step:
            return 0
        if "emix_actions" not in batch:
            return 0

        image_keys = self._resolve_image_keys(batch)
        if not image_keys:
            LOGGER.warning("[emix.debug] EMIX batch has no image features; skipping step %d", step)
            return 0

        branch_actions = [_as_numpy(value) for value in batch["emix_actions"]]
        if len(branch_actions) != 2:
            LOGGER.warning("[emix.debug] Expected two emix action branches, got %d", len(branch_actions))
            return 0
        batch_size = int(branch_actions[0].shape[0])
        branch_eef_poses = [_as_numpy(value) for value in batch.get("emix_eef_poses", [])]
        if len(branch_eef_poses) != 2:
            branch_eef_poses = [None, None]
        source_image_batch = batch.get("emix_source_images")
        if not (
            isinstance(source_image_batch, list)
            and len(source_image_batch) == 2
            and all(isinstance(branch, dict) for branch in source_image_batch)
        ):
            source_image_batch = None
        branch_states = [_as_numpy(value) for value in batch.get("emix_states", [])]
        if len(branch_states) != 2:
            fallback_state = _optional_numpy(batch.get("observation.state"))
            branch_states = [fallback_state, fallback_state]
        valid_mask = _optional_numpy(batch.get("emix_mask", batch.get("action_is_pad")))
        if valid_mask is not None and "action_is_pad" in batch and "emix_mask" not in batch:
            valid_mask = ~valid_mask.astype(bool)
        alphas = _optional_numpy(batch.get("emix_alpha"))
        sample_ids = _metadata_values(batch.get("sample_id"), batch_size)
        group_ids = _metadata_values(batch.get("group_id"), batch_size)
        member_ids = _metadata_values(batch.get("member_ids"), batch_size)

        exported = 0
        for sample_index in range(batch_size):
            if self.finished:
                break
            try:
                image_arrays = {
                    key: _image_sequence(batch[key], sample_index)
                    for key in image_keys
                }
                source_image_arrays = None
                if source_image_batch is not None:
                    source_image_arrays = [
                        {
                            key: _image_sequence(value, sample_index)
                            for key, value in branch.items()
                            if key in image_arrays
                        }
                        for branch in source_image_batch
                    ]
                actions = [branch[sample_index] for branch in branch_actions]
                eef_poses = [
                    pose[sample_index] if pose is not None else None
                    for pose in branch_eef_poses
                ]
                current_states = [
                    state[sample_index, 0] if state is not None else None
                    for state in branch_states
                ]
                sample_mask = valid_mask[sample_index] if valid_mask is not None else None
                alpha = float(alphas[sample_index]) if alphas is not None else None
                filename = self._export_sample(
                    step=step,
                    sample_index=sample_index,
                    image_arrays=image_arrays,
                    source_image_arrays=source_image_arrays,
                    actions=actions,
                    eef_poses=eef_poses,
                    current_states=current_states,
                    valid_mask=sample_mask,
                    alpha=alpha,
                    sample_id=sample_ids[sample_index],
                    group_id=group_ids[sample_index],
                    member_ids=member_ids[sample_index],
                )
                exported += 1
                self.exported_samples += 1
                LOGGER.info("[emix.debug] saved=%s", filename)
            except Exception:
                LOGGER.exception(
                    "[emix.debug] Failed to export step=%d sample=%d; continuing training",
                    step,
                    sample_index,
                )
        if exported > 0:
            self._next_due_step = step + self.settings.every_n_steps
        return exported

    def _resolve_image_keys(self, batch: dict[str, Any]) -> list[str]:
        configured = self.settings.image_keys
        if configured is not None:
            return [key for key in configured if key in batch]
        return sorted(key for key in batch if key.startswith("observation.images."))

    def _export_sample(
        self,
        *,
        step: int,
        sample_index: int,
        image_arrays: dict[str, list[np.ndarray]],
        source_image_arrays: list[dict[str, list[np.ndarray]]] | None,
        actions: list[np.ndarray],
        eef_poses: list[np.ndarray | None],
        current_states: list[np.ndarray | None],
        valid_mask: np.ndarray | None,
        alpha: float | None,
        sample_id: Any,
        group_id: Any,
        member_ids: Any,
    ) -> Path:
        if not image_arrays:
            raise ValueError("No image frames available in EMIX sample")

        obs_steps = min(
            len(frames) for frames in image_arrays.values()
        )
        obs_steps = min(obs_steps, self.settings.vis_n_obs_steps)
        if obs_steps <= 0:
            raise ValueError("No observation frames available in EMIX sample")

        action_planes = (
            ((0, 1), "-Z top view (XOY)", "X", "Y"),
            ((0, 2), "side view (XOZ)", "X", "Z"),
            ((1, 2), "side view (YOZ)", "Y", "Z"),
        )
        n_image_keys = len(image_arrays)
        n_columns = 3 * n_image_keys + len(action_planes)
        fig = plt.figure(
            figsize=(3.2 * n_columns, 3.8 * obs_steps + 1.2),
            constrained_layout=True,
        )
        grid = fig.add_gridspec(obs_steps, n_columns)

        for frame_index in range(obs_steps):
            column = 0
            for key, frames in image_arrays.items():
                mixed_image = frames[frame_index]
                has_sources = source_image_arrays is not None and all(
                    key in branch and frame_index < len(branch[key])
                    for branch in source_image_arrays
                )
                image_items = (
                    (
                        "member_0",
                        source_image_arrays[0][key][frame_index],
                    ),
                    (
                        "member_1",
                        source_image_arrays[1][key][frame_index],
                    ),
                    ("mixed", mixed_image),
                ) if has_sources else (("member_0", None), ("member_1", None), ("mixed", mixed_image))
                for image_kind, image in image_items:
                    ax = fig.add_subplot(grid[frame_index, column])
                    if image is None:
                        ax.text(0.5, 0.5, "source unavailable", ha="center", va="center")
                    else:
                        ax.imshow(image)
                    ax.set_title(
                        f"{key.removeprefix('observation.images.')} t={frame_index}\n{image_kind}"
                    )
                    ax.axis("off")
                    column += 1

            for plane_index, (plane, title, x_label, y_label) in enumerate(action_planes):
                action_ax = fig.add_subplot(grid[frame_index, column + plane_index])
                self._draw_action_plane(
                    action_ax,
                    actions,
                    eef_poses,
                    current_states,
                    valid_mask,
                    frame_index=frame_index,
                    plane=plane,
                    title=title,
                    x_label=x_label,
                    y_label=y_label,
                )

        alpha_text = "n/a" if alpha is None else f"{alpha:.4f}"
        fig.suptitle(
            f"EMIX input | step={step} sample={sample_index} | alpha={alpha_text}\n"
            f"group={group_id} sample_id={sample_id} members={member_ids}",
            fontsize=10,
        )
        filename = self.output_dir / f"step_{step:08d}_sample_{self.exported_samples:05d}.png"
        fig.savefig(filename, dpi=self.settings.dpi, bbox_inches="tight")
        plt.close(fig)

        record = {
            "file": filename.name,
            "step": int(step),
            "batch_sample_index": int(sample_index),
            "sample_id": _json_value(sample_id),
            "group_id": _json_value(group_id),
            "member_ids": _json_value(member_ids),
            "alpha": alpha,
            "policy_type": self.settings.policy_type,
            "observation_steps": int(obs_steps),
            "vis_n_obs_steps": int(self.settings.vis_n_obs_steps),
            "action_horizon_steps": [int(len(action)) for action in actions],
            "debug_action_steps": self.settings.max_action_steps,
            "image_keys": list(image_arrays),
            "action_xyz_indices": list(self.settings.action_xyz_indices),
            "action_is_delta": self.settings.action_is_delta,
            "actions": [_json_value(action) for action in actions],
        }
        with self._manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return filename

    def _draw_action_plane(
        self,
        ax,
        actions: list[np.ndarray],
        eef_poses: list[np.ndarray | None],
        current_states: list[np.ndarray | None],
        valid_mask: np.ndarray | None,
        frame_index: int,
        plane: tuple[int, int],
        title: str,
        x_label: str,
        y_label: str,
    ) -> None:
        colors = ("#1976D2", "#E65100")
        labels = ("action 0", "action 1")
        all_points: list[np.ndarray] = [np.zeros((1, 2), dtype=np.float64)]
        rendered_paths: list[tuple[int, np.ndarray, float]] = []
        for branch_index, action in enumerate(actions):
            values = np.asarray(action)
            if values.ndim != 2:
                raise ValueError(f"Expected action shape (T,D), got {values.shape}")
            available_end = len(values)
            if valid_mask is not None:
                available_end = min(available_end, int(np.asarray(valid_mask, dtype=bool).sum()))
            start = min(frame_index, max(available_end - 1, 0))
            action_end = min(
                available_end,
                start + (self.settings.max_action_steps or available_end),
            )
            steps = action_end - start
            if steps <= 0:
                continue
            if eef_poses[branch_index] is not None:
                # observation.eef_pose is an absolute [x, y, z, rx, ry, rz]
                # trajectory. Express every future pose relative to the
                # branch's current EEF position.
                pose = np.asarray(eef_poses[branch_index])
                if pose.ndim != 2 or pose.shape[1] < 3:
                    raise ValueError(f"Expected EEF pose shape (T,6), got {pose.shape}")
                trajectory_end = min(
                    len(pose),
                    action_end + 1,
                )
                xyz = pose[start + 1:trajectory_end, :3]
                current_xyz = pose[start, :3]
                points = np.vstack([np.zeros((1, 3)), xyz - current_xyz])
            else:
                xyz = values[start:action_end, list(self.settings.action_xyz_indices)]
                if self.settings.action_is_delta:
                    points = np.vstack([np.zeros((1, 3)), np.cumsum(xyz, axis=0)])
                else:
                    # Fallback for datasets without EEF poses.
                    current_state = current_states[branch_index]
                    if current_state is None:
                        raise ValueError(
                            "EEF pose or current state is required for action visualization"
                        )
                    current_state = np.asarray(current_state)
                    if current_state.ndim == 2:
                        current_state = current_state[min(frame_index, len(current_state) - 1)]
                    current_xyz = current_state[list(self.settings.action_xyz_indices)]
                    points = np.vstack([np.zeros((1, 3)), xyz - current_xyz])
            xy = points[:, [plane[0], plane[1]]]
            all_points.append(xy)
            rendered_paths.append((branch_index, xy, float(np.linalg.norm(points[-1]))))
        points = np.vstack(all_points)
        # Keep the current state at the exact center of the viewport.  The
        # plotted points are already expressed relative to that state, so a
        # symmetric square range around (0, 0) is sufficient.
        extent = max(float(np.max(np.abs(points))), 1e-4)
        extent *= 1.25
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        for branch_index, xy, full_distance in rendered_paths:
            color = colors[branch_index]
            ax.plot(xy[:, 0], xy[:, 1], color=color, marker="o", ms=3, label=labels[branch_index])
            projected_distance = float(np.linalg.norm(xy[-1]))
            visibility_floor = extent * self.settings.min_display_fraction
            annotation_point = xy[-1]
            if 1e-12 < projected_distance < visibility_floor:
                # Keep the solid path at its true scale, then add a dashed
                # direction-only ray so a small branch is still visible.
                display_end = xy[-1] / projected_distance * visibility_floor
                ax.plot(
                    [0.0, display_end[0]],
                    [0.0, display_end[1]],
                    color=color,
                    linestyle="--",
                    linewidth=2.0,
                    alpha=0.8,
                )
                ax.scatter(
                    [display_end[0]], [display_end[1]],
                    color=color, marker=">", s=28,
                    label=f"{labels[branch_index]} direction (scaled)",
                )
                annotation_point = display_end
            elif projected_distance <= 1e-12:
                ax.scatter(
                    [0.0], [0.0], color=color, marker="o", s=28,
                    label=f"{labels[branch_index]} stationary",
                )
            ax.annotate(
                f"end |ΔEEF|={full_distance:.3g}m",
                xy=annotation_point,
                xytext=(4, 4),
                textcoords="offset points",
                color=color,
                fontsize=8,
            )
        ax.scatter([0], [0], c="black", marker="x", s=35, label="current state")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"EEF trajectory ({title})")
        ax.set_xlabel(f"Δ{x_label} (m)")
        ax.set_ylabel(f"Δ{y_label} (m)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _optional_numpy(value: Any) -> np.ndarray | None:
    return None if value is None else _as_numpy(value)


def _metadata_values(value: Any, batch_size: int) -> list[Any]:
    if value is None:
        return [None] * batch_size
    if isinstance(value, (list, tuple)):
        return list(value)
    array = _as_numpy(value)
    return [array[index] for index in range(batch_size)]


def _image_sequence(value: Any, sample_index: int) -> list[np.ndarray]:
    array = _as_numpy(value)[sample_index]
    if array.ndim == 3:
        array = array[None, ...]
    frames: list[np.ndarray] = []
    for frame in array:
        if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        frame = np.asarray(frame, dtype=np.float32)
        if frame.size and float(frame.max()) <= 1.0:
            frame = frame * 255.0
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return frames


def _json_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
