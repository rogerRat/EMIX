"""First-frame object-range visualization for one LeRobot dataset.

Takes the first frame of each episode video and answers one question: where
can objects appear, and how often? Two modes:

* ``range`` (default) — occupancy heatmap. The per-pixel maximum brightness
  across episodes approximates the static (near-white) background; a pixel
  counts as occupied in an episode when it is darker than that background by
  more than a threshold. The per-pixel occupied fraction is rendered as a
  colormap over a grayscale background, so frequently-used placement spots
  glow hot and rarely-used spots stay dark — unlike a plain 1/N average,
  which the white background dominates. Pixels occupied in essentially every
  episode (the table, fixtures, the arm's home pose) are suppressed to dim
  gray so the object range stands out.
* ``average`` — the original 1/N weighted average of every first frame.

The module is strictly dataset-internal: it reads only one dataset root and
never combines frames across datasets. Every episode index is asserted to be
within the dataset bounds before the frame is read.

Runnable as a module:
    python -m embodied_mixup.visualization.first_frame_average \
        --root data/lerobot/<task> [--output outputs/first_frame/<task>_first_frame_range.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_VIDEO_KEY = "observation.images.head_camera"
# A pixel is "occupied" when it is this much darker than the per-pixel
# brightest-ever value (the static background estimate), in [0, 1] units.
DEFAULT_OCCUPANCY_THRESHOLD = 0.08
# Occupancy at or above this fraction marks always-present scenery (table,
# fixtures, arm home pose); it is rendered dim gray instead of peak colormap.
STATIC_OCCUPANCY = 0.98


def _load_dataset(dataset_root: Path):
    """Load one LeRobot dataset from its exact root (same fallback as cli.py)."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id=dataset_root.name, root=str(dataset_root))


def _frame_to_unit_float(frame) -> np.ndarray:
    """Convert one observation frame to float32 in [0, 1] with shape (H, W, C)."""
    array = np.asarray(frame, dtype=np.float32)
    assert array.ndim == 3 and array.shape[0] in (1, 3), (
        f"expected a (C, H, W) frame with 1 or 3 channels, got shape {array.shape}"
    )
    if array.max(initial=0.0) > 1.0:
        array = array / 255.0
    return np.transpose(array, (1, 2, 0))


def _iter_first_frames(dataset_root: Path, video_key: str):
    """Yield every episode's first frame as float32 [0, 1] (H, W, C)."""
    dataset = _load_dataset(Path(dataset_root))
    episodes = dataset.meta.episodes
    num_episodes = int(dataset.num_episodes)
    assert num_episodes >= 1, f"{dataset_root} has no episodes to visualize"
    assert episodes.num_rows == num_episodes, (
        f"episode table has {episodes.num_rows} rows but dataset reports "
        f"{num_episodes} episodes"
    )
    total_frames = len(dataset)
    for episode_index in range(num_episodes):
        row = episodes[episode_index]
        from_index = int(row["dataset_from_index"])
        length = int(row["length"])
        assert 0 <= from_index < total_frames, (
            f"episode {episode_index} starts out of bounds: "
            f"{from_index} not in [0, {total_frames})"
        )
        assert from_index + length <= total_frames, (
            f"episode {episode_index} extends out of bounds: "
            f"[{from_index}, {from_index + length}) exceeds {total_frames} frames"
        )
        yield dataset[from_index][video_key]


def average_first_frames(
    dataset_root: Path | str,
    *,
    video_key: str = DEFAULT_VIDEO_KEY,
) -> np.ndarray:
    """Average the first frame of every episode, each weighted 1/N.

    Returns a float32 (H, W, C) image in [0, 1]. ``N`` is the dataset's
    episode count; the weights are exactly ``1/N`` each and must sum to 1.
    """
    accumulated: np.ndarray | None = None
    count = 0
    for frame_source in _iter_first_frames(dataset_root, video_key):
        frame = _frame_to_unit_float(frame_source)
        if accumulated is None:
            accumulated = np.zeros_like(frame, dtype=np.float64)
        else:
            assert frame.shape == accumulated.shape, (
                f"first frame shape {frame.shape} differs from {accumulated.shape}; "
                "first frames must share one resolution"
            )
        accumulated += frame
        count += 1
    assert accumulated is not None and count >= 1, "no first frames were read"
    weights = np.full(count, 1.0 / count)
    assert np.isclose(weights.sum(), 1.0), "1/N weights must sum to 1"
    averaged = accumulated / count
    assert averaged.min() >= 0.0 and averaged.max() <= 1.0 + 1e-6, (
        "weighted average left the [0, 1] range"
    )
    return np.clip(averaged, 0.0, 1.0).astype(np.float32)


def first_frame_occupancy(
    dataset_root: Path | str,
    *,
    video_key: str = DEFAULT_VIDEO_KEY,
    threshold: float = DEFAULT_OCCUPANCY_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel object-occupancy fraction across every episode's first frame.

    The background is estimated as the per-pixel maximum brightness over all
    first frames (the scene background is the brightest static surface); a
    pixel is occupied in an episode when it is at least ``threshold`` darker
    than that estimate. Returns ``(background, occupancy)`` where
    ``background`` is float32 (H, W, C) in [0, 1] and ``occupancy`` is
    float32 (H, W) in [0, 1] — the fraction of episodes in which each pixel
    was occupied.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    background: np.ndarray | None = None
    occupied_count: np.ndarray | None = None
    count = 0
    for frame_source in _iter_first_frames(dataset_root, video_key):
        frame = _frame_to_unit_float(frame_source)
        if background is None:
            background = frame.copy()
            occupied_count = np.zeros(frame.shape[:2], dtype=np.float32)
        else:
            assert frame.shape == background.shape, (
                f"first frame shape {frame.shape} differs from {background.shape}; "
                "first frames must share one resolution"
            )
        np.maximum(background, frame, out=background)
        occupied_count += (frame.max(axis=2, initial=0.0) < background.max(axis=2) - threshold)
        count += 1
    assert background is not None and occupied_count is not None and count >= 1
    return background.astype(np.float32), occupied_count / count


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Expand a (H, W, 1) image to three channels, pass others through."""
    return np.repeat(image, 3, axis=2) if image.shape[2] == 1 else image


def save_first_frame_average(
    dataset_root: Path | str,
    output: Path | str,
    *,
    video_key: str = DEFAULT_VIDEO_KEY,
    mode: str = "range",
    threshold: float = DEFAULT_OCCUPANCY_THRESHOLD,
) -> Path:
    """Render the first-frame visualization of one dataset and write a PNG."""
    output = Path(output)
    if mode == "average":
        image = average_first_frames(dataset_root, video_key=video_key)
        rgb = np.rint(_to_rgb(image) * 255.0).astype(np.uint8)
        Image.fromarray(rgb, mode="RGB").save(output)
        return output
    if mode != "range":
        raise ValueError(f"unknown mode {mode!r}; expected 'range' or 'average'")

    background, occupancy = first_frame_occupancy(
        dataset_root, video_key=video_key, threshold=threshold
    )
    num_episodes = len(_load_dataset(Path(dataset_root)).meta.episodes)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 7.4))
    grayscale = _to_rgb(background).mean(axis=2)
    axis.imshow(grayscale, cmap="gray", vmin=0.0, vmax=1.0)

    # Always-present scenery (table, fixtures, arm home pose) is dimmed so the
    # graded object range is not drowned in saturated red.
    display = np.where(occupancy >= STATIC_OCCUPANCY, np.nan, occupancy)
    colored = axis.imshow(
        display, cmap="turbo", vmin=0.0, vmax=1.0, alpha=0.65, interpolation="nearest"
    )
    colorbar = figure.colorbar(colored, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("episode fraction with an object at the pixel")
    # Wrap the title so long task names stay inside the figure bounds.
    import textwrap

    axis.set_title(
        textwrap.fill(
            f"first-frame object range — {Path(dataset_root).name} "
            f"({num_episodes} episodes, threshold {threshold:.2f}; "
            f">= {STATIC_OCCUPANCY:.0%} occupancy dimmed as static)",
            width=64,
        ),
        fontsize=11,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Render to a temp file first, then rename: watchers/dev servers in the
    # output directory must never observe a half-written PNG.
    import os
    import tempfile

    descriptor, temporary = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.", suffix=".png")
    os.close(descriptor)
    try:
        figure.savefig(temporary, dpi=150)
        plt.close(figure)
        os.chmod(temporary, 0o644)
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="LeRobot dataset root")
    parser.add_argument(
        "--output", default=None,
        help="output PNG path (default: outputs/first_frame/<task>_first_frame_range.png)",
    )
    parser.add_argument("--video-key", default=DEFAULT_VIDEO_KEY)
    parser.add_argument(
        "--mode", choices=("range", "average"), default="range",
        help="'range' = occupancy heatmap (default), 'average' = 1/N blend",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_OCCUPANCY_THRESHOLD,
        help="occupancy darkness threshold in (0, 1) for --mode range",
    )
    args = parser.parse_args()
    output = Path(args.output) if args.output else (
        Path("outputs") / "first_frame" / f"{Path(args.root).name}_first_frame_range.png"
    )
    output = save_first_frame_average(
        args.root, output,
        video_key=args.video_key, mode=args.mode, threshold=args.threshold,
    )
    print(f"[first_frame_average] wrote {output} (mode={args.mode})")


if __name__ == "__main__":
    main()
