"""3D interactive scatter plot of gripper pick/place transition points.

Scans each episode's gripper signal for sustained transition trends: a
monotonic fall (1 -> 0, gripper closing) marks a pick, a monotonic rise
(0 -> 1, gripper opening) marks a place.  Renders episode trajectories
as thin lines plus pick and place points as two differently-colored marker
sets in a 3D work-space coordinate frame using Plotly WebGL.

@input:  trajectories dict mapping episode index -> {"xyz": (N, 3) array,
         "gripper_signal": raw (N,) gripper signal}.  Compatible with the
         trajectories dict produced by prepare_visualization_data().
@output: plotly.graph_objects.Figure (call .show() or .write_html()).
@scenario: Inspect where the policy/scripted demos grasp and release
          objects across the workspace, e.g. verifying that pick points
          concentrate on object spawn volumes while place points cluster
          at task target locations.

Standalone usage (no EmixIndex required)::

    python -m embodied_mixup.visualization.plot_actions \
        --root data/lerobot/place_bread_basket_left_cen_arm_near_side_cam
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .camera_presets import CAMERA_PRESETS
from .utils import _PLOTLY_FONT

logger = logging.getLogger(__name__)

# Categorical pair validated for CVD separation (dataviz palette check):
# blue for pick, orange for place.
PICK_COLOR = "#1f77b4"
PLACE_COLOR = "#ff7f0e"


def _find_trend_runs(
    signal: np.ndarray,
    min_steps: int = 3,
    min_swing: float = 0.2,
    eps: float = 1e-6,
) -> List[Dict[str, Any]]:
    """Find sustained monotonic gripper runs in a gradually-varying signal.

    The gripper rarely jumps straight from 0 to 1; it ramps through small
    steps (e.g. 0 -> 0.01 -> 0.02 -> ...).  A transition is therefore a
    *trend*: at least min_steps consecutive same-direction steps whose
    total swing reaches min_swing.  Flat frames (movement pause) do not
    break a run; a direction reversal closes it.

    @input:  signal: (N,) raw gripper signal (1 = open, 0 = closed).
             min_steps: minimum consecutive directional steps in a run.
             min_swing: minimum total |end - start| swing of a run.
             eps: step dead-band for treating a delta as flat.
    @output: List of run dicts with start/end frames (inclusive), direction
             (+1 rising/opening, -1 falling/closing) and swing.
    @scenario: Trend-based pick/place detection robust to slow scripted
              gripper ramps and single-frame jitter.
    """
    runs: List[Dict[str, Any]] = []
    n = len(signal)

    def _close_run(start: int, end: int, direction: int, steps: int) -> None:
        # Trim trailing flat frames so the reported span is the actual ramp.
        while end > start and abs(signal[end] - signal[end - 1]) <= eps:
            end -= 1
        swing = float(signal[end] - signal[start])
        if steps >= min_steps and abs(swing) >= min_swing:
            runs.append({
                "start": int(start),
                "end": int(end),
                "direction": int(direction),
                "swing": swing,
            })

    run_start = 0
    direction = 0
    steps = 0
    for i in range(1, n):
        delta = signal[i] - signal[i - 1]
        if abs(delta) <= eps:
            continue  # flat frame: pause inside the run, trend unchanged
        step_dir = 1 if delta > 0 else -1
        if direction == 0:
            run_start, direction, steps = i - 1, step_dir, 1
        elif step_dir == direction:
            steps += 1
        else:
            _close_run(run_start, i - 1, direction, steps)
            run_start, direction, steps = i - 1, step_dir, 1
    if direction != 0:
        _close_run(run_start, n - 1, direction, steps)
    return runs


def extract_pick_place_points(
    trajectories: Dict[int, dict],
    min_trend_steps: int = 3,
    min_swing: float = 0.2,
) -> Dict[str, Any]:
    """Detect gripper open->close (pick) and close->open (place) events.

    The gripper signal is normalized to [0, 1] where 1.0 is fully open
    and 0.0 is fully closed (RoboTwin's open_gripper pos convention).
    Transitions are detected as trends (see _find_trend_runs): a sustained
    monotonic fall 1 -> 0 is a pick, a sustained monotonic rise 0 -> 1 is
    a place.  The event point is the EEF position at the midpoint frame
    of the transition run.

    @input:  trajectories: episode index -> dict with "xyz" (N, 3) and
             optionally "gripper_signal" (raw, preferred) or "gripper"
             (sign-binarized fallback, detected via single-frame flips).
             min_trend_steps / min_swing: trend qualification parameters.
    @output: Dict with "pick" and "place" entries, each holding xyz (M, 3),
             episodes, frames (mid-run within-episode indices), transition
             span start/end frames, and gripper signal at both ends.
    @scenario: Shared detection step for the 3D pick/place plot; also
              usable standalone for sanity-checking episode segmentation.
    """

    def _empty_bucket() -> Dict[str, list]:
        return {
            "xyz": [],
            "episodes": [],
            "frames": [],
            "span_start": [],
            "span_end": [],
            "gripper_before": [],
            "gripper_after": [],
        }

    picks = _empty_bucket()
    places = _empty_bucket()

    for ep_idx in sorted(trajectories.keys()):
        traj = trajectories[ep_idx]
        xyz = np.asarray(traj["xyz"], dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] < 3 or len(xyz) < 2:
            continue

        if "gripper_signal" in traj:
            signal = np.asarray(traj["gripper_signal"], dtype=np.float64).reshape(-1)
        elif "gripper" in traj:
            # Sign-binarized {-1, 0, 1} or {0, 1} fallback: a trend is a
            # single-frame flip, so bypass the ramp-trend parameters.
            signal = np.sign(
                np.asarray(traj["gripper"], dtype=np.float64).reshape(-1)
            )
            flips = np.flatnonzero(signal[1:] != signal[:-1]) + 1
            events = [
                (int(t - 1), int(t), 1 if signal[t] > signal[t - 1] else -1)
                for t in flips
            ]
            for start, end, direction in events:
                bucket = places if direction > 0 else picks
                mid = (start + end) // 2
                bucket["xyz"].append(xyz[mid, :3])
                bucket["episodes"].append(int(ep_idx))
                bucket["frames"].append(mid)
                bucket["span_start"].append(start)
                bucket["span_end"].append(end)
                bucket["gripper_before"].append(float(signal[start]))
                bucket["gripper_after"].append(float(signal[end]))
            continue
        else:
            logger.warning("Episode %s has no gripper signal; skipped.", ep_idx)
            continue

        if len(signal) != len(xyz):
            logger.warning(
                "Episode %s gripper length (%d) != xyz length (%d); skipped.",
                ep_idx, len(signal), len(xyz),
            )
            continue

        for run in _find_trend_runs(signal, min_steps=min_trend_steps, min_swing=min_swing):
            # falling 1 -> 0 (closing) = pick ; rising 0 -> 1 (opening) = place
            bucket = places if run["direction"] > 0 else picks
            mid = (run["start"] + run["end"]) // 2
            bucket["xyz"].append(xyz[mid, :3])
            bucket["episodes"].append(int(ep_idx))
            bucket["frames"].append(mid)
            bucket["span_start"].append(run["start"])
            bucket["span_end"].append(run["end"])
            bucket["gripper_before"].append(float(signal[run["start"]]))
            bucket["gripper_after"].append(float(signal[run["end"]]))

    def _finalize(bucket: Dict[str, list]) -> Dict[str, Any]:
        n = len(bucket["xyz"])
        dtypes = {
            "episodes": np.int64, "frames": np.int64,
            "span_start": np.int64, "span_end": np.int64,
            "gripper_before": np.float64, "gripper_after": np.float64,
        }
        if n == 0:
            bucket["xyz"] = np.empty((0, 3))
            for key, dtype in dtypes.items():
                bucket[key] = np.empty(0, dtype=dtype)
        else:
            bucket["xyz"] = np.asarray(bucket["xyz"], dtype=np.float64)
            for key, dtype in dtypes.items():
                bucket[key] = np.asarray(bucket[key], dtype=dtype)
        return bucket

    result = {
        "pick": _finalize(picks),
        "place": _finalize(places),
        "min_trend_steps": min_trend_steps,
        "min_swing": min_swing,
        "n_episodes": len(trajectories),
    }
    logger.info(
        "Pick/place detection (trend: >=%d steps, swing >= %.2f): "
        "%d episodes -> %d picks, %d places",
        min_trend_steps, min_swing, result["n_episodes"],
        len(result["pick"]["xyz"]), len(result["place"]["xyz"]),
    )
    return result


def plot_pick_place_3d(
    trajectories: Dict[int, dict],
    max_episodes: int = 50,
    camera: Optional[Dict[str, Any]] = None,
    marker_size: int = 6,
    trajectory_opacity: float = 0.3,
    min_trend_steps: int = 3,
    min_swing: float = 0.2,
    pick_color: str = PICK_COLOR,
    place_color: str = PLACE_COLOR,
    event_opacity: float = 0.85,
    task_name: Optional[str] = None,
) -> Any:
    """
    Create an interactive 3D scatter plot of trajectories with pick/place points.

    Renders episode trajectory paths as thin semi-transparent lines, pick
    points (gripper trend 1 -> 0, closing) as blue markers, and place points
    (gripper trend 0 -> 1, opening) as orange markers.

    @input:  trajectories: episode index -> {"xyz": (N, 3), gripper signal};
             see extract_pick_place_points() for accepted keys.
             max_episodes: cap on number of episode trajectories to render.
             camera: optional scene.camera dict or preset name.
             marker_size: pixel size of pick/place markers.
             trajectory_opacity: alpha of trajectory lines (0-1).
             min_trend_steps / min_swing: transition trend parameters.
             pick_color / place_color: marker colors for the two event types.
             event_opacity: alpha of pick/place markers (0-1).
             task_name: dataset/task name shown in the plot title.
    @output: plotly.graph_objects.Figure ready for display or export.
    @scenario: Verify grasp/release spatial distribution against object
              spawn ranges and task target locations.
    """
    import plotly.graph_objects as go

    events = extract_pick_place_points(
        trajectories, min_trend_steps=min_trend_steps, min_swing=min_swing,
    )

    fig = go.Figure()

    # ------------------------------------------------------------------
    # Trace Group 1: Episode trajectories (thin semi-transparent lines)
    # ------------------------------------------------------------------
    ep_keys = sorted(trajectories.keys())[:max_episodes]
    logger.info(
        "Rendering %d / %d episode trajectories for pick/place 3D plot.",
        len(ep_keys), len(trajectories),
    )
    for ep_idx in ep_keys:
        xyz = np.asarray(trajectories[ep_idx]["xyz"], dtype=np.float64)
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="lines",
                line=dict(
                    color=f"rgba(180,180,180,{trajectory_opacity})",
                    width=1,
                ),
                name=f"Episode {ep_idx}",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # ------------------------------------------------------------------
    # Trace Group 2: Pick and place transition points
    # ------------------------------------------------------------------
    hover_template = (
        "%{customdata[0]}<br>"
        "Pos: (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>"
        "Episode: %{customdata[1]}<br>"
        "Transition frames: %{customdata[2]} - %{customdata[3]}<br>"
        "Gripper: %{customdata[4]:.2f} -> %{customdata[5]:.2f}"
        "<extra></extra>"
    )
    for event_name, color in (("pick", pick_color), ("place", place_color)):
        bucket = events[event_name]
        label = "Pick (gripper closes)" if event_name == "pick" else "Place (gripper opens)"
        if len(bucket["xyz"]) == 0:
            logger.warning("No %s events detected; trace omitted.", event_name)
            continue
        customdata = np.column_stack([
            np.full(len(bucket["xyz"]), label),
            bucket["episodes"],
            bucket["span_start"],
            bucket["span_end"],
            bucket["gripper_before"],
            bucket["gripper_after"],
        ])
        fig.add_trace(
            go.Scatter3d(
                x=bucket["xyz"][:, 0],
                y=bucket["xyz"][:, 1],
                z=bucket["xyz"][:, 2],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=color,
                    opacity=event_opacity,
                ),
                name=label,
                hovertemplate=hover_template,
                customdata=customdata,
            )
        )

    # ------------------------------------------------------------------
    # Layout configuration
    # ------------------------------------------------------------------
    if isinstance(camera, str) and camera in CAMERA_PRESETS:
        _camera = CAMERA_PRESETS[camera]
    elif isinstance(camera, dict):
        _camera = camera
    else:
        _camera = CAMERA_PRESETS["isometric"]

    fig.update_layout(
        scene=dict(
            xaxis_title=dict(text="X (m)", font=_PLOTLY_FONT),
            yaxis_title=dict(text="Y (m)", font=_PLOTLY_FONT),
            zaxis_title=dict(text="Z (m)", font=_PLOTLY_FONT),
            aspectmode="data",
            camera=_camera,
        ),
        title=dict(
            text=(
                f"{task_name} — " if task_name else ""
            ) + (
                f"Gripper Pick/Place Points in 3D Workspace "
                f"({len(events['pick']['xyz'])} picks, "
                f"{len(events['place']['xyz'])} places)"
            ),
            font=_PLOTLY_FONT,
        ),
        font=_PLOTLY_FONT,
        legend=dict(itemsizing="constant"),
        width=1200,
        height=900,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def _load_trajectories(dataset) -> Dict[int, dict]:
    """Build a minimal trajectories dict from a LeRobot dataset.

    @input:  dataset: LeRobotDataset-like object providing frame access.
    @output: episode index -> {"xyz", "gripper_signal"} per-frame arrays.
    @scenario: Standalone python -m entry path that skips EmixIndex building.
    """
    from ..candidates.loaders import (
        load_actions,
        load_eef_poses,
        load_episode_indices,
        load_gripper_signal,
    )

    episode_indices = load_episode_indices(dataset)
    poses = load_eef_poses(dataset)
    actions = load_actions(dataset)
    gripper_signal = load_gripper_signal(actions, dataset)

    trajectories: Dict[int, dict] = {}
    for ep_idx in np.unique(episode_indices):
        mask = episode_indices == ep_idx
        trajectories[int(ep_idx)] = {
            "xyz": poses[mask, :3],
            "gripper_signal": gripper_signal[mask],
        }
    return trajectories


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for standalone pick/place visualization."""
    import argparse

    from ..datasets.path_utils import (
        default_repo_id_from_dataset_root,
        require_lerobot_dataset_root,
    )

    parser = argparse.ArgumentParser(
        prog="python -m embodied_mixup.visualization.plot_actions",
        description="Plot gripper pick/place transition points over 3D trajectories.",
    )
    parser.add_argument("--root", required=True, help="LeRobot dataset root path.")
    parser.add_argument("--repo-id", default=None, help="Dataset repo id (derived from root if omitted).")
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: outputs/pick_place/<repo_id>_pick_place.html).")
    parser.add_argument("--camera", default="isometric",
                        help=f"Camera preset: {', '.join(CAMERA_PRESETS)}.")
    parser.add_argument("--max-episodes", type=int, default=50,
                        help="Max trajectory lines to render (default: 50).")
    parser.add_argument("--trend-steps", type=int, default=3,
                        help="Min consecutive same-direction gripper steps for a transition (default: 3).")
    parser.add_argument("--trend-swing", type=float, default=0.2,
                        help="Min total gripper swing for a transition (default: 0.2).")
    parser.add_argument("--marker-size", type=int, default=6)
    parser.add_argument("--png", default=None,
                        help="PNG snapshot path (default: <output>.png next to the HTML).")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip the PNG snapshot and only write the HTML.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dataset_root = require_lerobot_dataset_root(args.root)
    repo_id = (
        args.repo_id.rstrip("/")
        if args.repo_id
        else default_repo_id_from_dataset_root(dataset_root)
    )

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    logger.info("Loading LeRobot dataset: repo_id=%s root=%s", repo_id, dataset_root)
    dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root))

    trajectories = _load_trajectories(dataset)
    fig = plot_pick_place_3d(
        trajectories,
        max_episodes=args.max_episodes,
        camera=args.camera,
        marker_size=args.marker_size,
        min_trend_steps=args.trend_steps,
        min_swing=args.trend_swing,
        task_name=repo_id,
    )

    output_path = (
        Path(args.output)
        if args.output
        else Path("outputs") / "pick_place" / f"{repo_id}_pick_place.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Inline plotly.js so the HTML is self-contained (file:// offline use and
    # headless-Chrome PNG rendering without network access).
    fig.write_html(str(output_path), include_plotlyjs=True)
    logger.info("Pick/place plot saved to: %s", output_path)

    if not args.no_png:
        png_path = Path(args.png) if args.png else output_path.with_suffix(".png")
        try:
            _render_png_snapshot(output_path, png_path)
            logger.info("Pick/place PNG snapshot saved to: %s", png_path)
        except Exception as exc:
            logger.warning("PNG snapshot skipped (%s); HTML is still available.", exc)


def _render_png_snapshot(html_path: Path, png_path: Path) -> None:
    """Screenshot the plot HTML via the repo's headless-Chrome renderer.

    @input:  html_path: self-contained plot HTML written by main().
             png_path: target PNG path.
    @output: png_path written (parents created).
    @scenario: Reuse scripts/render_vis_png.py (the flow's vis phase) so
              pick/place snapshots match the other vis PNGs pixel-for-pixel
              in quality (2x DPI, content-cropped).
    """
    import importlib.util

    render_script = Path(__file__).resolve().parents[2] / "scripts" / "render_vis_png.py"
    if not render_script.is_file():
        raise FileNotFoundError(f"Renderer not found: {render_script}")
    spec = importlib.util.spec_from_file_location("render_vis_png", render_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Single 1200x900 figure: a viewport slightly larger than the layout
    # keeps the content crop tight after the renderer's bbox pass.
    module.render_html_to_png(html_path, png_path, width=1400, height=1100)


if __name__ == "__main__":
    main()
