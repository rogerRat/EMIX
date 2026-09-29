"""2D projection views (XY, XZ, YZ) of trajectory intersections and clusters.

Provides three subplots showing top-down (XY), front (XZ), and side (YZ)
projections of the workspace, each with trajectory paths as thin lines
and intersection midpoints colored by DBSCAN cluster label.

@input:  viz_data dict from prepare_visualization_data().
@output: plotly.graph_objects.Figure with 1x3 subplot grid.
@scenario: Detailed planar analysis when 3D depth occlusion obscures
          intersection distribution.  Each projection reveals
          cluster structure from a different orthogonal direction.
"""

import logging
from typing import Dict, List, Optional, Any

import numpy as np

from .utils import _get_cluster_colors, _PLOTLY_FONT

logger = logging.getLogger(__name__)

# Column indices for each projection plane.
_PROJECTIONS = [
    ("XY", 0, 1),  # top-down: x vs y
    ("XZ", 0, 2),  # front:    x vs z
    ("YZ", 1, 2),  # side:     y vs z
]


def plot_2d_projections(
    viz_data: Dict[str, Any],
    max_episodes: int = 50,
    marker_size: int = 4,
    noise_marker_size: int = 2,
    trajectory_opacity: float = 0.3,
    cluster_opacity: float = 0.8,
    noise_opacity: float = 0.3,
    episode_highlights: Optional[List[int]] = None,
) -> Any:
    """
    Create three 2D projection subplots (XY, XZ, YZ) of intersection clusters.

    Each subplot shows:
    - Episode trajectories as thin semi-transparent lines.
    - Intersection midpoints colored by DBSCAN cluster label.
    - Noise points (label == -1) as small gray dots.

    @input:  viz_data: output of prepare_visualization_data().
             max_episodes: cap on background trajectory rendering.
             marker_size: pixel size of cluster markers in 2D.
             noise_marker_size: pixel size of noise markers in 2D.
             trajectory_opacity: alpha of trajectory lines.
             cluster_opacity: alpha of cluster markers.
             noise_opacity: alpha of noise markers.
             episode_highlights: optional list of episode indices to
                                render with thicker, more opaque lines.
    @output: plotly.graph_objects.Figure with 1 row x 3 columns.
    @scenario: Planar analysis for understanding intersection geometry
              when perspective effects in 3D make patterns hard to read.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    trajectories = viz_data.get("trajectories", {})
    midpoints: np.ndarray = viz_data.get("midpoints", np.empty((0, 6)))
    labels: np.ndarray = viz_data.get("cluster_labels", np.empty(0, dtype=int))
    ep_keys = sorted(trajectories.keys())[:max_episodes]
    noise_mask = labels == -1 if len(labels) > 0 else np.zeros(0, dtype=bool)
    cluster_mask = ~noise_mask
    unique_labels = (
        sorted(int(x) for x in set(labels[cluster_mask]))
        if cluster_mask.any() else []
    )
    cluster_colors = _get_cluster_colors(len(unique_labels))

    highlight_set = set(episode_highlights or [])

    # Track trace indices for visibility toggle (JS layer)
    traj_indices_2d: list[int] = []
    noise_indices_2d: list[int] = []
    cluster_indices_2d: list[int] = []
    raw_rejected_indices_2d: list[int] = []
    raw_accepted_cluster_indices_2d: list[int] = []

    has_raw_2d = viz_data.get("has_raw_data", False)
    if has_raw_2d:
        raw_midpoints_all_2d: np.ndarray = viz_data.get("raw_midpoints", np.empty((0, 6)))
        raw_accepted_mask_2d: np.ndarray = viz_data.get("raw_accepted_mask", np.empty(0, dtype=bool))
        raw_cluster_ids_2d: np.ndarray = viz_data.get("raw_cluster_ids", np.empty(0, dtype=np.int64))
    else:
        raw_midpoints_all_2d = np.empty((0, 6))
        raw_accepted_mask_2d = np.empty(0, dtype=bool)
        raw_cluster_ids_2d = np.empty(0, dtype=np.int64)

    if has_raw_2d and len(raw_accepted_mask_2d) > 0:
        rejected_mask_2d = ~raw_accepted_mask_2d
        if raw_accepted_mask_2d.any():
            raw_unique_labels = sorted(
                int(x) for x in set(raw_cluster_ids_2d[raw_accepted_mask_2d]) if x >= 0
            )
            noise_in_raw_2d = raw_accepted_mask_2d & (raw_cluster_ids_2d == -1)
            all_raw_cids = list(raw_unique_labels)
            if noise_in_raw_2d.any():
                all_raw_cids.append(-1)
            raw_colors_2d = _get_cluster_colors(max(len(all_raw_cids), 1))
        else:
            all_raw_cids = []
            raw_colors_2d = []
    else:
        rejected_mask_2d = np.zeros(0, dtype=bool)
        all_raw_cids = []
        raw_colors_2d = []

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[p[0] for p in _PROJECTIONS],
        horizontal_spacing=0.08,
    )

    # ------------------------------------------------------------------
    # For each projection (col), render trajectories + intersections
    # ------------------------------------------------------------------
    for col, (plane_name, col_i, col_j) in enumerate(_PROJECTIONS, start=1):
        # Draw each RoboTwin actor's randomization footprint in this projection.
        for obj in viz_data.get("robotwin_object_ranges", []):
            bounds = {0: obj["x"], 1: obj["y"], 2: (0.74, 0.80)}
            lo_i, hi_i = bounds[col_i]
            lo_j, hi_j = bounds[col_j]
            fig.add_trace(
                go.Scatter(
                    x=[lo_i, hi_i, hi_i, lo_i, lo_i],
                    y=[lo_j, lo_j, hi_j, hi_j, lo_j],
                    mode="lines",
                    line=dict(color="rgba(255,80,20,0.65)", width=2),
                    name=f"Object range: {obj['name']}" if col == 1 else f"{obj['name']} ({plane_name})",
                    legendgroup=f"obj-{obj['name']}", showlegend=(col == 1),
                    hoverinfo="name",
                ), row=1, col=col,
            )
        # -- Trajectories --
        for ep_idx in ep_keys:
            xyz = trajectories[ep_idx]["xyz"]
            is_highlighted = ep_idx in highlight_set

            fig.add_trace(
                go.Scatter(
                    x=xyz[:, col_i],
                    y=xyz[:, col_j],
                    mode="lines",
                    line=dict(
                        color=(
                            f"rgba(255,100,100,{0.6})"
                            if is_highlighted
                            else f"rgba(180,180,180,{trajectory_opacity})"
                        ),
                        width=2 if is_highlighted else 1,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
            traj_indices_2d.append(len(fig.data) - 1)
        if noise_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=midpoints[noise_mask, col_i],
                    y=midpoints[noise_mask, col_j],
                    mode="markers",
                    marker=dict(
                        size=noise_marker_size,
                        color="gray",
                        opacity=noise_opacity,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
            noise_indices_2d.append(len(fig.data) - 1)
        if cluster_mask.any():
            for color_idx, cluster_id in enumerate(unique_labels):
                mask = labels == cluster_id

                fig.add_trace(
                    go.Scatter(
                        x=midpoints[mask, col_i],
                        y=midpoints[mask, col_j],
                        mode="markers",
                        marker=dict(
                            size=marker_size,
                            color=cluster_colors[color_idx],
                            opacity=cluster_opacity,
                        ),
                        name=f"C{cluster_id}" if col == 1 else f"C{cluster_id} ({plane_name})",
                        showlegend=(col == 1),  # legend only in first column
                        legendgroup=f"c{cluster_id}",
                    ),
                    row=1,
                    col=col,
                )
                cluster_indices_2d.append(len(fig.data) - 1)

        # -- Raw rejected points (visible only in Raw mode) --
        if rejected_mask_2d.any():
            fig.add_trace(
                go.Scatter(
                    x=raw_midpoints_all_2d[rejected_mask_2d, col_i],
                    y=raw_midpoints_all_2d[rejected_mask_2d, col_j],
                    mode="markers",
                    visible=False,
                    marker=dict(
                        size=noise_marker_size,
                        color="magenta",
                        opacity=0.5,
                    ),
                    showlegend=(col == 1),
                    name="Rejected (raw)" if col == 1 else f"Rejected (raw, {plane_name})",
                    legendgroup="raw_rejected",
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )
            raw_rejected_indices_2d.append(len(fig.data) - 1)

        # -- Raw accepted cluster points (visible only in Raw mode) --
        if has_raw_2d and raw_accepted_mask_2d.any():
            for color_idx, cid in enumerate(all_raw_cids):
                if cid == -1:
                    raw_cmask = noise_in_raw_2d
                    cname = "Raw accepted (no C)" if col == 1 else f"Raw accepted (no C, {plane_name})"
                    ccolor = "gray"
                else:
                    raw_cmask = raw_accepted_mask_2d & (raw_cluster_ids_2d == cid)
                    cname = f"Raw C{cid}" if col == 1 else f"Raw C{cid} ({plane_name})"
                    ccolor = raw_colors_2d[color_idx % len(raw_colors_2d)]

                if not raw_cmask.any():
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=raw_midpoints_all_2d[raw_cmask, col_i],
                        y=raw_midpoints_all_2d[raw_cmask, col_j],
                        mode="markers",
                        visible=False,
                        marker=dict(
                            size=marker_size,
                            color=ccolor,
                            opacity=cluster_opacity,
                        ),
                        showlegend=(col == 1),
                        name=cname if col == 1 else f"Raw C{cid} ({plane_name})",
                        legendgroup=f"raw_c{cid}",
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col,
                )
                raw_accepted_cluster_indices_2d.append(len(fig.data) - 1)

        fig.update_xaxes(title_text=f"{plane_name[0]} (m)", row=1, col=col)
        fig.update_yaxes(title_text=f"{plane_name[1]} (m)", row=1, col=col)
        # Lock aspect ratio to prevent deformation
        fig.update_yaxes(scaleanchor=f"x{col}", scaleratio=1.0, row=1, col=col)

    fig.update_layout(
        title=dict(text="2D Projections: Intersection Clusters (XY | XZ | YZ)", font=_PLOTLY_FONT),
        font=_PLOTLY_FONT,
        width=1600,
        height=600,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    # ------------------------------------------------------------------
    # Build button config for external JS layer
    # ------------------------------------------------------------------
    total_traces_2d = len(fig.data)

    def _make_vis_2d(show_noise: bool, show_clusters: bool,
                     show_raw_rej: bool, show_raw_clusters: bool) -> list[bool]:
        vis = [False] * total_traces_2d
        for i in traj_indices_2d:
            vis[i] = True
        for i in noise_indices_2d:
            vis[i] = show_noise
        for i in cluster_indices_2d:
            vis[i] = show_clusters
        for i in raw_rejected_indices_2d:
            vis[i] = show_raw_rej
        for i in raw_accepted_cluster_indices_2d:
            vis[i] = show_raw_clusters
        return vis

    cluster_filtered_vis_2d = _make_vis_2d(True, True, False, False)
    density_filtered_vis_2d = _make_vis_2d(False, True, False, False)  # 2D: density same as cluster
    cluster_raw_vis_2d = _make_vis_2d(False, False, True, True)
    density_raw_vis_2d = _make_vis_2d(False, False, True, True)  # 2D: density same as cluster

    button_config_2d = {
        "fig_id": "projections-2d",
        "visible_states": {
            "cluster_filtered": cluster_filtered_vis_2d,
            "density_filtered": density_filtered_vis_2d,
            "cluster_raw": cluster_raw_vis_2d,
            "density_raw": density_raw_vis_2d,
        },
        "has_raw_data": has_raw_2d,
    }

    return fig, button_config_2d
