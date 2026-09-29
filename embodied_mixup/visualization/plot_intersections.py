"""3D interactive scatter plot of trajectories, intersection midpoints,
and DBSCAN clustering results.

Renders episode trajectories as thin lines and intersection midpoints as
colored markers in a 3D work-space coordinate frame using Plotly WebGL.

@input:  viz_data dict from prepare_visualization_data().
@output: plotly.graph_objects.Figure (call .show() or .write_html()).
@scenario: Primary visualization for validating intersection detection
          and DBSCAN cluster quality in 3D physical space.
"""

import logging
from typing import Dict, List, Optional, Any

import numpy as np

from .utils import _get_cluster_colors, _PLOTLY_FONT, _compute_local_point_density
from .camera_presets import CAMERA_PRESETS

logger = logging.getLogger(__name__)


def plot_intersections_3d(
    viz_data: Dict[str, Any],
    max_episodes: int = 50,
    camera: Optional[Dict[str, Any]] = None,
    marker_size: int = 5,
    noise_marker_size: int = 3,
    trajectory_opacity: float = 0.3,
    cluster_opacity: float = 0.8,
    noise_opacity: float = 0.4,
) -> Any:
    """
    Create an interactive 3D scatter plot of trajectories and intersections.

    Renders episode trajectory paths as thin semi-transparent lines and
    intersection midpoints as markers colored by DBSCAN cluster label.
    Noise points (label == -1) are rendered as small gray dots.

    @input:  viz_data: output of prepare_visualization_data() with keys:
            trajectories, midpoints, cluster_labels, divergences, member_pairs.
             max_episodes: cap on number of episode trajectories to render
                          (performance constraint: 50 is smooth on most GPUs).
             camera: optional scene.camera dict; falls back to 'isometric' preset.
             marker_size: pixel size of cluster marker points.
             noise_marker_size: pixel size of noise (unclustered) markers.
             trajectory_opacity: alpha of trajectory lines (0-1).
             cluster_opacity: alpha of cluster markers (0-1).
             noise_opacity: alpha of noise markers (0-1).
    @output: plotly.graph_objects.Figure ready for display or export.
    @scenario: Primary 3D visualization for verifying intersection detection
              and DBSCAN clustering correctness in physical workspace.
    """
    import plotly.graph_objects as go

    trajectories = viz_data.get("trajectories", {})
    midpoints: np.ndarray = viz_data.get("midpoints", np.empty((0, 6)))
    labels: np.ndarray = viz_data.get("cluster_labels", np.empty(0, dtype=int))
    divergences: np.ndarray = viz_data.get("divergences", np.empty(0))
    member_pairs: List[Any] = viz_data.get("member_pairs", [])
    local_density: np.ndarray = viz_data.get("local_density", np.empty(0, dtype=np.int32))

    if len(local_density) != len(midpoints):
        logger.warning(
            "Local density array length (%d) does not match midpoint count (%d).",
            len(local_density),
            len(midpoints),
        )
        local_density = np.zeros(len(midpoints), dtype=np.int32)

    # Materialize member metadata once.  The previous per-cluster list
    # comprehensions repeatedly walked every VizMemberPair and allocated a
    # temporary Python object list before building each hover array.
    if member_pairs:
        member_ep_0 = np.fromiter(
            (p.member_episodes[0] for p in member_pairs), dtype=np.int64,
            count=len(member_pairs),
        )
        member_frame_0 = np.fromiter(
            (p.member_indices[0] for p in member_pairs), dtype=np.int64,
            count=len(member_pairs),
        )
        member_ep_1 = np.fromiter(
            (p.member_episodes[1] for p in member_pairs), dtype=np.int64,
            count=len(member_pairs),
        )
        member_frame_1 = np.fromiter(
            (p.member_indices[1] for p in member_pairs), dtype=np.int64,
            count=len(member_pairs),
        )
    else:
        member_ep_0 = member_frame_0 = member_ep_1 = member_frame_1 = np.empty(0, dtype=np.int64)

    fig = go.Figure()
    trajectory_trace_count = 0

    # RoboTwin object randomization volumes (transparent cuboids).
    for obj in viz_data.get("robotwin_object_ranges", []):
        x0, x1 = obj["x"]; y0, y1 = obj["y"]
        z0, z1 = 0.74, 0.80
        vertices = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
                    (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for a, b in edges:
            fig.add_trace(go.Scatter3d(x=[vertices[a][0], vertices[b][0]],
                                       y=[vertices[a][1], vertices[b][1]],
                                       z=[vertices[a][2], vertices[b][2]], mode="lines",
                                       line=dict(color="rgba(255,80,20,0.65)", width=4),
                                       name=f"Object range: {obj['name']}", legendgroup=f"obj-{obj['name']}",
                                       showlegend=(a == 0), hoverinfo="name"))
    cluster_trace_indices: list[int] = []
    # The density view reuses the filtered/raw point traces and only changes
    # marker colors.  Keeping these mappings lets the HTML layer switch modes
    # without embedding a second full copy of every point and customdata row.
    marker_states: dict[str, dict[str, dict[str, Any]]] = {
        "cluster_filtered": {},
        "density_filtered": {},
        "cluster_raw": {},
        "density_raw": {},
    }
    filtered_density_median = float(np.median(local_density)) if len(local_density) else 0.0
    filtered_density_min = int(np.min(local_density)) if len(local_density) else 0
    filtered_density_max = int(np.max(local_density)) if len(local_density) else 0
    filtered_density_hover = (
        "Pos: (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>"
        "Cluster: %{customdata[0]}<br>"
        "Nearby points: %{customdata[1]}<br>"
        "Divergence: %{customdata[2]:.4f}<br>"
        "Member 0: ep %{customdata[3]} frame %{customdata[4]}<br>"
        "Member 1: ep %{customdata[5]} frame %{customdata[6]}"
        "<extra></extra>"
    )

    # ------------------------------------------------------------------
    # Trace Group 1: Episode trajectories (thin semi-transparent lines)
    # ------------------------------------------------------------------
    ep_keys = sorted(trajectories.keys())[:max_episodes]
    logger.info(
        "Rendering %d / %d episode trajectories for 3D scatter plot.",
        len(ep_keys),
        len(trajectories),
    )

    for ep_idx in ep_keys:
        xyz = trajectories[ep_idx]["xyz"]
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
        trajectory_trace_count += 1

    # ------------------------------------------------------------------
    # Trace Group 2: Noise points (unclustered, label == -1)
    # ------------------------------------------------------------------
    if len(labels) > 0:
        noise_mask = labels == -1
    else:
        noise_mask = np.zeros(0, dtype=bool)
    cluster_mask = ~noise_mask

    if noise_mask.any():
        # Build customdata for hover: [density, divergence, member_0_ep, member_1_ep]
        customdata = np.column_stack([
            np.full(int(noise_mask.sum()), -1, dtype=int),
            local_density[noise_mask],
            divergences[noise_mask],
            member_ep_0[noise_mask],
            member_frame_0[noise_mask],
            member_ep_1[noise_mask],
            member_frame_1[noise_mask],
        ])

        fig.add_trace(
            go.Scatter3d(
                x=midpoints[noise_mask, 0],
                y=midpoints[noise_mask, 1],
                z=midpoints[noise_mask, 2],
                mode="markers",
                marker=dict(
                    size=noise_marker_size,
                    color="gray",
                    opacity=noise_opacity,
                ),
                name="Noise (unclustered)",
                hovertemplate=(
                    "Noise point<br>"
                    "Pos: (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>"
                    "Nearby points: %{customdata[1]}<br>"
                    "Divergence: %{customdata[2]:.4f}<br>"
                    "Member 0 ep: %{customdata[3]}<br>"
                    "Member 1 ep: %{customdata[5]}"
                    "<extra></extra>"
                ),
                customdata=customdata,
            )
        )
        noise_trace_idx = len(fig.data) - 1
        cluster_trace_indices.append(noise_trace_idx)
        marker_states["cluster_filtered"][str(noise_trace_idx)] = {
            "color": "gray", "showscale": False,
            "hovertemplate": fig.data[noise_trace_idx].hovertemplate,
        }
        marker_states["density_filtered"][str(noise_trace_idx)] = {
            "color": local_density[noise_mask].tolist(),
            "colorscale": "Viridis",
            "cmin": filtered_density_min,
            "cmax": filtered_density_max,
            "cmid": filtered_density_median,
            "showscale": True,
            "colorbar": {"title": {"text": "Nearby Points", "side": "right"}, "thickness": 20, "len": 0.6},
            "hovertemplate": filtered_density_hover,
        }

    # ------------------------------------------------------------------
    # Trace Group 3: Cluster points, one trace per cluster for legend
    # ------------------------------------------------------------------
    if cluster_mask.any():
        unique_labels = sorted(int(x) for x in set(labels[cluster_mask]))
        colors = _get_cluster_colors(len(unique_labels))

        for color_idx, cluster_id in enumerate(unique_labels):
            mask = labels == cluster_id
            n_pts = int(mask.sum())

            customdata = np.column_stack([
                np.full(n_pts, cluster_id, dtype=int),
                local_density[mask],
                divergences[mask],
                member_ep_0[mask],
                member_frame_0[mask],
                member_ep_1[mask],
                member_frame_1[mask],
            ])

            fig.add_trace(
                go.Scatter3d(
                    x=midpoints[mask, 0],
                    y=midpoints[mask, 1],
                    z=midpoints[mask, 2],
                    mode="markers",
                marker=dict(
                    size=marker_size,
                    color=colors[color_idx],
                    opacity=cluster_opacity,
                ),
                    name=f"Cluster {cluster_id}",
                    hovertemplate=(
                        "Cluster %{customdata[0]}<br>"
                        "Pos: (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>"
                        "Nearby points: %{customdata[1]}<br>"
                        "Divergence: %{customdata[2]:.4f}<br>"
                        "Member 0: ep %{customdata[3]} frame %{customdata[4]}<br>"
                        "Member 1: ep %{customdata[5]} frame %{customdata[6]}"
                        "<extra></extra>"
                    ),
                    customdata=customdata,
                )
            )
            cluster_trace_idx = len(fig.data) - 1
            cluster_trace_indices.append(cluster_trace_idx)
            marker_states["cluster_filtered"][str(cluster_trace_idx)] = {
                "color": colors[color_idx], "showscale": False,
                "hovertemplate": fig.data[cluster_trace_idx].hovertemplate,
            }
            marker_states["density_filtered"][str(cluster_trace_idx)] = {
                "color": local_density[mask].tolist(),
                "colorscale": "Viridis",
                "cmin": filtered_density_min,
                "cmax": filtered_density_max,
                "cmid": filtered_density_median,
                "showscale": False,
                "hovertemplate": filtered_density_hover,
            }

    # ------------------------------------------------------------------
    # Trace Group 4: Raw rejected midpoints (visible only in Raw mode)
    # ------------------------------------------------------------------
    raw_rejected_trace_idx: Optional[int] = None
    raw_accepted_cluster_indices: list[int] = []
    raw_density_masks: dict[int, np.ndarray] = {}
    has_raw = viz_data.get("has_raw_data", False)

    if has_raw:
        raw_midpoints_all: np.ndarray = viz_data.get("raw_midpoints", np.empty((0, 6)))
        raw_accepted_mask: np.ndarray = viz_data.get("raw_accepted_mask", np.empty(0, dtype=bool))
        raw_divs: np.ndarray = viz_data.get("raw_divergences", np.empty(0))
        raw_cluster_ids_arr: np.ndarray = viz_data.get("raw_cluster_ids", np.empty(0, dtype=np.int64))
        raw_ep_a_arr: np.ndarray = viz_data.get("raw_ep_a", np.empty(0, dtype=np.int64))
        raw_frame_a_arr: np.ndarray = viz_data.get("raw_frame_a", np.empty(0, dtype=np.int64))
        raw_ep_b_arr: np.ndarray = viz_data.get("raw_ep_b", np.empty(0, dtype=np.int64))
        raw_frame_b_arr: np.ndarray = viz_data.get("raw_frame_b", np.empty(0, dtype=np.int64))

        if len(raw_accepted_mask) > 0:
            rejected_mask = ~raw_accepted_mask
        else:
            rejected_mask = np.zeros(0, dtype=bool)

        raw_density_loc = (
            _compute_local_point_density(raw_midpoints_all[:, :3])
            if len(raw_midpoints_all) else np.empty(0, dtype=np.int32)
        )
        raw_density_median = float(np.median(raw_density_loc)) if len(raw_density_loc) else 0.0
        raw_density_min = int(np.min(raw_density_loc)) if len(raw_density_loc) else 0
        raw_density_max = int(np.max(raw_density_loc)) if len(raw_density_loc) else 0
        raw_density_hover = (
            "Raw pair<br>"
            "Pos: (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>"
            "Cluster: %{customdata[0]}<br>"
            "Nearby points: %{customdata[1]}<br>"
            "Divergence: %{customdata[2]:.4f}<br>"
            "Member 0: ep %{customdata[3]} frame %{customdata[4]}<br>"
            "Member 1: ep %{customdata[5]} frame %{customdata[6]}"
            "<extra></extra>"
        )

        # Raw rejected trace
        if rejected_mask.any():
            raw_rej_customdata = np.column_stack([
                raw_cluster_ids_arr[rejected_mask],
                raw_density_loc[rejected_mask],
                raw_divs[rejected_mask],
                raw_ep_a_arr[rejected_mask],
                raw_frame_a_arr[rejected_mask],
                raw_ep_b_arr[rejected_mask],
                raw_frame_b_arr[rejected_mask],
            ])
            fig.add_trace(
                go.Scatter3d(
                    x=raw_midpoints_all[rejected_mask, 0],
                    y=raw_midpoints_all[rejected_mask, 1],
                    z=raw_midpoints_all[rejected_mask, 2],
                    mode="markers",
                    visible=False,
                    marker=dict(
                        size=noise_marker_size,
                        color="magenta",
                        opacity=0.5,
                    ),
                    name="Rejected (raw only)",
                    hovertemplate=(
                        "Rejected raw pair<br>"
                        "Pos: (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>"
                        "Divergence: %{customdata[2]:.4f}<br>"
                        "Member 0: ep %{customdata[3]} frame %{customdata[4]}<br>"
                        "Member 1: ep %{customdata[5]} frame %{customdata[6]}"
                        "<extra></extra>"
                    ),
                    customdata=raw_rej_customdata,
                )
            )
            raw_rejected_trace_idx = len(fig.data) - 1
            marker_states["cluster_raw"][str(raw_rejected_trace_idx)] = {
                "color": "magenta", "showscale": False,
                "hovertemplate": fig.data[raw_rejected_trace_idx].hovertemplate,
            }
            raw_density_masks[raw_rejected_trace_idx] = rejected_mask

        # Raw accepted cluster traces (one per cluster ID)
        accepted_mask = raw_accepted_mask
        if accepted_mask.any():
            raw_unique_labels = sorted(
                int(x) for x in set(raw_cluster_ids_arr[accepted_mask]) if x >= 0
            )
            # Also handle raw accepted points that map to cluster -1
            noise_in_raw = accepted_mask & (raw_cluster_ids_arr == -1)
            all_cluster_ids = list(raw_unique_labels)
            if noise_in_raw.any():
                all_cluster_ids.append(-1)

            raw_colors = _get_cluster_colors(max(len(all_cluster_ids), 1))
            for color_idx, cid in enumerate(all_cluster_ids):
                if cid == -1:
                    raw_cmask = noise_in_raw
                    cname = "Raw accepted (no cluster)"
                    ccolor = "gray"
                else:
                    raw_cmask = accepted_mask & (raw_cluster_ids_arr == cid)
                    cname = f"Raw accepted (C{cid})"
                    ccolor = raw_colors[color_idx % len(raw_colors)]

                n_pts = int(raw_cmask.sum())
                if n_pts == 0:
                    continue

                raw_c_customdata = np.column_stack([
                    np.full(n_pts, cid, dtype=int),
                    raw_density_loc[raw_cmask],
                    raw_divs[raw_cmask],
                    raw_ep_a_arr[raw_cmask],
                    raw_frame_a_arr[raw_cmask],
                    raw_ep_b_arr[raw_cmask],
                    raw_frame_b_arr[raw_cmask],
                ])
                fig.add_trace(
                    go.Scatter3d(
                        x=raw_midpoints_all[raw_cmask, 0],
                        y=raw_midpoints_all[raw_cmask, 1],
                        z=raw_midpoints_all[raw_cmask, 2],
                        mode="markers",
                        visible=False,
                    marker=dict(
                        size=marker_size,
                        color=ccolor,
                        opacity=cluster_opacity,
                    ),
                        name=cname,
                        hovertemplate=(
                            "Raw accepted (C%{customdata[0]})<br>"
                            "Pos: (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>"
                            "Divergence: %{customdata[2]:.4f}<br>"
                            "Member 0: ep %{customdata[3]} frame %{customdata[4]}<br>"
                            "Member 1: ep %{customdata[5]} frame %{customdata[6]}"
                            "<extra></extra>"
                        ),
                        customdata=raw_c_customdata,
                    )
                )
                raw_cluster_trace_idx = len(fig.data) - 1
                raw_accepted_cluster_indices.append(raw_cluster_trace_idx)
                marker_states["cluster_raw"][str(raw_cluster_trace_idx)] = {
                    "color": ccolor, "showscale": False,
                    "hovertemplate": fig.data[raw_cluster_trace_idx].hovertemplate,
                }
                raw_density_masks[raw_cluster_trace_idx] = raw_cmask

        # Raw density trace
        if len(raw_midpoints_all) > 0:
            raw_density_colorbar_added = False
            for raw_idx, raw_cmask in raw_density_masks.items():
                marker_states["density_raw"][str(raw_idx)] = {
                    "color": raw_density_loc[raw_cmask].tolist(),
                    "colorscale": "Viridis",
                    "cmin": raw_density_min,
                    "cmax": raw_density_max,
                    "cmid": raw_density_median,
                    "showscale": not raw_density_colorbar_added,
                    "hovertemplate": raw_density_hover,
                }
                if not raw_density_colorbar_added:
                    marker_states["density_raw"][str(raw_idx)]["colorbar"] = {
                        "title": {"text": "Nearby Points", "side": "right"},
                        "thickness": 20,
                        "len": 0.6,
                    }
                    raw_density_colorbar_added = True

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
        title=dict(text="Trajectory Intersections in 3D Workspace", font=_PLOTLY_FONT),
        font=_PLOTLY_FONT,
        legend=dict(itemsizing="constant"),
        width=1200,
        height=900,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # ------------------------------------------------------------------
    # Build button config for external JS layer (no Plotly updatemenu)
    # ------------------------------------------------------------------
    total_traces = len(fig.data)
    traj_count = trajectory_trace_count

    # Identify trace index ranges for filtered data
    filtered_noise_idx: Optional[int] = cluster_trace_indices[0] if (
        cluster_trace_indices and len(labels) > 0 and (labels == -1).any()
    ) else None

    filtered_cluster_start = traj_count + (1 if filtered_noise_idx is not None else 0)
    filtered_cluster_end = filtered_cluster_start + len(cluster_trace_indices) - (
        1 if filtered_noise_idx is not None else 0
    )
    # Build 4 complete visibility arrays
    def _make_vis(show_traj: bool, show_filt_noise: bool,
                  show_filt_clusters: bool, show_filt_density: bool,
                  show_raw_rej: bool, show_raw_clusters: bool,
                  show_raw_density: bool) -> list[bool]:
        vis = [False] * total_traces
        # Trajectories always visible
        for i in range(traj_count):
            vis[i] = True
        # Filtered noise
        if filtered_noise_idx is not None:
            vis[filtered_noise_idx] = show_filt_noise
        # Filtered cluster traces
        for i in range(filtered_cluster_start, filtered_cluster_end):
            vis[i] = show_filt_clusters
        # Raw rejected
        if raw_rejected_trace_idx is not None:
            vis[raw_rejected_trace_idx] = show_raw_rej
        # Raw accepted clusters
        for i in raw_accepted_cluster_indices:
            vis[i] = show_raw_clusters
        return vis

    cluster_filtered_vis = _make_vis(
        True, True, True, False, False, False, False,
    )
    density_filtered_vis = _make_vis(
        True, True, True, True, False, False, False,
    )
    cluster_raw_vis = _make_vis(
        True, False, False, False, True, True, False,
    )
    density_raw_vis = _make_vis(
        True, False, False, False, True, True, True,
    )

    button_config = {
        "fig_id": "intersections-3d",
        "visible_states": {
            "cluster_filtered": cluster_filtered_vis,
            "density_filtered": density_filtered_vis,
            "cluster_raw": cluster_raw_vis,
            "density_raw": density_raw_vis,
        },
        "marker_states": marker_states,
        "has_raw_data": has_raw,
    }

    return fig, button_config


def plot_cluster_histogram(viz_data: Dict[str, Any]) -> Any:
    """Create a bar chart of cluster sizes sorted from largest to smallest.

    Each bar represents one cluster (or noise, label == -1), colored
    using the same palette as the 3D scatter plot.  Cluster labels are
    shown on the x-axis and point counts on the y-axis.

    @input:  viz_data from prepare_visualization_data() with keys
             cluster_labels and (optionally) cluster_info.
    @output: plotly.graph_objects.Figure.
    """
    import plotly.graph_objects as go

    labels: np.ndarray = viz_data.get("cluster_labels", np.empty(0, dtype=int))
    if len(labels) == 0:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text="Cluster Size Distribution (empty)", font=_PLOTLY_FONT),
            font=_PLOTLY_FONT,
        )
        return fig

    unique, counts = np.unique(labels, return_counts=True)

    # Separate noise (label == -1) from real clusters.
    noise_mask = unique == -1
    cluster_mask = ~noise_mask

    # ---- Color mapping: match the 3D plot's cluster_id → color scheme ----
    # The 3D plot sorts cluster IDs numerically, then assigns palette
    # colors in that order.  Replicate the same mapping here so bars
    # and scatter markers share identical colours.
    cluster_unique = unique[cluster_mask]
    cluster_counts = counts[cluster_mask]

    # Build cluster_id → color dict keyed by ID-sorted order.
    sorted_by_id = np.sort(cluster_unique)
    palette = _get_cluster_colors(len(sorted_by_id))
    cluster_color_map: dict[int, str] = {}
    for idx, cid in enumerate(sorted_by_id):
        cluster_color_map[int(cid)] = palette[idx % len(palette)]

    # Sort bars by count descending for display.
    order = np.argsort(-cluster_counts)
    sorted_cluster_ids = cluster_unique[order]
    sorted_cluster_counts = cluster_counts[order]

    # Build bar labels, values, and colours.
    bar_labels: list[str] = []
    bar_values: list[int] = []
    bar_colors: list[str] = []

    for idx, cid in enumerate(sorted_cluster_ids):
        bar_labels.append(f"Cluster {int(cid)}")
        bar_values.append(int(sorted_cluster_counts[idx]))
        bar_colors.append(cluster_color_map[int(cid)])

    if noise_mask.any():
        noise_count = int(counts[noise_mask][0])
        bar_labels.append("Noise")
        bar_values.append(noise_count)
        bar_colors.append("gray")

    fig = go.Figure(data=[
        go.Bar(
            x=bar_labels,
            y=bar_values,
            marker_color=bar_colors,
            text=bar_values,
            textposition="outside",
            hovertemplate="%{x}: %{y} points<extra></extra>",
        )
    ])
    fig.update_layout(
        title=dict(text="Cluster Size Distribution", font=_PLOTLY_FONT),
        xaxis_title=dict(text="Cluster", font=_PLOTLY_FONT),
        yaxis_title=dict(text="Number of Points", font=_PLOTLY_FONT),
        font=_PLOTLY_FONT,
        width=900,
        height=500,
        margin=dict(l=60, r=20, t=50, b=80),
        showlegend=False,
    )
    return fig
