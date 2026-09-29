"""Interactive hyperparameter tuning dashboard using Plotly Dash.

This version operates on EmixIndex and source dataset inputs only.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..candidates.emix_index import EmixIndex

from ..configs.config import DEFAULT_EMBODIED_MIXUP_CONFIG

logger = logging.getLogger(__name__)


def create_tuning_dashboard(
    index: "EmixIndex",
    dataset: Any = None,
    rebuild_fn: Optional[Callable] = None,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
) -> Any:
    """Create a Dash application for interactive hyperparameter tuning."""

    try:
        from dash import Dash, dcc, html, Input, Output
    except ImportError as exc:
        raise ImportError("Dash is required for the tuning dashboard.") from exc

    from .plot_intersections import plot_intersections_3d
    from .utils import prepare_visualization_data

    try:
        initial_viz = prepare_visualization_data(index, dataset)
        initial_fig = plot_intersections_3d(initial_viz)
    except Exception as exc:
        logger.warning("Could not pre-compute initial visualization: %s", exc)
        import plotly.graph_objects as go

        initial_fig = go.Figure()
        initial_fig.update_layout(title=f"Failed to load initial visualization: {exc}")

    meta = index.public.metadata
    config = meta.get("config", {})
    default_xyz = float(config.get("xyz_epsilon", DEFAULT_EMBODIED_MIXUP_CONFIG.xyz_epsilon))
    default_rpy = float(config.get("rpy_epsilon", DEFAULT_EMBODIED_MIXUP_CONFIG.rpy_epsilon))

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("Embodied Mixup: Intersection & Cluster Tuning", style={"textAlign": "center"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Search Parameters"),
                            html.Label("xyz_epsilon (m)"),
                            dcc.Slider(
                                id="xyz-eps",
                                min=0.005,
                                max=0.10,
                                step=0.005,
                                value=default_xyz,
                            ),
                            html.Br(),
                            html.Label("rpy_epsilon (rad)"),
                            dcc.Slider(
                                id="rpy-eps",
                                min=0.005,
                                max=0.10,
                                step=0.005,
                                value=default_rpy,
                            ),
                            html.Br(),
                            html.Label("cluster_selection_epsilon"),
                            dcc.Slider(
                                id="cluster-sel-eps",
                                min=0.01,
                                max=0.20,
                                step=0.01,
                                value=float(
                                    config.get(
                                        "hdbscan_cluster_selection_epsilon",
                                        DEFAULT_EMBODIED_MIXUP_CONFIG.hdbscan_cluster_selection_epsilon,
                                    )
                                ),
                            ),
                            html.Br(),
                            html.Label("hdbscan_min_samples"),
                            dcc.Slider(
                                id="hdbscan-min-samp",
                                min=1,
                                max=15,
                                step=1,
                                value=int(
                                    config.get(
                                        "hdbscan_min_samples",
                                        DEFAULT_EMBODIED_MIXUP_CONFIG.hdbscan_min_samples,
                                    )
                                ),
                            ),
                            html.Br(),
                            html.Label("hdbscan_min_cluster_size_ratio"),
                            dcc.Slider(
                                id="min-cluster-size-ratio",
                                min=0.001,
                                max=0.10,
                                step=0.001,
                                value=float(
                                    config.get(
                                        "hdbscan_min_cluster_size_ratio",
                                        DEFAULT_EMBODIED_MIXUP_CONFIG.hdbscan_min_cluster_size_ratio,
                                    )
                                ),
                            ),
                            html.Hr(),
                            html.Div(id="stats-panel"),
                        ],
                        style={"width": "28%", "display": "inline-block", "verticalAlign": "top", "padding": "15px"},
                    ),
                    html.Div(
                        [
                            dcc.Loading(
                                dcc.Graph(id="intersection-3d", figure=initial_fig),
                                type="cube",
                            )
                        ],
                        style={"width": "70%", "display": "inline-block"},
                    ),
                ]
            ),
        ]
    )

    if rebuild_fn is not None:

        @app.callback(
            Output("intersection-3d", "figure"),
            Output("stats-panel", "children"),
            Input("xyz-eps", "value"),
            Input("rpy-eps", "value"),
            Input("cluster-sel-eps", "value"),
            Input("hdbscan-min-samp", "value"),
            Input("min-cluster-size-ratio", "value"),
        )
        def update_plot(xyz_eps, rpy_eps, cluster_sel_eps, hdbscan_min_samp, min_cluster_size_ratio):
            try:
                new_index, viz_data = rebuild_fn(
                    xyz_eps=xyz_eps,
                    rpy_eps=rpy_eps,
                    cluster_selection_epsilon=cluster_sel_eps,
                    hdbscan_min_samples=hdbscan_min_samp,
                    hdbscan_min_cluster_size_ratio=min_cluster_size_ratio,
                )
            except Exception as exc:
                logger.error("Rebuild failed: %s", exc)
                import plotly.graph_objects as go

                error_fig = go.Figure()
                error_fig.update_layout(title=f"Rebuild Error: {exc}")
                return error_fig, html.Div(f"Error during rebuild: {exc}", style={"color": "red"})

            from .plot_intersections import plot_intersections_3d

            fig = plot_intersections_3d(viz_data)
            labels = viz_data.get("cluster_labels", np.empty(0, dtype=int))
            n_groups = len(new_index.content.cross_point_groups)
            n_clusters = len(set(labels[labels >= 0])) if len(labels) > 0 else 0
            n_noise = int((labels == -1).sum()) if len(labels) > 0 else 0
            noise_pct = (n_noise / max(n_groups, 1)) * 100.0

            stats = html.Div(
                [
                    html.B("Current Parameters:"),
                    html.Ul(
                        [
                            html.Li(f"xyz_epsilon = {xyz_eps:.3f} m"),
                            html.Li(f"rpy_epsilon = {rpy_eps:.3f} rad"),
                            html.Li(f"cluster_selection_epsilon = {cluster_sel_eps:.2f}"),
                            html.Li(f"hdbscan_min_samples = {hdbscan_min_samp}"),
                            html.Li(f"hdbscan_min_cluster_size_ratio = {min_cluster_size_ratio:.3f}"),
                        ]
                    ),
                    html.B("Results:"),
                    html.Ul(
                        [
                            html.Li(f"Total groups: {n_groups}"),
                            html.Li(f"Clusters: {n_clusters}"),
                            html.Li(f"Noise groups: {n_noise} ({noise_pct:.1f}%)"),
                        ]
                    ),
                ]
            )
            return fig, stats

    else:

        @app.callback(
            Output("stats-panel", "children"),
            Input("xyz-eps", "value"),
            Input("rpy-eps", "value"),
            Input("cluster-sel-eps", "value"),
            Input("hdbscan-min-samp", "value"),
            Input("min-cluster-size-ratio", "value"),
        )
        def show_static_warning(xyz_eps, rpy_eps, cluster_sel_eps, hdbscan_min_samp, min_cluster_size_ratio):
            return html.Div(
                [
                    html.P("Live rebuild is not available.", style={"color": "darkorange"}),
                    html.P(
                        "Current slider values: "
                        f"xyz={xyz_eps:.3f}, rpy={rpy_eps:.3f}, "
                        f"cluster_sel_eps={cluster_sel_eps:.2f}, "
                        f"min_samples={hdbscan_min_samp}, "
                        f"min_cluster_size_ratio={min_cluster_size_ratio:.3f}"
                    ),
                ]
            )

    return app


def make_rebuild_fn(dataset, config=None):
    """Create a rebuild callback for the tuning dashboard."""

    from ..configs import EmbodiedMixupConfig
    from ..candidates.emix_index_utils import build_emix_index
    from .utils import prepare_visualization_data

    base_config = config if config is not None else EmbodiedMixupConfig()

    def rebuild_fn(xyz_eps, rpy_eps, cluster_selection_epsilon, hdbscan_min_samples, hdbscan_min_cluster_size_ratio):
        rebuild_config = replace(
            base_config,
            xyz_epsilon=xyz_eps,
            rpy_epsilon=rpy_eps,
            hdbscan_min_cluster_size_ratio=hdbscan_min_cluster_size_ratio,
            hdbscan_min_samples=hdbscan_min_samples,
            hdbscan_cluster_selection_epsilon=cluster_selection_epsilon,
        )
        new_index = build_emix_index(dataset=dataset, config=rebuild_config)
        viz_data = prepare_visualization_data(new_index, dataset)
        return new_index, viz_data

    return rebuild_fn
