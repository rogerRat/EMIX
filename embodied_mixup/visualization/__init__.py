"""
Visualization subpackage for Embodied Mixup.

Provides interactive 3D/2D visualization of trajectory intersections
and DBSCAN clustering results using Plotly.

Modules:
    utils: Data preparation and color helpers for visualization.
    camera_presets: Preset camera views and view control utilities.
    plot_intersections: 3D scatter plot of trajectories, intersections, and clusters.
    plot_actions: 3D scatter plot of gripper pick/place transition points
        (standalone module, not imported here so `python -m` stays
        warning-free; import from .plot_actions directly).
    plot_divergence: Click-to-inspect camera image map for
        inspecting trajectory views at divergence points.
    plot_projections: 2D XY/XZ/YZ projection views.
    tuning_dashboard: Dash app for interactive hyperparameter tuning.
    first_frame_average: 1/N-weighted overlay of every episode's first frame
        (standalone module, not imported here so `python -m` stays warning-free).
"""

from .utils import prepare_visualization_data
from .camera_presets import CAMERA_PRESETS, set_camera_view
from .plot_intersections import plot_intersections_3d
from .plot_projections import plot_2d_projections
from .plot_divergence import build_click_image_map
from .tuning_dashboard import create_tuning_dashboard, make_rebuild_fn

__all__ = [
    "prepare_visualization_data",
    "CAMERA_PRESETS",
    "set_camera_view",
    "plot_intersections_3d",
    "plot_2d_projections",
    "build_click_image_map",
    "create_tuning_dashboard",
    "make_rebuild_fn",
]
