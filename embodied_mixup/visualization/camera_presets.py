"""Camera presets and view control utilities for 3D visualization.

Provides predefined camera viewpoints (front, top, side, isometric, close)
and a helper to convert Matlab-style elevation/azimuth angles to Plotly
camera configuration.

@input:  Preset name (str) or explicit elev/azim/distance angles.
@output: Plotly scene.camera dict compatible with fig.update_layout().
@scenario: Set 3D viewpoint for reproducible screenshots and controlled
          inspection of trajectory intersection visualizations.
"""

import math
import logging
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Predefined camera presets (Plotly scene.camera format)
# Each preset specifies 'up' direction, 'center' of view, and 'eye' position.
# These match the default Matlab 3D viewpoint conventions.
# ---------------------------------------------------------------------------

CAMERA_PRESETS: Dict[str, dict] = {
    "front": dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=2.5, z=0),
    ),
    "top": dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=2.5),
    ),
    "side": dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2.5, y=0, z=0),
    ),
    "isometric": dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=1.2),
    ),
    "close": dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.8, y=0.8, z=0.6),
    ),
}


def _spherical_to_cartesian(
    elev_deg: float,
    azim_deg: float,
    distance: float,
) -> Dict[str, float]:
    """
    Convert Matlab-style elevation/azimuth angles to Cartesian camera position.

    @input:  elev_deg: elevation angle in degrees (above XY plane).
             azim_deg: azimuth angle in degrees (from X axis in XY plane).
             distance: radial distance from center.
    @output: dict with 'x', 'y', 'z' keys representing eye position.
    @scenario: Internal helper for set_camera_view when explicit angles are given.

    Conversion formulas (Matlab convention):
        eye_x = distance * cos(elevation) * cos(azimuth)
        eye_y = distance * cos(elevation) * sin(azimuth)
        eye_z = distance * sin(elevation)
    """
    elev_rad = math.radians(elev_deg)
    azim_rad = math.radians(azim_deg)

    return dict(
        x=distance * math.cos(elev_rad) * math.cos(azim_rad),
        y=distance * math.cos(elev_rad) * math.sin(azim_rad),
        z=distance * math.sin(elev_rad),
    )


def set_camera_view(
    fig,
    preset: str = "isometric",
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    distance: Optional[float] = None,
) -> None:
    """
    Set the 3D camera viewpoint for a Plotly Figure.

    Supports named presets (see CAMERA_PRESETS) or explicit Matlab-style
    elevation/azimuth/distance angles.  When angles are provided they
    override the preset's eye position; all other preset fields (up, center)
    are preserved.

    @input:  fig: plotly.graph_objects.Figure with a 3D scene.
             preset: one of 'front', 'top', 'side', 'isometric', 'close'.
             elev: optional elevation angle in degrees (0 = horizon, 90 = top).
             azim: optional azimuth angle in degrees.
             distance: optional camera distance from center.
    @output: fig is mutated in-place (camera updated). Also returns fig.
    @scenario: Call before fig.show() or fig.write_image() to control viewpoint.

    Usage::

        >>> fig = plot_intersections_3d(viz_data)
        >>> set_camera_view(fig, preset="top")
        >>> set_camera_view(fig, elev=30, azim=-60, distance=2.5)
    """
    if preset not in CAMERA_PRESETS:
        logger.warning(
            "Unknown camera preset '%s', falling back to 'isometric'. "
            "Available: %s",
            preset,
            list(CAMERA_PRESETS.keys()),
        )
        preset = "isometric"

    camera = dict(CAMERA_PRESETS[preset])

    # Override eye position when explicit angles are provided.
    if elev is not None or azim is not None or distance is not None:
        _elev = elev if elev is not None else 30.0
        _azim = azim if azim is not None else -60.0
        _dist = distance if distance is not None else 2.5

        camera["eye"] = _spherical_to_cartesian(_elev, _azim, _dist)

    fig.update_layout(scene=dict(camera=camera))
    return fig
