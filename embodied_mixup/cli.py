"""Command-line interface for the canonical EMIX offline pipeline.

Provides four subcommands:
  build      - Build the canonical EmixIndex from a source LeRobot dataset
  visualize  - Generate interactive 3D/2D visualizations and save to HTML
  tune       - Launch Dash dashboard for interactive hyperparameter tuning
  train      - Launch local policy training with resume support

Usage:
  python -m embodied_mixup build --repo_id ID --root PATH [options]
  python -m embodied_mixup visualize --repo_id ID --root PATH [options]
  python -m embodied_mixup tune --repo_id ID --root PATH [options]
  python -m embodied_mixup train --config PATH [options]

@input:  Command-line arguments (argparse)
@output: Side effects (index cache, HTML files, Dash server)
@scenario: CLI entry point for the offline EMIX pipeline. Build computes the
          canonical index artifact, visualize generates Plotly figures, and
          tune launches an interactive dashboard for parameter exploration.
"""

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
import sys
import pkgutil
from pathlib import Path

from .datasets import default_repo_id_from_dataset_root, require_lerobot_dataset_root
from .configs.config import DEFAULT_EMBODIED_MIXUP_CONFIG

logger = logging.getLogger(__name__)
UTC_PLUS_8 = timezone(timedelta(hours=8))


def _setup_logging(level: str = "INFO") -> None:
    """Configure logging for the CLI."""
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC_PLUS_8).strftime("%Y%m%d%H%M%S")
    log_path = logs_dir / f"embodied_mixup_{stamp}.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[%(name)s] [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logger.debug("CLI logging initialized at %s", log_path)


def _default_emix_output_dir(repo_id: str) -> Path:
    """Resolve the default artifact output directory for Emix artifacts."""
    normalized_repo_id = repo_id.rstrip("/")
    return Path("data") / "emix_index" / normalized_repo_id


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    """Write ``payload`` to ``target`` via tmp file + rename.

    Editors and dev servers (e.g. VS Code Live Server) watch the output
    directory and reload pages on any sibling change; writing the final path
    in place exposes a partially written file to both watchers and HTTP
    readers. The rename makes each artifact appear exactly once, complete.
    """
    import os as _os
    import tempfile as _tempfile

    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = _tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with _os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        _os.chmod(temporary, 0o644)
        _os.replace(temporary, target)
    except BaseException:
        try:
            _os.unlink(temporary)
        except OSError:
            pass
        raise


def _ensure_local_plotly_bundle(output_path: str | Path) -> Path:
    """Write plotly.min.js next to the output HTML if it is missing."""
    bundle_bytes = pkgutil.get_data("plotly", "package_data/plotly.min.js")
    if bundle_bytes is None:
        raise RuntimeError("Could not locate plotly.min.js in the installed plotly package.")

    bundle_path = Path(output_path).resolve().parent / "plotly.min.js"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if not bundle_path.exists():
        _atomic_write_bytes(bundle_path, bundle_bytes)
    return bundle_path


def _make_visualization_asset_dir(output_path: str | Path) -> Path:
    """Create the timestamped click-image asset directory for one HTML file.

    @input:  output_path: target visualization HTML path.
    @output: Path to a newly named asset directory.
    @scenario: Store click-to-inspect images beside the HTML without
               overwriting assets from previous visualization runs.
    """
    html_path = Path(output_path)
    stamp = datetime.now(UTC_PLUS_8).strftime("%Y%m%d%H%M%S")
    safe_stem = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in html_path.stem
    )
    asset_dir = html_path.parent / f"{safe_stem}_{stamp}_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    return asset_dir


def _load_dataset(repo_id: str | None, root: str):
    """Load one source LeRobot dataset from its exact dataset root."""

    dataset_root = require_lerobot_dataset_root(root)
    resolved_repo_id = (
        default_repo_id_from_dataset_root(dataset_root)
        if repo_id is None or str(repo_id).strip() == ""
        else str(repo_id).rstrip("/")
    )

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            logger.error(
                "Failed to import lerobot. Install it with: pip install lerobot"
            )
            sys.exit(1)

    logger.info("Loading LeRobot dataset: repo_id=%s root=%s", resolved_repo_id, dataset_root)
    dataset = LeRobotDataset(repo_id=resolved_repo_id, root=str(dataset_root))
    logger.info(
        "Dataset loaded: %d episodes, %d frames",
        dataset.num_episodes,
        len(dataset),
    )
    return dataset


def _resolve_config(args):
    """Build the candidate-generation config from parsed CLI args."""
    from .configs import EmbodiedMixupConfig

    return EmbodiedMixupConfig(
        xyz_epsilon=args.xyz_eps,
        rpy_epsilon=args.rpy_eps,
        exclude_tail_seconds=getattr(args, "exclude_tail_seconds",
                                     DEFAULT_EMBODIED_MIXUP_CONFIG.exclude_tail_seconds),
        dataset_fps=getattr(args, "dataset_fps", DEFAULT_EMBODIED_MIXUP_CONFIG.dataset_fps),
        future_horizon=args.future_horizon,
        future_angle_thresh=args.future_angle_thresh,
        min_future_disp=args.min_future_disp,
        action_delta_threshold=args.action_delta_threshold,
        gripper_action_filter_enabled=not args.disable_gripper_action_filter,
        hdbscan_min_cluster_size_ratio=args.hdbscan_min_cluster_size_ratio,
        hdbscan_min_samples=args.hdbscan_min_samples,
        hdbscan_cluster_selection_epsilon=args.cluster_selection_epsilon,
        hdbscan_alpha=args.hdbscan_alpha,
        max_candidate_pairs=getattr(args, "max_candidate_pairs", None),
        max_candidate_pairs_per_episode=(
            None
            if getattr(args, "max_candidate_pairs_per_episode", None) == 0
            else getattr(args, "max_candidate_pairs_per_episode", None)
        ),
        exclude_start_radius=(
            None
            if getattr(args, "exclude_start_radius", None) == 0
            else getattr(args, "exclude_start_radius", None)
        ),
        max_pair_time_ratio=(
            None
            if getattr(args, "max_pair_time_ratio", None) == 0
            else getattr(args, "max_pair_time_ratio", None)
        ),
        training_candidates_total_budget=getattr(args, "training_candidates_total_budget", None),
        training_candidates_cap_per_cluster=getattr(
            args, "training_candidates_cap_per_cluster", None
        ),
        training_candidates_include_noise=getattr(
            args, "training_candidates_include_noise", False
        ),
        training_candidates_seed=getattr(args, "training_candidates_seed", 0),
    )


def cmd_build(args) -> None:
    """Build and persist the canonical EmixIndex artifact."""
    _setup_logging(args.log_level)
    dataset = _load_dataset(args.repo_id, args.root)
    config = _resolve_config(args)
    repo_id = (
        args.repo_id.rstrip("/")
        if getattr(args, "repo_id", None)
        else default_repo_id_from_dataset_root(args.root)
    )
    output_dir = (
        Path(args.output_dir)
        if getattr(args, "output_dir", None) is not None
        else _default_emix_output_dir(repo_id)
    )
    from .candidates.emix_index_utils import build_emix_index_streaming

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = build_emix_index_streaming(
        dataset=dataset, config=config, output_path=output_dir / "emix_index.json"
    )
    logger.info("EmixIndex written to: %s", index_path)


def cmd_visualize(args) -> None:
    """Build EmixIndex and generate Plotly visualizations."""
    _setup_logging(args.log_level)
    dataset = _load_dataset(args.repo_id, args.root)
    config = _resolve_config(args)
    from .candidates.emix_index_utils import build_emix_index, load_emix_index

    # Reuse a previously built canonical EmixIndex when one is supplied (the
    # flow builds the index in its own phase and passes it here), otherwise
    # rebuild it so `visualize` stays usable standalone.
    if getattr(args, "index", None):
        index = load_emix_index(args.index)
        logger.info("Loaded existing EmixIndex: %s", args.index)
    else:
        index = build_emix_index(dataset=dataset, config=config)

    from .visualization.utils import prepare_visualization_data
    from .visualization.plot_intersections import (
        plot_intersections_3d,
        plot_cluster_histogram,
    )
    from .visualization.plot_projections import plot_2d_projections
    from .visualization.plot_divergence import build_click_image_map

    # Normalize repo_id: derive it from the dataset root when omitted.
    repo_id = (
        args.repo_id.rstrip("/")
        if getattr(args, "repo_id", None)
        else default_repo_id_from_dataset_root(args.root)
    )

    logger.info("Preparing visualization data...")
    viz_data = prepare_visualization_data(
        index, dataset, config=config,
        max_points=getattr(args, "max_points", None),
    )
    if getattr(args, "vis_obj", None):
        from .visualization.robotwin_objects import attach_object_ranges
        attach_object_ranges(viz_data, args.vis_obj)

    # Determine output path (defaults to outputs/intersections/ — this
    # script's own folder, separate from the other visualization outputs).
    output_path = args.output
    if output_path is None:
        output_dir = Path("outputs") / "intersections"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"emix_visualization_{repo_id}.html")
    else:
        # Ensure parent directory exists for user-specified output paths
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating figures...")
    figures = {}
    button_configs = {}

    logger.info("  [1/4] 3D intersection scatter plot...")
    fig_3d, btn_cfg_3d = plot_intersections_3d(viz_data, camera=args.camera)
    figures["intersections_3d"] = fig_3d
    button_configs["intersections_3d"] = btn_cfg_3d

    logger.info("  [2/4] 2D projections (XY, XZ, YZ)...")
    fig_2d, btn_cfg_2d = plot_2d_projections(viz_data)
    figures["projections_2d"] = fig_2d
    button_configs["projections_2d"] = btn_cfg_2d

    logger.info("  [3/4] Cluster size histogram...")
    fig_hist = plot_cluster_histogram(viz_data)
    figures["cluster_histogram"] = fig_hist
    button_configs["cluster_histogram"] = {
        "fig_id": "cluster-histogram",
        "visible_states": {},
        "has_raw_data": False,
    }

    _skip_click_images = bool(getattr(args, "skip_click_images", False))
    if _skip_click_images:
        logger.info("  [4/4] Skipping click-to-inspect camera images (--skip-click-images)...")
        camera_keys = (
            list(index.public.source_datasets[0].media.camera_keys)
            if getattr(index.public, "source_datasets", None)
            and index.public.source_datasets
            and getattr(index.public.source_datasets[0], "media", None)
            else []
        )
        image_map_data = {
            "images_js": "{}",
            "imaged_positions_js": "[]",
            "camera_key": camera_keys[0] if camera_keys else "head_camera",
        }
        image_asset_dir = Path(output_path).parent
    else:
        logger.info("  [4/4] Pre-computing click-to-inspect camera images...")
        _total_budget = (
            args.click_image_total_budget
            if getattr(args, "click_image_total_budget", None) is not None
            else config.click_image_total_budget
        )
        _min_per_cluster = (
            args.click_image_min_per_cluster
            if getattr(args, "click_image_min_per_cluster", None) is not None
            else config.click_image_min_per_cluster
        )
        image_asset_dir = _make_visualization_asset_dir(output_path)
        image_map_data = build_click_image_map(
            index, dataset,
            total_budget=_total_budget,
            min_per_cluster=_min_per_cluster,
            asset_dir=image_asset_dir,
            html_dir=Path(output_path).parent,
        )

    # Write all figures to a single HTML file
    logger.info("Writing HTML output to: %s", output_path)
    try:
        bundle_path = _ensure_local_plotly_bundle(output_path)

        _atomic_write_bytes(
            Path(output_path),
            _build_visualization_html(
                figures, button_configs, image_map_data, bundle_path,
            ).encode("utf-8"),
        )
        logger.info("Visualization saved to: %s", output_path)
        logger.info("Referenced local Plotly bundle: %s", bundle_path)
        logger.info("Click-to-inspect image assets saved to: %s", image_asset_dir)
    except Exception as exc:
        logger.error("Failed to write HTML: %s", exc)
        # Fallback: save each figure separately
        for name, fig in figures.items():
            path = output_path.replace(".html", f"_{name}.html")
            fig.write_html(path, include_plotlyjs="directory")
            logger.info("Saved %s to: %s", name, path)


def _build_visualization_html(figures, button_configs, image_map_data, bundle_path):
    """Assemble the complete HTML page with Plotly figures, custom button
    controls, and click-to-inspect camera image panel.

    @input:  figures: dict of name -> plotly Figure.
             button_configs: dict of name -> button config dict.
             image_map_data: dict from build_click_image_map().
             bundle_path: path to local plotly.min.js.
    @output: Full HTML string.
    @scenario: Build a self-contained visualization page with dual
              toggle buttons (Color: Cluster/Density, Data: Action
              Filtered/Raw) and click-to-inspect camera images.
    """
    import json as _json

    page_title = "Embodied Mixup Visualization"

    # Read and inline plotly.min.js so the HTML is self-contained and
    # works when opened directly from the filesystem (file:// URLs for
    # external scripts are blocked by browser security policies).
    with open(bundle_path, "rb") as _fh:
        plotly_js_inline = _fh.read().decode("utf-8")

    # Generate Plotly divs
    plotly_divs = []
    for fig_idx, (name, fig) in enumerate(figures.items()):
        include_js = False
        div_html = fig.to_html(
            full_html=False,
            include_plotlyjs=include_js,
            div_id=button_configs[name]["fig_id"],
        )
        plotly_divs.append(f"<h2>{name.replace('_', ' ').title()}</h2>\n{div_html}")

    plotly_section = "\n".join(plotly_divs)

    # Serialize configs for JS
    btn_cfgs_json = _json.dumps(button_configs)
    image_map_json = image_map_data.get("images_js", "{}")
    imaged_positions_json = image_map_data.get("imaged_positions_js", "[]")
    camera_key = image_map_data.get("camera_key", "top")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{page_title}</title>
<script>{plotly_js_inline}</script>
<style>
  body {{
    font-family: 'Times New Roman', Times, serif;
    margin: 20px;
    background: #fff;
    color: #333;
  }}
  h2 {{
    margin-top: 30px;
    margin-bottom: 10px;
  }}
  .emix-controls {{
    margin: 10px 0 20px 0;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .emix-btn-group {{
    display: flex;
    align-items: center;
    gap: 4px;
  }}
  .emix-btn-label {{
    font-size: 13px;
    font-weight: bold;
    min-width: 55px;
    color: #555;
  }}
  .emix-btn {{
    padding: 4px 14px;
    border: 1px solid #aaa;
    border-radius: 3px;
    background: #f5f5f5;
    cursor: pointer;
    font-family: 'Times New Roman', Times, serif;
    font-size: 13px;
    transition: background 0.15s;
  }}
  .emix-btn.active {{
    background: #1f77b4;
    color: #fff;
    border-color: #1f77b4;
    font-weight: bold;
  }}
  .emix-btn:hover:not(.active) {{
    background: #e0e0e0;
  }}
  #image-panel {{
    display: none;
    margin: 20px 0;
    padding: 16px;
    border: 1px solid #ccc;
    border-radius: 6px;
    background: #fafafa;
    max-width: 900px;
  }}
  #image-panel .panel-header {{
    font-size: 14px;
    font-weight: bold;
    margin-bottom: 12px;
    color: #333;
  }}
  #image-panel .panel-images {{
    display: flex;
    gap: 24px;
    justify-content: center;
    flex-wrap: wrap;
  }}
  #image-panel .panel-img-wrap {{
    text-align: center;
  }}
  #image-panel .panel-img-wrap img {{
    border: 2px solid #ddd;
    border-radius: 4px;
    max-width: 480px;
    height: auto;
  }}
  #image-panel .panel-img-label {{
    font-size: 12px;
    margin-top: 4px;
    color: #666;
  }}
</style>
</head>
<body>
<h1>{page_title}</h1>

<div class="emix-controls" id="emix-controls">
  <div class="emix-btn-group">
    <span class="emix-btn-label">Color:</span>
    <button class="emix-btn active" data-group="color" data-value="cluster">Cluster</button>
    <button class="emix-btn" data-group="color" data-value="density">Density</button>
  </div>
  <div class="emix-btn-group" id="source-btn-group">
    <span class="emix-btn-label">Data:</span>
    <button class="emix-btn active" data-group="source" data-value="filtered">Action Filtered</button>
    <button class="emix-btn" data-group="source" data-value="raw">Raw</button>
  </div>
</div>

<div id="image-panel">
  <div class="panel-header" id="panel-info">Click a point in the 3D plot to inspect camera images.</div>
  <div class="panel-images">
    <div class="panel-img-wrap">
      <img id="img-a" src="" alt="Trajectory A" style="display:none;">
      <div class="panel-img-label" id="label-a"></div>
    </div>
    <div class="panel-img-wrap">
      <img id="img-b" src="" alt="Trajectory B" style="display:none;">
      <div class="panel-img-label" id="label-b"></div>
    </div>
  </div>
</div>

{plotly_section}

<script>
(function() {{
  var btnConfigs = {btn_cfgs_json};
  var imageMap = {image_map_json};
  var imagedPositions = {imaged_positions_json};
  var cameraKey = "{camera_key}";

  // ---- No click-to-inspect images (e.g. --skip-click-images or a source
  //      dataset without usable video frames): hide the image panel and do
  //      not bind the click-to-inspect handler, leaving a spatial-only page.
  var hasImageMap = Object.keys(imageMap).length > 0;
  if (!hasImageMap) {{
    var imagePanel = document.getElementById("image-panel");
    if (imagePanel) imagePanel.style.display = "none";
  }}

  // ---- State ----
  var colorMode = "cluster";
  var dataSource = "filtered";

  function getStateKey() {{
    return colorMode + "_" + dataSource;
  }}

  function applyVisibility(figId, visArray) {{
    var figEl = document.getElementById(figId);
    if (figEl && typeof Plotly !== "undefined") {{
      Plotly.update(figId, {{visible: visArray}});
    }}
  }}

  function applyMarkerState(figId, markerStates) {{
    var figEl = document.getElementById(figId);
    if (!figEl || typeof Plotly === "undefined" || !markerStates) return;
    Object.keys(markerStates).forEach(function(index) {{
      var state = markerStates[index];
      var traceIndex = Number(index);
      var update = {{
        "marker.color": [state.color],
        "marker.showscale": [Boolean(state.showscale)]
      }};
      if (state.colorscale !== undefined) update["marker.colorscale"] = [state.colorscale];
      if (state.cmin !== undefined) update["marker.cmin"] = [state.cmin];
      if (state.cmax !== undefined) update["marker.cmax"] = [state.cmax];
      if (state.cmid !== undefined) update["marker.cmid"] = [state.cmid];
      update["marker.colorbar"] = [state.colorbar || null];
      if (state.hovertemplate !== undefined) update.hovertemplate = [state.hovertemplate];
      Plotly.restyle(figId, update, [traceIndex]);
    }});
  }}

  function updateAllViews() {{
    var stateKey = getStateKey();
    Object.keys(btnConfigs).forEach(function(name) {{
      var cfg = btnConfigs[name];
      var visArray = cfg.visible_states[stateKey];
      if (visArray) {{
        applyVisibility(cfg.fig_id, visArray);
      }}
      applyMarkerState(cfg.fig_id, cfg.marker_states ? cfg.marker_states[stateKey] : null);
    }});
  }}

  // ---- Button handlers ----
  var buttons = document.querySelectorAll(".emix-btn");
  buttons.forEach(function(btn) {{
    btn.addEventListener("click", function() {{
      var group = this.getAttribute("data-group");
      var value = this.getAttribute("data-value");

      // Update active state within this group
      var siblings = document.querySelectorAll(
        '.emix-btn[data-group="' + group + '"]'
      );
      siblings.forEach(function(s) {{ s.classList.remove("active"); }});
      this.classList.add("active");

      // Update JS state
      if (group === "color") {{
        colorMode = value;
      }} else if (group === "source") {{
        dataSource = value;
      }}

      updateAllViews();
    }});
  }});

  // ---- Hide source button group if no raw data ----
  (function() {{
    var hasRaw = false;
    Object.keys(btnConfigs).forEach(function(name) {{
      if (btnConfigs[name].has_raw_data) hasRaw = true;
    }});
    if (!hasRaw) {{
      var sourceGroup = document.getElementById("source-btn-group");
      if (sourceGroup) sourceGroup.style.display = "none";
    }}
  }})();

  // ---- Click-to-inspect: 3D plot (with nearest-neighbor fallback) ----
  var fig3dId = btnConfigs["intersections_3d"] ?
                btnConfigs["intersections_3d"].fig_id : "intersections-3d";
  var fig3dEl = document.getElementById(fig3dId);

  function findNearestImagedPoint(x, y, z) {{
    if (!imagedPositions || imagedPositions.length === 0) return null;
    var best = null;
    var bestDist2 = Infinity;
    for (var i = 0; i < imagedPositions.length; i++) {{
      var p = imagedPositions[i];
      var dx = p.x - x;
      var dy = p.y - y;
      var dz = p.z - z;
      var d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < bestDist2) {{
        bestDist2 = d2;
        best = p;
      }}
    }}
    return best;
  }}

  function showImagePanel(imgs, epA, frameA, epB, frameB, clusterId, divergence, isApprox) {{
    var panel = document.getElementById("image-panel");
    var imgA = document.getElementById("img-a");
    var imgB = document.getElementById("img-b");
    var labelA = document.getElementById("label-a");
    var labelB = document.getElementById("label-b");
    var panelInfo = document.getElementById("panel-info");

    imgA.src = imgs.img_a;
    imgA.style.display = "";
    imgB.src = imgs.img_b;
    imgB.style.display = "";
    var imgEpA = imgs.ep_a !== undefined && imgs.ep_a !== null ? imgs.ep_a : epA;
    var imgFrameA = imgs.frame_a !== undefined && imgs.frame_a !== null ? imgs.frame_a : frameA;
    var imgEpB = imgs.ep_b !== undefined && imgs.ep_b !== null ? imgs.ep_b : epB;
    var imgFrameB = imgs.frame_b !== undefined && imgs.frame_b !== null ? imgs.frame_b : frameB;
    labelA.textContent = "Ep " + imgEpA + " Frame " + imgFrameA;
    labelB.textContent = "Ep " + imgEpB + " Frame " + imgFrameB;
    if (imgs.width_a && imgs.height_a) {{
      labelA.textContent += " | " + imgs.width_a + "x" + imgs.height_a;
    }}
    if (imgs.width_b && imgs.height_b) {{
      labelB.textContent += " | " + imgs.width_b + "x" + imgs.height_b;
    }}

    var approxNote = isApprox
      ? " (nearest pre-computed point shown; exact match unavailable)"
      : "";
    panelInfo.textContent =
      "Cluster " + (imgs.cluster_id !== undefined ? imgs.cluster_id : clusterId) +
      " | Divergence: " + (imgs.divergence !== undefined ? imgs.divergence.toFixed(5) : divergence.toFixed(5)) +
      " | Camera: " + cameraKey +
      approxNote;
    panel.style.display = "block";
  }}

  function showNoImagePanel(epA, frameA, epB, frameB) {{
    var panel = document.getElementById("image-panel");
    var imgA = document.getElementById("img-a");
    var imgB = document.getElementById("img-b");
    var labelA = document.getElementById("label-a");
    var labelB = document.getElementById("label-b");
    var panelInfo = document.getElementById("panel-info");

    imgA.style.display = "none";
    imgB.style.display = "none";
    labelA.textContent = "";
    labelB.textContent = "";
    panelInfo.textContent =
      "No pre-computed or nearby images for Ep " + epA + " Frame " + frameA +
      " / Ep " + epB + " Frame " + frameB +
      " (camera: " + cameraKey + ")";
    panel.style.display = "block";
  }}

  if (hasImageMap && fig3dEl && typeof Plotly !== "undefined") {{
    fig3dEl.on("plotly_click", function(data) {{
      if (!data.points || data.points.length === 0) return;
      var pt = data.points[0];
      var cd = pt.customdata;
      if (!cd || cd.length < 7) return;

      // customdata layout (both filtered and raw traces use same format):
      // [cluster_id, density, divergence, ep_a, frame_a, ep_b, frame_b]
      var epA = cd[3], frameA = cd[4];
      var epB = cd[5], frameB = cd[6];
      var clusterId = cd[0];
      var divergence = cd[2];

      var key = epA + "_" + frameA + "_" + epB + "_" + frameB;
      var imgs = imageMap[key];

      if (imgs) {{
        showImagePanel(imgs, epA, frameA, epB, frameB, clusterId, divergence, false);
      }} else {{
        // Exact match not found - try nearest-neighbor fallback.
        var nearest = findNearestImagedPoint(pt.x, pt.y, pt.z);
        if (nearest && imageMap[nearest.key]) {{
          showImagePanel(imageMap[nearest.key], epA, frameA, epB, frameB, clusterId, divergence, true);
        }} else {{
          showNoImagePanel(epA, frameA, epB, frameB);
        }}
      }}
    }});
  }}
}})();
</script>
</body>
</html>"""


def cmd_tune(args) -> None:
    """Launch interactive Dash dashboard for hyperparameter tuning."""
    _setup_logging(args.log_level)
    dataset = _load_dataset(args.repo_id, args.root)
    config = _resolve_config(args)
    from .candidates.emix_index_utils import build_emix_index

    index = build_emix_index(dataset=dataset, config=config)

    from .visualization.tuning_dashboard import create_tuning_dashboard, make_rebuild_fn

    rebuild_fn = make_rebuild_fn(dataset, config=config)
    app = create_tuning_dashboard(
        index,
        dataset=dataset,
        rebuild_fn=rebuild_fn,
        host=args.host,
        port=args.port,
        debug=args.debug,
    )

    logger.info(
        "Launching tuning dashboard at http://%s:%d ...", args.host, args.port
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


def cmd_train(args) -> None:
    """Launch local policy training with resume support."""
    from .policies import train as train_module

    resume_cfg = train_module._build_resume_config(args)
    train_module.train(
        args.config,
        resume_cfg=resume_cfg,
        vis_n_obs_steps=args.vis_n_obs_steps,
    )


def cmd_eval(args) -> None:
    """Launch the RoboTwin evaluation bridge from a unified config."""
    from evaluation.evaluate import main as evaluate_main

    evaluate_main(args)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all subcommands."""
    parser.add_argument(
        "--repo_id", type=str, default=None,
        help="LeRobot dataset repository ID (e.g., stack_cube_rgb_bh_500ep_84x84_eef_nohandcamera_nonoop)",
    )
    parser.add_argument(
        "--root", type=str, required=True,
        help="Exact LeRobot dataset root directory containing meta/info.json.",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )


def _add_index_args(parser: argparse.ArgumentParser) -> None:
    """Add index-building arguments shared by build/visualize/tune."""
    parser.add_argument(
        "--xyz_eps", type=float, default=DEFAULT_EMBODIED_MIXUP_CONFIG.xyz_epsilon,
        help="Position similarity threshold in meters (default: 0.02)",
    )
    parser.add_argument(
        "--rpy_eps", type=float, default=DEFAULT_EMBODIED_MIXUP_CONFIG.rpy_epsilon,
        help="Orientation similarity threshold in radians (default: 0.26, ~15 deg)",
    )
    parser.add_argument(
        "--exclude_tail_seconds", type=float,
        default=DEFAULT_EMBODIED_MIXUP_CONFIG.exclude_tail_seconds,
        help="Exclude each episode's final seconds from pairing; the trimmed "
             "home-return tail otherwise pairs with episode starts "
             "(default: 0.1, ~2 frames at 17 fps)",
    )
    parser.add_argument(
        "--dataset_fps", type=float, default=DEFAULT_EMBODIED_MIXUP_CONFIG.dataset_fps,
        help="Dataset fps used to convert --exclude_tail_seconds to frames (default: 17)",
    )
    parser.add_argument(
        "--future_horizon", type=int, default=DEFAULT_EMBODIED_MIXUP_CONFIG.future_horizon,
        help="Short-horizon look-ahead in frames for future motion direction (default: 8).",
    )
    parser.add_argument(
        "--future_angle_thresh", type=float, default=DEFAULT_EMBODIED_MIXUP_CONFIG.future_angle_thresh,
        help="Minimum future xyz direction angle in radians (default: pi/6).",
    )
    parser.add_argument(
        "--min_future_disp", type=float, default=DEFAULT_EMBODIED_MIXUP_CONFIG.min_future_disp,
        help="Minimum future xyz displacement magnitude for stable angle estimation "
             "(default: 0.002 m).",
    )
    parser.add_argument(
        "--action_delta_threshold",
        type=float,
        default=DEFAULT_EMBODIED_MIXUP_CONFIG.action_delta_threshold,
        help="Minimum current-frame action L2 divergence required by the action filter "
             "(default: 0.05).",
    )
    parser.add_argument(
        "--disable_gripper_action_filter", action="store_true",
        help="Disable diagnostic short-horizon gripper transition mismatch computation.",
    )
    parser.add_argument(
        "--cluster_selection_epsilon", type=float, default=DEFAULT_EMBODIED_MIXUP_CONFIG.hdbscan_cluster_selection_epsilon,
        help="HDBSCAN post-processing merge threshold in standardized 6D space "
             "(default: 0.05). Higher values merge more nearby micro-clusters.",
    )
    parser.add_argument(
        "--hdbscan_min_cluster_size_ratio", type=float, default=DEFAULT_EMBODIED_MIXUP_CONFIG.hdbscan_min_cluster_size_ratio,
        help="HDBSCAN min_cluster_size as fraction of total pairs "
             "(default: 0.01 = 1%%).",
    )
    parser.add_argument(
        "--hdbscan_min_samples", type=int, default=DEFAULT_EMBODIED_MIXUP_CONFIG.hdbscan_min_samples,
        help="HDBSCAN min_samples for core-point classification (default: 3). "
             "Lower values (1-3) are more aggressive in sparse regions.",
    )
    parser.add_argument(
        "--hdbscan_alpha", type=float, default=DEFAULT_EMBODIED_MIXUP_CONFIG.hdbscan_alpha,
        help="HDBSCAN cluster selection stability (default: 1.0). "
             ">1 = larger/more-stable clusters, <1 = smaller clusters.",
    )
    parser.add_argument(
        "--max_candidate_pairs", type=int, default=None,
        help="Absolute hard cap on candidate pairs entering clustering and "
             "the index; beyond the cap a seeded uniform subsample is kept "
             "(default: none, the per-episode ratio decides).",
    )
    parser.add_argument(
        "--max_candidate_pairs_per_episode", type=float,
        default=DEFAULT_EMBODIED_MIXUP_CONFIG.max_candidate_pairs_per_episode,
        help="Candidate-pair cap scaled with collection size: pairs allowed "
             "per episode (default: 1000). Pass 0 together with "
             "--max_candidate_pairs to disable capping entirely.",
    )
    parser.add_argument(
        "--exclude_start_radius", type=float,
        default=DEFAULT_EMBODIED_MIXUP_CONFIG.exclude_start_radius,
        help="Drop candidate pairs whose midpoint xyz is within this radius "
             "(m) of the arm's shared start point (median of episode first "
             "frames). Default: disabled. Pass e.g. 0.02 to filter the home "
             "cluster; 0 disables an explicitly configured radius.",
    )
    parser.add_argument(
        "--max_pair_time_ratio", type=float,
        default=DEFAULT_EMBODIED_MIXUP_CONFIG.max_pair_time_ratio,
        help="Drop candidate pairs whose two frames are further apart in "
             "within-episode time than this ratio of the collection's mean "
             "episode length (e.g. 0.6 keeps spans under 60%%). Default: "
             "disabled; 0 disables an explicitly configured ratio.",
    )
    parser.add_argument(
        "--training_candidates_total_budget", type=int, default=None,
        help="Optional total budget metadata for training-group selection.",
    )
    parser.add_argument(
        "--training_candidates_cap_per_cluster", type=int, default=None,
        help="Optional explicit hard cap per cluster recorded in EmixIndex metadata.",
    )
    parser.add_argument(
        "--training_candidates_include_noise", action="store_true",
        help="Keep noise groups trainable instead of marking them aborted.",
    )
    parser.add_argument(
        "--training_candidates_seed", type=int, default=0,
        help="Random seed recorded in EmixIndex metadata.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="embodied_mixup",
        description="Embodied Mixup offline pipeline: index building, "
                    "clustering, visualization, and training.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- build ---
    build_sub = subparsers.add_parser(
        "build",
        help="Build the canonical EmixIndex from a LeRobot dataset.",
    )
    _add_common_args(build_sub)
    _add_index_args(build_sub)
    build_sub.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for emix_index.json. Defaults to data/emix_index/<repo_id>/.",
    )
    build_sub.set_defaults(func=cmd_build)

    # --- visualize ---
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Build index and generate interactive Plotly visualizations.",
    )
    _add_common_args(viz_parser)
    _add_index_args(viz_parser)
    viz_parser.add_argument(
        "--output", type=str, default=None,
        help="Output HTML file path (default: emix_visualization_{repo_id}.html).",
    )
    viz_parser.add_argument(
        "--index", type=str, default=None,
        help="Path to an existing emix_index.json to visualize; skips rebuilding "
             "the index from the dataset.",
    )
    viz_parser.add_argument(
        "--camera", type=str, default="isometric",
        choices=["front", "top", "side", "isometric", "close"],
        help="Camera preset for 3D views (default: isometric).",
    )
    viz_parser.add_argument(
        "--grid_resolution", type=int, default=30,
        help="Grid resolution for volume/density cloud rendering (default: 30).",
    )
    viz_parser.add_argument(
        "--density_sigma", type=float, default=1.5,
        help="Gaussian smoothing sigma for density cloud (default: 1.5). "
             "Higher = softer/blurrier clouds.",
    )
    viz_parser.add_argument(
        "--click-image-total-budget", type=int, default=None,
        help="Max pairs to pre-compute camera images for click-to-inspect "
             "(default: 100).",
    )
    viz_parser.add_argument(
        "--click-image-min-per-cluster", type=int, default=None,
        help="Minimum image pairs per cluster for click-to-inspect "
             "(default: 10). May expand total budget.",
    )
    viz_parser.add_argument(
        "--skip-click-images", action="store_true",
        help="Skip pre-computing click-to-inspect camera images. The HTML keeps "
             "only the spatial intersection figures (3D scatter, 2D projections, "
             "cluster histogram) and hides the image panel. Use this when the "
             "source dataset has no video frames or the intersection-frame "
             "feature is not wanted.",
    )
    viz_parser.add_argument(
        "--max-points", type=int, default=10000,
        help="Cap the number of cross-point groups rendered in the HTML. The "
             "full EmixIndex can hold hundreds of thousands of groups (each a "
             "hoverable WebGL marker), which bloats the HTML to hundreds of MB "
             "and makes it laggy; capping to ~10k keeps the file small and "
             "smooth. A deterministic spread sample is taken so the rendered "
             "points span the whole workspace. The raw-pair accepted set is "
             "capped to the same sample (default: 10000).",
    )
    viz_parser.add_argument(
        "--vis-obj", dest="vis_obj", type=str, default=None,
        help="RoboTwin task path (without or with .py) whose non-robot object "
             "randomization ranges are overlaid on the 3D and 2D plots.",
    )
    viz_parser.set_defaults(func=cmd_visualize)

    # --- tune ---
    tune_parser = subparsers.add_parser(
        "tune",
        help="Launch Dash interactive tuning dashboard.",
    )
    _add_common_args(tune_parser)
    _add_index_args(tune_parser)
    tune_parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Dash server host (default: 127.0.0.1).",
    )
    tune_parser.add_argument(
        "--port", type=int, default=8050,
        help="Dash server port (default: 8050).",
    )
    tune_parser.add_argument(
        "--debug", action="store_true",
        help="Enable Dash debug mode (hot reload, error display).",
    )
    tune_parser.add_argument(
        "--grid_resolution", type=int, default=30,
        help="Grid resolution for volume/density cloud rendering (default: 30).",
    )
    tune_parser.add_argument(
        "--density_sigma", type=float, default=1.5,
        help="Gaussian smoothing sigma for density cloud (default: 1.5). "
             "Higher = softer/blurrier clouds.",
    )
    tune_parser.set_defaults(func=cmd_tune)

    # --- train ---
    train_parser = subparsers.add_parser(
        "train",
        help="Launch local policy training with resume support.",
    )
    train_parser.add_argument(
        "--config", type=str, required=True,
        help="Path to local YAML config.",
    )
    train_parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Checkpoint directory or pretrained_model path to resume from.",
    )
    train_parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override YAML output_dir with a new directory.",
    )
    train_parser.add_argument(
        "--seed", type=int, default=0,
        help="Global random seed.",
    )
    train_parser.add_argument(
        "--start_step", type=int, default=None,
        help="Last completed step number for resumed runs.",
    )
    train_parser.add_argument(
        "--total_steps", type=int, default=None,
        help="Total step count for this run.",
    )
    train_parser.add_argument(
        "--vis_n_obs_steps", type=int, default=None,
        help=(
            "Override the number of observation frames rendered by EMIX debug "
            "visualization; defaults to the policy n_obs_steps."
        ),
    )
    train_parser.set_defaults(func=cmd_train)

    # --- eval ---
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a LeRobot checkpoint in RoboTwin using the unified config.",
    )
    eval_parser.add_argument("--config", type=str, required=True)
    eval_parser.add_argument("--benchmark-root", default=None)
    eval_parser.add_argument("--task-name", default=None)
    eval_parser.add_argument("--task-config", default=None)
    eval_parser.add_argument("--piper-distance", type=float, default=None)
    eval_parser.add_argument("--checkpoint", default=None)
    eval_parser.add_argument("--dataset-root", default=None)
    eval_parser.add_argument("--repo-id", default=None)
    eval_parser.add_argument("--episodes", type=int, default=None)
    eval_parser.add_argument("--seed", type=int, default=None)
    eval_parser.add_argument("--device", default=None)
    eval_parser.add_argument("--output-dir", default=None)
    eval_parser.set_defaults(func=cmd_eval)

    return parser


def main(argv=None) -> None:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
