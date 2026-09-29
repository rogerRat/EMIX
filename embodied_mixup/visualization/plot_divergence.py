"""Divergence point camera image map for click-to-inspect interaction.

Pre-computes camera images for cross-point groups in an EmixIndex, writes
them into an asset directory, and returns a JSON-serializable image map for
static HTML lookup.

The image map is keyed by a composite string "ep_frame_ep_frame" that
uniquely identifies each pair of frames, so a Plotly click event handler
can look up the corresponding camera images in O(1) time.

Selection is budget-aware and cluster-fair: each cluster is guaranteed at
least `min_per_cluster` image pairs (the total budget is expanded when
necessary), and remaining budget is distributed across clusters in
proportion to cluster size.

@input:  EmixIndex, LeRobotDataset-like object, optional camera key,
         total_budget, min_per_cluster.
@output: Dict with images_js (JSON string), total_pairs, camera_key,
         imaged_positions_js (JSON string for nearest-neighbor fallback),
         and asset_dir.
@scenario: Embed camera snapshots alongside trajectories so users can
          click any intersection point in the 3D plot and see what the
          robot saw from both trajectories at that divergence point.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..candidates.emix_index import EmixIndex

logger = logging.getLogger(__name__)

# Default max image dimension in pixels.  Zero preserves the source image
# resolution, which avoids quality loss from pre-computed image downscaling.
_DEFAULT_MAX_IMAGE_DIM = 0


def build_click_image_map(
    index: EmixIndex,
    dataset,
    camera_key: Optional[str] = None,
    max_image_dim: int = _DEFAULT_MAX_IMAGE_DIM,
    total_budget: int = 100,
    min_per_cluster: int = 10,
    asset_dir: str | Path | None = None,
    html_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Pre-compute camera image files for cross-point groups.

    Selection is cluster-fair: each cluster gets at least
    *min_per_cluster* pairs, then remaining budget is distributed
    proportionally to cluster size.  The total budget is silently
    expanded when the per-cluster floor demands it.

    @input:  index: EmixIndex with cross_point_groups.
             dataset: LeRobotDataset-like object supporting __getitem__.
             camera_key: observation.images.{key} to extract (default:
                         first key from source media contract).
             max_image_dim: max width/height in pixels. 0 preserves source
                            resolution. Aspect ratio is preserved and the
                            image is never upscaled beyond source.
             total_budget: upper bound on total pre-computed pairs
                           (may be expanded to satisfy min_per_cluster).
             min_per_cluster: floor on pairs per cluster.
             asset_dir: directory where PNG images are written. None keeps
                        the historical in-memory-only failure behavior.
             html_dir: directory used to make image paths relative to HTML.
    @output: Dict with:
             - images_js: JSON string mapping composite key to
               {img_a, img_b, cluster_id, divergence, x, y, z}.
             - imaged_positions_js: JSON string - list of {key, x, y, z}
               for nearest-neighbour fallback in the browser.
             - total_pairs: number of pairs in the map.
             - camera_key: camera key used.
             - skipped: number of pairs skipped due to save errors.
             - asset_dir: image asset directory, or None.
    @scenario: Save click-to-inspect images outside the HTML so the browser
               displays indexed source-quality PNG files instead of
               low-resolution embedded images.
    """
    if max_image_dim < 0:
        raise ValueError(f"max_image_dim must be non-negative, got {max_image_dim}")

    resolved_asset_dir: Path | None = None
    resolved_html_dir: Path | None = None
    if asset_dir is not None:
        resolved_asset_dir = Path(asset_dir)
        resolved_asset_dir.mkdir(parents=True, exist_ok=True)
        resolved_html_dir = Path(html_dir) if html_dir is not None else resolved_asset_dir.parent

    if camera_key is None:
        source_datasets = getattr(index.public, "source_datasets", [])
        camera_keys = (
            source_datasets[0].media.camera_keys if source_datasets else []
        )
        if camera_keys:
            camera_key = camera_keys[0]
        else:
            camera_key = "top"

    feature_key = f"observation.images.{camera_key}"

    # Load EEF poses so we can embed midpoints for nearest-neighbor
    # fallback in the browser.
    from ..candidates.loaders import load_eef_poses
    poses = load_eef_poses(dataset)

    groups: list = index.content.cross_point_groups
    # ------------------------------------------------------------------
    # 1.  Partition groups by cluster label.
    # ------------------------------------------------------------------
    cluster_buckets: dict[int, list] = {}
    for group in groups:
        if len(group.members) != 2:
            continue
        cid = int(group.group_metrics.get("cluster_id", -1))
        cluster_buckets.setdefault(cid, []).append(group)

    cluster_ids = sorted(cluster_buckets.keys())
    n_clusters = len(cluster_ids)
    if n_clusters == 0:
        logger.warning("build_click_image_map: no valid groups found.")
        return {
            "images_js": "{}",
            "imaged_positions_js": "[]",
            "total_pairs": 0,
            "camera_key": camera_key,
            "skipped": 0,
            "asset_dir": str(resolved_asset_dir) if resolved_asset_dir else None,
        }

    # ------------------------------------------------------------------
    # 2.  Determine per-cluster allocation.
    # ------------------------------------------------------------------
    # Each cluster gets its fair share: at least min_per_cluster.
    # If total_budget is insufficient to satisfy the floor, expand it.
    required_for_floor = n_clusters * min_per_cluster
    effective_budget = max(total_budget, required_for_floor)
    if effective_budget > total_budget:
        logger.info(
            "Expanding click-image budget %d -> %d to satisfy "
            "min_per_cluster=%d across %d clusters.",
            total_budget, effective_budget, min_per_cluster, n_clusters,
        )

    # Floor allocation: every cluster gets min_per_cluster (capped at
    # what is actually available in that cluster).
    cluster_sizes = {cid: len(bucket) for cid, bucket in cluster_buckets.items()}
    allocated: dict[int, int] = {}
    for cid in cluster_ids:
        allocated[cid] = min(min_per_cluster, cluster_sizes[cid])

    spent = sum(allocated.values())
    remaining_budget = max(0, effective_budget - spent)

    # Distribute remaining budget proportionally to cluster size.
    if remaining_budget > 0:
        total_size = sum(cluster_sizes.values())
        # Allocate proportional shares (at least 1 per cluster that still
        # has room, when budget permits).
        extra_shares: list[tuple[int, int]] = []  # (cid, points)
        for cid in cluster_ids:
            room = cluster_sizes[cid] - allocated[cid]
            if room <= 0:
                continue
            if total_size > 0:
                share = max(1, int(round(remaining_budget * cluster_sizes[cid] / total_size)))
            else:
                share = 1
            extra_shares.append((cid, min(share, room)))

        # Cap total extra to remaining_budget (rounding may overshoot).
        extra_shares.sort(key=lambda x: -x[1])  # largest first
        for cid, share in extra_shares:
            if remaining_budget <= 0:
                break
            give = min(share, remaining_budget, cluster_sizes[cid] - allocated[cid])
            if give > 0:
                allocated[cid] += give
                remaining_budget -= give

    logger.info(
        "Click-image allocation: budget=%d clusters=%d -> %s",
        effective_budget, n_clusters,
        {cid: allocated[cid] for cid in cluster_ids},
    )

    # ------------------------------------------------------------------
    # 3.  Save images for selected groups.
    # ------------------------------------------------------------------
    image_map: Dict[str, Dict[str, Any]] = {}
    imaged_positions: List[Dict[str, Any]] = []
    skipped = 0

    logger.info(
        "Pre-computing camera images for up to %d groups (camera=%s, max_image_dim=%d)...",
        sum(allocated.values()), camera_key, max_image_dim,
    )

    for cid in cluster_ids:
        bucket = cluster_buckets[cid]
        quota = allocated.get(cid, 0)
        if quota <= 0:
            continue
        # Take groups evenly distributed across the bucket (not just the
        # first *quota* which may be spatially biased).
        if quota >= len(bucket):
            selected = bucket
        else:
            step = len(bucket) / quota
            indices = [int(i * step) for i in range(quota)]
            selected = [bucket[i] for i in indices]

        for group in selected:
            m0 = group.members[0]
            m1 = group.members[1]
            f0 = int(m0.source_frame_index)
            f1 = int(m1.source_frame_index)
            ep0 = int(m0.source_episode_index)
            ep1 = int(m1.source_episode_index)
            cluster_id = int(group.group_metrics.get("cluster_id", -1))
            divergence = float(group.group_metrics.get("action_divergence", 0.0))

            # Midpoint for nearest-neighbor fallback.
            p0 = poses[f0].astype(np.float64)
            p1 = poses[f1].astype(np.float64)
            mx, my, mz = float((p0[0] + p1[0]) / 2.0), float((p0[1] + p1[1]) / 2.0), float((p0[2] + p1[2]) / 2.0)

            composite_key = f"{ep0}_{f0}_{ep1}_{f1}"

            frame_a_stem = f"{composite_key}_a"
            frame_b_stem = f"{composite_key}_b"
            img_a = _load_and_save_frame(
                dataset,
                f0,
                feature_key,
                max_image_dim,
                resolved_asset_dir,
                resolved_html_dir,
                frame_a_stem,
            )
            img_b = _load_and_save_frame(
                dataset,
                f1,
                feature_key,
                max_image_dim,
                resolved_asset_dir,
                resolved_html_dir,
                frame_b_stem,
            )

            if img_a is None or img_b is None:
                skipped += 1
                placeholder = _make_placeholder_image(
                    max_image_dim,
                    resolved_asset_dir,
                    resolved_html_dir,
                    f"{composite_key}_missing",
                )
                image_map[composite_key] = {
                    "img_a": img_a["src"] if img_a is not None else placeholder,
                    "img_b": img_b["src"] if img_b is not None else placeholder,
                    "cluster_id": cluster_id,
                    "divergence": divergence,
                    "ep_a": ep0, "frame_a": f0,
                    "ep_b": ep1, "frame_b": f1,
                    "x": mx, "y": my, "z": mz,
                    "width_a": img_a["width"] if img_a is not None else None,
                    "height_a": img_a["height"] if img_a is not None else None,
                    "width_b": img_b["width"] if img_b is not None else None,
                    "height_b": img_b["height"] if img_b is not None else None,
                }
            else:
                image_map[composite_key] = {
                    "img_a": img_a["src"],
                    "img_b": img_b["src"],
                    "cluster_id": cluster_id,
                    "divergence": divergence,
                    "ep_a": ep0, "frame_a": f0,
                    "ep_b": ep1, "frame_b": f1,
                    "x": mx, "y": my, "z": mz,
                    "width_a": img_a["width"],
                    "height_a": img_a["height"],
                    "width_b": img_b["width"],
                    "height_b": img_b["height"],
                }
            imaged_positions.append({"key": composite_key, "x": mx, "y": my, "z": mz})

    logger.info(
        "build_click_image_map: %d pairs processed, %d skipped",
        len(image_map), skipped,
    )

    return {
        "images_js": json.dumps(image_map),
        "imaged_positions_js": json.dumps(imaged_positions),
        "total_pairs": len(image_map),
        "camera_key": camera_key,
        "skipped": skipped,
        "asset_dir": str(resolved_asset_dir) if resolved_asset_dir else None,
    }


def _load_and_save_frame(
    dataset,
    frame_idx: int,
    feature_key: str,
    max_image_dim: int,
    asset_dir: Path | None,
    html_dir: Path | None,
    file_stem: str,
) -> Optional[Dict[str, Any]]:
    """Load a single dataset frame, extract the camera image, and save it.

    @input:  dataset: LeRobotDataset-like object.
             frame_idx: global frame index.
             feature_key: e.g. "observation.images.top".
             max_image_dim: max width/height in pixels. 0 preserves source
                            resolution. The image is never upscaled.
             asset_dir: target directory for PNG files.
             html_dir: output HTML directory for relative paths.
             file_stem: safe file stem without extension.
    @output: Dict with src, width, and height on success; None on failure.
    @scenario: Persist a click-to-inspect frame as a standalone PNG asset.
    """
    if asset_dir is None or html_dir is None:
        logger.debug("No asset_dir/html_dir provided for frame %d", frame_idx)
        return None

    try:
        item = dataset[frame_idx]
    except Exception as exc:
        logger.debug("Failed to load frame %d: %s", frame_idx, exc)
        return None

    if feature_key not in item:
        logger.debug("Feature '%s' not found in frame %d", feature_key, frame_idx)
        return None

    try:
        value = item[feature_key]
        img = _to_uint8_hwc(value)
        if img is None:
            return None
        return _save_image_file(img, max_image_dim, asset_dir, html_dir, file_stem)
    except Exception as exc:
        logger.debug("Failed to save frame %d: %s", frame_idx, exc)
        return None


def _to_uint8_hwc(value) -> Optional[np.ndarray]:
    """Convert a dataset image value to uint8 HWC numpy array.

    @input:  value: torch tensor, numpy array, PIL image, or array-like image.
    @output: uint8 HWC numpy array with three channels, or None on failure.
    @scenario: Normalize LeRobot camera values before PNG export.

    Handles torch tensors (CHW float32 0-1), numpy arrays (HWC uint8 or
    CHW float), and PIL Images.
    """
    # Torch tensor
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    else:
        value = np.asarray(value)

    if value.ndim == 2:
        value = np.stack([value] * 3, axis=-1)

    if value.ndim != 3:
        return None

    # Detect CHW vs HWC: if first dim is small (1 or 3), assume CHW
    if value.shape[0] in (1, 3) and value.shape[-1] not in (1, 3):
        value = np.transpose(value, (1, 2, 0))

    # Float [0,1] -> uint8 [0,255]
    if np.issubdtype(value.dtype, np.floating):
        value = np.clip(value * 255.0, 0, 255)

    value = value.astype(np.uint8)

    # Ensure 3-channel RGB
    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=-1)

    return value


def _save_image_file(
    img: np.ndarray,
    max_image_dim: int,
    asset_dir: Path,
    html_dir: Path,
    file_stem: str,
) -> Optional[Dict[str, Any]]:
    """Save a uint8 HWC image as PNG and return its browser source path.

    @input:  img: uint8 HWC image.
             max_image_dim: max width/height in pixels. 0 preserves source
                            resolution. The image is never upscaled.
             asset_dir: target directory for PNG files.
             html_dir: output HTML directory for relative paths.
             file_stem: safe file stem without extension.
    @output: Dict with relative src, width, and height; None on failure.
    @scenario: Write a source-quality click-to-inspect image outside HTML.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL (Pillow) is required for image export.")
        return None

    pil_img = Image.fromarray(img)
    src_w, src_h = pil_img.size

    if max_image_dim > 0:
        scale = min(max_image_dim / max(src_w, src_h), 1.0)
    else:
        scale = 1.0

    if scale < 1.0:
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        pil_img = pil_img.resize(
            (new_w, new_h),
            Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BILINEAR,
        )

    output_path = asset_dir / f"{file_stem}.png"
    pil_img.save(output_path, format="PNG")
    rel_src = os.path.relpath(output_path, start=html_dir)
    rel_src = Path(rel_src).as_posix()
    out_w, out_h = pil_img.size
    return {"src": rel_src, "width": out_w, "height": out_h}


def _make_placeholder_image(
    max_image_dim: int,
    asset_dir: Path | None,
    html_dir: Path | None,
    file_stem: str,
) -> str:
    """Generate a gray placeholder PNG and return its browser source path.

    @input:  max_image_dim: desired placeholder size. 0 uses 256 pixels.
             asset_dir: target directory for PNG files.
             html_dir: output HTML directory for relative paths.
             file_stem: safe file stem without extension.
    @output: Relative image source path, or empty string on failure.
    @scenario: Keep the image panel usable when one frame cannot be loaded.
    """
    if asset_dir is None or html_dir is None:
        return ""

    try:
        from PIL import Image
    except ImportError:
        return ""

    size = max_image_dim if max_image_dim > 0 else 256
    img = Image.new("RGB", (size, size), color=(128, 128, 128))
    output_path = asset_dir / f"{file_stem}.png"
    img.save(output_path, format="PNG")
    rel_src = os.path.relpath(output_path, start=html_dir)
    return Path(rel_src).as_posix()
