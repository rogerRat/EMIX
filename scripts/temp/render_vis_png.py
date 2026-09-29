"""Render an EMIX visualization HTML into a crisp PNG via headless Chrome.

The flow generates a Plotly HTML page per task and needs a high-quality static
image to inspect clustering without opening a browser. Chrome's default
``--screenshot`` renders at 1x with a default viewport, which produces a small,
blurry image, so this script deliberately uses non-default settings:

  * ``--force-device-scale-factor`` = 2 (retina-crisp text and markers)
  * a large explicit window (width/height), so the full stacked page is laid
    out before capture
  * ``--virtual-time-budget`` so Plotly's WebGL scene and post-plot tasks
    finish rendering before the frame is grabbed
  * ``--hide-scrollbars`` so no scrollbar gutter shows in the output

After Chrome writes the raw screenshot, PIL crops it to the content bounding
box (plus a small margin), so the PNG is exactly the visualization with no
blank viewport padding. A blank page (render failure) is treated as an error,
not silently saved.

Usage:
  python scripts/render_vis_png.py <input.html> <output.png> \
      [--width 2400] [--height 2600] [--device-scale-factor 2] \
      [--virtual-time-budget 60000] [--margin 24] [--timeout 240]

Environment:
  Runs in the `emix` conda env (needs PIL); Chrome is located on PATH or via
  the CHROME_BIN environment variable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageChops

CHROME_CANDIDATES = (
    "google-chrome-stable",
    "google-chrome",
    "chromium",
    "chromium-browser",
    "chrome",
)


def _find_chrome() -> str:
    """Locate a Chrome/Chromium executable, honoring CHROME_BIN first."""
    import os

    candidate = os.environ.get("CHROME_BIN")
    if candidate:
        resolved = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if resolved:
            return resolved
    for name in CHROME_CANDIDATES:
        resolved = shutil.which(name)
        if resolved:
            return resolved
    raise FileNotFoundError(
        "No Chrome/Chromium binary found on PATH (tried "
        f"{', '.join(CHROME_CANDIDATES)}). Set CHROME_BIN to the executable path."
    )


def _content_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Return the bounding box of non-white content, or None for a blank page."""
    white = Image.new("RGB", image.size, (255, 255, 255))
    difference = ImageChops.difference(image.convert("RGB"), white)
    bbox = difference.getbbox()
    if bbox is None:
        return None
    # Reject trivial content (e.g. a single stray pixel) as effectively blank.
    left, top, right, bottom = bbox
    if (right - left) < 50 or (bottom - top) < 50:
        return None
    return bbox


def render_html_to_png(
    html_path: Path,
    png_path: Path,
    *,
    width: int = 2400,
    height: int = 2600,
    device_scale_factor: int = 2,
    virtual_time_budget: int = 60_000,
    margin: int = 24,
    timeout: int = 240,
) -> Path:
    """Screenshot an HTML page with headless Chrome and crop to its content."""
    chrome = _find_chrome()
    html_uri = html_path.resolve().as_uri()
    png_path = png_path.resolve()
    png_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="emix-vis-") as work_dir:
        raw_png = Path(work_dir) / "raw.png"
        command = [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--enable-unsafe-swiftshader",
            "--hide-scrollbars",
            "--force-color-profile=srgb",
            f"--force-device-scale-factor={device_scale_factor}",
            f"--window-size={width},{height}",
            f"--virtual-time-budget={virtual_time_budget}",
            f"--screenshot={raw_png}",
            html_uri,
        ]
        subprocess.run(command, check=True, timeout=timeout, capture_output=True)

        if not raw_png.is_file() or raw_png.stat().st_size == 0:
            raise RuntimeError(f"Chrome produced no screenshot for {html_uri}")

        image = Image.open(raw_png).convert("RGB")
        bbox = _content_bbox(image)
        if bbox is None:
            raise RuntimeError(f"Screenshot of {html_uri} is blank; refusing to save a blank PNG")

        left, top, right, bottom = bbox
        left = max(0, left - margin)
        top = max(0, top - margin)
        right = min(image.width, right + margin)
        bottom = min(image.height, bottom + margin)
        image.crop((left, top, right, bottom)).save(png_path, format="PNG")
        return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input HTML file to screenshot.")
    parser.add_argument("output", type=Path, help="Output PNG file path.")
    parser.add_argument("--width", type=int, default=2400, help="Chrome window width (CSS px).")
    parser.add_argument("--height", type=int, default=2600, help="Chrome window height (CSS px).")
    parser.add_argument(
        "--device-scale-factor", type=int, default=2,
        help="Hi-DPI scale factor for crisp output (default: 2).",
    )
    parser.add_argument(
        "--virtual-time-budget", type=int, default=60_000,
        help="Milliseconds of virtual time to let the page render before capture.",
    )
    parser.add_argument(
        "--margin", type=int, default=24,
        help="Extra device pixels of white margin kept around cropped content.",
    )
    parser.add_argument(
        "--timeout", type=int, default=240,
        help="Wall-clock seconds to allow Chrome before killing it.",
    )
    args = parser.parse_args()

    output = render_html_to_png(
        args.input,
        args.output,
        width=args.width,
        height=args.height,
        device_scale_factor=args.device_scale_factor,
        virtual_time_budget=args.virtual_time_budget,
        margin=args.margin,
        timeout=args.timeout,
    )
    print(f"Rendered {output} ({output.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
