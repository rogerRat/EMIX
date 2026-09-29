"""Visual comparison of the training-time Gaussian noise on real archived frames.

Loads frames straight from the archived lerobot AV1 videos (the exact bytes the
training dataloader decodes), applies the same additive N(0, variance) noise in
the [0,1] domain as embodied_mixup.policies.utils.image_noise, and renders a
grid: rows = tasks/frames, columns = original + several variance levels with
the run's var=0.0025 highlighted.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from embodied_mixup.policies.utils.image_noise import ImageNoiseConfig, add_image_noise  # noqa: E402

DATA = Path("archive/DP-10tasks-500ep-emix-baseline-cen-arm-top-cam-20260911/data/lerobot")
OUT = Path("outputs/analysis/noise_visual_compare.png")

# (task, frame timestamp seconds into the concatenated chunk video)
SAMPLES = [
    ("place_can_basket", 1.0),
    ("place_dual_shoes", 2.0),
    ("stack_blocks_two", 1.5),
]
VARIANCES = [0.000625, 0.0025, 0.01]  # sigma = 0.025 / 0.05 / 0.1

SCALE = 4  # 84x84 -> 336x336 per panel


def grab_frame(video: Path, t: float) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(t), "-i", str(video), "-frames:v", "1", tmp],
        check=True, capture_output=True,
    )
    img = np.asarray(Image.open(tmp).convert("RGB"), dtype=np.float32) / 255.0
    Path(tmp).unlink()
    return img


def add_noise(img: np.ndarray, variance: float) -> np.ndarray:
    batch = {"observation.images.head_camera": __import__("torch").from_numpy(img).permute(2, 0, 1)[None]}
    noisy = add_image_noise(batch, ImageNoiseConfig(enabled=True, variance=variance), seed=0, step=0)
    return noisy["observation.images.head_camera"][0].permute(1, 2, 0).numpy()


def main() -> None:
    rows = []
    for task, t in SAMPLES:
        video = DATA / f"{task}_left_cen_arm_top_cam" / "videos/observation.images.head_camera/chunk-000/file-000.mp4"
        orig = grab_frame(video, t)
        panels = [(f"{task}  original", orig)]
        for var in VARIANCES:
            mark = "  <-- 本轮 var=0.0025" if var == 0.0025 else ""
            panels.append((f"var={var}  (sigma={var**0.5:.3f}, 255域={var**0.5*255:.1f}){mark}", add_noise(orig, var)))
        rows.append(panels)

    n_cols = len(rows[0])
    cell_w, cell_h = 84 * SCALE, 84 * SCALE + 26
    canvas = Image.new("RGB", (n_cols * cell_w, len(rows) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for r, panels in enumerate(rows):
        for c, (label, img) in enumerate(panels):
            panel = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize(
                (84 * SCALE, 84 * SCALE), Image.NEAREST
            )
            x, y = c * cell_w, r * cell_h
            canvas.paste(panel, (x, y))
            draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline="gray")
            draw.text((x + 6, y + 84 * SCALE + 6), label, fill="black")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
