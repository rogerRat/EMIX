"""Plot the symmetric Beta(alpha; beta, beta) densities used by EMIX."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def beta_pdf(x: np.ndarray, shape: float) -> np.ndarray:
    """Evaluate Beta(shape, shape) probability density values."""

    normalization = math.gamma(shape) ** 2 / math.gamma(2.0 * shape)
    return np.power(x, shape - 1.0) * np.power(1.0 - x, shape - 1.0) / normalization


def main() -> None:
    shapes = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6)
    x = np.linspace(1e-5, 1.0 - 1e-5, 4000)
    x_center = np.linspace(1e-3, 1.0 - 1e-3, 4000)
    output_path = Path("outputs/beta_alpha_distributions.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (center_ax, full_ax) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for shape in shapes:
        label = f"beta={shape:.1f}"
        center_ax.plot(x_center, beta_pdf(x_center, shape), linewidth=2.0, label=label)
        full_ax.plot(x, beta_pdf(x, shape), linewidth=2.0, label=label)

    for ax in (center_ax, full_ax):
        ax.axvline(0.5, color="#666666", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("Alpha (weight of member_0)")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.25)

    center_ax.set_title("Central region (linear scale)")
    center_ax.set_ylabel("Probability density")
    center_ax.set_ylim(0.0, 3.0)
    center_ax.text(0.505, 0.95, "mean = 0.5", transform=center_ax.transAxes, ha="left", va="top", color="#555555")
    center_ax.legend(title="Beta shape", ncol=2)

    full_ax.set_title("Full range (log scale)")
    full_ax.set_yscale("log")
    full_ax.set_ylabel("Probability density (log)")
    full_ax.text(0.505, 0.95, "mean = 0.5", transform=full_ax.transAxes, ha="left", va="top", color="#555555")

    fig.suptitle("EMIX Alpha distribution: symmetric Beta(beta, beta)", fontsize=16)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
