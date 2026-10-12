
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Results from SkyWave experiments
    noise_levels = np.array([0.0, 0.1, 0.2, 0.3, 0.5])

    clean_rmse = np.array([
        0.1106, 0.1412, 0.2092, 0.2891, 0.4582
    ])

    robust_rmse = np.array([
        0.1385, 0.1504, 0.1832, 0.2281, 0.3340
    ])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        noise_levels,
        clean_rmse,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="#2563eb",
        label="Clean LSTM",
    )

    ax.plot(
        noise_levels,
        robust_rmse,
        marker="s",
        linewidth=2.5,
        markersize=8,
        color="#a855f7",
        label="Noise-Robust LSTM",
    )

    ax.set_title(
        "LSTM Robustness Under Measurement Noise",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    ax.set_xlabel("Gaussian Noise (σ)", fontsize=12)
    ax.set_ylabel("Root Mean Squared Error (RMSE)", fontsize=12)

    ax.set_xticks(noise_levels)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=11)

    ax.annotate(
        "Noise-Robust LSTM takes the lead",
        xy=(0.2, 0.1832),
        xytext=(0.24, 0.12),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#6b7280"},
    )

    fig.tight_layout()

    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "figure_5_noise_robustness.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Figure saved to: {output_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
