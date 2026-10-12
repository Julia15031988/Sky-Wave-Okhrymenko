from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    noise = np.array([0.0, 0.1, 0.2, 0.3, 0.5])

    cv = [1.2810, 3.3997, 3.5828, 3.6289, 3.6740]
    ctrv = [0.0376, 0.1443, 0.3065, 0.4744, 0.7691]
    clean = [0.1106, 0.1440, 0.2130, 0.2932, 0.4622]
    robust = [0.1385, 0.1528, 0.1871, 0.2327, 0.3392]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 9),
        sharex=True,
    )

    models = [
        ("Kalman CV", cv, "o"),
        ("Kalman CTRV (EKF)", ctrv, "s"),
        ("Clean LSTM", clean, "^"),
        ("Noise-Robust LSTM", robust, "D"),
    ]

    for name, values, marker in models:
        axes[0].plot(
            noise, values,
            marker=marker,
            linewidth=2,
            label=name,
        )

    axes[0].set_title("All Four Models")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for name, values, marker in models[1:]:
        axes[1].plot(
            noise, values,
            marker=marker,
            linewidth=2,
            label=name,
        )

    axes[1].set_title("Detailed View: CTRV vs LSTM")
    axes[1].set_xlabel("Gaussian Noise Standard Deviation")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xticks(noise)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(
        "SkyWave: Physics-Based vs Learning-Based Prediction",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout()

    output = Path("figures/figure_6_four_model_comparison.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output, dpi=300, bbox_inches="tight")

    print(f"Saved: {output}")

    plt.show()


if __name__ == "__main__":
    main()
