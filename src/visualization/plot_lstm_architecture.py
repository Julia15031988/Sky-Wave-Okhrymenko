
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def draw_box(x, title, detail, color):
        box = FancyBboxPatch(
            (x, 2.0),
            2.5,
            2.0,
            boxstyle="round,pad=0.15,rounding_size=0.25",
            linewidth=2,
            edgecolor=color,
            facecolor="#161d33",
        )
        ax.add_patch(box)

        ax.text(
            x + 1.25, 3.35, title,
            ha="center", va="center",
            fontsize=15, fontweight="bold",
            color="white",
        )

        ax.text(
            x + 1.25, 2.65, detail,
            ha="center", va="center",
            fontsize=12, color=color,
        )

    draw_box(0.3, "INPUT", "30 × 2 positions", "#38bdf8")
    draw_box(3.6, "LSTM", "2 layers | hidden 64", "#c084fc")
    draw_box(6.9, "LINEAR", "64 → 40 features", "#f9a8d4")
    draw_box(10.2, "OUTPUT", "20 × 2 positions", "#5eead4")

    for start, end in [(2.95, 3.5), (6.25, 6.8), (9.55, 10.1)]:
        arrow = FancyArrowPatch(
            (start, 3.0),
            (end, 3.0),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2,
            color="#cbd5e1",
        )
        ax.add_patch(arrow)

    ax.text(
        6.5, 5.35,
        "SKYWAVE | LSTM TRAJECTORY PREDICTOR",
        ha="center", va="center",
        fontsize=20, fontweight="bold",
        color="white",
    )

    ax.text(
        6.5, 1.0,
        "53,288 trainable parameters  |  "
        "30 observed positions → 20 predicted positions",
        ha="center",
        fontsize=12,
        color="#cbd5e1",
    )

    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "figure_2_lstm_architecture.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    print(f"Figure saved to: {output_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
