import matplotlib.pyplot as plt
import numpy as np

from src.config import TrajectoryConfig
from src.data_generation.noise import add_gaussian_noise
from src.data_generation.trajectory_generator import TrajectoryGenerator


def plot_trajectories(
    true_trajectory: np.ndarray,
    noisy_trajectory: np.ndarray,
) -> None:
    """Visualize true and noisy 2D trajectories."""

    plt.figure(figsize=(9, 6))

    plt.plot(
        true_trajectory[:, 0],
        true_trajectory[:, 1],
        label="True trajectory",
        color="blue",
        linewidth=2,
    )

    plt.scatter(
        noisy_trajectory[:, 0],
        noisy_trajectory[:, 1],
        label="Noisy observations",
        color="red",
        alpha=0.5,
        s=15,
    )

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("SkyWave: True vs Noisy Trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    config = TrajectoryConfig()

    generator = TrajectoryGenerator(config)
    true_trajectory = generator.generate_linear()

    noisy_trajectory = add_gaussian_noise(
        true_trajectory,
        noise_std=config.noise_std,
        random_seed=config.random_seed,
    )

    plot_trajectories(true_trajectory, noisy_trajectory)
