import numpy as np


def add_gaussian_noise(
    trajectory: np.ndarray,
    noise_std: float = 0.1,
    random_seed: int = 42,
) -> np.ndarray:
    """Add Gaussian measurement noise to a trajectory."""

    rng = np.random.default_rng(random_seed)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=trajectory.shape,
    )

    noisy_trajectory = trajectory + noise

    return noisy_trajectory


if __name__ == "__main__":
    from src.config import TrajectoryConfig
    from src.data_generation.trajectory_generator import TrajectoryGenerator

    config = TrajectoryConfig()

    generator = TrajectoryGenerator(config)
    trajectory = generator.generate_linear()

    noisy_trajectory = add_gaussian_noise(
        trajectory,
        noise_std=config.noise_std,
        random_seed=config.random_seed,
    )

    print("Original trajectory:")
    print(trajectory[:5])

    print("\nNoisy trajectory:")
    print(noisy_trajectory[:5])
