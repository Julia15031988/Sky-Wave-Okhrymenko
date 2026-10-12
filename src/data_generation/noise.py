import numpy as np

from src.config import TrajectoryConfig


class NoiseGenerator:
    def __init__(self, config: TrajectoryConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def add_gaussian_noise(
        self,
        trajectory: np.ndarray,
    ) -> np.ndarray:
        """Add Gaussian measurement noise to a trajectory."""
        noise = self.rng.normal(
            loc=0.0,
            scale=self.config.noise_std,
            size=trajectory.shape,
        )

        return trajectory + noise
