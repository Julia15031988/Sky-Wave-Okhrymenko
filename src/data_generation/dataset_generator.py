
import numpy as np

from src.config import TrajectoryConfig
from src.data_generation.trajectory_generator import TrajectoryGenerator


class TrajectoryDatasetGenerator:
    """Generate synthetic trajectory datasets for machine learning."""

    def __init__(self, config: TrajectoryConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def generate_circular_dataset(
        self,
        num_trajectories: int,
    ) -> np.ndarray:
        """Generate circular trajectories with varied motion parameters."""
        if num_trajectories < 1:
            raise ValueError("num_trajectories must be positive")

        trajectories = []

        for _ in range(num_trajectories):
            radius = self.rng.uniform(2.0, 10.0)
            angular_speed = self.rng.uniform(0.1, 1.0)

            generator = TrajectoryGenerator(self.config)

            trajectory = generator.generate_circular(
                radius=radius,
                angular_speed=angular_speed,
            )

            trajectories.append(trajectory)

        return np.stack(trajectories)


def create_sequences(
    trajectories: np.ndarray,
    input_steps: int = 30,
    output_steps: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Create input and target windows from trajectories."""
    if input_steps < 1 or output_steps < 1:
        raise ValueError("Window sizes must be positive")

    if trajectories.ndim != 3 or trajectories.shape[2] != 2:
        raise ValueError("Expected shape (N, T, 2)")

    if trajectories.shape[1] < input_steps + output_steps:
        raise ValueError(
            "Trajectory length must be at least input_steps + output_steps"
        )

    x_sequences = []
    y_sequences = []

    for trajectory in trajectories:
        total_steps = len(trajectory)

        for start in range(
            total_steps - input_steps - output_steps + 1
        ):
            input_end = start + input_steps
            output_end = input_end + output_steps

            x_sequences.append(
                trajectory[start:input_end]
            )

            y_sequences.append(
                trajectory[input_end:output_end]
            )

    return (
        np.asarray(x_sequences),
        np.asarray(y_sequences),
    )
