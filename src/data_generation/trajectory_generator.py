import numpy as np

from src.config import TrajectoryConfig


class TrajectoryGenerator:
    """Generate synthetic 2D motion trajectories."""

    def __init__(self, config: TrajectoryConfig) -> None:
        self.config = config

    def generate_linear(self) -> np.ndarray:
        """Generate motion with constant velocity."""

        time = (
            np.arange(self.config.num_steps)
            * self.config.time_step
        )

        x = self.config.initial_x + self.config.initial_vx * time
        y = self.config.initial_y + self.config.initial_vy * time

        trajectory = np.column_stack((x, y))

        return trajectory
