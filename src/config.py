from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryConfig:
    """Configuration for synthetic trajectory generation."""

    time_step: float = 0.1
    num_steps: int = 200
    initial_x: float = 0.0
    initial_y: float = 0.0
    initial_vx: float = 1.0
    initial_vy: float = 1.0
    noise_std: float = 0.1
    random_seed: int = 42
