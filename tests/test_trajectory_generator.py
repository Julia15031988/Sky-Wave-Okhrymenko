import numpy as np

from src.config import TrajectoryConfig
from src.data_generation.trajectory_generator import TrajectoryGenerator


def test_linear_trajectory_shape():
    config = TrajectoryConfig(num_steps=200)
    generator = TrajectoryGenerator(config)

    trajectory = generator.generate_linear()

    assert trajectory.shape == (200, 2)


def test_linear_trajectory_coordinates():
    config = TrajectoryConfig(
        num_steps=3,
        time_step=0.1,
        initial_x=0.0,
        initial_y=0.0,
        initial_vx=1.0,
        initial_vy=2.0,
    )

    generator = TrajectoryGenerator(config)
    trajectory = generator.generate_linear()

    expected = np.array([
        [0.0, 0.0],
        [0.1, 0.2],
        [0.2, 0.4],
    ])

    np.testing.assert_allclose(trajectory, expected)
