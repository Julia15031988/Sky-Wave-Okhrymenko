import numpy as np

from src.config import TrajectoryConfig
from src.data_generation.noise import NoiseGenerator


def test_noise_generator_shape():
    config = TrajectoryConfig()
    generator = NoiseGenerator(config)

    trajectory = np.zeros((100, 2))

    noisy_trajectory = generator.add_gaussian_noise(
        trajectory
    )

    assert noisy_trajectory.shape == trajectory.shape

def test_noise_generator_reproducibility():
    config = TrajectoryConfig(random_seed=42)
    trajectory = np.zeros((100, 2))

    generator_1 = NoiseGenerator(config)
    generator_2 = NoiseGenerator(config)

    result_1 = generator_1.add_gaussian_noise(trajectory)
    result_2 = generator_2.add_gaussian_noise(trajectory)

    np.testing.assert_array_equal(result_1, result_2)


def test_noise_generator_zero_noise():
    config = TrajectoryConfig(noise_std=0.0)
    generator = NoiseGenerator(config)

    trajectory = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    result = generator.add_gaussian_noise(trajectory)

    np.testing.assert_array_equal(result, trajectory)


def test_noise_generator_changes_trajectory():
    config = TrajectoryConfig(noise_std=0.5)
    generator = NoiseGenerator(config)

    trajectory = np.zeros((100, 2))

    result = generator.add_gaussian_noise(trajectory)

    assert not np.array_equal(result, trajectory)
