import numpy as np

from src.data_generation.noise import add_gaussian_noise


def test_noise_preserves_shape():
    trajectory = np.zeros((200, 2))

    noisy = add_gaussian_noise(trajectory)

    assert noisy.shape == (200, 2)


def test_noise_is_reproducible():
    trajectory = np.zeros((10, 2))

    first = add_gaussian_noise(
        trajectory,
        random_seed=42,
    )

    second = add_gaussian_noise(
        trajectory,
        random_seed=42,
    )

    np.testing.assert_array_equal(first, second)


def test_zero_noise():
    trajectory = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    noisy = add_gaussian_noise(
        trajectory,
        noise_std=0.0,
    )

    np.testing.assert_array_equal(noisy, trajectory)
