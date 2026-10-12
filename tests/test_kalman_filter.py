import numpy as np

from src.models.kalman_filter import KalmanTrajectoryPredictor


def test_kalman_returns_two_coordinates():
    kalman = KalmanTrajectoryPredictor(
        time_step=0.1,
        noise_std=0.1,
    )

    result = kalman.update(np.array([5.0, 3.0]))

    assert result.shape == (2,)
    assert np.all(np.isfinite(result))


def test_kalman_filters_entire_trajectory():
    kalman = KalmanTrajectoryPredictor(
        time_step=0.1,
        noise_std=0.1,
    )

    measurements = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    result = kalman.filter_trajectory(measurements)

    assert result.shape == measurements.shape
    assert np.all(np.isfinite(result))
