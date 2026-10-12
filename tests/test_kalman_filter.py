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

def test_predict_next_position():
    kalman = KalmanTrajectoryPredictor(
        time_step=0.1,
        noise_std=0.1,
    )

    # Set a known state: x, y, vx, vy
    kalman.filter.x = np.array([5.0, 3.0, 1.0, 2.0])

    # Save the original state
    original_state = kalman.filter.x.copy()

    # Predict the next position
    prediction = kalman.predict_next_position()

    # Check the expected coordinates
    expected = np.array([5.1, 3.2])

    np.testing.assert_allclose(prediction, expected)

    # Ensure prediction does not modify the filter state
    np.testing.assert_array_equal(
        kalman.filter.x,
        original_state,
    )

def test_predict_future_trajectory():
    kalman = KalmanTrajectoryPredictor(
        time_step=0.1,
        noise_std=0.1,
    )

    # Set a known state: x, y, vx, vy
    kalman.filter.x = np.array([5.0, 3.0, 1.0, 2.0])

    # Save the original state
    original_state = kalman.filter.x.copy()

    # Predict three future positions
    predictions = kalman.predict_future_trajectory(3)

    expected = np.array([
        [5.1, 3.2],
        [5.2, 3.4],
        [5.3, 3.6],
    ])

    # Verify predicted coordinates
    np.testing.assert_allclose(predictions, expected)

    # Verify the original filter state is unchanged
    np.testing.assert_array_equal(
        kalman.filter.x,
        original_state,
    )
