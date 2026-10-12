import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanTrajectoryPredictor:
    def __init__(
        self,
        time_step: float,
        noise_std: float,
        acceleration_std: float = 0.1,
    ):
        self.time_step = time_step

        # Initialize Kalman Filter
        # State: [x, y, vx, vy]
        # Measurement: [x, y]
        self.filter = KalmanFilter(
            dim_x=4,
            dim_z=2,
        )

        # F: State transition matrix
        self.filter.F = np.array([
            [1, 0, time_step, 0],
            [0, 1, 0, time_step],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        # H: Measurement matrix
        self.filter.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # R: Measurement noise covariance
        self.filter.R = np.array([
            [noise_std**2, 0],
            [0, noise_std**2],
        ])

        # Q: Process noise covariance
        dt = time_step

        G = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt],
        ])

        self.filter.Q = acceleration_std**2 * G @ G.T

        # Initial state: [x, y, vx, vy]
        self.filter.x = np.zeros(4)

        # P: Initial state uncertainty
        self.filter.P = np.diag([
            1.0,
            1.0,
            100.0,
            100.0,
        ])

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Predict and correct the state using a measurement."""

        self.filter.predict()
        self.filter.update(measurement)

        return self.filter.x[:2].copy()

    def filter_trajectory(
            self,
            measurements: np.ndarray,
    ) -> np.ndarray:
        """Estimate positions from noisy trajectory measurements."""

        estimated_positions = []

        for measurement in measurements:
            estimated_position = self.update(measurement)
            estimated_positions.append(estimated_position)

        return np.array(estimated_positions)
