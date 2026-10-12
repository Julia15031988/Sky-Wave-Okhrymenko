
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter


class KalmanTurnRatePredictor:
    """Extended Kalman Filter with a constant turn-rate motion model."""

    def __init__(
        self,
        time_step: float,
        noise_std: float,
        acceleration_std: float = 0.1,
        turn_rate_std: float = 0.05,
    ) -> None:
        self.dt = time_step

        self.filter = ExtendedKalmanFilter(
            dim_x=5,
            dim_z=2,
        )

        self.filter.x = np.zeros(5)

        self.filter.P = np.diag([
            1.0,
            1.0,
            100.0,
            100.0,
            1.0,
        ])

        self.filter.R = np.eye(2) * noise_std**2

        self.filter.Q = np.diag([
            1e-6,
            1e-6,
            acceleration_std**2 * self.dt**2,
            acceleration_std**2 * self.dt**2,
            turn_rate_std**2 * self.dt**2,
        ])

    @staticmethod
    def measurement_function(state):
        return state[:2]

    @staticmethod
    def measurement_jacobian(state):
        return np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ])

    def transition_function(self, state):
        x, y, vx, vy, omega = state

        angle = omega * self.dt

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        if abs(omega) < 1e-6:
            new_x = x + vx * self.dt
            new_y = y + vy * self.dt
        else:
            new_x = (
                x
                + (sin_a / omega) * vx
                - ((1 - cos_a) / omega) * vy
            )
            new_y = (
                y
                + ((1 - cos_a) / omega) * vx
                + (sin_a / omega) * vy
            )

        new_vx = cos_a * vx - sin_a * vy
        new_vy = sin_a * vx + cos_a * vy

        return np.array([
            new_x,
            new_y,
            new_vx,
            new_vy,
            omega,
        ])

    def transition_jacobian(self, state):
        epsilon = 1e-5
        jacobian = np.zeros((5, 5))

        for i in range(5):
            plus = state.copy()
            minus = state.copy()

            plus[i] += epsilon
            minus[i] -= epsilon

            jacobian[:, i] = (
                self.transition_function(plus)
                - self.transition_function(minus)
            ) / (2 * epsilon)

        return jacobian

    def update(self, measurement):
        state = self.filter.x.copy()

        F = self.transition_jacobian(state)

        self.filter.x = self.transition_function(state)
        self.filter.P = (
            F @ self.filter.P @ F.T + self.filter.Q
        )

        self.filter.update(
            measurement,
            HJacobian=self.measurement_jacobian,
            Hx=self.measurement_function,
        )

        return self.filter.x[:2].copy()

    def filter_trajectory(self, measurements):
        estimates = []

        for measurement in measurements:
            estimates.append(self.update(measurement))

        return np.asarray(estimates)

    def predict_future_trajectory(self, steps):
        state = self.filter.x.copy()
        predictions = []

        for _ in range(steps):
            state = self.transition_function(state)
            predictions.append(state[:2].copy())

        return np.asarray(predictions)
