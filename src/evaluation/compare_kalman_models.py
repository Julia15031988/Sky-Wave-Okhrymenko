
import numpy as np

from src.config import TrajectoryConfig
from src.data_generation.trajectory_generator import TrajectoryGenerator
from src.models.kalman_filter import KalmanTrajectoryPredictor
from src.models.kalman_turn_rate import KalmanTurnRatePredictor


def calculate_rmse(predictions, targets):
    return float(
        np.sqrt(np.mean((predictions - targets) ** 2))
    )


def main():
    config = TrajectoryConfig()
    generator = TrajectoryGenerator(config)

    # Generate circular motion
    clean = generator.generate_circular(
        radius=5.0,
        angular_speed=0.3,
    )

    # Add reproducible Gaussian noise
    rng = np.random.default_rng(42)
    noisy = clean + rng.normal(
        0.0,
        0.1,
        size=clean.shape,
    )

    observed = noisy[:30]
    target = clean[30:50]

    # Model 1: Constant Velocity Kalman
    kalman_cv = KalmanTrajectoryPredictor(
        time_step=0.1,
        noise_std=0.1,
    )

    kalman_cv.filter_trajectory(observed)

    cv_predictions = kalman_cv.predict_future_trajectory(
        steps=20,
    )

    # Model 2: Constant Turn Rate EKF
    kalman_ctrv = KalmanTurnRatePredictor(
        time_step=0.1,
        noise_std=0.1,
    )

    kalman_ctrv.filter_trajectory(observed)

    ctrv_predictions = kalman_ctrv.predict_future_trajectory(
        steps=20,
    )

    cv_rmse = calculate_rmse(cv_predictions, target)
    ctrv_rmse = calculate_rmse(ctrv_predictions, target)

    print("=== SKYWAVE: KALMAN CV vs CTRV ===")
    print("Trajectory: Circular")
    print("Radius: 5.0")
    print("Angular speed: 0.3")
    print("History: 30 steps")
    print("Forecast: 20 steps")
    print("Noise sigma: 0.1")
    print()
    print(f"Kalman CV RMSE:   {cv_rmse:.4f}")
    print(f"Kalman CTRV RMSE: {ctrv_rmse:.4f}")
    print()
    print(
        "Estimated turn rate:",
        round(float(kalman_ctrv.filter.x[4]), 4),
    )


if __name__ == "__main__":
    main()
