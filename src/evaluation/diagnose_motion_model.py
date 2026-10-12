
import numpy as np

from src.config import TrajectoryConfig
from src.data_generation.trajectory_generator import TrajectoryGenerator
from src.models.kalman_filter import KalmanTrajectoryPredictor


def evaluate_trajectory(
    trajectory: np.ndarray,
    noise_std: float = 0.1,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)

    noisy = trajectory + rng.normal(
        0.0,
        noise_std,
        size=trajectory.shape,
    )

    observed = noisy[:30]
    target = trajectory[30:50]

    predictor = KalmanTrajectoryPredictor(
        time_step=0.1,
        noise_std=noise_std,
    )

    predictor.filter_trajectory(observed)

    predictions = predictor.predict_future_trajectory(
        steps=20,
    )

    return float(
        np.sqrt(np.mean((predictions - target) ** 2))
    )


def main():
    config = TrajectoryConfig()
    generator = TrajectoryGenerator(config)

    linear = generator.generate_linear()

    circular = generator.generate_circular(
        radius=5.0,
        angular_speed=0.3,
    )

    linear_rmse = evaluate_trajectory(linear)
    circular_rmse = evaluate_trajectory(circular)

    print("=== KALMAN CV: MOTION MODEL DIAGNOSTIC ===")
    print("History: 30 steps")
    print("Forecast: 20 steps")
    print("Noise sigma: 0.1")
    print()
    print(f"Linear trajectory RMSE:   {linear_rmse:.4f}")
    print(f"Circular trajectory RMSE: {circular_rmse:.4f}")


if __name__ == "__main__":
    main()
