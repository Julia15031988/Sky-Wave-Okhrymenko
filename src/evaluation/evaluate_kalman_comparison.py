
import numpy as np

from src.evaluation.comparison_dataset import prepare_comparison_data
from src.models.kalman_filter import KalmanTrajectoryPredictor


def evaluate_kalman(
    observations: np.ndarray,
    targets: np.ndarray,
    time_step: float = 0.1,
    noise_std: float = 0.1,
) -> tuple[float, float]:
    """Evaluate Kalman forecasts on shared test windows."""

    predictions = []

    for observed_window in observations:
        # Reset the filter for every independent forecast.
        predictor = KalmanTrajectoryPredictor(
            time_step=time_step,
            noise_std=noise_std,
        )

        # Estimate state from 30 observed positions.
        predictor.filter_trajectory(observed_window)

        # Forecast 20 future positions without measurements.
        future = predictor.predict_future_trajectory(
            steps=targets.shape[1],
        )

        predictions.append(future)

    predictions = np.asarray(predictions)

    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    return float(mae), float(rmse)


def main():
    noise_std = 0.1

    observations, targets = prepare_comparison_data(
        noise_std=noise_std,
    )

    mae, rmse = evaluate_kalman(
        observations,
        targets,
        time_step=0.1,
        noise_std=noise_std,
    )

    print("=== Kalman Filter: Shared Comparison Dataset ===")
    print(f"Test windows: {len(observations)}")
    print(f"Input steps: {observations.shape[1]}")
    print(f"Forecast steps: {targets.shape[1]}")
    print(f"Noise sigma: {noise_std}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
