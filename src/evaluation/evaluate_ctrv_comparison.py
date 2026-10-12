
import numpy as np

from src.evaluation.comparison_dataset import prepare_comparison_data
from src.models.kalman_turn_rate import KalmanTurnRatePredictor


def evaluate_ctrv(
    observations: np.ndarray,
    targets: np.ndarray,
    time_step: float = 0.1,
    noise_std: float = 0.1,
) -> tuple[float, float]:
    predictions = []

    for observed_window in observations:
        predictor = KalmanTurnRatePredictor(
            time_step=time_step,
            noise_std=noise_std,
        )

        predictor.filter_trajectory(observed_window)

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

    mae, rmse = evaluate_ctrv(
        observations,
        targets,
        time_step=0.1,
        noise_std=noise_std,
    )

    print("=== SKYWAVE: CTRV SHARED DATASET ===")
    print(f"Test windows: {len(observations)}")
    print(f"History: {observations.shape[1]} steps")
    print(f"Forecast: {targets.shape[1]} steps")
    print(f"Noise sigma: {noise_std}")
    print()
    print(f"CTRV MAE:  {mae:.4f}")
    print(f"CTRV RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
