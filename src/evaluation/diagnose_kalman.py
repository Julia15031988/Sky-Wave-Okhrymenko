
import numpy as np

from src.evaluation.comparison_dataset import prepare_comparison_data
from src.models.kalman_filter import KalmanTrajectoryPredictor


def predict_kalman(observed, initialize_velocity=False):
    predictor = KalmanTrajectoryPredictor(
        time_step=0.1,
        noise_std=0.1,
    )

    if initialize_velocity:
        position = observed[0]
        velocity = (observed[1] - observed[0]) / 0.1

        predictor.filter.x = np.array([
            position[0],
            position[1],
            velocity[0],
            velocity[1],
        ])

    predictor.filter_trajectory(observed)
    return predictor.predict_future_trajectory(steps=20)


def main():
    observations, targets = prepare_comparison_data(
        noise_std=0.1,
    )

    original_predictions = []
    initialized_predictions = []

    for observed in observations:
        original_predictions.append(
            predict_kalman(observed, initialize_velocity=False)
        )
        initialized_predictions.append(
            predict_kalman(observed, initialize_velocity=True)
        )

    original_predictions = np.asarray(original_predictions)
    initialized_predictions = np.asarray(initialized_predictions)

    original_rmse = np.sqrt(
        np.mean((original_predictions - targets) ** 2)
    )

    initialized_rmse = np.sqrt(
        np.mean((initialized_predictions - targets) ** 2)
    )

    print("=== KALMAN INITIALIZATION: FULL DATASET ===")
    print(f"Test windows: {len(observations)}")
    print(f"Original RMSE:    {original_rmse:.4f}")
    print(f"Initialized RMSE: {initialized_rmse:.4f}")


if __name__ == "__main__":
    main()
