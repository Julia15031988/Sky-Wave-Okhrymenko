import matplotlib.pyplot as plt
import numpy as np
from src.evaluation.metrics import calculate_mae, calculate_rmse

from src.config import TrajectoryConfig
from src.data_generation.noise import NoiseGenerator
from src.data_generation.trajectory_generator import TrajectoryGenerator
from src.models.kalman_filter import KalmanTrajectoryPredictor


def main():
    config = TrajectoryConfig()

    # Generate the true trajectory
    generator = TrajectoryGenerator(config)
    true_trajectory = generator.generate_linear()

    # Add Gaussian noise
    noise_generator = NoiseGenerator(config)
    noisy_trajectory = noise_generator.add_gaussian_noise(
        true_trajectory
    )

    # Use the first 150 points as observed history
    history_steps = 150
    forecast_steps = 20

    observed = noisy_trajectory[:history_steps]

    # Initialize and run Kalman Filter
    kalman = KalmanTrajectoryPredictor(
        time_step=config.time_step,
        noise_std=config.noise_std,
    )

    filtered = kalman.filter_trajectory(observed)

    # Predict 20 future positions without measurements
    forecast = kalman.predict_future_trajectory(
        forecast_steps
    )

    # Future ground truth is used only for evaluation
    future_true = true_trajectory[
        history_steps:history_steps + forecast_steps
    ]
    # Evaluate future trajectory prediction
    forecast_mae = calculate_mae(future_true, forecast)
    forecast_rmse = calculate_rmse(future_true, forecast)

    print("\n--- SkyWave: Future Forecast Evaluation ---")
    print(f"Forecast horizon: {forecast_steps} steps")
    print(f"MAE:  {forecast_mae:.4f}")
    print(f"RMSE: {forecast_rmse:.4f}")

    # Display the final part of the observed trajectory
    visible_start = 120

    plt.figure(figsize=(10, 7))

    plt.plot(
        true_trajectory[visible_start:history_steps, 0],
        true_trajectory[visible_start:history_steps, 1],
        label="True history",
        color="blue",
    )

    plt.scatter(
        observed[visible_start:, 0],
        observed[visible_start:, 1],
        label="Noisy observations",
        color="red",
        s=12,
        alpha=0.5,
    )

    plt.plot(
        filtered[visible_start:, 0],
        filtered[visible_start:, 1],
        label="Kalman filtered",
        color="green",
    )

    plt.plot(
        future_true[:, 0],
        future_true[:, 1],
        label="True future",
        color="blue",
        linestyle="--",
    )

    plt.plot(
        forecast[:, 0],
        forecast[:, 1],
        label="Kalman forecast",
        color="purple",
        linestyle="--",
        marker="o",
        markersize=3,
    )

    plt.scatter(
        filtered[-1, 0],
        filtered[-1, 1],
        color="black",
        s=70,
        label="Forecast starting point",
        zorder=5,
    )

    plt.title("SkyWave: Kalman Future Trajectory Forecast")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()
