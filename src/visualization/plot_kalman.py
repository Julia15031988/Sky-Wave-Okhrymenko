import matplotlib.pyplot as plt

from src.config import TrajectoryConfig
from src.data_generation.noise import NoiseGenerator
from src.data_generation.trajectory_generator import TrajectoryGenerator
from src.models.kalman_filter import KalmanTrajectoryPredictor
from src.evaluation.metrics import calculate_mae, calculate_rmse


def main() -> None:
    # 1. Experiment configuration
    config = TrajectoryConfig()

    # 2. Generate true trajectory
    generator = TrajectoryGenerator(config)
    true_trajectory = generator.generate_linear()

    # 3. Add measurement noise
    noise_generator = NoiseGenerator(config)
    noisy_trajectory = noise_generator.add_gaussian_noise(true_trajectory)

    # 4. Create Kalman Filter
    kalman = KalmanTrajectoryPredictor(
        time_step=config.time_step,
        noise_std=config.noise_std,
    )

    # 5. Estimate trajectory
    kalman_trajectory = kalman.filter_trajectory(noisy_trajectory)

    # Evaluate noisy measurements
    noisy_mae = calculate_mae(true_trajectory, noisy_trajectory)
    noisy_rmse = calculate_rmse(true_trajectory, noisy_trajectory)

    # Evaluate Kalman estimates
    kalman_mae = calculate_mae(true_trajectory, kalman_trajectory)
    kalman_rmse = calculate_rmse(true_trajectory, kalman_trajectory)

    print("\n--- SkyWave: Kalman Filter Evaluation ---")

    print("\nNoisy observations:")
    print(f"MAE:  {noisy_mae:.4f}")
    print(f"RMSE: {noisy_rmse:.4f}")

    print("\nKalman estimates:")
    print(f"MAE:  {kalman_mae:.4f}")
    print(f"RMSE: {kalman_rmse:.4f}")

    # 6. Visualize results
    plt.figure(figsize=(10, 6))

    plt.plot(
        true_trajectory[:, 0],
        true_trajectory[:, 1],
        label="True trajectory",
        color="blue",
        linewidth=2,
    )

    plt.scatter(
        noisy_trajectory[:, 0],
        noisy_trajectory[:, 1],
        label="Noisy observations",
        color="red",
        alpha=0.4,
        s=15,
    )

    plt.plot(
        kalman_trajectory[:, 0],
        kalman_trajectory[:, 1],
        label="Kalman estimate",
        color="green",
        linewidth=2,
    )

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("SkyWave: Kalman Filter Trajectory Estimation")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # Zoom into a small section of the trajectory
    plt.xlim(5, 8)
    plt.ylim(5, 8)

    plt.show(block=True)


if __name__ == "__main__":
    main()
