
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import TrajectoryConfig
from src.data_generation.data_split import split_trajectories
from src.data_generation.dataset_generator import (
    TrajectoryDatasetGenerator,
    create_sequences,
)
from src.evaluation.metrics import calculate_mae, calculate_rmse
from src.models.lstm_model import LSTMTrajectoryPredictor


def plot_lstm_forecast() -> None:
    """Visualize LSTM predictions on an unseen circular trajectory."""

    config = TrajectoryConfig()

    # Recreate the same dataset used during training.
    generator = TrajectoryDatasetGenerator(config)

    trajectories = generator.generate_circular_dataset(
        num_trajectories=100,
    )

    # Use only the held-out test trajectories.
    _, _, test_data = split_trajectories(
        trajectories,
        random_seed=config.random_seed,
    )

    X_test, y_test = create_sequences(
        test_data,
        input_steps=30,
        output_steps=20,
    )

    # Select one test example.
    sample_index = 0

    history = X_test[sample_index]
    true_future = y_test[sample_index]

    # Load the trained LSTM model.
    model = LSTMTrajectoryPredictor()

    state_dict = torch.load(
        "models/lstm_circular_best.pt",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    # Prepare input: (batch, time_steps, coordinates).
    input_tensor = torch.tensor(
        history,
        dtype=torch.float32,
    ).unsqueeze(0)

    # Predict without updating model weights.
    with torch.no_grad():
        prediction = model(input_tensor)

    predicted_future = prediction.squeeze(0).numpy()

    # Calculate forecast errors.
    mae = calculate_mae(
        true_future,
        predicted_future,
    )

    rmse = calculate_rmse(
        true_future,
        predicted_future,
    )

    print("--- SkyWave: LSTM Circular Forecast ---")
    print(f"History length: {len(history)} steps")
    print(f"Forecast horizon: {len(true_future)} steps")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Plot the entire selected test trajectory.
    full_trajectory = test_data[0]

    plt.figure(figsize=(10, 7))

    plt.plot(
        full_trajectory[:, 0],
        full_trajectory[:, 1],
        color="royalblue",
        alpha=0.35,
        label="Ground Truth Trajectory",
    )

    plt.plot(
        history[:, 0],
        history[:, 1],
        color="green",
        linewidth=2.5,
        label="Observed History (30 steps)",
    )

    plt.plot(
        true_future[:, 0],
        true_future[:, 1],
        color="darkorange",
        linestyle="--",
        linewidth=2.5,
        label="True Future (20 steps)",
    )

    # Connect the prediction to the last observed point.
    predicted_path = np.vstack(
        (history[-1], predicted_future)
    )

    plt.plot(
        predicted_path[:, 0],
        predicted_path[:, 1],
        color="purple",
        linewidth=2.5,
        marker="o",
        markersize=3,
        label="LSTM Forecast",
    )

    plt.scatter(
        history[-1, 0],
        history[-1, 1],
        color="black",
        s=65,
        zorder=5,
        label="Forecast Start",
    )

    plt.title(
        "SkyWave: LSTM Circular Trajectory Forecast"
    )
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_lstm_forecast()
