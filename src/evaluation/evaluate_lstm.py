
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import TrajectoryConfig
from src.data_generation.data_split import split_trajectories
from src.data_generation.dataset_generator import (
    TrajectoryDatasetGenerator,
    create_sequences,
)
from src.data_generation.torch_dataset import TrajectoryDataset
from src.evaluation.metrics import calculate_mae, calculate_rmse
from src.models.lstm_model import LSTMTrajectoryPredictor


def evaluate_lstm() -> None:
    """Evaluate the trained LSTM on all held-out test windows."""

    config = TrajectoryConfig()

    generator = TrajectoryDatasetGenerator(config)

    trajectories = generator.generate_circular_dataset(
        num_trajectories=100,
    )

    _, _, test_data = split_trajectories(
        trajectories,
        random_seed=config.random_seed,
    )

    X_test, y_test = create_sequences(
        test_data,
        input_steps=30,
        output_steps=20,
    )

    test_loader = DataLoader(
        TrajectoryDataset(X_test, y_test),
        batch_size=64,
        shuffle=False,
    )

    model_path = Path("models/lstm_circular_best.pt")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    model = LSTMTrajectoryPredictor()

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)

            all_predictions.append(
                predictions.numpy()
            )

            all_targets.append(
                batch_y.numpy()
            )

    predicted = np.concatenate(
        all_predictions,
        axis=0,
    )

    actual = np.concatenate(
        all_targets,
        axis=0,
    )

    mae = calculate_mae(actual, predicted)
    rmse = calculate_rmse(actual, predicted)

    print("\n--- SkyWave: LSTM Test Evaluation ---")
    print(f"Test trajectories: {len(test_data)}")
    print(f"Test windows: {len(X_test)}")
    print(f"History length: {X_test.shape[1]} steps")
    print(f"Forecast horizon: {y_test.shape[1]} steps")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("-------------------------------------")


if __name__ == "__main__":
    evaluate_lstm()
