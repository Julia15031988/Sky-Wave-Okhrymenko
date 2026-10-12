
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import TrajectoryConfig
from src.data_generation.data_split import split_trajectories
from src.data_generation.dataset_generator import TrajectoryDatasetGenerator
from src.data_generation.noisy_sequences import create_noisy_sequences
from src.data_generation.torch_dataset import TrajectoryDataset
from src.evaluation.metrics import calculate_mae, calculate_rmse
from src.models.lstm_model import LSTMTrajectoryPredictor


def evaluate_model(
    checkpoint_path: str,
    test_loader: DataLoader,
) -> tuple[float, float]:
    model = LSTMTrajectoryPredictor()

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predicted = model(X_batch)

            predictions.append(predicted.numpy())
            targets.append(y_batch.numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    mae = calculate_mae(targets, predictions)
    rmse = calculate_rmse(targets, predictions)

    return mae, rmse


def main() -> None:
    config = TrajectoryConfig()

    generator = TrajectoryDatasetGenerator(config)
    trajectories = generator.generate_circular_dataset(
        num_trajectories=100
    )

    _, _, test_trajectories = split_trajectories(
        trajectories,
        random_seed=42,
    )

    X_test, y_test = create_noisy_sequences(
        test_trajectories,
        input_steps=30,
        output_steps=20,
        noise_std=0.1,
        random_seed=44,
    )

    test_loader = DataLoader(
        TrajectoryDataset(X_test, y_test),
        batch_size=64,
        shuffle=False,
    )

    models = {
        "Clean LSTM": "models/lstm_circular_best.pt",
        "Noise-Robust LSTM": "models/lstm_circular_noisy_best.pt",
    }

    print("\n--- SkyWave: LSTM Model Comparison ---")
    print(f"Test trajectories: {len(test_trajectories)}")
    print(f"Test windows: {len(X_test)}")
    print("Input noise standard deviation: 0.1")
    print("Forecast horizon: 20 steps\n")

    for name, checkpoint in models.items():
        mae, rmse = evaluate_model(checkpoint, test_loader)

        print(f"{name}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}\n")

    print("-------------------------------------")


if __name__ == "__main__":
    main()
