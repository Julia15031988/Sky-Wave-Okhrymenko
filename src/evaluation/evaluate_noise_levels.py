
import numpy as np
import torch

from src.config import TrajectoryConfig
from src.data_generation.data_split import split_trajectories
from src.data_generation.dataset_generator import (
    TrajectoryDatasetGenerator,
)
from src.data_generation.noisy_sequences import create_noisy_sequences
from src.evaluation.metrics import calculate_mae, calculate_rmse
from src.models.lstm_model import LSTMTrajectoryPredictor


NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]

CHECKPOINTS = {
    "Clean LSTM": "models/lstm_circular_best.pt",
    "Noise-Robust LSTM": "models/lstm_circular_noisy_best.pt",
}


def load_model(checkpoint_path: str) -> LSTMTrajectoryPredictor:
    model = LSTMTrajectoryPredictor()

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()
    return model


def predict(
    model: LSTMTrajectoryPredictor,
    X: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(
                X[start:start + batch_size]
            ).float()

            output = model(batch)
            predictions.append(output.numpy())

    return np.concatenate(predictions, axis=0)


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

    models = {
        name: load_model(path)
        for name, path in CHECKPOINTS.items()
    }

    print("\n--- SkyWave: Noise Robustness Evaluation ---")
    print(f"Test trajectories: {len(test_trajectories)}")
    print("Forecast horizon: 20 steps")
    print("Same test trajectories and noise seed for both models\n")

    for noise_std in NOISE_LEVELS:
        X_test, y_test = create_noisy_sequences(
            test_trajectories,
            input_steps=30,
            output_steps=20,
            noise_std=noise_std,
            random_seed=44,
        )

        print(f"Noise std = {noise_std:.1f}")

        for name, model in models.items():
            predicted = predict(model, X_test)

            mae = calculate_mae(y_test, predicted)
            rmse = calculate_rmse(y_test, predicted)

            print(
                f"  {name:<18} "
                f"MAE: {mae:.4f} | RMSE: {rmse:.4f}"
            )

        print()

    print("-------------------------------------------")


if __name__ == "__main__":
    main()
