
import numpy as np
import torch

from src.evaluation.comparison_dataset import prepare_comparison_data
from src.evaluation.evaluate_kalman_comparison import evaluate_kalman
from src.models.lstm_model import LSTMTrajectoryPredictor


def evaluate_lstm(
    model_path: str,
    observations: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, float]:
    """Evaluate a trained LSTM on the shared test dataset."""

    device = torch.device("cpu")

    model = LSTMTrajectoryPredictor(
        input_size=2,
        hidden_size=64,
        num_layers=2,
        output_steps=20,
    ).to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model.eval()

    predictions = []

    with torch.inference_mode():
        for start in range(0, len(observations), 256):
            batch = torch.from_numpy(
                observations[start:start + 256]
            ).float().to(device)

            output = model(batch)
            predictions.append(output.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    return float(mae), float(rmse)


def main():
    noise_std = 0.1

    observations, targets = prepare_comparison_data(
        noise_std=noise_std,
    )

    print("=== SKYWAVE: HEAD-TO-HEAD COMPARISON ===")
    print(f"Test windows: {len(observations)}")
    print(f"History: {observations.shape[1]} steps")
    print(f"Forecast: {targets.shape[1]} steps")
    print(f"Gaussian noise sigma: {noise_std}")
    print()

    kalman_mae, kalman_rmse = evaluate_kalman(
        observations,
        targets,
        time_step=0.1,
        noise_std=noise_std,
    )

    clean_mae, clean_rmse = evaluate_lstm(
        "models/lstm_circular_best.pt",
        observations,
        targets,
    )

    noisy_mae, noisy_rmse = evaluate_lstm(
        "models/lstm_circular_noisy_best.pt",
        observations,
        targets,
    )

    print(f"{'Model':<22} {'MAE':>10} {'RMSE':>10}")
    print("-" * 44)

    results = [
        ("Kalman CV", kalman_mae, kalman_rmse),
        ("Clean LSTM", clean_mae, clean_rmse),
        ("Noise-Robust LSTM", noisy_mae, noisy_rmse),
    ]

    for name, mae, rmse in results:
        print(f"{name:<22} {mae:>10.4f} {rmse:>10.4f}")


if __name__ == "__main__":
    main()
