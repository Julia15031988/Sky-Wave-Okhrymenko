
import numpy as np
import torch

from src.evaluation.comparison_dataset import prepare_comparison_data
from src.evaluation.evaluate_kalman_comparison import evaluate_kalman
from src.evaluation.evaluate_ctrv_comparison import evaluate_ctrv
from src.models.lstm_model import LSTMTrajectoryPredictor


def evaluate_lstm(
    model_path: str,
    observations: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, float]:
    model = LSTMTrajectoryPredictor(
        input_size=2,
        hidden_size=64,
        num_layers=2,
        output_steps=20,
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model.eval()

    predictions = []

    with torch.inference_mode():
        for start in range(0, len(observations), 256):
            batch = torch.from_numpy(
                observations[start:start + 256]
            ).float()

            output = model(batch)
            predictions.append(output.numpy())

    predictions = np.concatenate(predictions, axis=0)

    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    return float(mae), float(rmse)


def main():
    noise_std = 0.1

    observations, targets = prepare_comparison_data(
        noise_std=noise_std,
    )

    print("=== SKYWAVE: FOUR-MODEL COMPARISON ===")
    print(f"Test windows: {len(observations)}")
    print(f"History: {observations.shape[1]} steps")
    print(f"Forecast: {targets.shape[1]} steps")
    print(f"Noise sigma: {noise_std}")
    print()

    cv_mae, cv_rmse = evaluate_kalman(
        observations,
        targets,
        time_step=0.1,
        noise_std=noise_std,
    )

    ctrv_mae, ctrv_rmse = evaluate_ctrv(
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

    robust_mae, robust_rmse = evaluate_lstm(
        "models/lstm_circular_noisy_best.pt",
        observations,
        targets,
    )

    results = [
        ("Kalman CV", cv_mae, cv_rmse),
        ("Kalman CTRV", ctrv_mae, ctrv_rmse),
        ("Clean LSTM", clean_mae, clean_rmse),
        ("Noise-Robust LSTM", robust_mae, robust_rmse),
    ]

    print(f"{'Model':<22} {'MAE':>10} {'RMSE':>10}")
    print("-" * 44)

    for name, mae, rmse in results:
        print(f"{name:<22} {mae:>10.4f} {rmse:>10.4f}")


if __name__ == "__main__":
    main()
