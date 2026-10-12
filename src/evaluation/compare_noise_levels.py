
from src.evaluation.compare_models import evaluate_lstm
from src.evaluation.comparison_dataset import prepare_comparison_data
from src.evaluation.evaluate_ctrv_comparison import evaluate_ctrv
from src.evaluation.evaluate_kalman_comparison import evaluate_kalman


def main():
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]

    print("=== SKYWAVE: FOUR-MODEL NOISE ROBUSTNESS ===")
    print("History: 30 steps | Forecast: 20 steps")
    print()

    print(
        f"{'Noise':<8} {'CV':>10} {'CTRV':>10} "
        f"{'Clean LSTM':>13} {'Robust LSTM':>13}"
    )
    print("-" * 60)

    for noise_std in noise_levels:
        observations, targets = prepare_comparison_data(
            noise_std=noise_std,
            seed=42,
        )

        # Kalman CV
        _, cv_rmse = evaluate_kalman(
            observations,
            targets,
            time_step=0.1,
            noise_std=noise_std,
        )

        # Kalman CTRV
        _, ctrv_rmse = evaluate_ctrv(
            observations,
            targets,
            time_step=0.1,
            noise_std=noise_std,
        )

        # Clean LSTM
        _, clean_rmse = evaluate_lstm(
            "models/lstm_circular_best.pt",
            observations,
            targets,
        )

        # Noise-Robust LSTM
        _, robust_rmse = evaluate_lstm(
            "models/lstm_circular_noisy_best.pt",
            observations,
            targets,
        )

        print(
            f"{noise_std:<8.1f} "
            f"{cv_rmse:>10.4f} "
            f"{ctrv_rmse:>10.4f} "
            f"{clean_rmse:>13.4f} "
            f"{robust_rmse:>13.4f}"
        )


if __name__ == "__main__":
    main()
