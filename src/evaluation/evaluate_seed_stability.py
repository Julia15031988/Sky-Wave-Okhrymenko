
import numpy as np

from src.evaluation.compare_models import evaluate_lstm
from src.evaluation.comparison_dataset import prepare_comparison_data
from src.evaluation.evaluate_ctrv_comparison import evaluate_ctrv
from src.evaluation.evaluate_kalman_comparison import evaluate_kalman


def main():
    seeds = [42, 123, 2026]
    noise_levels = [0.1, 0.5]

    models = [
        "Kalman CV",
        "Kalman CTRV",
        "Clean LSTM",
        "Noise-Robust LSTM",
    ]

    print("=== SKYWAVE: SEED STABILITY EXPERIMENT ===")

    for noise_std in noise_levels:
        results = {name: [] for name in models}

        print(f"\nNoise sigma: {noise_std}")

        for seed in seeds:
            observations, targets = prepare_comparison_data(
                noise_std=noise_std,
                seed=seed,
            )

            _, cv_rmse = evaluate_kalman(
                observations,
                targets,
                time_step=0.1,
                noise_std=noise_std,
            )

            _, ctrv_rmse = evaluate_ctrv(
                observations,
                targets,
                time_step=0.1,
                noise_std=noise_std,
            )

            _, clean_rmse = evaluate_lstm(
                "models/lstm_circular_best.pt",
                observations,
                targets,
            )

            _, robust_rmse = evaluate_lstm(
                "models/lstm_circular_noisy_best.pt",
                observations,
                targets,
            )

            scores = [
                cv_rmse,
                ctrv_rmse,
                clean_rmse,
                robust_rmse,
            ]

            for name, score in zip(models, scores):
                results[name].append(score)

            print(f"  Seed {seed}: completed", flush=True)

        print()
        print(
            f"{'Model':<22} {'Mean RMSE':>12} {'Std RMSE':>12}"
        )
        print("-" * 48)

        for name in models:
            values = np.asarray(results[name])

            print(
                f"{name:<22} "
                f"{values.mean():>12.4f} "
                f"{values.std(ddof=1):>12.4f}"
            )


if __name__ == "__main__":
    main()
