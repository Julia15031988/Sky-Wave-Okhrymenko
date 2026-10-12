
import numpy as np

from src.config import TrajectoryConfig
from src.data_generation.data_split import split_trajectories
from src.data_generation.dataset_generator import TrajectoryDatasetGenerator


def prepare_comparison_data(
    noise_std: float = 0.1,
    input_steps: int = 30,
    output_steps: int = 20,
    seed: int = 42,
):
    """
    Prepare identical test windows for Kalman and LSTM.

    Gaussian noise is added once to each full trajectory,
    before creating overlapping observation windows.
    Future targets remain clean.
    """

    config = TrajectoryConfig()
    generator = TrajectoryDatasetGenerator(config)

    trajectories = generator.generate_circular_dataset(100)

    _, _, test_trajectories = split_trajectories(
        trajectories,
        random_seed=42,
    )

    rng = np.random.default_rng(seed)

    noisy_trajectories = [
        trajectory
        + rng.normal(
            loc=0.0,
            scale=noise_std,
            size=trajectory.shape,
        )
        for trajectory in test_trajectories
    ]

    observed_windows = []
    target_windows = []

    for noisy, clean in zip(noisy_trajectories, test_trajectories):
        total_windows = len(clean) - input_steps - output_steps + 1

        for start in range(total_windows):
            observed_windows.append(
                noisy[start : start + input_steps]
            )

            target_windows.append(
                clean[
                    start + input_steps :
                    start + input_steps + output_steps
                ]
            )

    return (
        np.asarray(observed_windows, dtype=np.float32),
        np.asarray(target_windows, dtype=np.float32),
    )


if __name__ == "__main__":
    X, y = prepare_comparison_data()

    print("Comparison dataset ready!")
    print(f"Input shape:  {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of samples: {len(X)}")
