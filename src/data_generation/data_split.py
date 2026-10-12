
import numpy as np


def split_trajectories(
    trajectories: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split complete trajectories into train, validation, and test sets."""

    if trajectories.ndim != 3 or trajectories.shape[2] != 2:
        raise ValueError("Expected shape (N, T, 2)")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    if train_ratio + val_ratio >= 1:
        raise ValueError("Train and validation ratios must sum to less than 1")

    num_trajectories = len(trajectories)

    if num_trajectories < 3:
        raise ValueError("At least 3 trajectories are required")

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(num_trajectories)

    train_end = int(num_trajectories * train_ratio)
    val_end = train_end + int(num_trajectories * val_ratio)

    if train_end == 0 or val_end == train_end or val_end == num_trajectories:
        raise ValueError("Each split must contain at least one trajectory")

    train_data = trajectories[indices[:train_end]]
    val_data = trajectories[indices[train_end:val_end]]
    test_data = trajectories[indices[val_end:]]

    return train_data, val_data, test_data
