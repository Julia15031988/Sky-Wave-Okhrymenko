import numpy as np


def calculate_mae(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    """Calculate mean absolute error."""
    return float(np.mean(np.abs(true_values - predicted_values)))


def calculate_rmse(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    """Calculate root mean squared error."""
    return float(
        np.sqrt(np.mean((true_values - predicted_values) ** 2))
    )
