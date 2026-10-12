import numpy as np
import pytest

from src.evaluation.metrics import calculate_mae, calculate_rmse


def test_mae_perfect_prediction():
    true = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 2.0, 3.0])

    assert calculate_mae(true, predicted) == 0.0


def test_mae_with_error():
    true = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 2.0, 4.0])

    assert calculate_mae(true, predicted) == pytest.approx(1 / 3)


def test_rmse_with_error():
    true = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 2.0, 4.0])

    assert calculate_rmse(true, predicted) == pytest.approx(
        np.sqrt(1 / 3)
    )
