
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """PyTorch dataset for trajectory forecasting."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        if X.ndim != 3 or y.ndim != 3:
            raise ValueError("X and y must be 3D arrays")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if X.shape[2] != 2 or y.shape[2] != 2:
            raise ValueError("Expected 2D coordinates")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]
