
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import TrajectoryConfig
from src.data_generation.data_split import split_trajectories
from src.data_generation.dataset_generator import (
    TrajectoryDatasetGenerator,
    create_sequences,
)
from src.data_generation.torch_dataset import TrajectoryDataset
from src.models.lstm_model import LSTMTrajectoryPredictor


def train_lstm(
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> None:
    """Train and validate an LSTM for trajectory forecasting."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Training parameters must be positive")

    torch.manual_seed(42)

    config = TrajectoryConfig()

    generator = TrajectoryDatasetGenerator(config)
    trajectories = generator.generate_circular_dataset(
        num_trajectories=100,
    )

    train_data, val_data, _ = split_trajectories(
        trajectories,
        random_seed=config.random_seed,
    )

    X_train, y_train = create_sequences(train_data)
    X_val, y_val = create_sequences(val_data)

    train_loader = DataLoader(
        TrajectoryDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TrajectoryDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = LSTMTrajectoryPredictor().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    best_val_loss = float("inf")
    model_path = Path("models/lstm_circular_best.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print("-" * 55)

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

            train_loss_sum += (
                loss.item() * batch_X.size(0)
            )

        train_loss = train_loss_sum / len(X_train)

        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)

                val_loss_sum += (
                    loss.item() * batch_X.size(0)
                )

        val_loss = val_loss_sum / len(X_val)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

    print("-" * 55)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {model_path}")


if __name__ == "__main__":
    train_lstm()
