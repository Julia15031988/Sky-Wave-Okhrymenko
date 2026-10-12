
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import TrajectoryConfig
from src.data_generation.data_split import split_trajectories
from src.data_generation.dataset_generator import TrajectoryDatasetGenerator
from src.data_generation.noisy_sequences import create_noisy_sequences
from src.data_generation.torch_dataset import TrajectoryDataset
from src.models.lstm_model import LSTMTrajectoryPredictor


def main() -> None:
    torch.manual_seed(42)

    config = TrajectoryConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate clean trajectories and split BEFORE windowing.
    generator = TrajectoryDatasetGenerator(config)
    trajectories = generator.generate_circular_dataset(
        num_trajectories=100
    )

    train_trajectories, val_trajectories, _ = split_trajectories(
        trajectories,
        random_seed=42,
    )

    # Noisy history -> clean future.
    X_train, y_train = create_noisy_sequences(
        train_trajectories,
        noise_std=0.1,
        random_seed=42,
    )

    X_val, y_val = create_noisy_sequences(
        val_trajectories,
        noise_std=0.1,
        random_seed=43,
    )

    train_loader = DataLoader(
        TrajectoryDataset(X_train, y_train),
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        TrajectoryDataset(X_val, y_val),
        batch_size=32,
        shuffle=False,
    )

    model = LSTMTrajectoryPredictor().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    checkpoint_path = Path("models/lstm_circular_noisy_best.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs = 20

    print("\n--- SkyWave: Noise-Robust LSTM Training ---")
    print(f"Device: {device}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print("Input noise standard deviation: 0.1")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)

                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    print("------------------------------------------")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
