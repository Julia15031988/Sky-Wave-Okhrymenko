
import torch
from torch import nn


class LSTMTrajectoryPredictor(nn.Module):
    """Predict future 2D positions from historical trajectories."""

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_steps: int = 20,
    ):
        super().__init__()

        if input_size < 1 or hidden_size < 1:
            raise ValueError("Input and hidden sizes must be positive")

        if num_layers < 1 or output_steps < 1:
            raise ValueError("Layers and output steps must be positive")

        self.output_steps = output_steps

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=output_steps * 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted future positions."""
        _, (hidden_state, _) = self.lstm(x)

        last_hidden = hidden_state[-1]

        predictions = self.fc(last_hidden)

        return predictions.reshape(
            x.size(0),
            self.output_steps,
            2,
        )
