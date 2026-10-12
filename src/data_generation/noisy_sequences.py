
import numpy as np

from src.data_generation.dataset_generator import create_sequences


def create_noisy_sequences(
    trajectories: np.ndarray,
    input_steps: int = 30,
    output_steps: int = 20,
    noise_std: float = 0.1,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy input histories and clean future targets."""

    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    if not np.isfinite(noise_std):
        raise ValueError("noise_std must be finite")

    # Build input and target windows from clean trajectories.
    X_clean, y_clean = create_sequences(
        trajectories,
        input_steps=input_steps,
        output_steps=output_steps,
    )

    # Generate reproducible Gaussian measurement noise.
    rng = np.random.default_rng(random_seed)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=X_clean.shape,
    )

    X_noisy = X_clean + noise

    return X_noisy.astype(np.float32), y_clean.astype(np.float32)
