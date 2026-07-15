"""Smoke test: train SpectralAutoencoder on bundled embedding benchmark."""

import numpy as np
import pytest

from data import load_benchmark
from mesie.ai.models import ModelConfig, SpectralAutoencoder
from mesie.ai.training import TrainingConfig, TrainingPipeline


def _embedding_matrix_from_benchmark() -> np.ndarray:
    data = load_benchmark("embedding_training_data")
    rows = []
    for sample in data["samples"]:
        amp = np.asarray(sample["amplitudes"], dtype=float)
        # Fixed-size summary features for autoencoder input
        bands = 16
        idx = np.linspace(0, len(amp) - 1, bands).astype(int)
        rows.append(amp[idx])
    return np.vstack(rows)


def test_training_pipeline_on_embedding_benchmark():
    X = _embedding_matrix_from_benchmark()
    assert X.shape[0] == 200

    model = SpectralAutoencoder(
        ModelConfig(input_dim=X.shape[1], latent_dim=8, hidden_layers=[])
    )
    pipeline = TrainingPipeline(
        TrainingConfig(epochs=5, batch_size=16, early_stopping_patience=3, seed=42)
    )
    result = pipeline.train_autoencoder(model, X)

    assert len(result.train_losses) >= 1
    assert result.best_val_loss < float("inf")
    reconstructed = model.reconstruct(X[:8])
    assert reconstructed.shape == (8, X.shape[1])