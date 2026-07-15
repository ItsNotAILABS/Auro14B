"""Neural spectral encoders for deep embedding generation.

Provides CNN-based and Autoencoder-based encoders that produce
fixed-length embeddings from spectral records using PyTorch.
Falls back to a lightweight NumPy implementation when torch is unavailable.

These encoders complement the statistical SpectralVectorizer with learned
representations capable of capturing complex spectral patterns.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from mesie.core.records import MultiElementRecord
from mesie.io.loaders import RecordInput, load_record

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


# =============================================================================
# Shared utilities
# =============================================================================


def _record_to_signal(record: MultiElementRecord, target_length: int = 128) -> np.ndarray:
    """Extract and resample the first component's amplitude to a fixed length.

    Args:
        record: Spectral record.
        target_length: Desired signal length.

    Returns:
        1D numpy array of shape (target_length,).
    """
    if not record.components:
        return np.zeros(target_length, dtype=np.float32)
    amp = np.abs(record.components[0].amplitude).astype(np.float32)
    if len(amp) == target_length:
        return amp
    # Linear interpolation to target length
    x_old = np.linspace(0, 1, len(amp))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, amp).astype(np.float32)


# =============================================================================
# NumPy fallback encoder (always available)
# =============================================================================


class _NumpyAutoencoderFallback:
    """Minimal autoencoder using NumPy matrix multiplications.

    Provides encode/decode/train without requiring PyTorch, suitable for
    lightweight deployments and deterministic reproducibility.
    """

    def __init__(self, input_dim: int = 128, latent_dim: int = 32, seed: int = 42) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self._rng = np.random.default_rng(seed)
        # Xavier initialization
        scale_enc = np.sqrt(2.0 / (input_dim + latent_dim))
        scale_dec = np.sqrt(2.0 / (latent_dim + input_dim))
        self._W_enc = (self._rng.standard_normal((input_dim, latent_dim)) * scale_enc).astype(np.float32)
        self._b_enc = np.zeros(latent_dim, dtype=np.float32)
        self._W_dec = (self._rng.standard_normal((latent_dim, input_dim)) * scale_dec).astype(np.float32)
        self._b_dec = np.zeros(input_dim, dtype=np.float32)
        self._trained = False

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent space."""
        z = x @ self._W_enc + self._b_enc
        return np.maximum(z, 0)  # ReLU

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode from latent space."""
        return z @ self._W_dec + self._b_dec

    def train_step(self, batch: np.ndarray, lr: float = 1e-3) -> float:
        """Single gradient descent step. Returns MSE loss."""
        z = self.encode(batch)
        recon = self.decode(z)
        error = recon - batch
        loss = float(np.mean(error ** 2))

        # Backprop through linear layers (simplified)
        grad_dec = z.T @ error / len(batch)
        grad_b_dec = error.mean(axis=0)

        # Gradient through ReLU
        raw_z = batch @ self._W_enc + self._b_enc
        relu_mask = (raw_z > 0).astype(np.float32)
        dz = (error @ self._W_dec.T) * relu_mask
        grad_enc = batch.T @ dz / len(batch)
        grad_b_enc = dz.mean(axis=0)

        self._W_enc -= lr * grad_enc
        self._b_enc -= lr * grad_b_enc
        self._W_dec -= lr * grad_dec
        self._b_dec -= lr * grad_b_dec

        return loss

    def fit(self, data: np.ndarray, epochs: int = 50, lr: float = 1e-3, batch_size: int = 32) -> List[float]:
        """Train on data array. Returns loss history."""
        losses: List[float] = []
        n = len(data)
        for _ in range(epochs):
            indices = self._rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                batch = data[indices[start:start + batch_size]]
                epoch_loss += self.train_step(batch, lr)
                n_batches += 1
            losses.append(epoch_loss / max(n_batches, 1))
        self._trained = True
        return losses


# =============================================================================
# PyTorch CNN Encoder
# =============================================================================


if HAS_TORCH:

    class _SpectralCNN(nn.Module):
        """1D CNN that maps a spectral signal to a fixed-length embedding."""

        def __init__(self, input_length: int = 128, embedding_dim: int = 64) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Sequential(
                nn.Linear(128, embedding_dim),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass: (batch, 1, length) -> (batch, embedding_dim)."""
            h = self.conv(x)
            h = h.squeeze(-1)
            return self.fc(h)

    class _SpectralAutoencoderTorch(nn.Module):
        """1D autoencoder that learns compressed spectral embeddings."""

        def __init__(self, input_dim: int = 128, latent_dim: int = 32) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Returns (reconstruction, latent)."""
            z = self.encoder(x)
            recon = self.decoder(z)
            return recon, z

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """Encode to latent space."""
            return self.encoder(x)


# =============================================================================
# Public API: NeuralSpectralEncoder
# =============================================================================


class NeuralSpectralEncoder:
    """Neural network encoder for spectral fingerprinting.

    Supports two modes:
        - ``"cnn"``: 1D CNN on resampled amplitude signal.
        - ``"autoencoder"``: Autoencoder latent space as embedding.

    Falls back to a NumPy autoencoder when PyTorch is unavailable.

    Args:
        mode: Encoder mode — ``"cnn"`` or ``"autoencoder"``.
        input_length: Fixed signal length for resampling.
        embedding_dim: Output embedding dimensionality.
        seed: Random seed for reproducibility.
    """

    MODES = ("cnn", "autoencoder")

    def __init__(
        self,
        mode: str = "autoencoder",
        input_length: int = 128,
        embedding_dim: int = 64,
        seed: int = 42,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'")

        self.mode = mode
        self.input_length = input_length
        self.embedding_dim = embedding_dim
        self.seed = seed
        self._use_torch = HAS_TORCH
        self._model = None
        self._is_fitted = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the underlying model."""
        if self._use_torch:
            torch.manual_seed(self.seed)
            if self.mode == "cnn":
                self._model = _SpectralCNN(
                    input_length=self.input_length,
                    embedding_dim=self.embedding_dim,
                )
            else:
                self._model = _SpectralAutoencoderTorch(
                    input_dim=self.input_length,
                    latent_dim=self.embedding_dim,
                )
            self._model.eval()
        else:
            # NumPy fallback (autoencoder only; CNN falls back to autoencoder)
            self._model = _NumpyAutoencoderFallback(
                input_dim=self.input_length,
                latent_dim=self.embedding_dim,
                seed=self.seed,
            )

    def _prepare_signal(self, record: RecordInput) -> np.ndarray:
        """Load and resample a record to fixed-length signal."""
        rec = load_record(record)
        return _record_to_signal(rec, self.input_length)

    def encode(self, record: RecordInput) -> np.ndarray:
        """Encode a single spectral record to an embedding vector.

        Args:
            record: Spectral record input.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        signal = self._prepare_signal(record)

        if self._use_torch:
            with torch.no_grad():
                tensor = torch.from_numpy(signal).unsqueeze(0)  # (1, input_length)
                if self.mode == "cnn":
                    tensor = tensor.unsqueeze(1)  # (1, 1, input_length)
                    emb = self._model(tensor)
                else:
                    emb = self._model.encode(tensor)
                return emb.squeeze(0).numpy()
        else:
            return self._model.encode(signal.reshape(1, -1)).squeeze(0)

    def encode_batch(self, records: Sequence[RecordInput]) -> np.ndarray:
        """Encode a batch of records.

        Args:
            records: Sequence of spectral records.

        Returns:
            2D numpy array of shape (n_records, embedding_dim).
        """
        signals = np.array([self._prepare_signal(r) for r in records], dtype=np.float32)

        if self._use_torch:
            with torch.no_grad():
                tensor = torch.from_numpy(signals)
                if self.mode == "cnn":
                    tensor = tensor.unsqueeze(1)  # (batch, 1, input_length)
                    emb = self._model(tensor)
                else:
                    emb = self._model.encode(tensor)
                return emb.numpy()
        else:
            return self._model.encode(signals)

    def fit(
        self,
        records: Sequence[RecordInput],
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> List[float]:
        """Train the encoder on a collection of spectral records.

        With the PyTorch backend, this is only supported for autoencoder mode.
        With the NumPy fallback backend (including ``mode="cnn"`` fallback),
        this trains the fallback autoencoder implementation.

        Args:
            records: Training records.
            epochs: Number of training epochs.
            lr: Learning rate.
            batch_size: Batch size.

        Returns:
            List of per-epoch average losses.
        """
        signals = np.array([self._prepare_signal(r) for r in records], dtype=np.float32)

        if self._use_torch:
            if self.mode != "autoencoder":
                raise NotImplementedError("fit() is only supported for mode='autoencoder' with torch backend")

            model = self._model

            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            dataset = torch.from_numpy(signals)
            losses: List[float] = []

            for _ in range(epochs):
                perm = torch.randperm(len(dataset))
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, len(dataset), batch_size):
                    batch = dataset[perm[start:start + batch_size]]
                    recon, _ = model(batch)
                    loss = torch.nn.functional.mse_loss(recon, batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1
                losses.append(epoch_loss / max(n_batches, 1))

            model.eval()
            self._is_fitted = True
            return losses
        else:
            loss_history = self._model.fit(signals, epochs=epochs, lr=lr, batch_size=batch_size)
            self._is_fitted = True
            return loss_history

    @property
    def is_fitted(self) -> bool:
        """Whether the encoder has been trained."""
        return self._is_fitted

    @property
    def backend(self) -> str:
        """Active computation backend ('torch' or 'numpy')."""
        return "torch" if self._use_torch else "numpy"
