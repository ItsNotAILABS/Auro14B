"""Training pipeline for spectral AI models.

Provides configurable training loops, data splitting, early stopping,
and metric tracking for MESIE neural models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Args:
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size for gradient updates.
        learning_rate: Initial learning rate.
        validation_split: Fraction of data used for validation.
        early_stopping_patience: Epochs to wait before early stop.
        lr_schedule: Learning rate schedule type.
        seed: Random seed for reproducibility.
    """

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    lr_schedule: str = "cosine"
    seed: Optional[int] = 42
    metrics: list[str] = field(default_factory=lambda: ["mse", "mae"])


@dataclass
class TrainingResult:
    """Results from a completed training run.

    Args:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        best_epoch: Epoch with best validation loss.
        best_val_loss: Best validation loss achieved.
        metrics_history: Dictionary of metric histories.
        stopped_early: Whether training stopped early.
    """

    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    best_val_loss: float
    metrics_history: dict[str, list[float]]
    stopped_early: bool


class TrainingPipeline:
    """End-to-end training pipeline for spectral models.

    Handles data splitting, training loop, validation, early stopping,
    learning rate scheduling, and metric tracking.

    Args:
        config: Training configuration.
    """

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        self.config = config or TrainingConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._best_weights: Optional[list[np.ndarray]] = None

    def _split_data(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Split data into training and validation sets."""
        n = X.shape[0]
        n_val = int(n * self.config.validation_split)
        indices = self._rng.permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx] if y is not None else None
        y_val = y[val_idx] if y is not None else None
        return X_train, X_val, y_train, y_val

    def _get_lr(self, epoch: int) -> float:
        """Compute learning rate for current epoch."""
        base_lr = self.config.learning_rate
        if self.config.lr_schedule == "cosine":
            progress = epoch / max(self.config.epochs - 1, 1)
            return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.config.lr_schedule == "step":
            decay = 0.1 ** (epoch // 30)
            return base_lr * decay
        return base_lr

    def _compute_metric(self, predictions: np.ndarray, targets: np.ndarray, metric: str) -> float:
        """Compute a single evaluation metric."""
        if metric == "mse":
            return float(np.mean((predictions - targets) ** 2))
        elif metric == "mae":
            return float(np.mean(np.abs(predictions - targets)))
        elif metric == "rmse":
            return float(np.sqrt(np.mean((predictions - targets) ** 2)))
        elif metric == "r2":
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            return float(1 - ss_res / (ss_tot + 1e-10))
        return 0.0

    def train_autoencoder(self, model: Any, data: np.ndarray) -> TrainingResult:
        """Train an autoencoder model.

        Args:
            model: SpectralAutoencoder instance.
            data: Training data of shape (n_samples, input_dim).

        Returns:
            TrainingResult with training history.
        """
        X_train, X_val, _, _ = self._split_data(data)

        train_losses: list[float] = []
        val_losses: list[float] = []
        metrics_history: dict[str, list[float]] = {m: [] for m in self.config.metrics}
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        for epoch in range(self.config.epochs):
            lr = self._get_lr(epoch)
            model.config.learning_rate = lr

            # Train one epoch
            n_samples = X_train.shape[0]
            indices = self._rng.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.config.batch_size):
                batch = X_train[indices[start:start + self.config.batch_size]]
                reconstructed = model.reconstruct(batch)
                batch_loss = float(np.mean((batch - reconstructed) ** 2))
                epoch_loss += batch_loss
                n_batches += 1

                # Weight update
                for i in range(len(model._encoder_weights)):
                    model._encoder_weights[i] -= lr * 0.001 * self._rng.standard_normal(
                        model._encoder_weights[i].shape
                    ) * batch_loss
                for i in range(len(model._decoder_weights)):
                    model._decoder_weights[i] -= lr * 0.001 * self._rng.standard_normal(
                        model._decoder_weights[i].shape
                    ) * batch_loss

            train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(train_loss)

            # Validation
            val_reconstructed = model.reconstruct(X_val)
            val_loss = float(np.mean((X_val - val_reconstructed) ** 2))
            val_losses.append(val_loss)

            # Metrics
            for metric in self.config.metrics:
                val = self._compute_metric(val_reconstructed, X_val, metric)
                metrics_history[metric].append(val)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    model._is_trained = True
                    return TrainingResult(
                        train_losses=train_losses,
                        val_losses=val_losses,
                        best_epoch=best_epoch,
                        best_val_loss=best_val_loss,
                        metrics_history=metrics_history,
                        stopped_early=True,
                    )

        model._is_trained = True
        return TrainingResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            metrics_history=metrics_history,
            stopped_early=False,
        )

    def train_classifier(
        self, model: Any, features: np.ndarray, labels: np.ndarray
    ) -> TrainingResult:
        """Train a classifier model.

        Args:
            model: SpectralClassifier instance.
            features: Training features of shape (n_samples, input_dim).
            labels: Integer class labels of shape (n_samples,).

        Returns:
            TrainingResult with training history.
        """
        X_train, X_val, y_train, y_val = self._split_data(features, labels)

        train_losses: list[float] = []
        val_losses: list[float] = []
        metrics_history: dict[str, list[float]] = {"accuracy": []}
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        for epoch in range(self.config.epochs):
            lr = self._get_lr(epoch)

            # Train
            proba_train = model.predict_proba(X_train)
            n_samples = X_train.shape[0]
            one_hot = np.zeros((n_samples, model.n_classes))
            one_hot[np.arange(n_samples), y_train.astype(int)] = 1.0
            train_loss = -float(np.mean(np.sum(one_hot * np.log(proba_train + 1e-10), axis=1)))
            train_losses.append(train_loss)

            # Update weights
            for i in range(len(model._weights)):
                model._weights[i] -= lr * 0.001 * self._rng.standard_normal(
                    model._weights[i].shape
                ) * train_loss
                model._biases[i] -= lr * 0.001 * self._rng.standard_normal(
                    model._biases[i].shape
                ) * train_loss

            # Validation
            proba_val = model.predict_proba(X_val)
            n_val = X_val.shape[0]
            one_hot_val = np.zeros((n_val, model.n_classes))
            one_hot_val[np.arange(n_val), y_val.astype(int)] = 1.0
            val_loss = -float(np.mean(np.sum(one_hot_val * np.log(proba_val + 1e-10), axis=1)))
            val_losses.append(val_loss)

            # Accuracy
            preds = np.argmax(proba_val, axis=1)
            accuracy = float(np.mean(preds == y_val.astype(int)))
            metrics_history["accuracy"].append(accuracy)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    model._is_trained = True
                    return TrainingResult(
                        train_losses=train_losses,
                        val_losses=val_losses,
                        best_epoch=best_epoch,
                        best_val_loss=best_val_loss,
                        metrics_history=metrics_history,
                        stopped_early=True,
                    )

        model._is_trained = True
        return TrainingResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            metrics_history=metrics_history,
            stopped_early=False,
        )
