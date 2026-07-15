"""Probing tasks for evaluating representation quality.

Linear and non-linear probes for testing what information
is captured in pretrained representations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class LinearProbe:
    """Linear probe for evaluating frozen representations.

    Trains a simple linear classifier on top of frozen
    representations to assess downstream task utility.

    Attributes:
        input_dim: Representation dimension.
        num_classes: Number of output classes.
        weight: Linear weight matrix.
        bias: Bias vector.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
    ):
        """Initialize linear probe.

        Args:
            input_dim: Input dimension.
            num_classes: Number of classes.
            lr: Learning rate.
            weight_decay: L2 regularization.
            max_epochs: Maximum training epochs.
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        # Initialize weights
        scale = np.sqrt(2.0 / (input_dim + num_classes))
        self.weight = np.random.randn(input_dim, num_classes) * scale
        self.bias = np.zeros(num_classes)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """Train linear probe.

        Args:
            X: Training features [N, D].
            y: Training labels [N].
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Training history.
        """
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        n_samples = X.shape[0]
        batch_size = min(256, n_samples)

        for epoch in range(self.max_epochs):
            # Shuffle
            perm = np.random.permutation(n_samples)
            epoch_loss = 0.0
            correct = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = perm[i:i + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                # Forward
                logits = np.dot(X_batch, self.weight) + self.bias
                probs = self._softmax(logits)

                # Loss
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(len(y_batch)), y_batch.astype(int)] = 1
                loss = -np.mean(np.sum(one_hot * np.log(probs + 1e-10), axis=-1))

                # Gradients
                grad_logits = (probs - one_hot) / len(y_batch)
                grad_weight = np.dot(X_batch.T, grad_logits) + self.weight_decay * self.weight
                grad_bias = np.mean(grad_logits, axis=0)

                # Update
                self.weight -= self.lr * grad_weight
                self.bias -= self.lr * grad_bias

                epoch_loss += loss * len(y_batch)
                correct += np.sum(np.argmax(logits, axis=-1) == y_batch)

            train_acc = correct / n_samples
            history["train_loss"].append(epoch_loss / n_samples)
            history["train_acc"].append(train_acc)

            if X_val is not None and y_val is not None:
                val_acc = self.evaluate(X_val, y_val)["accuracy"]
                history["val_acc"].append(val_acc)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes.

        Args:
            X: Features [N, D].

        Returns:
            Predicted labels [N].
        """
        logits = np.dot(X, self.weight) + self.bias
        return np.argmax(logits, axis=-1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate probe.

        Args:
            X: Features.
            y: Labels.

        Returns:
            Evaluation metrics.
        """
        predictions = self.predict(X)
        accuracy = float(np.mean(predictions == y))
        return {"accuracy": accuracy}

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)


class KNNProbe:
    """K-Nearest Neighbors probe.

    Non-parametric evaluation of representation quality.
    No training required — directly evaluates neighborhood structure.

    Attributes:
        k: Number of neighbors.
        metric: Distance metric.
    """

    def __init__(
        self,
        k: int = 20,
        metric: str = "cosine",
        temperature: float = 0.07,
    ):
        """Initialize KNN probe.

        Args:
            k: Number of neighbors.
            metric: Distance metric ('cosine', 'euclidean').
            temperature: Temperature for weighted voting.
        """
        self.k = k
        self.metric = metric
        self.temperature = temperature
        self._train_features: Optional[np.ndarray] = None
        self._train_labels: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store training data.

        Args:
            X: Training features [N, D].
            y: Training labels [N].
        """
        if self.metric == "cosine":
            self._train_features = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)
        else:
            self._train_features = X
        self._train_labels = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using KNN voting.

        Args:
            X: Query features [M, D].

        Returns:
            Predicted labels [M].
        """
        if self._train_features is None:
            raise ValueError("Must call fit() first")

        if self.metric == "cosine":
            X_norm = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)
            similarities = np.dot(X_norm, self._train_features.T)
            # Higher similarity = closer
            top_k_indices = np.argsort(similarities, axis=-1)[:, -self.k:]
        else:
            # Euclidean distance
            diffs = X[:, None] - self._train_features[None, :]
            distances = np.sum(diffs ** 2, axis=-1)
            top_k_indices = np.argsort(distances, axis=-1)[:, :self.k]

        # Weighted voting
        predictions = []
        for i in range(len(X)):
            neighbor_labels = self._train_labels[top_k_indices[i]]

            if self.metric == "cosine":
                weights = similarities[i, top_k_indices[i]]
                weights = np.exp(weights / self.temperature)
            else:
                dists = distances[i, top_k_indices[i]]
                weights = np.exp(-dists / (self.temperature + 1e-10))

            # Weighted vote
            classes = np.unique(neighbor_labels)
            class_weights = np.zeros(len(classes))
            for j, cls in enumerate(classes):
                mask = neighbor_labels == cls
                class_weights[j] = np.sum(weights[mask])

            predictions.append(classes[np.argmax(class_weights)])

        return np.array(predictions)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate KNN probe."""
        predictions = self.predict(X)
        return {"accuracy": float(np.mean(predictions == y))}


class MLPProbe:
    """MLP probe for non-linear evaluation.

    Trains a small MLP on frozen representations to assess
    what non-linear information is captured.

    Attributes:
        hidden_dims: Hidden layer dimensions.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 0.001,
        max_epochs: int = 50,
        dropout: float = 0.1,
    ):
        """Initialize MLP probe.

        Args:
            input_dim: Input dimension.
            num_classes: Number of classes.
            hidden_dims: Hidden layer dimensions.
            lr: Learning rate.
            max_epochs: Maximum epochs.
            dropout: Dropout rate.
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims or [256, 128]
        self.lr = lr
        self.max_epochs = max_epochs
        self.dropout = dropout

        # Build MLP
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        dims = [input_dim] + self.hidden_dims + [num_classes]
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            self.weights.append(np.random.randn(dims[i], dims[i+1]) * scale)
            self.biases.append(np.zeros(dims[i+1]))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """Train MLP probe."""
        history: Dict[str, List[float]] = {"train_acc": [], "val_acc": []}
        n_samples = X.shape[0]
        batch_size = min(256, n_samples)

        for epoch in range(self.max_epochs):
            perm = np.random.permutation(n_samples)
            correct = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = perm[i:i + batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                # Forward
                activations = [X_batch]
                h = X_batch
                for j in range(len(self.weights)):
                    h = np.dot(h, self.weights[j]) + self.biases[j]
                    if j < len(self.weights) - 1:
                        h = np.maximum(0, h)  # ReLU
                        # Dropout
                        if self.dropout > 0:
                            mask = np.random.binomial(1, 1 - self.dropout, h.shape)
                            h = h * mask / (1 - self.dropout)
                    activations.append(h)

                # Softmax + loss
                logits = activations[-1]
                probs = self._softmax(logits)
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(len(y_batch)), y_batch.astype(int)] = 1

                # Backprop through layers
                grad = (probs - one_hot) / len(y_batch)
                for j in range(len(self.weights) - 1, -1, -1):
                    grad_w = np.dot(activations[j].T, grad)
                    grad_b = np.mean(grad, axis=0)

                    if j > 0:
                        grad = np.dot(grad, self.weights[j].T)
                        # ReLU backward
                        grad = grad * (activations[j] > 0)

                    self.weights[j] -= self.lr * grad_w
                    self.biases[j] -= self.lr * grad_b

                correct += np.sum(np.argmax(logits, axis=-1) == y_batch)

            history["train_acc"].append(correct / n_samples)
            if X_val is not None:
                history["val_acc"].append(self.evaluate(X_val, y_val)["accuracy"])

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict."""
        h = X
        for j in range(len(self.weights)):
            h = np.dot(h, self.weights[j]) + self.biases[j]
            if j < len(self.weights) - 1:
                h = np.maximum(0, h)
        return np.argmax(h, axis=-1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate."""
        return {"accuracy": float(np.mean(self.predict(X) == y))}

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)


class ModalityClassificationProbe:
    """Probe for modality classification.

    Tests whether representations encode modality information
    (they should, to some degree, but not exclusively).

    Attributes:
        modality_names: List of modality names.
    """

    def __init__(
        self,
        input_dim: int,
        modality_names: Optional[List[str]] = None,
    ):
        """Initialize modality probe.

        Args:
            input_dim: Representation dimension.
            modality_names: List of modalities.
        """
        self.modality_names = modality_names or [
            "seismic", "vibration", "eeg", "ecg", "audio", "rf", "synthetic"
        ]
        self.num_modalities = len(self.modality_names)
        self.probe = LinearProbe(input_dim, self.num_modalities)

    def fit(
        self, embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Train modality classification probe.

        Args:
            embeddings: Per-modality embeddings.

        Returns:
            Training results.
        """
        X_list = []
        y_list = []
        for i, (name, emb) in enumerate(embeddings.items()):
            X_list.append(emb)
            y_list.append(np.full(len(emb), i))

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        # Split train/test
        n = len(X)
        perm = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx = perm[:split]
        test_idx = perm[split:]

        self.probe.fit(X[train_idx], y[train_idx])
        results = self.probe.evaluate(X[test_idx], y[test_idx])

        return results

    def evaluate(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate modality classification."""
        return self.fit(embeddings)


class FrequencyResolutionProbe:
    """Probe for frequency resolution of representations.

    Tests whether representations can distinguish signals
    at different frequencies — measures spectral resolution.

    Attributes:
        input_dim: Representation dimension.
        num_frequency_bins: Number of frequency bins to distinguish.
    """

    def __init__(
        self,
        input_dim: int,
        num_frequency_bins: int = 32,
        sample_rate: float = 1.0,
    ):
        """Initialize frequency resolution probe.

        Args:
            input_dim: Representation dimension.
            num_frequency_bins: Number of bins.
            sample_rate: Sampling rate.
        """
        self.input_dim = input_dim
        self.num_frequency_bins = num_frequency_bins
        self.sample_rate = sample_rate
        self.probe = LinearProbe(input_dim, num_frequency_bins)

    def generate_test_signals(
        self,
        num_per_bin: int = 100,
        signal_length: int = 1024,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate test signals at specific frequencies.

        Args:
            num_per_bin: Samples per frequency bin.
            signal_length: Signal length.

        Returns:
            Tuple of (signals [N, T], frequency_bin_labels [N]).
        """
        nyquist = self.sample_rate / 2
        freq_edges = np.linspace(0, nyquist, self.num_frequency_bins + 1)

        signals = []
        labels = []
        t = np.arange(signal_length) / self.sample_rate

        for bin_idx in range(self.num_frequency_bins):
            f_low = freq_edges[bin_idx]
            f_high = freq_edges[bin_idx + 1]

            for _ in range(num_per_bin):
                freq = np.random.uniform(f_low, f_high)
                phase = np.random.uniform(0, 2 * np.pi)
                amplitude = np.random.uniform(0.5, 1.5)

                signal = amplitude * np.sin(2 * np.pi * freq * t + phase)
                signal += np.random.randn(signal_length) * 0.1

                signals.append(signal)
                labels.append(bin_idx)

        return np.array(signals), np.array(labels)

    def evaluate(
        self,
        encode_fn,
        num_per_bin: int = 100,
        signal_length: int = 1024,
    ) -> Dict[str, float]:
        """Evaluate frequency resolution.

        Args:
            encode_fn: Function that encodes signals to representations.
            num_per_bin: Samples per frequency bin.
            signal_length: Signal length.

        Returns:
            Resolution metrics.
        """
        signals, labels = self.generate_test_signals(num_per_bin, signal_length)

        # Encode
        embeddings = encode_fn(signals)

        # Split
        n = len(embeddings)
        perm = np.random.permutation(n)
        split = int(0.8 * n)

        X_train = embeddings[perm[:split]]
        y_train = labels[perm[:split]]
        X_test = embeddings[perm[split:]]
        y_test = labels[perm[split:]]

        # Train probe
        self.probe.fit(X_train, y_train)
        results = self.probe.evaluate(X_test, y_test)

        # Additional metrics
        results["num_frequency_bins"] = self.num_frequency_bins
        results["frequency_resolution_hz"] = self.sample_rate / (2 * self.num_frequency_bins)

        return results
