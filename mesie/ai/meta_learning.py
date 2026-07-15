"""Meta-learning framework for few-shot spectral intelligence.

Provides MAML-inspired meta-learning, prototypical networks, and
rapid adaptation mechanisms for spectral data with limited labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class MetaStrategy(Enum):
    """Meta-learning strategies available."""

    MAML = "maml"
    PROTOTYPICAL = "prototypical"
    MATCHING = "matching"
    REPTILE = "reptile"
    METRIC = "metric"


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning."""

    inner_lr: float = 0.01
    outer_lr: float = 0.001
    n_inner_steps: int = 5
    n_way: int = 5
    k_shot: int = 1
    query_size: int = 15
    strategy: MetaStrategy = MetaStrategy.MAML
    embedding_dim: int = 64
    task_embedding_dim: int = 32
    adaptation_steps: int = 3


@dataclass
class MetaTask:
    """Represents a meta-learning task (episode)."""

    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray
    task_id: str = ""

    @property
    def n_way(self) -> int:
        return len(np.unique(self.support_y))

    @property
    def k_shot(self) -> int:
        return len(self.support_x) // self.n_way


@dataclass
class MetaResult:
    """Result from meta-learning evaluation."""

    accuracy: float
    loss: float
    per_class_accuracy: dict[int, float] = field(default_factory=dict)
    adaptation_steps_used: int = 0
    confidence_scores: Optional[np.ndarray] = None


class PrototypicalNetwork:
    """Prototypical network for few-shot spectral classification.

    Learns an embedding space where classification is performed by
    computing distances to class prototypes (mean embeddings).
    """

    def __init__(self, input_dim: int, embedding_dim: int = 64) -> None:
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self._encoder_weights: list[np.ndarray] = []
        self._is_trained = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize encoder weights."""
        dims = [self.input_dim, 128, self.embedding_dim]
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            self._encoder_weights.append(
                np.random.randn(dims[i], dims[i + 1]) * scale
            )

    def embed(self, x: np.ndarray) -> np.ndarray:
        """Embed input data into learned space."""
        h = x
        for w in self._encoder_weights:
            h = np.maximum(0, h @ w)
        return h

    def compute_prototypes(
        self, support_x: np.ndarray, support_y: np.ndarray
    ) -> dict[int, np.ndarray]:
        """Compute class prototypes from support set."""
        embeddings = self.embed(support_x)
        prototypes = {}
        for cls in np.unique(support_y):
            mask = support_y == cls
            prototypes[int(cls)] = embeddings[mask].mean(axis=0)
        return prototypes

    def predict(
        self, query_x: np.ndarray, prototypes: dict[int, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict classes for query set using prototypes."""
        embeddings = self.embed(query_x)
        classes = sorted(prototypes.keys())
        proto_matrix = np.array([prototypes[c] for c in classes])

        # Euclidean distances
        distances = np.zeros((len(embeddings), len(classes)))
        for i, emb in enumerate(embeddings):
            for j, proto in enumerate(proto_matrix):
                distances[i, j] = np.linalg.norm(emb - proto)

        # Convert to probabilities via softmax on negative distances
        neg_dist = -distances
        exp_d = np.exp(neg_dist - neg_dist.max(axis=1, keepdims=True))
        probs = exp_d / exp_d.sum(axis=1, keepdims=True)

        predictions = np.array([classes[i] for i in np.argmax(probs, axis=1)])
        confidence = probs.max(axis=1)
        return predictions, confidence

    def evaluate_task(self, task: MetaTask) -> MetaResult:
        """Evaluate on a single meta-learning task."""
        prototypes = self.compute_prototypes(task.support_x, task.support_y)
        predictions, confidence = self.predict(task.query_x, prototypes)
        accuracy = float(np.mean(predictions == task.query_y))

        per_class = {}
        for cls in np.unique(task.query_y):
            mask = task.query_y == cls
            per_class[int(cls)] = float(np.mean(predictions[mask] == cls))

        return MetaResult(
            accuracy=accuracy,
            loss=1.0 - accuracy,
            per_class_accuracy=per_class,
            confidence_scores=confidence,
        )


class MAMLAdapter:
    """Model-Agnostic Meta-Learning adapter for spectral models.

    Implements inner-loop adaptation via gradient-free approximation
    for rapid task-specific fine-tuning.
    """

    def __init__(self, config: MetaLearningConfig) -> None:
        self.config = config
        self._base_weights: Optional[np.ndarray] = None
        self._adapted_weights: Optional[np.ndarray] = None
        self._task_history: list[MetaResult] = []

    def initialize(self, input_dim: int, output_dim: int) -> None:
        """Initialize base model parameters."""
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self._base_weights = np.random.randn(input_dim, output_dim) * scale

    def inner_adapt(
        self, support_x: np.ndarray, support_y: np.ndarray
    ) -> np.ndarray:
        """Perform inner-loop adaptation on support set."""
        if self._base_weights is None:
            raise ValueError("Must call initialize() first")

        weights = self._base_weights.copy()
        for _ in range(self.config.n_inner_steps):
            # Gradient-free perturbation-based adaptation
            predictions = support_x @ weights
            error = support_y.reshape(-1, 1) - predictions if support_y.ndim == 1 else support_y - predictions
            gradient_approx = support_x.T @ error / len(support_x)
            weights += self.config.inner_lr * gradient_approx

        self._adapted_weights = weights
        return weights

    def predict_adapted(self, query_x: np.ndarray) -> np.ndarray:
        """Make predictions with adapted weights."""
        if self._adapted_weights is None:
            raise ValueError("Must call inner_adapt() first")
        return query_x @ self._adapted_weights

    def evaluate_task(self, task: MetaTask) -> MetaResult:
        """Evaluate on a meta-learning task with inner adaptation."""
        if self._base_weights is None:
            n_classes = len(np.unique(task.support_y))
            self.initialize(task.support_x.shape[1], n_classes)

        self.inner_adapt(task.support_x, task.support_y)
        raw_output = self.predict_adapted(task.query_x)

        # Convert to class predictions
        predictions = np.argmax(raw_output, axis=1) if raw_output.ndim > 1 else (raw_output > 0.5).astype(int).flatten()

        accuracy = float(np.mean(predictions == task.query_y))
        result = MetaResult(
            accuracy=accuracy,
            loss=1.0 - accuracy,
            adaptation_steps_used=self.config.n_inner_steps,
        )
        self._task_history.append(result)
        return result


class TaskDistribution:
    """Generates meta-learning tasks from spectral datasets."""

    def __init__(self, n_way: int = 5, k_shot: int = 1, query_size: int = 15) -> None:
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size

    def sample_task(
        self, data: np.ndarray, labels: np.ndarray, rng: Optional[np.random.RandomState] = None
    ) -> MetaTask:
        """Sample a random N-way K-shot task."""
        if rng is None:
            rng = np.random.RandomState()

        unique_classes = np.unique(labels)
        if len(unique_classes) < self.n_way:
            selected_classes = unique_classes
        else:
            selected_classes = rng.choice(unique_classes, self.n_way, replace=False)

        support_x_list, support_y_list = [], []
        query_x_list, query_y_list = [], []

        for cls in selected_classes:
            cls_indices = np.where(labels == cls)[0]
            selected = rng.choice(cls_indices, min(self.k_shot + self.query_size, len(cls_indices)), replace=False)

            support_idx = selected[: self.k_shot]
            query_idx = selected[self.k_shot: self.k_shot + self.query_size]

            support_x_list.append(data[support_idx])
            support_y_list.extend([cls] * len(support_idx))
            query_x_list.append(data[query_idx])
            query_y_list.extend([cls] * len(query_idx))

        return MetaTask(
            support_x=np.vstack(support_x_list),
            support_y=np.array(support_y_list),
            query_x=np.vstack(query_x_list) if query_x_list else np.empty((0, data.shape[1])),
            query_y=np.array(query_y_list),
        )

    def sample_tasks(
        self, data: np.ndarray, labels: np.ndarray, n_tasks: int, seed: int = 42
    ) -> list[MetaTask]:
        """Sample multiple tasks."""
        rng = np.random.RandomState(seed)
        return [self.sample_task(data, labels, rng) for _ in range(n_tasks)]
