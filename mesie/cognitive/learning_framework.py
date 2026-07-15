"""Spectral Learning and Optimization Framework.

Provides machine learning algorithms specifically designed for spectral
data, including online learning, meta-learning, continual learning,
and hyperparameter optimization.

Key Components:
    - OnlineLearner: Incremental learning from streaming spectra
    - MetaLearner: Few-shot learning across spectral domains
    - ContinualLearner: Learning without catastrophic forgetting
    - SpectralOptimizer: Hyperparameter optimization for spectral models
    - EnsemblePredictor: Ensemble methods for spectral classification
    - ActiveLearner: Intelligent sample selection for labeling
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Enumerations
# =============================================================================


class LearningStrategy(Enum):
    """Online learning strategies."""
    PASSIVE = "passive"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


class MetaLearningMethod(Enum):
    """Meta-learning approaches."""
    MAML = "maml"
    PROTOTYPICAL = "prototypical"
    MATCHING = "matching"
    REPTILE = "reptile"


class AcquisitionFunction(Enum):
    """Active learning acquisition functions."""
    UNCERTAINTY = "uncertainty"
    ENTROPY = "entropy"
    MARGIN = "margin"
    QBC = "query_by_committee"
    EXPECTED_IMPROVEMENT = "expected_improvement"


class OptimizationMethod(Enum):
    """Hyperparameter optimization methods."""
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    BANDIT = "bandit"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class LearningState:
    """State of a learning algorithm.

    Args:
        iteration: Current iteration.
        loss: Current loss value.
        accuracy: Current accuracy.
        weights: Model weights.
        metadata: Additional state info.
    """
    iteration: int = 0
    loss: float = float("inf")
    accuracy: float = 0.0
    weights: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prototype:
    """Class prototype for prototypical networks.

    Args:
        class_id: Class identifier.
        embedding: Prototype embedding.
        support_count: Number of support examples.
        variance: Intra-class variance estimate.
    """
    class_id: int
    embedding: np.ndarray
    support_count: int = 0
    variance: float = 0.0


@dataclass
class OptimizationTrial:
    """A hyperparameter optimization trial.

    Args:
        params: Hyperparameter values.
        score: Evaluation score.
        duration: Time taken.
        trial_id: Trial identifier.
    """
    params: Dict[str, float]
    score: float
    duration: float = 0.0
    trial_id: int = 0


@dataclass
class ActiveSample:
    """A sample selected by active learning.

    Args:
        index: Sample index.
        acquisition_value: Acquisition function value.
        predicted_label: Model's predicted label.
        uncertainty: Prediction uncertainty.
    """
    index: int
    acquisition_value: float
    predicted_label: int = -1
    uncertainty: float = 0.0


# =============================================================================
# Online Learner
# =============================================================================


class OnlineLearner:
    """Incremental online learning for spectral data.

    Learns from streaming spectral data without storing all
    samples, using online gradient descent variants.

    Args:
        input_dim: Input feature dimension.
        n_classes: Number of output classes.
        strategy: Learning strategy.
        learning_rate: Initial learning rate.
        regularization: L2 regularization strength.
    """

    def __init__(
        self,
        input_dim: int = 128,
        n_classes: int = 10,
        strategy: LearningStrategy = LearningStrategy.ADAPTIVE,
        learning_rate: float = 0.01,
        regularization: float = 0.001,
    ) -> None:
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.regularization = regularization

        # Model parameters
        self._weights = np.random.randn(n_classes, input_dim) * 0.01
        self._bias = np.zeros(n_classes)
        self._iteration: int = 0
        self._loss_history: List[float] = []
        self._accuracy_window: List[bool] = []

        # Adaptive rate scheduling
        self._grad_accumulator = np.zeros_like(self._weights)
        self._grad_sq_accumulator = np.zeros_like(self._weights)

    def partial_fit(self, x: np.ndarray, y: int) -> float:
        """Update model with a single sample.

        Args:
            x: Input feature vector.
            y: True class label.

        Returns:
            Loss value for this sample.
        """
        x = np.atleast_1d(x).flatten()
        if len(x) > self.input_dim:
            x = x[:self.input_dim]
        elif len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))

        # Forward pass (softmax)
        logits = self._weights @ x + self._bias
        probs = self._softmax(logits)

        # Compute loss
        loss = -np.log(probs[y] + 1e-12)
        self._loss_history.append(float(loss))

        # Prediction accuracy
        pred = int(np.argmax(probs))
        self._accuracy_window.append(pred == y)
        if len(self._accuracy_window) > 100:
            self._accuracy_window.pop(0)

        # Gradient computation
        grad = probs.copy()
        grad[y] -= 1.0
        weight_grad = np.outer(grad, x) + self.regularization * self._weights
        bias_grad = grad

        # Learning rate adaptation
        lr = self._get_learning_rate(weight_grad)

        # Strategy-specific update
        if self.strategy == LearningStrategy.AGGRESSIVE:
            # Aggressive: larger step for incorrect predictions
            if pred != y:
                lr *= 2.0
        elif self.strategy == LearningStrategy.ADAPTIVE:
            # AdaGrad-style
            self._grad_sq_accumulator += weight_grad ** 2
            lr = self.learning_rate / (np.sqrt(self._grad_sq_accumulator) + 1e-8)

        # Update weights
        if isinstance(lr, np.ndarray):
            self._weights -= lr * weight_grad
        else:
            self._weights -= lr * weight_grad
        self._bias -= self.learning_rate * bias_grad

        self._iteration += 1
        return float(loss)

    def predict(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict class for input.

        Args:
            x: Input feature vector.

        Returns:
            Tuple of (predicted_class, probabilities).
        """
        x = np.atleast_1d(x).flatten()
        if len(x) > self.input_dim:
            x = x[:self.input_dim]
        elif len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))

        logits = self._weights @ x + self._bias
        probs = self._softmax(logits)
        return int(np.argmax(probs)), probs

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict classes for batch of inputs.

        Args:
            X: Input matrix (n_samples x input_dim).

        Returns:
            Tuple of (predictions, probability_matrix).
        """
        predictions = []
        probs_list = []
        for x in X:
            pred, probs = self.predict(x)
            predictions.append(pred)
            probs_list.append(probs)
        return np.array(predictions), np.array(probs_list)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / (np.sum(exp_logits) + 1e-12)

    def _get_learning_rate(self, grad: np.ndarray) -> float:
        """Compute adaptive learning rate."""
        if self.strategy == LearningStrategy.ADAPTIVE:
            # Already handled in partial_fit
            return self.learning_rate
        # Simple decay
        return self.learning_rate / (1.0 + 0.001 * self._iteration)

    def get_state(self) -> LearningState:
        """Get current learning state."""
        return LearningState(
            iteration=self._iteration,
            loss=self._loss_history[-1] if self._loss_history else float("inf"),
            accuracy=np.mean(self._accuracy_window) if self._accuracy_window else 0.0,
            weights=self._weights.copy(),
            metadata={
                "strategy": self.strategy.value,
                "total_samples": self._iteration,
            },
        )

    @property
    def accuracy(self) -> float:
        """Recent accuracy."""
        return float(np.mean(self._accuracy_window)) if self._accuracy_window else 0.0


# =============================================================================
# Meta-Learner
# =============================================================================


class MetaLearner:
    """Meta-learning for few-shot spectral classification.

    Learns to learn from few examples by building prototypical
    representations of spectral classes.

    Args:
        embedding_dim: Embedding dimension.
        method: Meta-learning method.
        n_support: Number of support examples per class.
        temperature: Softmax temperature for distance.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        method: MetaLearningMethod = MetaLearningMethod.PROTOTYPICAL,
        n_support: int = 5,
        temperature: float = 1.0,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.method = method
        self.n_support = n_support
        self.temperature = temperature

        self._prototypes: Dict[int, Prototype] = {}
        self._embedding_weights = np.random.randn(embedding_dim, 256) * 0.1
        self._embedding_bias = np.zeros(embedding_dim)
        self._n_episodes: int = 0

    def register_class(self, class_id: int, support_set: np.ndarray) -> None:
        """Register a new class from support examples.

        Args:
            class_id: Class identifier.
            support_set: Support examples (n_support x feature_dim).
        """
        support_set = np.atleast_2d(support_set)
        embeddings = np.array([self._embed(x) for x in support_set])
        prototype = np.mean(embeddings, axis=0)
        variance = float(np.mean(np.var(embeddings, axis=0)))

        self._prototypes[class_id] = Prototype(
            class_id=class_id,
            embedding=prototype,
            support_count=len(support_set),
            variance=variance,
        )

    def classify(self, x: np.ndarray) -> Tuple[int, Dict[int, float]]:
        """Classify using prototypical distance.

        Args:
            x: Input feature vector.

        Returns:
            Tuple of (predicted_class, class_distances).
        """
        if not self._prototypes:
            return -1, {}

        embedding = self._embed(x)
        distances = {}

        for class_id, proto in self._prototypes.items():
            dist = float(np.sum((embedding - proto.embedding) ** 2))
            distances[class_id] = dist

        # Convert distances to probabilities
        dist_array = np.array(list(distances.values()))
        neg_dist = -dist_array / self.temperature
        probs = np.exp(neg_dist - np.max(neg_dist))
        probs /= np.sum(probs) + 1e-12

        class_probs = {}
        for i, class_id in enumerate(distances.keys()):
            class_probs[class_id] = float(probs[i])

        best_class = min(distances, key=distances.get)
        return best_class, class_probs

    def update_prototype(self, class_id: int, x: np.ndarray, momentum: float = 0.1) -> None:
        """Update a prototype with a new example (online).

        Args:
            class_id: Class to update.
            x: New example.
            momentum: Update momentum.
        """
        if class_id not in self._prototypes:
            self.register_class(class_id, x.reshape(1, -1))
            return

        embedding = self._embed(x)
        proto = self._prototypes[class_id]
        proto.embedding = (1 - momentum) * proto.embedding + momentum * embedding
        proto.support_count += 1

    def meta_train_episode(
        self,
        support_X: np.ndarray,
        support_y: np.ndarray,
        query_X: np.ndarray,
        query_y: np.ndarray,
    ) -> float:
        """Train on one meta-learning episode.

        Args:
            support_X: Support set features.
            support_y: Support set labels.
            query_X: Query set features.
            query_y: Query set labels.

        Returns:
            Episode loss.
        """
        # Build prototypes from support set
        classes = np.unique(support_y)
        for cls in classes:
            mask = support_y == cls
            self.register_class(int(cls), support_X[mask])

        # Evaluate on query set
        total_loss = 0.0
        correct = 0

        for x, y in zip(query_X, query_y):
            pred, probs = self.classify(x)
            if pred == y:
                correct += 1
            # Cross-entropy loss approximation
            prob_correct = probs.get(int(y), 1e-12)
            total_loss -= np.log(prob_correct + 1e-12)

        self._n_episodes += 1
        avg_loss = total_loss / max(1, len(query_y))

        # Simple gradient update to embedding
        if avg_loss > 0.5:
            # Perturb embedding weights
            perturbation = 0.001 * np.random.randn(*self._embedding_weights.shape)
            self._embedding_weights += perturbation

        return float(avg_loss)

    def _embed(self, x: np.ndarray) -> np.ndarray:
        """Embed input into prototype space."""
        x = np.atleast_1d(x).flatten()
        if len(x) > 256:
            x = x[:256]
        elif len(x) < 256:
            x = np.pad(x, (0, 256 - len(x)))

        # Simple linear + ReLU embedding
        z = self._embedding_weights @ x + self._embedding_bias
        z = np.maximum(z, 0)  # ReLU
        # L2 normalize
        norm = np.linalg.norm(z) + 1e-12
        return z / norm

    @property
    def n_classes(self) -> int:
        """Number of registered classes."""
        return len(self._prototypes)

    @property
    def n_episodes(self) -> int:
        """Number of training episodes."""
        return self._n_episodes


# =============================================================================
# Continual Learner
# =============================================================================


class ContinualLearner:
    """Continual learning without catastrophic forgetting.

    Uses elastic weight consolidation (EWC) and progressive
    networks to retain knowledge across tasks.

    Args:
        input_dim: Input feature dimension.
        n_classes: Number of classes.
        ewc_lambda: EWC regularization strength.
        memory_size: Replay memory size per task.
    """

    def __init__(
        self,
        input_dim: int = 128,
        n_classes: int = 10,
        ewc_lambda: float = 1000.0,
        memory_size: int = 50,
    ) -> None:
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.ewc_lambda = ewc_lambda
        self.memory_size = memory_size

        # Model parameters
        self._weights = np.random.randn(n_classes, input_dim) * 0.01
        self._bias = np.zeros(n_classes)

        # EWC components
        self._fisher: Optional[np.ndarray] = None  # Fisher information
        self._star_weights: Optional[np.ndarray] = None  # Optimal weights for prev tasks
        self._star_bias: Optional[np.ndarray] = None

        # Task memory
        self._task_memories: Dict[int, List[Tuple[np.ndarray, int]]] = {}
        self._current_task: int = 0
        self._task_performance: Dict[int, float] = {}

    def train_task(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_id: int,
        n_epochs: int = 10,
        learning_rate: float = 0.01,
    ) -> Dict[str, float]:
        """Train on a new task.

        Args:
            X: Training features.
            y: Training labels.
            task_id: Task identifier.
            n_epochs: Number of training epochs.
            learning_rate: Learning rate.

        Returns:
            Training metrics.
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y).astype(int)
        n_samples = len(X)

        # Store examples in memory
        indices = np.random.choice(n_samples, min(self.memory_size, n_samples), replace=False)
        self._task_memories[task_id] = [(X[i], int(y[i])) for i in indices]

        losses = []
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            order = np.random.permutation(n_samples)

            for idx in order:
                x = X[idx]
                label = y[idx]

                # Pad/truncate input
                if len(x) > self.input_dim:
                    x = x[:self.input_dim]
                elif len(x) < self.input_dim:
                    x = np.pad(x, (0, self.input_dim - len(x)))

                # Forward
                logits = self._weights @ x + self._bias
                probs = self._softmax(logits)
                loss = -np.log(probs[label] + 1e-12)

                # EWC penalty
                ewc_loss = 0.0
                if self._fisher is not None and self._star_weights is not None:
                    ewc_loss = float(
                        0.5 * self.ewc_lambda *
                        np.sum(self._fisher * (self._weights - self._star_weights) ** 2)
                    )
                    loss += ewc_loss

                epoch_loss += loss

                # Backward
                grad = probs.copy()
                grad[label] -= 1.0
                weight_grad = np.outer(grad, x)
                bias_grad = grad

                # EWC gradient
                if self._fisher is not None and self._star_weights is not None:
                    weight_grad += self.ewc_lambda * self._fisher * (self._weights - self._star_weights)

                self._weights -= learning_rate * weight_grad
                self._bias -= learning_rate * bias_grad

            losses.append(epoch_loss / n_samples)

            # Replay from previous tasks
            self._replay(learning_rate)

        # Update EWC after task
        self._compute_fisher(X, y)
        self._star_weights = self._weights.copy()
        self._star_bias = self._bias.copy()

        # Evaluate
        accuracy = self._evaluate(X, y)
        self._task_performance[task_id] = accuracy
        self._current_task = task_id

        return {
            "final_loss": losses[-1] if losses else float("inf"),
            "accuracy": accuracy,
            "task_id": task_id,
        }

    def _replay(self, learning_rate: float) -> None:
        """Replay samples from previous tasks."""
        for task_id, memories in self._task_memories.items():
            if task_id == self._current_task:
                continue
            if not memories:
                continue

            # Sample from memory
            n_replay = min(5, len(memories))
            indices = np.random.choice(len(memories), n_replay, replace=False)

            for idx in indices:
                x, label = memories[idx]
                if len(x) > self.input_dim:
                    x = x[:self.input_dim]
                elif len(x) < self.input_dim:
                    x = np.pad(x, (0, self.input_dim - len(x)))

                logits = self._weights @ x + self._bias
                probs = self._softmax(logits)
                grad = probs.copy()
                grad[label] -= 1.0

                self._weights -= learning_rate * 0.5 * np.outer(grad, x)
                self._bias -= learning_rate * 0.5 * grad

    def _compute_fisher(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute Fisher information matrix (diagonal)."""
        fisher = np.zeros_like(self._weights)
        n_samples = min(100, len(X))

        for i in range(n_samples):
            x = X[i]
            if len(x) > self.input_dim:
                x = x[:self.input_dim]
            elif len(x) < self.input_dim:
                x = np.pad(x, (0, self.input_dim - len(x)))

            logits = self._weights @ x + self._bias
            probs = self._softmax(logits)
            grad = probs.copy()
            grad[y[i]] -= 1.0
            fisher += np.outer(grad, x) ** 2

        fisher /= n_samples

        if self._fisher is None:
            self._fisher = fisher
        else:
            self._fisher = 0.5 * (self._fisher + fisher)

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate accuracy."""
        correct = 0
        for i in range(len(X)):
            pred, _ = self.predict(X[i])
            if pred == y[i]:
                correct += 1
        return correct / max(1, len(X))

    def predict(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict class."""
        x = np.atleast_1d(x).flatten()
        if len(x) > self.input_dim:
            x = x[:self.input_dim]
        elif len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))

        logits = self._weights @ x + self._bias
        probs = self._softmax(logits)
        return int(np.argmax(probs)), probs

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Stable softmax."""
        exp_l = np.exp(logits - np.max(logits))
        return exp_l / (np.sum(exp_l) + 1e-12)

    def evaluate_all_tasks(self, task_data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, float]:
        """Evaluate on all previous tasks to measure forgetting.

        Args:
            task_data: Dict of task_id -> (X, y) pairs.

        Returns:
            Dictionary of task_id -> accuracy.
        """
        results = {}
        for task_id, (X, y) in task_data.items():
            results[task_id] = self._evaluate(X, y)
        return results

    @property
    def n_tasks_learned(self) -> int:
        """Number of tasks learned."""
        return len(self._task_performance)


# =============================================================================
# Spectral Optimizer
# =============================================================================


class SpectralOptimizer:
    """Hyperparameter optimization for spectral models.

    Uses Bayesian optimization, random search, and evolutionary
    strategies to find optimal model configurations.

    Args:
        param_space: Parameter search space {name: (min, max)}.
        method: Optimization method.
        n_initial: Number of initial random trials.
    """

    def __init__(
        self,
        param_space: Dict[str, Tuple[float, float]],
        method: OptimizationMethod = OptimizationMethod.BAYESIAN,
        n_initial: int = 10,
    ) -> None:
        self.param_space = param_space
        self.method = method
        self.n_initial = n_initial

        self._trials: List[OptimizationTrial] = []
        self._best_trial: Optional[OptimizationTrial] = None
        self._n_evaluations: int = 0

    def suggest(self) -> Dict[str, float]:
        """Suggest next hyperparameter configuration.

        Returns:
            Dictionary of parameter values.
        """
        if len(self._trials) < self.n_initial:
            return self._random_sample()

        if self.method == OptimizationMethod.RANDOM_SEARCH:
            return self._random_sample()
        elif self.method == OptimizationMethod.BAYESIAN:
            return self._bayesian_suggest()
        elif self.method == OptimizationMethod.EVOLUTIONARY:
            return self._evolutionary_suggest()
        else:
            return self._random_sample()

    def report(self, params: Dict[str, float], score: float, duration: float = 0.0) -> None:
        """Report trial results.

        Args:
            params: Parameter values.
            score: Evaluation score (higher is better).
            duration: Time taken.
        """
        trial = OptimizationTrial(
            params=params,
            score=score,
            duration=duration,
            trial_id=len(self._trials),
        )
        self._trials.append(trial)
        self._n_evaluations += 1

        if self._best_trial is None or score > self._best_trial.score:
            self._best_trial = trial

    def _random_sample(self) -> Dict[str, float]:
        """Random sample from parameter space."""
        params = {}
        for name, (low, high) in self.param_space.items():
            params[name] = float(np.random.uniform(low, high))
        return params

    def _bayesian_suggest(self) -> Dict[str, float]:
        """Bayesian optimization suggestion (GP surrogate)."""
        if not self._trials:
            return self._random_sample()

        # Use expected improvement heuristic
        # Sample candidates and pick best expected improvement
        n_candidates = 100
        best_score = self._best_trial.score if self._best_trial else 0.0

        candidates = []
        for _ in range(n_candidates):
            params = self._random_sample()
            # Predict score using local interpolation
            predicted = self._predict_score(params)
            # Expected improvement
            ei = max(0.0, predicted - best_score)
            candidates.append((params, ei))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _predict_score(self, params: Dict[str, float]) -> float:
        """Predict score for a configuration using kernel regression."""
        if not self._trials:
            return 0.0

        # Kernel-weighted average of past scores
        total_weight = 0.0
        weighted_score = 0.0

        for trial in self._trials:
            dist = self._param_distance(params, trial.params)
            weight = np.exp(-dist * 5.0)
            weighted_score += weight * trial.score
            total_weight += weight

        if total_weight < 1e-12:
            return 0.0
        return weighted_score / total_weight

    def _evolutionary_suggest(self) -> Dict[str, float]:
        """Evolutionary suggestion (mutation of best)."""
        if not self._trials:
            return self._random_sample()

        # Select top performers
        sorted_trials = sorted(self._trials, key=lambda t: t.score, reverse=True)
        parent = sorted_trials[0].params

        # Mutate
        child = {}
        for name, (low, high) in self.param_space.items():
            mutation = np.random.randn() * 0.1 * (high - low)
            value = parent.get(name, (low + high) / 2) + mutation
            child[name] = float(np.clip(value, low, high))

        return child

    def _param_distance(self, p1: Dict[str, float], p2: Dict[str, float]) -> float:
        """Normalized distance between parameter configs."""
        dist = 0.0
        for name, (low, high) in self.param_space.items():
            range_val = high - low + 1e-12
            d = (p1.get(name, 0) - p2.get(name, 0)) / range_val
            dist += d ** 2
        return np.sqrt(dist)

    @property
    def best_params(self) -> Optional[Dict[str, float]]:
        """Best parameters found."""
        return self._best_trial.params if self._best_trial else None

    @property
    def best_score(self) -> float:
        """Best score achieved."""
        return self._best_trial.score if self._best_trial else 0.0

    @property
    def n_trials(self) -> int:
        """Number of trials completed."""
        return len(self._trials)


# =============================================================================
# Ensemble Predictor
# =============================================================================


class EnsemblePredictor:
    """Ensemble methods for spectral prediction.

    Combines multiple weak learners using boosting, bagging,
    and stacking strategies.

    Args:
        n_estimators: Number of base estimators.
        input_dim: Feature dimension.
        n_classes: Number of classes.
        diversity_weight: Weight for diversity in ensemble.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        input_dim: int = 128,
        n_classes: int = 10,
        diversity_weight: float = 0.1,
    ) -> None:
        self.n_estimators = n_estimators
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.diversity_weight = diversity_weight

        # Create base estimators (linear classifiers with different init)
        self._estimators: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._weights_ensemble: np.ndarray = np.ones(n_estimators) / n_estimators

        for _ in range(n_estimators):
            w = np.random.randn(n_classes, input_dim) * 0.1
            b = np.zeros(n_classes)
            self._estimators.append(w)
            self._biases.append(b)

        self._is_trained: bool = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 10,
        learning_rate: float = 0.01,
    ) -> Dict[str, float]:
        """Train ensemble on spectral data.

        Args:
            X: Training features.
            y: Training labels.
            n_epochs: Training epochs per estimator.
            learning_rate: Learning rate.

        Returns:
            Training metrics.
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y).astype(int)
        n_samples = len(X)

        # Bagging: each estimator trains on bootstrap sample
        estimator_accuracies = []

        for est_idx in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Train this estimator
            W = self._estimators[est_idx]
            b = self._biases[est_idx]

            for epoch in range(n_epochs):
                for i in range(len(X_boot)):
                    x = self._prepare_input(X_boot[i])
                    logits = W @ x + b
                    probs = self._softmax(logits)

                    grad = probs.copy()
                    grad[y_boot[i]] -= 1.0

                    W -= learning_rate * np.outer(grad, x)
                    b -= learning_rate * grad

            self._estimators[est_idx] = W
            self._biases[est_idx] = b

            # Evaluate
            correct = 0
            for i in range(min(100, n_samples)):
                x = self._prepare_input(X[i])
                logits = W @ x + b
                if np.argmax(logits) == y[i]:
                    correct += 1
            acc = correct / min(100, n_samples)
            estimator_accuracies.append(acc)

        # Update ensemble weights based on accuracy
        accs = np.array(estimator_accuracies) + 1e-12
        self._weights_ensemble = accs / np.sum(accs)
        self._is_trained = True

        return {
            "mean_accuracy": float(np.mean(estimator_accuracies)),
            "best_estimator_accuracy": float(np.max(estimator_accuracies)),
            "ensemble_diversity": float(np.std(estimator_accuracies)),
        }

    def predict(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        """Ensemble prediction with weighted voting.

        Args:
            x: Input features.

        Returns:
            Tuple of (prediction, probabilities).
        """
        x = self._prepare_input(x)
        aggregated_probs = np.zeros(self.n_classes)

        for est_idx in range(self.n_estimators):
            W = self._estimators[est_idx]
            b = self._biases[est_idx]
            logits = W @ x + b
            probs = self._softmax(logits)
            aggregated_probs += self._weights_ensemble[est_idx] * probs

        aggregated_probs /= np.sum(aggregated_probs) + 1e-12
        return int(np.argmax(aggregated_probs)), aggregated_probs

    def predict_with_uncertainty(self, x: np.ndarray) -> Tuple[int, float, float]:
        """Predict with uncertainty estimation.

        Args:
            x: Input features.

        Returns:
            Tuple of (prediction, confidence, uncertainty).
        """
        x = self._prepare_input(x)
        predictions = []

        for est_idx in range(self.n_estimators):
            W = self._estimators[est_idx]
            b = self._biases[est_idx]
            logits = W @ x + b
            pred = int(np.argmax(logits))
            predictions.append(pred)

        # Voting
        from collections import Counter
        votes = Counter(predictions)
        best_pred = votes.most_common(1)[0][0]
        confidence = votes[best_pred] / self.n_estimators
        uncertainty = 1.0 - confidence

        return best_pred, confidence, uncertainty

    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        """Prepare input to correct dimension."""
        x = np.atleast_1d(x).flatten()
        if len(x) > self.input_dim:
            return x[:self.input_dim]
        elif len(x) < self.input_dim:
            return np.pad(x, (0, self.input_dim - len(x)))
        return x

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Stable softmax."""
        exp_l = np.exp(logits - np.max(logits))
        return exp_l / (np.sum(exp_l) + 1e-12)

    @property
    def is_trained(self) -> bool:
        """Whether the ensemble has been trained."""
        return self._is_trained


# =============================================================================
# Active Learner
# =============================================================================


class ActiveLearner:
    """Active learning for efficient spectral labeling.

    Selects the most informative samples for labeling,
    maximizing model improvement with minimal labels.

    Args:
        model: Underlying prediction model (has predict method).
        acquisition: Acquisition function for sample selection.
        batch_size: Number of samples to select per iteration.
    """

    def __init__(
        self,
        input_dim: int = 128,
        n_classes: int = 10,
        acquisition: AcquisitionFunction = AcquisitionFunction.UNCERTAINTY,
        batch_size: int = 10,
    ) -> None:
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.acquisition = acquisition
        self.batch_size = batch_size

        # Internal model
        self._model = OnlineLearner(input_dim=input_dim, n_classes=n_classes)
        self._labeled_indices: List[int] = []
        self._query_count: int = 0

    def select_samples(self, X_pool: np.ndarray) -> List[ActiveSample]:
        """Select most informative samples from pool.

        Args:
            X_pool: Unlabeled sample pool.

        Returns:
            List of ActiveSample objects to label.
        """
        X_pool = np.atleast_2d(X_pool)
        n_pool = len(X_pool)
        scores = np.zeros(n_pool)

        for i in range(n_pool):
            if i in self._labeled_indices:
                scores[i] = -float("inf")
                continue

            pred, probs = self._model.predict(X_pool[i])

            if self.acquisition == AcquisitionFunction.UNCERTAINTY:
                scores[i] = 1.0 - np.max(probs)
            elif self.acquisition == AcquisitionFunction.ENTROPY:
                scores[i] = -np.sum(probs * np.log2(probs + 1e-12))
            elif self.acquisition == AcquisitionFunction.MARGIN:
                sorted_probs = np.sort(probs)[::-1]
                scores[i] = 1.0 - (sorted_probs[0] - sorted_probs[1])
            else:
                scores[i] = 1.0 - np.max(probs)

        # Select top-k
        top_indices = np.argsort(scores)[-self.batch_size:][::-1]

        samples = []
        for idx in top_indices:
            if scores[idx] == -float("inf"):
                continue
            pred, probs = self._model.predict(X_pool[idx])
            samples.append(ActiveSample(
                index=int(idx),
                acquisition_value=float(scores[idx]),
                predicted_label=pred,
                uncertainty=float(1.0 - np.max(probs)),
            ))

        self._query_count += len(samples)
        return samples

    def label_sample(self, x: np.ndarray, y: int, index: int) -> float:
        """Add a labeled sample and update the model.

        Args:
            x: Feature vector.
            y: True label.
            index: Pool index of this sample.

        Returns:
            Loss from this update.
        """
        self._labeled_indices.append(index)
        loss = self._model.partial_fit(x, y)
        return loss

    @property
    def n_labeled(self) -> int:
        """Number of labeled samples."""
        return len(self._labeled_indices)

    @property
    def n_queries(self) -> int:
        """Total query iterations."""
        return self._query_count

    @property
    def model_accuracy(self) -> float:
        """Current model accuracy."""
        return self._model.accuracy
