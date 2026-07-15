"""Explainability and interpretability for spectral AI models.

Provides feature importance, attention visualization, SHAP-style
explanations, and causal attribution for spectral predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

import numpy as np


class ExplanationType(Enum):
    """Types of model explanations."""

    FEATURE_IMPORTANCE = "feature_importance"
    ATTENTION_MAP = "attention_map"
    GRADIENT = "gradient"
    PERTURBATION = "perturbation"
    COUNTERFACTUAL = "counterfactual"
    PROTOTYPE = "prototype"


@dataclass
class Explanation:
    """Structured model explanation."""

    explanation_type: ExplanationType
    feature_attributions: np.ndarray
    confidence: float
    top_features: list[int] = field(default_factory=list)
    description: str = ""
    counterfactual: Optional[np.ndarray] = None
    prototype_idx: Optional[int] = None

    @property
    def n_important_features(self) -> int:
        threshold = np.percentile(np.abs(self.feature_attributions), 90)
        return int(np.sum(np.abs(self.feature_attributions) >= threshold))


@dataclass
class SpectralFeatureImportance:
    """Feature importance scores for spectral bands."""

    band_importances: np.ndarray
    frequency_ranges: list[tuple[float, float]] = field(default_factory=list)
    global_importance: Optional[np.ndarray] = None
    interaction_scores: Optional[np.ndarray] = None


class PerturbationExplainer:
    """Perturbation-based model explanation (LIME-style).

    Explains predictions by observing how perturbing input features
    affects model output, without requiring model gradients.
    """

    def __init__(
        self, n_perturbations: int = 100, perturbation_scale: float = 0.1
    ) -> None:
        self.n_perturbations = n_perturbations
        self.perturbation_scale = perturbation_scale

    def explain(
        self, model_fn: Any, x: np.ndarray, baseline: Optional[np.ndarray] = None
    ) -> Explanation:
        """Generate explanation for a single sample."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if baseline is None:
            baseline = np.zeros_like(x)

        original_pred = np.asarray(model_fn(x)).flatten()
        importances = np.zeros(x.shape[1])

        for feature_idx in range(x.shape[1]):
            perturbed = np.tile(x, (self.n_perturbations, 1))
            perturbed[:, feature_idx] += np.random.randn(self.n_perturbations) * self.perturbation_scale

            perturbed_preds = np.asarray(model_fn(perturbed)).flatten()
            mean_pred = original_pred.mean() if original_pred.ndim > 0 else float(original_pred)
            importances[feature_idx] = np.abs(perturbed_preds.mean() - mean_pred)

        # Normalize
        max_imp = importances.max()
        if max_imp > 0:
            importances /= max_imp

        top_features = list(np.argsort(importances)[::-1][:10])

        return Explanation(
            explanation_type=ExplanationType.PERTURBATION,
            feature_attributions=importances,
            confidence=float(importances.max()),
            top_features=top_features,
            description=f"Top features: {top_features[:5]}",
        )


class GradientExplainer:
    """Gradient-based explanation using finite differences.

    Approximates gradients without requiring automatic differentiation
    by using finite differences for spectral model attribution.
    """

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.epsilon = epsilon

    def explain(self, model_fn: Any, x: np.ndarray) -> Explanation:
        """Compute gradient-based attribution."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        gradients = np.zeros(x.shape[1])
        base_pred = np.asarray(model_fn(x)).flatten().mean()

        for i in range(x.shape[1]):
            x_plus = x.copy()
            x_plus[0, i] += self.epsilon
            pred_plus = np.asarray(model_fn(x_plus)).flatten().mean()
            gradients[i] = (pred_plus - base_pred) / self.epsilon

        # Input * gradient for attribution
        attributions = np.abs(gradients * x.flatten())
        max_attr = attributions.max()
        if max_attr > 0:
            attributions /= max_attr

        top_features = list(np.argsort(attributions)[::-1][:10])

        return Explanation(
            explanation_type=ExplanationType.GRADIENT,
            feature_attributions=attributions,
            confidence=float(attributions.max()),
            top_features=top_features,
        )


class CounterfactualExplainer:
    """Generates counterfactual explanations.

    Finds minimal perturbations that change model predictions,
    revealing decision boundaries of spectral classifiers.
    """

    def __init__(
        self, max_iterations: int = 100, step_size: float = 0.01, target_class: Optional[int] = None
    ) -> None:
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.target_class = target_class

    def explain(
        self, model_fn: Any, x: np.ndarray, target_pred: Optional[float] = None
    ) -> Explanation:
        """Find counterfactual by iterative perturbation."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        original_pred = np.asarray(model_fn(x)).flatten()
        if target_pred is None:
            # Flip prediction direction
            target_pred = 1.0 - original_pred.mean()

        counterfactual = x.copy()
        for _ in range(self.max_iterations):
            current_pred = np.asarray(model_fn(counterfactual)).flatten().mean()
            if abs(current_pred - target_pred) < 0.1:
                break

            # Random direction toward target
            direction = np.random.randn(*counterfactual.shape)
            direction /= np.linalg.norm(direction)

            # Try perturbation
            candidate = counterfactual + self.step_size * direction
            new_pred = np.asarray(model_fn(candidate)).flatten().mean()

            if abs(new_pred - target_pred) < abs(current_pred - target_pred):
                counterfactual = candidate

        # Attribution = difference from original
        diff = np.abs(counterfactual - x).flatten()
        max_diff = diff.max()
        if max_diff > 0:
            diff /= max_diff

        return Explanation(
            explanation_type=ExplanationType.COUNTERFACTUAL,
            feature_attributions=diff,
            confidence=float(1.0 - abs(np.asarray(model_fn(counterfactual)).flatten().mean() - target_pred)),
            counterfactual=counterfactual,
            top_features=list(np.argsort(diff)[::-1][:10]),
        )


class SpectralAttentionVisualizer:
    """Visualizes attention patterns in spectral transformer models."""

    def __init__(self) -> None:
        self._attention_history: list[np.ndarray] = []

    def compute_attention_scores(
        self, query: np.ndarray, key: np.ndarray
    ) -> np.ndarray:
        """Compute scaled dot-product attention scores."""
        d_k = query.shape[-1]
        scores = query @ key.T / np.sqrt(d_k)
        # Softmax
        exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attention = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        self._attention_history.append(attention)
        return attention

    def aggregate_attention(self, multi_head_attention: np.ndarray) -> np.ndarray:
        """Aggregate multi-head attention into single importance map."""
        if multi_head_attention.ndim == 3:
            return multi_head_attention.mean(axis=0)
        return multi_head_attention

    def get_feature_importance_from_attention(
        self, attention_map: np.ndarray
    ) -> SpectralFeatureImportance:
        """Convert attention map to feature importance."""
        if attention_map.ndim == 2:
            importance = attention_map.mean(axis=0)
        else:
            importance = attention_map.flatten()

        return SpectralFeatureImportance(band_importances=importance)

    @property
    def attention_history(self) -> list[np.ndarray]:
        return self._attention_history
