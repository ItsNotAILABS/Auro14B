"""Transfer learning and domain adaptation for spectral models.

Provides utilities for adapting pre-trained spectral models to new
domains (e.g., from earthquake spectra to structural monitoring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class DomainInfo:
    """Metadata about a spectral domain.

    Args:
        name: Human-readable domain name.
        frequency_range: Tuple of (min_freq, max_freq) Hz.
        typical_components: Expected number of components.
        characteristic_features: Key spectral features of this domain.
    """

    name: str
    frequency_range: tuple[float, float] = (0.01, 100.0)
    typical_components: int = 3
    characteristic_features: list[str] = field(default_factory=list)


class TransferAdapter:
    """Adapter for transferring spectral models between domains.

    Applies feature alignment, normalization mapping, and weight
    adaptation to repurpose a pre-trained model for a new domain.

    Args:
        source_domain: Source domain info.
        target_domain: Target domain info.
        adaptation_strength: How strongly to adapt (0=no change, 1=full).
    """

    def __init__(
        self,
        source_domain: Optional[DomainInfo] = None,
        target_domain: Optional[DomainInfo] = None,
        adaptation_strength: float = 0.5,
    ) -> None:
        self.source_domain = source_domain or DomainInfo(name="generic")
        self.target_domain = target_domain or DomainInfo(name="generic")
        self.adaptation_strength = np.clip(adaptation_strength, 0.0, 1.0)
        self._source_stats: Optional[dict[str, np.ndarray]] = None
        self._target_stats: Optional[dict[str, np.ndarray]] = None
        self._transform_matrix: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, source_data: np.ndarray, target_data: np.ndarray) -> None:
        """Compute domain alignment statistics.

        Args:
            source_data: Representative data from source domain.
            target_data: Representative data from target domain.
        """
        source_data = np.atleast_2d(source_data)
        target_data = np.atleast_2d(target_data)

        self._source_stats = {
            "mean": np.mean(source_data, axis=0),
            "std": np.std(source_data, axis=0) + 1e-10,
            "cov": np.cov(source_data.T) if source_data.shape[0] > 1 else np.eye(source_data.shape[1]),
        }
        self._target_stats = {
            "mean": np.mean(target_data, axis=0),
            "std": np.std(target_data, axis=0) + 1e-10,
            "cov": np.cov(target_data.T) if target_data.shape[0] > 1 else np.eye(target_data.shape[1]),
        }

        # Compute whitening-coloring transform (CORAL-style)
        dim = source_data.shape[1]
        source_cov = self._source_stats["cov"]
        target_cov = self._target_stats["cov"]

        if source_cov.ndim < 2:
            source_cov = np.eye(dim)
        if target_cov.ndim < 2:
            target_cov = np.eye(dim)

        # Simplified domain alignment via mean/std matching
        self._transform_matrix = np.diag(self._target_stats["std"] / self._source_stats["std"])
        self._is_fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform source-domain data to target domain.

        Args:
            data: Source domain data.

        Returns:
            Adapted data aligned to target domain.
        """
        if not self._is_fitted:
            return data

        data = np.atleast_2d(data)

        # Normalize to source stats
        normalized = (data - self._source_stats["mean"]) / self._source_stats["std"]

        # Apply domain transform with adaptation strength
        if self._transform_matrix is not None:
            adapted = normalized @ self._transform_matrix
        else:
            adapted = normalized

        # Denormalize to target stats
        result = adapted * self._target_stats["std"] + self._target_stats["mean"]

        # Blend based on adaptation strength
        blended = (1 - self.adaptation_strength) * data + self.adaptation_strength * result
        return blended

    @property
    def is_fitted(self) -> bool:
        """Whether the adapter has been fitted."""
        return self._is_fitted


class DomainAdaptation:
    """Multi-strategy domain adaptation for spectral intelligence.

    Supports multiple adaptation strategies including statistical
    alignment, feature matching, and adversarial adaptation.

    Args:
        strategy: Adaptation strategy ('coral', 'mmd', 'normalization').
        n_components: Number of components for dimensionality reduction.
    """

    def __init__(
        self,
        strategy: str = "coral",
        n_components: Optional[int] = None,
    ) -> None:
        self.strategy = strategy
        self.n_components = n_components
        self._source_features: Optional[np.ndarray] = None
        self._target_features: Optional[np.ndarray] = None
        self._alignment_matrix: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> None:
        """Fit the domain adaptation model.

        Args:
            source_features: Feature matrix from source domain.
            target_features: Feature matrix from target domain.
        """
        source_features = np.atleast_2d(source_features)
        target_features = np.atleast_2d(target_features)

        self._source_features = source_features
        self._target_features = target_features

        if self.strategy == "coral":
            self._fit_coral(source_features, target_features)
        elif self.strategy == "mmd":
            self._fit_mmd(source_features, target_features)
        elif self.strategy == "normalization":
            self._fit_normalization(source_features, target_features)

        self._is_fitted = True

    def _fit_coral(self, source: np.ndarray, target: np.ndarray) -> None:
        """CORAL: Correlation alignment."""
        dim = source.shape[1]
        cs = np.cov(source.T) + np.eye(dim) * 1e-6
        ct = np.cov(target.T) + np.eye(dim) * 1e-6

        # Whitening source
        u_s, s_s, _ = np.linalg.svd(cs)
        whiten = u_s @ np.diag(1.0 / np.sqrt(s_s + 1e-10)) @ u_s.T

        # Coloring to target
        u_t, s_t, _ = np.linalg.svd(ct)
        color = u_t @ np.diag(np.sqrt(s_t + 1e-10)) @ u_t.T

        self._alignment_matrix = whiten @ color

    def _fit_mmd(self, source: np.ndarray, target: np.ndarray) -> None:
        """Maximum Mean Discrepancy alignment."""
        # Simple mean-matching approach
        source_mean = np.mean(source, axis=0)
        target_mean = np.mean(target, axis=0)
        shift = target_mean - source_mean
        dim = source.shape[1]
        self._alignment_matrix = np.eye(dim)
        self._mmd_shift = shift

    def _fit_normalization(self, source: np.ndarray, target: np.ndarray) -> None:
        """Z-score normalization alignment."""
        self._source_mean = np.mean(source, axis=0)
        self._source_std = np.std(source, axis=0) + 1e-10
        self._target_mean = np.mean(target, axis=0)
        self._target_std = np.std(target, axis=0) + 1e-10
        dim = source.shape[1]
        self._alignment_matrix = np.diag(self._target_std / self._source_std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data from source to target domain.

        Args:
            data: Source domain feature matrix.

        Returns:
            Domain-adapted feature matrix.
        """
        if not self._is_fitted:
            return data

        data = np.atleast_2d(data)

        if self.strategy == "coral":
            source_mean = np.mean(self._source_features, axis=0)
            target_mean = np.mean(self._target_features, axis=0)
            centered = data - source_mean
            aligned = centered @ self._alignment_matrix
            return aligned + target_mean

        elif self.strategy == "mmd":
            return data + self._mmd_shift

        elif self.strategy == "normalization":
            normalized = (data - self._source_mean) / self._source_std
            return normalized * self._target_std + self._target_mean

        return data

    def compute_domain_distance(self, source: np.ndarray, target: np.ndarray) -> float:
        """Compute distance between two domains.

        Args:
            source: Source domain features.
            target: Target domain features.

        Returns:
            Scalar distance metric.
        """
        source_mean = np.mean(np.atleast_2d(source), axis=0)
        target_mean = np.mean(np.atleast_2d(target), axis=0)
        return float(np.linalg.norm(source_mean - target_mean))

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted
