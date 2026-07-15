"""Spectral distance and similarity metrics."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def spectral_rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute root mean squared error between two amplitude arrays.

    Args:
        reference: Reference amplitude array.
        candidate: Candidate amplitude array.

    Returns:
        RMSE value.
    """
    diff = np.asarray(reference, dtype=float) - np.asarray(candidate, dtype=float)
    return float(np.sqrt(np.mean(diff ** 2)))


def spectral_mae(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute mean absolute error between two amplitude arrays.

    Args:
        reference: Reference amplitude array.
        candidate: Candidate amplitude array.

    Returns:
        MAE value.
    """
    diff = np.asarray(reference, dtype=float) - np.asarray(candidate, dtype=float)
    return float(np.mean(np.abs(diff)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def log_spectral_distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute log spectral distance between two amplitude arrays.

    Args:
        reference: Reference amplitude array.
        candidate: Candidate amplitude array.

    Returns:
        Mean absolute log difference.
    """
    eps = 1e-12
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    return float(np.mean(np.abs(np.log(np.abs(ref) + eps) - np.log(np.abs(cand) + eps))))


def band_weighted_error(
    frequency: np.ndarray,
    error: np.ndarray,
    band_weights: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> float:
    """Compute frequency-band weighted error.

    Args:
        frequency: Frequency array.
        error: Absolute error array.
        band_weights: List of (low_freq, high_freq, weight) tuples.

    Returns:
        Weighted mean error across bands.
    """
    if not band_weights:
        return float(np.mean(error))
    weighted_sum = 0.0
    total_weight = 0.0
    for low, high, weight in band_weights:
        mask = (frequency >= low) & (frequency < high)
        if not np.any(mask):
            continue
        w = max(float(weight), 0.0)
        weighted_sum += float(np.mean(error[mask])) * w
        total_weight += w
    if total_weight <= 0:
        return float(np.mean(error))
    return weighted_sum / total_weight


def coherence_similarity(components_a: np.ndarray, components_b: np.ndarray) -> float:
    """Compute coherence similarity between multi-component stacks.

    Args:
        components_a: 2D array (n_components x n_frequencies) for record A.
        components_b: 2D array (n_components x n_frequencies) for record B.

    Returns:
        Coherence similarity score in [0, 1].
    """
    if components_a.ndim != 2 or components_b.ndim != 2:
        return 0.0
    mean_a = np.mean(components_a, axis=0)
    mean_b = np.mean(components_b, axis=0)
    return cosine_similarity(mean_a, mean_b)
