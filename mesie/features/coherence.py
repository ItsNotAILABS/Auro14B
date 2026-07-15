"""Coherence computation for multi-component records."""

from __future__ import annotations

from typing import List

import numpy as np

from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.processing.interpolate import interpolate_component


def compute_coherence(record: MultiElementRecord) -> float:
    """Compute inter-component coherence for a multi-component record.

    Coherence measures how consistently components behave across frequency.
    Higher coherence = more similar component shapes.

    Args:
        record: Input multi-element record.

    Returns:
        Coherence score in (0, 1]. Returns 1.0 for single-component records.
    """
    if len(record.components) <= 1:
        return 1.0

    base_grid = record.components[0].frequency
    stack = np.vstack([
        interpolate_component(c, base_grid).amplitude for c in record.components
    ])
    return float(1.0 / (1.0 + np.mean(np.std(stack, axis=0))))


def compute_pairwise_coherence(record: MultiElementRecord) -> List[List[float]]:
    """Compute pairwise coherence between all component pairs.

    Args:
        record: Input multi-element record.

    Returns:
        2D matrix of pairwise coherence values.
    """
    n = len(record.components)
    if n == 0:
        return []

    base_grid = record.components[0].frequency
    amplitudes = [
        interpolate_component(c, base_grid).amplitude for c in record.components
    ]

    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                denom = float(np.linalg.norm(amplitudes[i]) * np.linalg.norm(amplitudes[j]))
                if denom > 1e-12:
                    matrix[i][j] = float(np.dot(amplitudes[i], amplitudes[j]) / denom)
    return matrix
