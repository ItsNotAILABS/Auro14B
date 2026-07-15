"""Spectral vectorization for embedding generation."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.io.loaders import RecordInput, load_record
from mesie.features.electro_spectral import ElectroSpectralLayer


class SpectralVectorizer:
    """Convert spectral records into fixed-size embedding vectors.

    Extracts statistical and spectral features to produce vectors suitable
    for machine learning, clustering, retrieval, and cognitive systems.

    Args:
        n_bands: Number of frequency bands for band energy features.
        include_statistics: Whether to include statistical features.
        include_spectral: Whether to include spectral features.
    """

    def __init__(
        self,
        n_bands: int = 8,
        include_statistics: bool = True,
        include_spectral: bool = True,
    ) -> None:
        self.n_bands = n_bands
        self.include_statistics = include_statistics
        self.include_spectral = include_spectral
        self._electro = ElectroSpectralLayer()

    def _compute_features(self, record: MultiElementRecord) -> np.ndarray:
        """Extract feature vector from a record."""
        features: List[float] = []

        if not record.components:
            return np.zeros(self._expected_dim())

        # Aggregate amplitude
        comp = record.components[0]
        freq = comp.frequency
        amp = np.abs(comp.amplitude)

        if self.include_statistics:
            features.extend([
                float(np.mean(amp)),
                float(np.std(amp)),
                float(np.max(amp)),
                float(np.min(amp)),
                float(np.median(amp)),
            ])

        if self.include_spectral:
            total = max(float(np.sum(amp)), 1e-12)
            centroid = float(np.sum(freq * amp) / total)
            spread = float(np.sqrt(np.sum(((freq - centroid) ** 2) * amp) / total))
            peak_freq = float(freq[np.argmax(amp)])

            # Spectral entropy
            p = amp / total
            p = p[p > 0]
            entropy = float(-np.sum(p * np.log(p + 1e-12)))

            features.extend([centroid, spread, peak_freq, entropy])

            # Band energies
            if len(freq) > 1:
                band_edges = np.linspace(freq[0], freq[-1], self.n_bands + 1)
                for i in range(self.n_bands):
                    mask = (freq >= band_edges[i]) & (freq < band_edges[i + 1])
                    features.append(float(np.sum(amp[mask] ** 2)))
            else:
                features.extend([0.0] * self.n_bands)

        return np.array(features, dtype=float)

    def _expected_dim(self) -> int:
        """Return expected feature vector dimensionality."""
        dim = 0
        if self.include_statistics:
            dim += 5
        if self.include_spectral:
            dim += 4 + self.n_bands
        return dim

    @property
    def embedding_dim(self) -> int:
        """Embedding vector dimensionality."""
        return self._expected_dim()

    def transform(self, record: RecordInput) -> np.ndarray:
        """Transform a single record into an embedding vector.

        Args:
            record: Input spectral record.

        Returns:
            1D numpy array embedding.
        """
        rec = load_record(record)
        return self._compute_features(rec)

    def fit_transform(self, record: RecordInput) -> np.ndarray:
        """Fit and transform a single record (alias for transform).

        Args:
            record: Input spectral record.

        Returns:
            1D numpy array embedding.
        """
        return self.transform(record)

    def batch_transform(self, records: Sequence[RecordInput]) -> np.ndarray:
        """Transform multiple records into an embedding matrix.

        Args:
            records: Sequence of input records.

        Returns:
            2D numpy array of shape (n_records, embedding_dim).
        """
        vectors = [self.transform(r) for r in records]
        return np.vstack(vectors)
