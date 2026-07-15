"""Agent state and anomaly detection adapters for cognitive architectures."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mesie.core.records import MultiElementRecord
from mesie.io.loaders import RecordInput, load_record
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.features.electro_spectral import ElectroSpectralLayer


class AgentStateSpectralAdapter:
    """Adapt spectral records into agent state representations.

    Converts spectral data into state vectors that agents can use
    for self-monitoring, comparison, and decision-making.

    Args:
        vectorizer: SpectralVectorizer for embedding computation.
    """

    def __init__(self, vectorizer: Optional[SpectralVectorizer] = None) -> None:
        self.vectorizer = vectorizer or SpectralVectorizer()
        self._electro = ElectroSpectralLayer()

    def to_state_vector(self, record: RecordInput) -> np.ndarray:
        """Convert a record into an agent state vector.

        Args:
            record: Input spectral record.

        Returns:
            State vector combining embedding and spectral features.
        """
        rec = load_record(record)
        embedding = self.vectorizer.transform(rec)
        sig = self._electro.compute_signature(rec)

        state_features = np.array([
            sig.spectral_centroid,
            sig.spectral_spread,
            sig.frequency_resonance,
            sig.coherence_signature,
            sig.harmonic_alignment,
        ], dtype=float)

        return np.concatenate([embedding, state_features])

    def compare_states(self, state_a: RecordInput, state_b: RecordInput) -> float:
        """Compare two agent states via spectral similarity.

        Args:
            state_a: First state record.
            state_b: Second state record.

        Returns:
            Similarity score in [0, 1].
        """
        va = self.to_state_vector(state_a)
        vb = self.to_state_vector(state_b)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def to_state_dict(self, record: RecordInput) -> Dict[str, Any]:
        """Convert a record into a structured agent state dictionary.

        Args:
            record: Input spectral record.

        Returns:
            Dictionary with state information.
        """
        rec = load_record(record)
        sig = self._electro.compute_signature(rec)
        embedding = self.vectorizer.transform(rec)

        return {
            "record_id": rec.record_id,
            "state_vector": embedding.tolist(),
            "centroid": sig.spectral_centroid,
            "spread": sig.spectral_spread,
            "resonance": sig.frequency_resonance,
            "coherence": sig.coherence_signature,
            "representation": rec.representation,
        }


class SpectralAnomalyAdapter:
    """Detect spectral anomalies for cognitive architecture alerting.

    Maintains a baseline of normal spectral patterns and flags
    deviations as anomalies.

    Args:
        vectorizer: SpectralVectorizer for embedding computation.
        threshold: Distance threshold for anomaly detection.
    """

    def __init__(
        self,
        vectorizer: Optional[SpectralVectorizer] = None,
        threshold: float = 2.0,
    ) -> None:
        self.vectorizer = vectorizer or SpectralVectorizer()
        self.threshold = threshold
        self._baseline_embeddings: List[np.ndarray] = []
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_std: Optional[float] = None

    def fit_baseline(self, records: Sequence[RecordInput]) -> None:
        """Fit anomaly detector with baseline (normal) records.

        Args:
            records: Sequence of normal/baseline records.
        """
        self._baseline_embeddings = [self.vectorizer.transform(r) for r in records]
        if self._baseline_embeddings:
            matrix = np.vstack(self._baseline_embeddings)
            self._baseline_mean = np.mean(matrix, axis=0)
            distances = np.linalg.norm(matrix - self._baseline_mean, axis=1)
            self._baseline_std = float(np.std(distances)) if len(distances) > 1 else 1.0

    def score_anomaly(self, record: RecordInput) -> float:
        """Compute anomaly score for a record.

        Args:
            record: Input record to evaluate.

        Returns:
            Anomaly score (higher = more anomalous). Returns 0.0 if no baseline.
        """
        if self._baseline_mean is None:
            return 0.0
        embedding = self.vectorizer.transform(record)
        distance = float(np.linalg.norm(embedding - self._baseline_mean))
        std = max(self._baseline_std or 1.0, 1e-12)
        return distance / std

    def is_anomaly(self, record: RecordInput) -> bool:
        """Check if a record is anomalous relative to baseline.

        Args:
            record: Input record to evaluate.

        Returns:
            True if anomaly score exceeds threshold.
        """
        return self.score_anomaly(record) > self.threshold
