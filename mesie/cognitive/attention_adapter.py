"""Spectral attention adapter for cognitive architectures."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mesie.core.records import MultiElementRecord
from mesie.io.loaders import RecordInput, load_record
from mesie.embeddings.vectorizers import SpectralVectorizer


class SpectralAttentionAdapter:
    """Adapt spectral features into attention weights for cognitive systems.

    Computes attention-compatible weights from spectral properties,
    allowing cognitive architectures to focus on relevant spectral features.

    Args:
        vectorizer: SpectralVectorizer for embedding computation.
    """

    def __init__(self, vectorizer: Optional[SpectralVectorizer] = None) -> None:
        self.vectorizer = vectorizer or SpectralVectorizer()

    def compute_attention_weights(
        self,
        records: Sequence[RecordInput],
        query: Optional[RecordInput] = None,
    ) -> np.ndarray:
        """Compute attention weights over a set of records.

        If a query is provided, weights are based on similarity to query.
        Otherwise, weights are based on spectral energy/importance.

        Args:
            records: Sequence of records to weight.
            query: Optional query record for similarity-based attention.

        Returns:
            1D array of attention weights summing to 1.
        """
        if not records:
            return np.array([], dtype=float)

        embeddings = np.vstack([self.vectorizer.transform(r) for r in records])

        if query is not None:
            query_emb = self.vectorizer.transform(query)
            # Compute similarity-based attention
            similarities = np.array([
                float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-12))
                for emb in embeddings
            ])
            # Softmax
            exp_sim = np.exp(similarities - np.max(similarities))
            weights = exp_sim / (np.sum(exp_sim) + 1e-12)
        else:
            # Energy-based attention
            norms = np.linalg.norm(embeddings, axis=1)
            weights = norms / (np.sum(norms) + 1e-12)

        return weights

    def to_attention_dict(
        self,
        records: Sequence[RecordInput],
        query: Optional[RecordInput] = None,
    ) -> Dict[str, float]:
        """Compute named attention weights for records.

        Args:
            records: Records to weight.
            query: Optional query record.

        Returns:
            Dictionary mapping record IDs to attention weights.
        """
        loaded = [load_record(r) for r in records]
        weights = self.compute_attention_weights(records, query)
        return {rec.record_id: float(w) for rec, w in zip(loaded, weights)}
