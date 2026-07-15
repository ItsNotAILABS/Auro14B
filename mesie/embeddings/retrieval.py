"""Spectral embedding retrieval and nearest-neighbor search."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from mesie.core.records import MultiElementRecord
from mesie.io.loaders import RecordInput, load_record
from mesie.embeddings.vectorizers import SpectralVectorizer


class SpectralRetriever:
    """Retrieve similar spectral records using embedding-based search.

    Maintains an index of record embeddings for nearest-neighbor retrieval.

    Args:
        vectorizer: SpectralVectorizer instance for embedding computation.
    """

    def __init__(self, vectorizer: Optional[SpectralVectorizer] = None) -> None:
        self.vectorizer = vectorizer or SpectralVectorizer()
        self._embeddings: List[np.ndarray] = []
        self._record_ids: List[str] = []

    def index(self, records: Sequence[RecordInput]) -> None:
        """Index records for retrieval.

        Args:
            records: Records to add to the index.
        """
        for r in records:
            rec = load_record(r)
            emb = self.vectorizer.transform(rec)
            self._embeddings.append(emb)
            self._record_ids.append(rec.record_id)

    def query(
        self,
        record: RecordInput,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Find most similar records by embedding distance.

        Args:
            record: Query record.
            top_k: Number of results to return.

        Returns:
            List of (record_id, distance) tuples sorted by distance ascending.
        """
        if not self._embeddings:
            return []

        query_emb = self.vectorizer.transform(record)
        distances = []
        for i, emb in enumerate(self._embeddings):
            dist = float(np.linalg.norm(query_emb - emb))
            distances.append((self._record_ids[i], dist))

        distances.sort(key=lambda x: x[1])
        return distances[:top_k]

    @property
    def size(self) -> int:
        """Number of indexed records."""
        return len(self._embeddings)
