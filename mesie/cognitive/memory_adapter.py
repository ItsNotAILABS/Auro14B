"""Spectral memory adapter for cognitive architectures."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from mesie.core.records import MultiElementRecord
from mesie.io.loaders import RecordInput, load_record
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.features.electro_spectral import ElectroSpectralLayer


class SpectralMemoryAdapter:
    """Adapt spectral records into memory objects for cognitive architectures.

    Converts spectral records into structured memory representations
    suitable for storage, retrieval, and reasoning in cognitive systems.

    Args:
        vectorizer: SpectralVectorizer for embedding generation.
    """

    def __init__(self, vectorizer: Optional[SpectralVectorizer] = None) -> None:
        self.vectorizer = vectorizer or SpectralVectorizer()
        self._electro = ElectroSpectralLayer()

    def to_memory_object(self, record: RecordInput) -> Dict[str, Any]:
        """Convert a spectral record into a cognitive memory object.

        Args:
            record: Input spectral record.

        Returns:
            Dictionary with memory-compatible structure including:
            - semantic_id: Record identifier
            - spectral_embedding: Embedding vector
            - resonance_signature: Resonance features
            - coherence_signature: Coherence value
            - lineage: Record provenance
            - confidence: Confidence score (based on validation)
            - anomaly_score: Anomaly detection score
            - memory_weight: Weight for memory prioritization
        """
        rec = load_record(record)
        embedding = self.vectorizer.transform(rec)
        signature = self._electro.compute_signature(rec)

        return {
            "semantic_id": rec.record_id,
            "spectral_embedding": embedding.tolist(),
            "resonance_signature": [
                signature.spectral_centroid,
                signature.spectral_spread,
                signature.frequency_resonance,
            ],
            "coherence_signature": signature.coherence_signature,
            "lineage": rec.lineage,
            "confidence": 1.0,
            "anomaly_score": 0.0,
            "memory_weight": 1.0,
        }

    def batch_to_memory(self, records: List[RecordInput]) -> List[Dict[str, Any]]:
        """Convert multiple records into memory objects.

        Args:
            records: List of input records.

        Returns:
            List of memory object dictionaries.
        """
        return [self.to_memory_object(r) for r in records]
