"""On-device spectral fingerprint library with live updates.

Enables continuous learning and updating of fingerprint libraries
on-device without cloud dependency. Supports streaming pipelines
for real-time fingerprint acquisition and matching.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class UpdateMode(Enum):
    """Fingerprint library update strategies."""

    APPEND = "append"
    ROLLING = "rolling"
    WEIGHTED_MERGE = "weighted_merge"
    SELECTIVE = "selective"


class MatchConfidence(Enum):
    """Confidence levels for fingerprint matches."""

    EXACT = "exact"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NO_MATCH = "no_match"


@dataclass
class FingerprintEntry:
    """A spectral fingerprint in the library.

    Attributes:
        fingerprint_id: Unique fingerprint identifier.
        label: Human-readable label/class name.
        spectral_vector: Normalized spectral representation.
        frequency_range: (min_hz, max_hz) range covered.
        sample_count: Number of samples that built this fingerprint.
        confidence: Confidence in this fingerprint's accuracy.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        metadata: Additional fingerprint context.
    """

    label: str
    spectral_vector: np.ndarray
    fingerprint_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    frequency_range: Tuple[float, float] = (0.0, 1000.0)
    sample_count: int = 1
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LibraryConfig:
    """Configuration for the fingerprint library.

    Args:
        max_entries: Maximum fingerprints stored on device.
        update_mode: How new observations update existing entries.
        match_threshold: Cosine similarity threshold for matching.
        vector_dim: Dimension of spectral fingerprint vectors.
        enable_compression: Whether to compress stored vectors.
        decay_rate: Exponential decay for old fingerprints (0=no decay).
    """

    max_entries: int = 10000
    update_mode: UpdateMode = UpdateMode.WEIGHTED_MERGE
    match_threshold: float = 0.85
    vector_dim: int = 128
    enable_compression: bool = False
    decay_rate: float = 0.0


class FingerprintLibrary:
    """On-device spectral fingerprint library with live streaming updates.

    Maintains a searchable library of spectral fingerprints that can be
    continuously updated from live sensor streams without cloud connectivity.
    Supports O(n) linear scan matching optimized for edge hardware.

    Args:
        config: Library configuration.
    """

    def __init__(self, config: Optional[LibraryConfig] = None) -> None:
        self.config = config or LibraryConfig()
        self._entries: OrderedDict[str, FingerprintEntry] = OrderedDict()
        self._label_index: Dict[str, List[str]] = {}
        self._total_queries = 0
        self._total_updates = 0

    def add(
        self,
        label: str,
        spectral_vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FingerprintEntry:
        """Add a new fingerprint to the library.

        If library is at capacity, the oldest entry is evicted (LRU).

        Args:
            label: Classification label for this fingerprint.
            spectral_vector: Normalized spectral vector.
            metadata: Optional metadata.

        Returns:
            The created FingerprintEntry.
        """
        vector = self._normalize_vector(spectral_vector)

        entry = FingerprintEntry(
            label=label,
            spectral_vector=vector,
            metadata=metadata or {},
        )

        # Evict oldest if at capacity
        if len(self._entries) >= self.config.max_entries:
            oldest_id = next(iter(self._entries))
            self._remove_from_index(oldest_id)
            del self._entries[oldest_id]

        self._entries[entry.fingerprint_id] = entry
        self._label_index.setdefault(label, []).append(entry.fingerprint_id)

        return entry

    def match(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[FingerprintEntry, float]]:
        """Find the closest matching fingerprints.

        Performs cosine similarity search across the library.

        Args:
            query_vector: Query spectral vector.
            top_k: Number of top matches to return.

        Returns:
            List of (entry, similarity_score) tuples, descending by score.
        """
        self._total_queries += 1
        query = self._normalize_vector(query_vector)

        scores: List[Tuple[FingerprintEntry, float]] = []
        for entry in self._entries.values():
            sim = self._cosine_similarity(query, entry.spectral_vector)
            if sim >= self.config.match_threshold:
                scores.append((entry, float(sim)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def classify(self, query_vector: np.ndarray) -> Tuple[str, MatchConfidence, float]:
        """Classify a spectral vector against the library.

        Args:
            query_vector: Input spectral vector to classify.

        Returns:
            Tuple of (label, confidence_level, similarity_score).
        """
        matches = self.match(query_vector, top_k=1)
        if not matches:
            return ("unknown", MatchConfidence.NO_MATCH, 0.0)

        entry, score = matches[0]
        confidence = self._score_to_confidence(score)
        return (entry.label, confidence, score)

    def update_from_stream(
        self,
        label: str,
        spectral_vector: np.ndarray,
    ) -> FingerprintEntry:
        """Update library from a streaming observation.

        Uses the configured update mode to incorporate new data:
        - APPEND: Always add as new entry
        - ROLLING: Replace oldest entry with same label
        - WEIGHTED_MERGE: Merge with closest same-label entry
        - SELECTIVE: Only update if significantly different

        Args:
            label: Label for the observation.
            spectral_vector: Observed spectral vector.

        Returns:
            The updated or newly created entry.
        """
        self._total_updates += 1
        vector = self._normalize_vector(spectral_vector)

        if self.config.update_mode == UpdateMode.APPEND:
            return self.add(label, vector)

        # Find existing entries with same label
        existing_ids = self._label_index.get(label, [])
        if not existing_ids:
            return self.add(label, vector)

        if self.config.update_mode == UpdateMode.ROLLING:
            # Replace oldest same-label entry
            oldest_id = existing_ids[0]
            entry = self._entries[oldest_id]
            entry.spectral_vector = vector
            entry.sample_count += 1
            entry.updated_at = time.time()
            return entry

        if self.config.update_mode == UpdateMode.WEIGHTED_MERGE:
            # Find closest same-label entry and merge
            best_id = None
            best_sim = -1.0
            for eid in existing_ids:
                entry = self._entries[eid]
                sim = self._cosine_similarity(vector, entry.spectral_vector)
                if sim > best_sim:
                    best_sim = sim
                    best_id = eid

            if best_id is not None and best_sim > self.config.match_threshold:
                entry = self._entries[best_id]
                # Weighted average based on sample count
                w = entry.sample_count / (entry.sample_count + 1)
                entry.spectral_vector = self._normalize_vector(
                    w * entry.spectral_vector + (1 - w) * vector
                )
                entry.sample_count += 1
                entry.updated_at = time.time()
                return entry
            return self.add(label, vector)

        if self.config.update_mode == UpdateMode.SELECTIVE:
            # Only add if sufficiently novel
            for eid in existing_ids:
                entry = self._entries[eid]
                sim = self._cosine_similarity(vector, entry.spectral_vector)
                if sim > self.config.match_threshold:
                    # Not novel enough, skip
                    return entry
            return self.add(label, vector)

        return self.add(label, vector)

    def get_by_label(self, label: str) -> List[FingerprintEntry]:
        """Get all fingerprints with a given label.

        Args:
            label: Label to filter by.

        Returns:
            List of matching entries.
        """
        ids = self._label_index.get(label, [])
        return [self._entries[eid] for eid in ids if eid in self._entries]

    def export_vectors(self) -> Dict[str, np.ndarray]:
        """Export all fingerprint vectors as a dictionary.

        Returns:
            Dict mapping fingerprint_id to spectral_vector.
        """
        return {
            eid: entry.spectral_vector.copy()
            for eid, entry in self._entries.items()
        }

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length, resizing if needed."""
        vector = np.asarray(vector, dtype=np.float64).ravel()
        if len(vector) != self.config.vector_dim:
            # Resize via interpolation
            x_old = np.linspace(0, 1, len(vector))
            x_new = np.linspace(0, 1, self.config.vector_dim)
            vector = np.interp(x_new, x_old, vector)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two unit vectors."""
        return float(np.dot(a, b))

    def _score_to_confidence(self, score: float) -> MatchConfidence:
        """Map similarity score to confidence level."""
        if score >= 0.98:
            return MatchConfidence.EXACT
        elif score >= 0.92:
            return MatchConfidence.HIGH
        elif score >= 0.85:
            return MatchConfidence.MODERATE
        elif score >= 0.7:
            return MatchConfidence.LOW
        return MatchConfidence.NO_MATCH

    def _remove_from_index(self, fingerprint_id: str) -> None:
        """Remove a fingerprint ID from the label index."""
        entry = self._entries.get(fingerprint_id)
        if entry and entry.label in self._label_index:
            ids = self._label_index[entry.label]
            if fingerprint_id in ids:
                ids.remove(fingerprint_id)
            if not ids:
                del self._label_index[entry.label]

    @property
    def size(self) -> int:
        """Number of fingerprints in the library."""
        return len(self._entries)

    @property
    def labels(self) -> List[str]:
        """All unique labels in the library."""
        return list(self._label_index.keys())

    @property
    def stats(self) -> Dict[str, Any]:
        """Library usage statistics."""
        return {
            "size": self.size,
            "labels": len(self._label_index),
            "total_queries": self._total_queries,
            "total_updates": self._total_updates,
            "capacity_used": self.size / self.config.max_entries,
        }
