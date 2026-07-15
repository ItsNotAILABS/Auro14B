"""Spectral memory store with lineage-aware retrieval.

Implements a compressed spectral history and retrieval mechanism that enables
agents to query: "Have I seen a pattern like this before?" using k-NN or
learned retrieval over embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MemoryEntry:
    """A single entry in the spectral memory store.

    Attributes
    ----------
    timestamp : float
        Time index when this memory was recorded.
    embedding : ndarray, shape (embed_dim,)
        Compressed spectral embedding at this time.
    event_type : str
        Type of event associated with this memory (e.g., 'resonance',
        'drift', 'anomaly', 'normal').
    metadata : dict
        Additional metadata (severity, entity_id, etc.).
    importance : float
        Computed importance score for memory prioritization.
    """

    timestamp: float
    embedding: np.ndarray
    event_type: str = "normal"
    metadata: Dict = field(default_factory=dict)
    importance: float = 1.0


@dataclass
class LineageQuery:
    """A query against the spectral memory store.

    Attributes
    ----------
    query_embedding : ndarray, shape (embed_dim,)
        Current embedding to find matches for.
    top_k : int
        Number of nearest memories to retrieve.
    event_filter : str or None
        Optional filter by event type.
    time_range : tuple or None
        Optional (start, end) time range filter.
    """

    query_embedding: np.ndarray
    top_k: int = 5
    event_filter: Optional[str] = None
    time_range: Optional[Tuple[float, float]] = None


@dataclass
class RetrievalResult:
    """Result from a lineage query against spectral memory.

    Attributes
    ----------
    entries : list of MemoryEntry
        Retrieved memory entries, sorted by relevance.
    distances : ndarray
        Distance/similarity scores for each retrieved entry.
    context_embedding : ndarray or None
        Aggregated context embedding from retrieved memories.
    """

    entries: List[MemoryEntry]
    distances: np.ndarray
    context_embedding: Optional[np.ndarray] = None


class SpectralMemoryStore:
    """Compressed spectral history with lineage-aware retrieval.

    Maintains a memory store M = {(t_i, z_{t_i}, event_i)} that enables
    agents to reason about temporal lineage by querying past spectral states.

    Supports:
    - k-NN retrieval over embeddings
    - Event-type filtering
    - Time-range filtering
    - Importance-weighted retrieval
    - Memory consolidation (compression of old memories)

    Parameters
    ----------
    capacity : int
        Maximum number of entries to store.
    embed_dim : int
        Dimensionality of stored embeddings.
    consolidation_threshold : int
        When capacity is reached, consolidate oldest entries.
    distance_metric : str
        Metric for retrieval: 'l2', 'cosine', 'dot'.
    importance_decay : float
        Exponential decay rate for importance over time.
    """

    def __init__(
        self,
        capacity: int = 10000,
        embed_dim: int = 64,
        consolidation_threshold: int = 8000,
        distance_metric: str = "cosine",
        importance_decay: float = 0.999,
    ):
        self.capacity = capacity
        self.embed_dim = embed_dim
        self.consolidation_threshold = consolidation_threshold
        self.distance_metric = distance_metric
        self.importance_decay = importance_decay

        self._entries: List[MemoryEntry] = []
        self._embedding_index: Optional[np.ndarray] = None
        self._index_dirty: bool = True

    @property
    def size(self) -> int:
        """Current number of entries in the store."""
        return len(self._entries)

    def store(
        self,
        timestamp: float,
        embedding: np.ndarray,
        event_type: str = "normal",
        metadata: Optional[Dict] = None,
        importance: Optional[float] = None,
    ) -> None:
        """Store a new spectral memory entry.

        Parameters
        ----------
        timestamp : float
            Time index for this memory.
        embedding : ndarray, shape (embed_dim,)
            Spectral embedding to store.
        event_type : str
            Event type label.
        metadata : dict or None
            Additional metadata.
        importance : float or None
            Explicit importance score. If None, computed automatically.
        """
        embedding = np.asarray(embedding, dtype=np.float64).ravel()

        if importance is None:
            importance = self._compute_importance(embedding, event_type)

        entry = MemoryEntry(
            timestamp=timestamp,
            embedding=embedding,
            event_type=event_type,
            metadata=metadata or {},
            importance=importance,
        )

        self._entries.append(entry)
        self._index_dirty = True

        # Consolidate if over capacity
        if len(self._entries) > self.capacity:
            self._consolidate()

    def query(self, query: LineageQuery) -> RetrievalResult:
        """Query the memory store for similar past spectral states.

        Parameters
        ----------
        query : LineageQuery
            Query specification.

        Returns
        -------
        RetrievalResult
            Retrieved entries with distances and context.
        """
        if len(self._entries) == 0:
            return RetrievalResult(
                entries=[],
                distances=np.array([]),
                context_embedding=None,
            )

        # Apply filters
        candidates = self._apply_filters(query)

        if len(candidates) == 0:
            return RetrievalResult(
                entries=[],
                distances=np.array([]),
                context_embedding=None,
            )

        # Compute distances
        query_emb = np.asarray(query.query_embedding, dtype=np.float64).ravel()
        candidate_embeddings = np.array([c.embedding for c in candidates])
        distances = self._compute_distances(query_emb, candidate_embeddings)

        # Sort by distance (ascending = most similar first)
        sorted_indices = np.argsort(distances)
        top_k = min(query.top_k, len(sorted_indices))
        top_indices = sorted_indices[:top_k]

        retrieved_entries = [candidates[i] for i in top_indices]
        retrieved_distances = distances[top_indices]

        # Compute aggregated context embedding
        context_embedding = self._aggregate_context(retrieved_entries, retrieved_distances)

        return RetrievalResult(
            entries=retrieved_entries,
            distances=retrieved_distances,
            context_embedding=context_embedding,
        )

    def query_simple(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        event_filter: Optional[str] = None,
    ) -> RetrievalResult:
        """Simplified query interface.

        Parameters
        ----------
        embedding : ndarray
            Query embedding.
        top_k : int
            Number of results.
        event_filter : str or None
            Optional event type filter.

        Returns
        -------
        RetrievalResult
        """
        return self.query(
            LineageQuery(
                query_embedding=embedding,
                top_k=top_k,
                event_filter=event_filter,
            )
        )

    def get_lineage(
        self, current_embedding: np.ndarray, window: int = 10
    ) -> np.ndarray:
        """Get a lineage-conditioned representation.

        Combines the current embedding with retrieved past context for
        policy conditioning.

        Parameters
        ----------
        current_embedding : ndarray, shape (embed_dim,)
            Current spectral embedding z_t.
        window : int
            Number of past memories to retrieve.

        Returns
        -------
        ndarray, shape (2 * embed_dim,)
            Concatenation of current embedding and aggregated past context.
        """
        current_embedding = np.asarray(current_embedding, dtype=np.float64).ravel()

        result = self.query_simple(current_embedding, top_k=window)

        if result.context_embedding is not None:
            return np.concatenate([current_embedding, result.context_embedding])
        else:
            return np.concatenate([current_embedding, np.zeros_like(current_embedding)])

    def decay_importance(self) -> None:
        """Apply importance decay to all entries."""
        for entry in self._entries:
            entry.importance *= self.importance_decay

    def get_event_counts(self) -> Dict[str, int]:
        """Get counts of each event type in memory."""
        counts: Dict[str, int] = {}
        for entry in self._entries:
            counts[entry.event_type] = counts.get(entry.event_type, 0) + 1
        return counts

    def clear(self) -> None:
        """Clear all entries from the memory store."""
        self._entries = []
        self._embedding_index = None
        self._index_dirty = True

    def _compute_importance(self, embedding: np.ndarray, event_type: str) -> float:
        """Compute importance score for a new entry."""
        base_importance = 1.0

        # Higher importance for non-normal events
        event_weights = {
            "normal": 1.0,
            "resonance": 3.0,
            "drift": 2.5,
            "anomaly": 4.0,
            "harmonic_shift": 2.0,
        }
        base_importance *= event_weights.get(event_type, 1.0)

        # Higher importance for entries dissimilar to recent history
        if len(self._entries) > 0:
            recent = self._entries[-min(10, len(self._entries)):]
            recent_embeddings = np.array([e.embedding for e in recent])
            distances = self._compute_distances(embedding, recent_embeddings)
            novelty = float(np.mean(distances))
            base_importance *= (1.0 + novelty)

        return base_importance

    def _apply_filters(self, query: LineageQuery) -> List[MemoryEntry]:
        """Apply query filters to entries."""
        candidates = self._entries

        if query.event_filter is not None:
            candidates = [
                e for e in candidates if e.event_type == query.event_filter
            ]

        if query.time_range is not None:
            start, end = query.time_range
            candidates = [
                e for e in candidates if start <= e.timestamp <= end
            ]

        return candidates

    def _compute_distances(
        self, query: np.ndarray, candidates: np.ndarray
    ) -> np.ndarray:
        """Compute distances between query and candidate embeddings."""
        if self.distance_metric == "l2":
            return np.sqrt(np.sum((candidates - query) ** 2, axis=1))
        elif self.distance_metric == "cosine":
            query_norm = np.linalg.norm(query)
            cand_norms = np.linalg.norm(candidates, axis=1)
            if query_norm < 1e-10:
                return np.ones(len(candidates))
            dots = candidates @ query
            similarities = dots / (cand_norms * query_norm + 1e-10)
            return 1.0 - similarities  # Convert to distance
        elif self.distance_metric == "dot":
            dots = candidates @ query
            return -dots  # Negate so higher dot = lower distance
        else:
            return np.sqrt(np.sum((candidates - query) ** 2, axis=1))

    def _aggregate_context(
        self, entries: List[MemoryEntry], distances: np.ndarray
    ) -> Optional[np.ndarray]:
        """Aggregate retrieved entries into a context embedding."""
        if len(entries) == 0:
            return None

        embeddings = np.array([e.embedding for e in entries])

        # Inverse-distance weighting
        weights = 1.0 / (distances + 1e-8)
        weights = weights / np.sum(weights)

        context = np.average(embeddings, axis=0, weights=weights)
        return context

    def _consolidate(self) -> None:
        """Consolidate memory when over capacity.

        Removes lowest-importance entries while preserving event diversity.
        """
        if len(self._entries) <= self.consolidation_threshold:
            return

        # Sort by importance (keep highest)
        sorted_entries = sorted(
            self._entries, key=lambda e: e.importance, reverse=True
        )

        # Keep top entries up to consolidation threshold
        self._entries = sorted_entries[: self.consolidation_threshold]
        self._index_dirty = True
