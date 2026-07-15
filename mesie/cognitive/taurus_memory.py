"""TAURUS — Temporal Adaptive Retrieval and Unified Storage.

A persistent, attention-weighted spectral memory system for cognitive
architectures. TAURUS provides long-term and working memory with
temporal decay, importance-based consolidation, and attention-driven
retrieval for spectral intelligence systems.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class MemoryTrace:
    """A single memory trace stored in TAURUS.

    Args:
        embedding: Vector representation of the stored spectral data.
        context: Metadata and contextual information.
        timestamp: Time of storage (epoch seconds).
        importance: Importance weight for retention priority.
        access_count: Number of times this trace has been retrieved.
        last_accessed: Timestamp of most recent retrieval.
        decay_rate: Rate at which this memory fades over time.
        semantic_tag: Optional semantic label for categorization.
    """

    embedding: np.ndarray
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    decay_rate: float = 0.01
    semantic_tag: str = ""

    def effective_strength(self, current_time: Optional[float] = None) -> float:
        """Compute the effective strength of this memory trace.

        Combines importance, recency, and access frequency into a
        unified strength score subject to temporal decay.

        Args:
            current_time: Reference time for decay calculation.

        Returns:
            Effective strength score in [0, inf).
        """
        now = current_time or time.time()
        age = max(0.0, now - self.timestamp)
        recency = max(0.0, now - self.last_accessed)

        # Temporal decay
        temporal_factor = np.exp(-self.decay_rate * age)
        # Recency boost
        recency_factor = np.exp(-0.001 * recency)
        # Access frequency boost (logarithmic)
        frequency_factor = 1.0 + np.log1p(self.access_count)

        return float(self.importance * temporal_factor * recency_factor * frequency_factor)


@dataclass
class RetrievalResult:
    """Result from a TAURUS memory retrieval operation.

    Args:
        trace: The retrieved memory trace.
        similarity: Cosine similarity to the query.
        effective_strength: Current effective strength of the trace.
        rank: Position in the retrieval results.
    """

    trace: MemoryTrace
    similarity: float
    effective_strength: float
    rank: int


class TaurusMemoryStore:
    """TAURUS long-term spectral memory store.

    Provides persistent storage with temporal decay, attention-weighted
    retrieval, and importance-based consolidation. Designed as the
    primary memory backend for cognitive spectral systems.

    Args:
        capacity: Maximum number of memory traces.
        consolidation_threshold: Minimum strength for trace retention.
        attention_temperature: Softmax temperature for attention-based retrieval.
    """

    def __init__(
        self,
        capacity: int = 1000,
        consolidation_threshold: float = 0.1,
        attention_temperature: float = 1.0,
    ) -> None:
        self.capacity = capacity
        self.consolidation_threshold = consolidation_threshold
        self.attention_temperature = attention_temperature
        self._traces: List[MemoryTrace] = []
        self._store_count: int = 0
        self._retrieval_count: int = 0

    def store(
        self,
        embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
        semantic_tag: str = "",
        decay_rate: float = 0.01,
    ) -> MemoryTrace:
        """Store a new memory trace in TAURUS.

        Args:
            embedding: Vector representation to store.
            context: Optional contextual metadata.
            importance: Importance weight for retention.
            semantic_tag: Optional semantic label.
            decay_rate: Temporal decay rate for this trace.

        Returns:
            The created MemoryTrace.
        """
        embedding = np.atleast_1d(embedding).flatten().astype(float)

        trace = MemoryTrace(
            embedding=embedding,
            context=context or {},
            importance=importance,
            semantic_tag=semantic_tag,
            decay_rate=decay_rate,
        )

        self._traces.append(trace)
        self._store_count += 1

        # Consolidation: evict weakest traces if over capacity
        if len(self._traces) > self.capacity:
            self._consolidate()

        return trace

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 5,
        semantic_filter: Optional[str] = None,
        min_strength: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Retrieve memory traces by similarity with attention weighting.

        Combines cosine similarity with effective strength to rank
        results. Uses attention temperature to control the sharpness
        of the retrieval distribution.

        Args:
            query: Query embedding vector.
            top_k: Maximum number of results to return.
            semantic_filter: Optional filter by semantic tag.
            min_strength: Minimum effective strength for inclusion.

        Returns:
            List of RetrievalResult ordered by relevance.
        """
        if not self._traces:
            return []

        self._retrieval_count += 1
        query = np.atleast_1d(query).flatten().astype(float)
        query_norm = np.linalg.norm(query) + 1e-12
        current_time = time.time()
        min_str = min_strength or self.consolidation_threshold

        candidates: List[tuple] = []
        for trace in self._traces:
            # Apply semantic filter
            if semantic_filter and trace.semantic_tag != semantic_filter:
                continue

            strength = trace.effective_strength(current_time)
            if strength < min_str:
                continue

            # Cosine similarity
            trace_norm = np.linalg.norm(trace.embedding) + 1e-12
            min_len = min(len(query), len(trace.embedding))
            similarity = float(
                np.dot(query[:min_len], trace.embedding[:min_len])
                / (query_norm * trace_norm)
            )

            # Combined score: similarity weighted by strength
            score = similarity * strength / self.attention_temperature
            candidates.append((trace, similarity, strength, score))

        # Sort by combined score descending
        candidates.sort(key=lambda x: x[3], reverse=True)

        results = []
        for rank, (trace, sim, strength, _) in enumerate(candidates[:top_k]):
            # Update access metadata
            trace.access_count += 1
            trace.last_accessed = current_time
            results.append(
                RetrievalResult(
                    trace=trace,
                    similarity=sim,
                    effective_strength=strength,
                    rank=rank,
                )
            )

        return results

    def retrieve_by_attention(
        self,
        query: np.ndarray,
        attention_weights: np.ndarray,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieve memory traces using external attention weights.

        Uses provided attention weights to modulate the query before
        performing similarity-based retrieval.

        Args:
            query: Query embedding vector.
            attention_weights: Attention weights to apply to the query.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult ordered by attention-weighted relevance.
        """
        query = np.atleast_1d(query).flatten().astype(float)
        attention_weights = np.atleast_1d(attention_weights).flatten().astype(float)

        # Modulate query by attention
        min_len = min(len(query), len(attention_weights))
        modulated_query = query.copy()
        modulated_query[:min_len] *= attention_weights[:min_len]

        return self.retrieve(modulated_query, top_k=top_k)

    def _consolidate(self) -> None:
        """Consolidate memory by evicting weakest traces."""
        current_time = time.time()
        # Compute strengths
        strengths = [
            (i, t.effective_strength(current_time))
            for i, t in enumerate(self._traces)
        ]
        strengths.sort(key=lambda x: x[1])

        # Remove weakest traces until at capacity
        n_remove = len(self._traces) - self.capacity
        indices_to_remove = sorted(
            [idx for idx, _ in strengths[:n_remove]], reverse=True
        )
        for idx in indices_to_remove:
            self._traces.pop(idx)

    def get_attention_analysis(self) -> Dict[str, Any]:
        """Analyze the attention distribution across stored memories.

        Returns:
            Dictionary with attention analysis metrics:
            - attention_entropy: How distributed the memory strengths are.
            - maximum_attention: Strength of the strongest memory.
            - attention_sparsity: Fraction of near-zero strength memories.
            - total_traces: Number of stored traces.
            - mean_importance: Average importance across traces.
        """
        if not self._traces:
            return {
                "attention_entropy": 0.0,
                "maximum_attention": 0.0,
                "attention_sparsity": 1.0,
                "total_traces": 0,
                "mean_importance": 0.0,
            }

        current_time = time.time()
        strengths = np.array([
            t.effective_strength(current_time) for t in self._traces
        ])

        # Normalize to probability distribution
        total = np.sum(strengths) + 1e-12
        probs = strengths / total

        # Entropy
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        # Max attention
        max_attention = float(np.max(strengths))
        # Sparsity (fraction below threshold)
        sparsity = float(np.mean(strengths < self.consolidation_threshold))
        # Mean importance
        mean_importance = float(np.mean([t.importance for t in self._traces]))

        return {
            "attention_entropy": entropy,
            "maximum_attention": max_attention,
            "attention_sparsity": sparsity,
            "total_traces": len(self._traces),
            "mean_importance": mean_importance,
        }

    def clear(self) -> None:
        """Clear all stored memory traces."""
        self._traces.clear()

    @property
    def size(self) -> int:
        """Current number of stored traces."""
        return len(self._traces)

    @property
    def store_count(self) -> int:
        """Total number of store operations performed."""
        return self._store_count

    @property
    def retrieval_count(self) -> int:
        """Total number of retrieval operations performed."""
        return self._retrieval_count


class TaurusWorkingMemory:
    """Short-term working memory with rapid access and limited capacity.

    Complements TaurusMemoryStore by providing a fast-access buffer
    for active spectral processing. Items in working memory are
    promoted to long-term storage based on importance and usage.

    Args:
        capacity: Maximum working memory slots.
        promotion_threshold: Access count threshold for LTM promotion.
        long_term_store: Optional TaurusMemoryStore for promotion.
    """

    def __init__(
        self,
        capacity: int = 7,
        promotion_threshold: int = 3,
        long_term_store: Optional[TaurusMemoryStore] = None,
    ) -> None:
        self.capacity = capacity
        self.promotion_threshold = promotion_threshold
        self.long_term_store = long_term_store
        self._slots: List[MemoryTrace] = []

    def hold(
        self,
        embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
        semantic_tag: str = "",
    ) -> MemoryTrace:
        """Hold an item in working memory.

        Args:
            embedding: Vector to hold.
            context: Associated context.
            importance: Importance weight.
            semantic_tag: Semantic label.

        Returns:
            The created or updated MemoryTrace.
        """
        embedding = np.atleast_1d(embedding).flatten().astype(float)
        trace = MemoryTrace(
            embedding=embedding,
            context=context or {},
            importance=importance,
            semantic_tag=semantic_tag,
            decay_rate=0.1,  # Faster decay for working memory
        )

        self._slots.append(trace)

        # Evict oldest if over capacity (FIFO with promotion)
        while len(self._slots) > self.capacity:
            evicted = self._slots.pop(0)
            # Promote to long-term if accessed enough
            if (
                self.long_term_store is not None
                and evicted.access_count >= self.promotion_threshold
            ):
                self.long_term_store.store(
                    embedding=evicted.embedding,
                    context=evicted.context,
                    importance=evicted.importance * 1.5,  # Boost for promoted items
                    semantic_tag=evicted.semantic_tag,
                )

        return trace

    def scan(self, query: np.ndarray) -> Optional[RetrievalResult]:
        """Scan working memory for the best match.

        Args:
            query: Query vector.

        Returns:
            Best matching RetrievalResult or None if empty.
        """
        if not self._slots:
            return None

        query = np.atleast_1d(query).flatten().astype(float)
        query_norm = np.linalg.norm(query) + 1e-12

        best_sim = -1.0
        best_trace = None
        for trace in self._slots:
            min_len = min(len(query), len(trace.embedding))
            trace_norm = np.linalg.norm(trace.embedding) + 1e-12
            sim = float(
                np.dot(query[:min_len], trace.embedding[:min_len])
                / (query_norm * trace_norm)
            )
            if sim > best_sim:
                best_sim = sim
                best_trace = trace

        if best_trace is not None:
            best_trace.access_count += 1
            best_trace.last_accessed = time.time()
            return RetrievalResult(
                trace=best_trace,
                similarity=best_sim,
                effective_strength=best_trace.effective_strength(),
                rank=0,
            )
        return None

    @property
    def size(self) -> int:
        """Number of items in working memory."""
        return len(self._slots)

    def clear(self) -> None:
        """Clear working memory."""
        self._slots.clear()
