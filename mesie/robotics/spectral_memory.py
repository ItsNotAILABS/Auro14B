"""Recursive self-improving spectral memory for autonomous agents.

Enables agents to build and query their own spectral memory efficiently.
Supports recursive containment, self-organization, and continuous
refinement of stored spectral knowledge.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class MemoryTier(Enum):
    """Memory storage tiers with different access characteristics."""

    WORKING = "working"       # Fast, small, volatile
    SHORT_TERM = "short_term"  # Moderate speed, recent items
    LONG_TERM = "long_term"   # Slow, large, persistent
    CONSOLIDATED = "consolidated"  # Compressed, highly refined


class ConsolidationStrategy(Enum):
    """How memories are consolidated from short to long term."""

    FREQUENCY = "frequency"     # Most accessed survive
    RECENCY = "recency"         # Most recent survive
    IMPORTANCE = "importance"   # Highest scored survive
    SPECTRAL_DIVERSITY = "spectral_diversity"  # Maximize coverage


@dataclass
class MemoryConfig:
    """Configuration for spectral memory system.

    Args:
        working_capacity: Max entries in working memory.
        short_term_capacity: Max entries in short-term memory.
        long_term_capacity: Max entries in long-term memory.
        vector_dim: Dimension of spectral memory vectors.
        consolidation_strategy: How to move from short to long term.
        consolidation_threshold: Access count before consolidation.
        decay_rate: Exponential decay for unused memories.
        enable_recursion: Allow memories to reference other memories.
        max_recursion_depth: Maximum nesting depth for recursive queries.
    """

    working_capacity: int = 64
    short_term_capacity: int = 1024
    long_term_capacity: int = 100000
    vector_dim: int = 128
    consolidation_strategy: ConsolidationStrategy = ConsolidationStrategy.IMPORTANCE
    consolidation_threshold: int = 5
    decay_rate: float = 0.001
    enable_recursion: bool = True
    max_recursion_depth: int = 3


@dataclass
class MemoryEntry:
    """A single entry in spectral memory.

    Attributes:
        memory_id: Unique memory identifier.
        vector: Spectral representation vector.
        label: Optional semantic label.
        tier: Current storage tier.
        access_count: Number of times accessed.
        importance_score: Computed importance (0-1).
        created_at: Creation timestamp.
        last_accessed: Last access timestamp.
        parent_ids: IDs of parent memories (for recursion).
        child_ids: IDs of child memories (for recursion).
        metadata: Additional context.
    """

    vector: np.ndarray
    memory_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    label: str = ""
    tier: MemoryTier = MemoryTier.WORKING
    access_count: int = 0
    importance_score: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Result of a memory query.

    Attributes:
        entries: Matched memory entries with scores.
        query_time_ms: Time taken to execute query.
        depth_reached: Recursion depth reached.
        total_scanned: Total entries scanned.
    """

    entries: List[Tuple[MemoryEntry, float]]
    query_time_ms: float = 0.0
    depth_reached: int = 0
    total_scanned: int = 0


class SpectralMemory:
    """Recursive self-improving spectral memory for autonomous agents.

    Provides a tiered memory system where agents can store, query,
    and recursively refine spectral representations. Supports
    self-organization via consolidation and importance scoring.

    Args:
        config: Memory system configuration.
    """

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self.config = config or MemoryConfig()
        self._working: Dict[str, MemoryEntry] = {}
        self._short_term: Dict[str, MemoryEntry] = {}
        self._long_term: Dict[str, MemoryEntry] = {}
        self._consolidated: Dict[str, MemoryEntry] = {}
        self._access_log: deque = deque(maxlen=10000)
        self._consolidation_count = 0

    def store(
        self,
        vector: np.ndarray,
        label: str = "",
        importance: float = 0.5,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Store a new spectral memory.

        Enters working memory first, then migrates through tiers
        based on access patterns and consolidation strategy.

        Args:
            vector: Spectral vector to store.
            label: Optional semantic label.
            importance: Initial importance score (0-1).
            parent_id: Parent memory ID for recursive structure.
            metadata: Additional context.

        Returns:
            The created MemoryEntry.
        """
        vector = self._normalize(vector)

        entry = MemoryEntry(
            vector=vector,
            label=label,
            importance_score=importance,
            tier=MemoryTier.WORKING,
            metadata=metadata or {},
        )

        # Link to parent if recursive
        if parent_id and self.config.enable_recursion:
            entry.parent_ids.append(parent_id)
            parent = self._find_entry(parent_id)
            if parent:
                parent.child_ids.append(entry.memory_id)

        # Evict from working memory if full
        if len(self._working) >= self.config.working_capacity:
            self._evict_working()

        self._working[entry.memory_id] = entry
        return entry

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        tiers: Optional[List[MemoryTier]] = None,
        recursive: bool = False,
    ) -> QueryResult:
        """Query spectral memory for similar entries.

        Searches across specified tiers using cosine similarity.
        Optionally follows recursive links to related memories.

        Args:
            vector: Query vector.
            top_k: Number of results to return.
            tiers: Which tiers to search (default: all).
            recursive: Whether to follow memory links.

        Returns:
            QueryResult with matched entries and scores.
        """
        start_time = time.time()
        vector = self._normalize(vector)

        if tiers is None:
            tiers = list(MemoryTier)

        # Gather candidates from selected tiers
        candidates: List[MemoryEntry] = []
        for tier in tiers:
            store = self._get_store(tier)
            candidates.extend(store.values())

        # Score all candidates
        scored: List[Tuple[MemoryEntry, float]] = []
        for entry in candidates:
            sim = float(np.dot(vector, entry.vector))
            scored.append((entry, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:top_k]

        # Update access counts
        for entry, _ in results:
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._access_log.append(entry.memory_id)

        # Recursive expansion
        depth = 0
        if recursive and self.config.enable_recursion:
            results, depth = self._expand_recursive(results, vector, top_k)

        elapsed_ms = (time.time() - start_time) * 1000

        return QueryResult(
            entries=results,
            query_time_ms=elapsed_ms,
            depth_reached=depth,
            total_scanned=len(candidates),
        )

    def consolidate(self) -> int:
        """Run memory consolidation cycle.

        Migrates entries from working → short_term → long_term
        based on the configured consolidation strategy.

        Returns:
            Number of entries migrated.
        """
        migrated = 0

        # Working → Short-term: entries accessed enough times
        to_promote = [
            (mid, entry) for mid, entry in self._working.items()
            if entry.access_count >= self.config.consolidation_threshold
        ]
        for mid, entry in to_promote:
            if len(self._short_term) >= self.config.short_term_capacity:
                self._evict_short_term()
            entry.tier = MemoryTier.SHORT_TERM
            self._short_term[mid] = entry
            del self._working[mid]
            migrated += 1

        # Short-term → Long-term: high importance entries
        to_archive = [
            (mid, entry) for mid, entry in self._short_term.items()
            if entry.importance_score >= 0.7
            and entry.access_count >= self.config.consolidation_threshold * 2
        ]
        for mid, entry in to_archive:
            if len(self._long_term) >= self.config.long_term_capacity:
                self._evict_long_term()
            entry.tier = MemoryTier.LONG_TERM
            self._long_term[mid] = entry
            del self._short_term[mid]
            migrated += 1

        self._consolidation_count += 1
        return migrated

    def refine(self, memory_id: str, new_vector: np.ndarray) -> Optional[MemoryEntry]:
        """Refine an existing memory with new information.

        Performs weighted merge of the existing memory vector with
        new observation, improving the memory over time.

        Args:
            memory_id: ID of memory to refine.
            new_vector: New observation vector.

        Returns:
            Updated entry, or None if not found.
        """
        entry = self._find_entry(memory_id)
        if entry is None:
            return None

        new_vector = self._normalize(new_vector)

        # Weighted merge — existing memory has more weight with more access
        weight = entry.access_count / (entry.access_count + 1)
        entry.vector = self._normalize(
            weight * entry.vector + (1 - weight) * new_vector
        )
        entry.access_count += 1
        entry.last_accessed = time.time()

        # Boost importance on refinement
        entry.importance_score = min(1.0, entry.importance_score + 0.05)

        return entry

    def create_recursive_group(
        self,
        vectors: List[np.ndarray],
        group_label: str = "",
    ) -> MemoryEntry:
        """Create a recursive memory group with parent-child structure.

        Stores multiple related memories and creates a parent
        summary memory that represents the group.

        Args:
            vectors: List of spectral vectors to group.
            group_label: Label for the group.

        Returns:
            Parent MemoryEntry summarizing the group.
        """
        # Create child entries
        child_ids = []
        for vec in vectors:
            child = self.store(vec, label=f"{group_label}_child")
            child_ids.append(child.memory_id)

        # Create parent as mean of children
        mean_vec = self._normalize(
            np.mean([self._normalize(v) for v in vectors], axis=0)
        )
        parent = self.store(
            mean_vec,
            label=group_label,
            importance=0.7,
        )
        parent.child_ids = child_ids

        # Link children to parent
        for cid in child_ids:
            child_entry = self._find_entry(cid)
            if child_entry:
                child_entry.parent_ids.append(parent.memory_id)

        return parent

    def _expand_recursive(
        self,
        results: List[Tuple[MemoryEntry, float]],
        query: np.ndarray,
        top_k: int,
        depth: int = 0,
    ) -> Tuple[List[Tuple[MemoryEntry, float]], int]:
        """Expand results by following recursive links."""
        if depth >= self.config.max_recursion_depth:
            return results, depth

        expanded_ids = set(e.memory_id for e, _ in results)
        new_entries: List[Tuple[MemoryEntry, float]] = []

        for entry, score in results:
            related_ids = entry.child_ids + entry.parent_ids
            for rid in related_ids:
                if rid not in expanded_ids:
                    related = self._find_entry(rid)
                    if related:
                        sim = float(np.dot(query, related.vector))
                        new_entries.append((related, sim * 0.9))  # Discount
                        expanded_ids.add(rid)

        if new_entries:
            all_results = results + new_entries
            all_results.sort(key=lambda x: x[1], reverse=True)
            return all_results[:top_k], depth + 1

        return results, depth

    def _find_entry(self, memory_id: str) -> Optional[MemoryEntry]:
        """Find a memory entry across all tiers."""
        for store in [self._working, self._short_term,
                      self._long_term, self._consolidated]:
            if memory_id in store:
                return store[memory_id]
        return None

    def _get_store(self, tier: MemoryTier) -> Dict[str, MemoryEntry]:
        """Get the store dict for a given tier."""
        if tier == MemoryTier.WORKING:
            return self._working
        elif tier == MemoryTier.SHORT_TERM:
            return self._short_term
        elif tier == MemoryTier.LONG_TERM:
            return self._long_term
        return self._consolidated

    def _evict_working(self) -> None:
        """Evict least important entry from working memory."""
        if not self._working:
            return
        least = min(self._working.values(), key=lambda e: e.importance_score)
        del self._working[least.memory_id]

    def _evict_short_term(self) -> None:
        """Evict from short-term based on strategy."""
        if not self._short_term:
            return
        if self.config.consolidation_strategy == ConsolidationStrategy.RECENCY:
            victim = min(self._short_term.values(), key=lambda e: e.last_accessed)
        elif self.config.consolidation_strategy == ConsolidationStrategy.FREQUENCY:
            victim = min(self._short_term.values(), key=lambda e: e.access_count)
        else:
            victim = min(self._short_term.values(), key=lambda e: e.importance_score)
        del self._short_term[victim.memory_id]

    def _evict_long_term(self) -> None:
        """Evict least important from long-term memory."""
        if not self._long_term:
            return
        victim = min(self._long_term.values(), key=lambda e: e.importance_score)
        del self._long_term[victim.memory_id]

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length, resizing if needed."""
        vector = np.asarray(vector, dtype=np.float64).ravel()
        if len(vector) != self.config.vector_dim:
            x_old = np.linspace(0, 1, len(vector))
            x_new = np.linspace(0, 1, self.config.vector_dim)
            vector = np.interp(x_new, x_old, vector)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    @property
    def total_memories(self) -> int:
        """Total memories across all tiers."""
        return (
            len(self._working) + len(self._short_term)
            + len(self._long_term) + len(self._consolidated)
        )

    @property
    def stats(self) -> Dict[str, Any]:
        """Memory system statistics."""
        return {
            "working": len(self._working),
            "short_term": len(self._short_term),
            "long_term": len(self._long_term),
            "consolidated": len(self._consolidated),
            "total": self.total_memories,
            "consolidation_cycles": self._consolidation_count,
        }
