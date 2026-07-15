"""TaurusMemory — Native working + persistent memory with resonance decay.

Implements TAURUS (Temporal Adaptive Retrieval and Unified Storage) at the
native level without NumPy. Uses resonance-weighted storage, temporal decay,
and helix compression for efficient spectral memory management in sovereign
swarm deployments.
"""

from __future__ import annotations

import array
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from phantom_native.sovereign_tensor import SovereignTensor


class TaurusMemory:
    """Native working + persistent memory with resonance decay + helix compression.

    Attributes:
        working_memory: Short-term memory buffer (bounded capacity).
        long_term: Persistent memory keyed by QSHA hash.
        capacity: Maximum working memory slots.
        decay_rate: Multiplicative decay applied to aging memories.
    """

    def __init__(self, capacity: int = 64, decay_rate: float = 0.95):
        self.working_memory: List[_MemoryEntry] = []
        self.long_term: Dict[str, _MemoryEntry] = {}
        self.capacity = capacity
        self.decay_rate = decay_rate

    def store(
        self,
        tensor: SovereignTensor,
        key: Optional[str] = None,
        importance: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store tensor with resonance weighting and decay.

        Args:
            tensor: SovereignTensor to store.
            key: Optional QSHA key. Auto-generated if not provided.
            importance: Priority weight for retention.
            context: Optional metadata dictionary.

        Returns:
            The storage key (QSHA hash).
        """
        if key is None:
            key = "qsha:" + format(hash(tensor.to_bytes()) & 0xFFFFFFFFFFFFFFFF, "016x")

        # Resonance boost based on memory activity
        resonance_score = tensor.resonance * importance * (
            1.0 + len(self.working_memory) * 0.02
        )

        entry = _MemoryEntry(
            tensor=tensor,
            key=key,
            resonance_score=resonance_score,
            importance=importance,
            timestamp=time.time(),
            context=context or {},
        )

        # Evict oldest if at capacity (with decay)
        if len(self.working_memory) >= self.capacity:
            self._apply_decay()
            self.working_memory.pop(0)

        self.working_memory.append(entry)
        self.long_term[key] = entry

        return key

    def recall_by_key(self, key: str) -> Optional[SovereignTensor]:
        """Retrieve a tensor from long-term memory by key."""
        entry = self.long_term.get(key)
        if entry is not None:
            entry.access_count += 1
            entry.last_accessed = time.time()
            return entry.tensor
        return None

    def recall_top_k(self, k: int = 8) -> List[SovereignTensor]:
        """Return highest-resonance items from working memory.

        Uses MESIE spectral matching principles — items with higher
        resonance scores are considered more relevant.
        """
        sorted_mem = sorted(
            self.working_memory,
            key=lambda e: e.resonance_score,
            reverse=True,
        )
        return [e.tensor for e in sorted_mem[:k]]

    def recall_by_similarity(
        self, query: SovereignTensor, top_k: int = 5
    ) -> List[Tuple[str, SovereignTensor, float]]:
        """Retrieve memories by cosine similarity to query tensor.

        Returns list of (key, tensor, similarity_score) tuples.
        """
        results: List[Tuple[str, SovereignTensor, float]] = []
        q_norm = query.norm()
        if q_norm == 0:
            return results

        for entry in self.working_memory:
            t = entry.tensor
            if len(t.data) != len(query.data):
                continue
            dot = sum(a * b for a, b in zip(query.data, t.data))
            t_norm = t.norm()
            if t_norm == 0:
                continue
            sim = dot / (q_norm * t_norm)
            results.append((entry.key, t, sim))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def compress_helix(self, tensor: SovereignTensor) -> SovereignTensor:
        """Helix-style compression: rotate + downsample by factor 2.

        Pairs adjacent elements and averages them, halving the tensor size.
        """
        n = len(tensor.data)
        if n < 2:
            return tensor
        compressed_n = n // 2
        compressed = array.array("f", [0.0] * compressed_n)
        for i in range(compressed_n):
            compressed[i] = (tensor.data[2 * i] + tensor.data[2 * i + 1]) * 0.5
        meta = dict(tensor.spectral_meta)
        meta["helix_compressed"] = True
        meta["original_size"] = n
        return SovereignTensor(list(compressed), (compressed_n,), meta)

    def consolidate(self, threshold: float = 0.3) -> int:
        """Consolidate working memory: promote high-resonance, drop low.

        Items below threshold are removed from working memory but
        remain in long-term storage for potential future recall.

        Returns:
            Number of entries removed from working memory.
        """
        before = len(self.working_memory)
        self.working_memory = [
            e for e in self.working_memory if e.resonance_score >= threshold
        ]
        return before - len(self.working_memory)

    def size(self) -> Dict[str, int]:
        """Return memory utilization stats."""
        return {
            "working_memory": len(self.working_memory),
            "long_term": len(self.long_term),
            "capacity": self.capacity,
        }

    def _apply_decay(self) -> None:
        """Apply temporal decay to all working memory entries."""
        for entry in self.working_memory:
            age = time.time() - entry.timestamp
            decay = self.decay_rate ** (age / 60.0)  # decay per minute
            entry.resonance_score *= decay


class _MemoryEntry:
    """Internal memory entry for TAURUS storage."""

    __slots__ = (
        "tensor",
        "key",
        "resonance_score",
        "importance",
        "timestamp",
        "last_accessed",
        "access_count",
        "context",
    )

    def __init__(
        self,
        tensor: SovereignTensor,
        key: str,
        resonance_score: float,
        importance: float,
        timestamp: float,
        context: Dict[str, Any],
    ):
        self.tensor = tensor
        self.key = key
        self.resonance_score = resonance_score
        self.importance = importance
        self.timestamp = timestamp
        self.last_accessed = timestamp
        self.access_count = 0
        self.context = context
