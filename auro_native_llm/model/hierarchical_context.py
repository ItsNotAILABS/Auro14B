"""Hierarchical long-context memory for AURO.

The current long-context facade can accept a large token stream but ultimately
selects chunks with token-overlap heuristics. This module upgrades selection to
a model-aware, coarse-to-fine memory process:

macro regions -> meso regions -> micro regions -> bounded raw-token retrieval

Chunk representations come from AURO's learned token-embedding space. The query
combines the current tail with active working-memory state. Selected history is
also summarized into hidden space so the long-context wrapper can prime working
memory *before* the transformer pass, allowing retrieved history to influence
adaptive MoE compute on the same cycle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable, Sequence

import numpy as np

from auro_native_llm.context import ContextEnvelope


def _hash_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


@dataclass(frozen=True)
class HierarchicalChunk:
    id: str
    level: str
    start: int
    end: int
    token_count: int
    sha256: str
    score: float
    similarity: float
    recency: float
    diversity: float


@dataclass(frozen=True)
class HierarchicalContextReceipt:
    schema: str
    accepted_tokens: int
    dense_tokens: int
    retrieved_tokens: int
    recent_tokens: int
    truncated_input_tokens: int
    macro_candidates: int
    meso_candidates: int
    micro_candidates: int
    selected_micro_ids: list[str]
    accepted_sha256: str
    dense_sha256: str
    summary_norm: float
    query_norm: float
    working_memory_query_used: bool
    working_memory_primed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HierarchicalContextMemory:
    """Coarse-to-fine retrieval over AURO token embeddings."""

    schema = "auro.hierarchical-context-memory.v1"

    def __init__(
        self,
        model,
        *,
        envelope: ContextEnvelope,
        macro_size: int = 8192,
        meso_size: int = 2048,
        micro_size: int = 512,
        macro_keep: int = 8,
        meso_per_macro: int = 4,
        embedding_sample: int = 256,
        working_memory_query_weight: float = 0.35,
    ) -> None:
        self.model = model
        self.envelope = envelope
        self.macro_size = max(int(macro_size), int(meso_size))
        self.meso_size = max(int(meso_size), int(micro_size))
        self.micro_size = max(64, int(micro_size))
        self.macro_keep = max(1, int(macro_keep))
        self.meso_per_macro = max(1, int(meso_per_macro))
        self.embedding_sample = max(16, int(embedding_sample))
        self.working_memory_query_weight = float(np.clip(working_memory_query_weight, 0.0, 1.0))
        self.last_summary = np.zeros(int(model.config.hidden_dim), dtype=np.float64)
        self.last_receipt: HierarchicalContextReceipt | None = None
        self.last_chunks: list[HierarchicalChunk] = []

    @property
    def hidden_dim(self) -> int:
        return int(self.model.config.hidden_dim)

    def _embedding_table(self) -> np.ndarray:
        return np.asarray(self.model.core.embedding.token_embeddings)

    def _sample_ids(self, ids: np.ndarray) -> np.ndarray:
        arr = np.asarray(ids, dtype=np.int64).reshape(-1)
        if arr.size <= self.embedding_sample:
            return arr
        positions = np.linspace(0, arr.size - 1, self.embedding_sample).astype(np.int64)
        return arr[positions]

    def _vector(self, ids: np.ndarray) -> np.ndarray:
        arr = self._sample_ids(ids)
        if not arr.size:
            return np.zeros(self.hidden_dim, dtype=np.float64)
        table = self._embedding_table()
        arr = np.clip(arr, 0, table.shape[0] - 1)
        vector = np.asarray(table[arr], dtype=np.float64).mean(axis=0)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-12 else np.zeros_like(vector)

    @staticmethod
    def _diversity(ids: np.ndarray) -> float:
        arr = np.asarray(ids).reshape(-1)
        if not arr.size:
            return 0.0
        return float(np.unique(arr).size) / float(arr.size)

    def _working_memory_vector(self) -> tuple[np.ndarray, bool]:
        delta = getattr(self.model, "delta_attention", None)
        memory = getattr(delta, "working_memory", None) if delta is not None else None
        if memory is None or not getattr(memory, "initialized", False):
            return np.zeros(self.hidden_dim, dtype=np.float64), False
        vector = np.asarray(memory.context_vector(), dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        return (vector / norm if norm > 1e-12 else vector), True

    def _query_vector(self, recent: np.ndarray) -> tuple[np.ndarray, bool]:
        tail = recent[-min(2048, int(recent.size)) :]
        token_query = self._vector(tail)
        memory_query, used = self._working_memory_vector()
        if used:
            w = self.working_memory_query_weight
            query = (1.0 - w) * token_query + w * memory_query
        else:
            query = token_query
        norm = float(np.linalg.norm(query))
        return (query / norm if norm > 1e-12 else query), used

    def _score(
        self,
        ids: np.ndarray,
        query: np.ndarray,
        *,
        start: int,
        total: int,
    ) -> tuple[float, float, float, float, np.ndarray]:
        vector = self._vector(ids)
        similarity = float(np.dot(vector, query)) if np.any(vector) and np.any(query) else 0.0
        recency = float(start + len(ids)) / float(max(1, total))
        diversity = self._diversity(ids)
        score = 0.72 * similarity + 0.18 * recency + 0.10 * diversity
        return float(score), similarity, recency, diversity, vector

    def _chunk_rows(
        self,
        history: np.ndarray,
        query: np.ndarray,
        *,
        size: int,
        level: str,
        base_start: int = 0,
    ) -> list[tuple[HierarchicalChunk, np.ndarray, np.ndarray]]:
        rows: list[tuple[HierarchicalChunk, np.ndarray, np.ndarray]] = []
        total = int(history.size) + int(base_start)
        for local_start in range(0, int(history.size), int(size)):
            arr = history[local_start : local_start + size]
            start = int(base_start + local_start)
            end = start + int(arr.size)
            score, similarity, recency, diversity, vector = self._score(
                arr, query, start=start, total=total
            )
            chunk = HierarchicalChunk(
                id=f"{level}:{start}:{end}",
                level=level,
                start=start,
                end=end,
                token_count=int(arr.size),
                sha256=_hash_ids(arr),
                score=round(score, 8),
                similarity=round(similarity, 8),
                recency=round(recency, 8),
                diversity=round(diversity, 8),
            )
            rows.append((chunk, arr.copy(), vector))
        return rows

    @staticmethod
    def _top(rows, count: int):
        return sorted(rows, key=lambda row: (-row[0].score, -row[0].start))[: max(0, int(count))]

    def _retrieve_hierarchy(
        self,
        history: np.ndarray,
        query: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[HierarchicalChunk], dict[str, int]]:
        macro_rows = self._chunk_rows(history, query, size=self.macro_size, level="macro")
        top_macro = self._top(macro_rows, self.macro_keep)

        meso_rows = []
        for macro, arr, _ in top_macro:
            meso_rows.extend(
                self._chunk_rows(
                    arr,
                    query,
                    size=self.meso_size,
                    level="meso",
                    base_start=macro.start,
                )
            )
        top_meso = self._top(meso_rows, self.macro_keep * self.meso_per_macro)

        micro_rows = []
        for meso, arr, _ in top_meso:
            micro_rows.extend(
                self._chunk_rows(
                    arr,
                    query,
                    size=self.micro_size,
                    level="micro",
                    base_start=meso.start,
                )
            )

        micro_slots = max(1, self.envelope.retrieval_budget // self.micro_size)
        selected = self._top(micro_rows, micro_slots)
        # Retrieval order follows chronology, not score order.
        selected = sorted(selected, key=lambda row: row[0].start)
        retrieved = (
            np.concatenate([row[1] for row in selected])
            if selected
            else np.empty(0, dtype=np.int64)
        )
        if retrieved.size > self.envelope.retrieval_budget:
            retrieved = retrieved[-self.envelope.retrieval_budget :]

        if selected:
            weights = np.asarray([max(0.0, row[0].score) + 1e-4 for row in selected], dtype=np.float64)
            weights /= weights.sum()
            summary = np.sum(
                np.stack([row[2] for row in selected], axis=0) * weights[:, None],
                axis=0,
            )
            norm = float(np.linalg.norm(summary))
            if norm > 1e-12:
                summary /= norm
        else:
            summary = np.zeros(self.hidden_dim, dtype=np.float64)

        all_chunks = [row[0] for row in macro_rows] + [row[0] for row in meso_rows] + [row[0] for row in micro_rows]
        counts = {
            "macro": len(macro_rows),
            "meso": len(meso_rows),
            "micro": len(micro_rows),
        }
        return retrieved, summary, all_chunks, counts

    def ingest(
        self,
        token_ids: Sequence[int] | np.ndarray,
    ) -> tuple[np.ndarray, HierarchicalContextReceipt, list[HierarchicalChunk]]:
        raw = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        truncated = max(0, int(raw.size) - self.envelope.accepted_limit)
        accepted = raw[-self.envelope.accepted_limit :]

        if accepted.size <= self.envelope.dense_window:
            dense = accepted.copy()
            self.last_summary = self._vector(dense)
            receipt = HierarchicalContextReceipt(
                schema=self.schema,
                accepted_tokens=int(accepted.size),
                dense_tokens=int(dense.size),
                retrieved_tokens=0,
                recent_tokens=int(dense.size),
                truncated_input_tokens=truncated,
                macro_candidates=0,
                meso_candidates=0,
                micro_candidates=0,
                selected_micro_ids=[],
                accepted_sha256=_hash_ids(accepted),
                dense_sha256=_hash_ids(dense),
                summary_norm=float(np.linalg.norm(self.last_summary)),
                query_norm=float(np.linalg.norm(self.last_summary)),
                working_memory_query_used=False,
                working_memory_primed=False,
            )
            self.last_receipt = receipt
            self.last_chunks = []
            return dense, receipt, []

        recent_budget = self.envelope.dense_window - self.envelope.retrieval_budget
        recent = accepted[-recent_budget:]
        history = accepted[:-recent_budget]
        query, used_working_memory = self._query_vector(recent)
        retrieved, summary, chunks, counts = self._retrieve_hierarchy(history, query)
        dense = np.concatenate([retrieved, recent])[-self.envelope.dense_window :]
        selected_micro = [
            chunk.id for chunk in chunks
            if chunk.level == "micro" and any(
                chunk.start <= pos < chunk.end
                for pos in []
            )
        ]
        # Recover selection IDs from chronological overlap with retrieved hashes.
        selected_hashes: list[str] = []
        if retrieved.size:
            for start in range(0, int(retrieved.size), self.micro_size):
                selected_hashes.append(_hash_ids(retrieved[start : start + self.micro_size]))
        selected_micro = [
            chunk.id for chunk in chunks
            if chunk.level == "micro" and chunk.sha256 in selected_hashes
        ]

        self.last_summary = summary
        receipt = HierarchicalContextReceipt(
            schema=self.schema,
            accepted_tokens=int(accepted.size),
            dense_tokens=int(dense.size),
            retrieved_tokens=int(retrieved.size),
            recent_tokens=int(recent.size),
            truncated_input_tokens=truncated,
            macro_candidates=counts["macro"],
            meso_candidates=counts["meso"],
            micro_candidates=counts["micro"],
            selected_micro_ids=selected_micro,
            accepted_sha256=_hash_ids(accepted),
            dense_sha256=_hash_ids(dense),
            summary_norm=float(np.linalg.norm(summary)),
            query_norm=float(np.linalg.norm(query)),
            working_memory_query_used=used_working_memory,
            working_memory_primed=False,
        )
        self.last_receipt = receipt
        self.last_chunks = chunks
        return dense, receipt, chunks

    def prime_working_memory(self) -> bool:
        """Prime active working memory from retrieved long-context summary."""
        if not np.any(self.last_summary):
            return False
        delta = getattr(self.model, "delta_attention", None)
        memory = getattr(delta, "working_memory", None) if delta is not None else None
        if memory is None:
            return False
        memory.step(self.last_summary)
        if self.last_receipt is not None:
            self.last_receipt = HierarchicalContextReceipt(
                **{
                    **self.last_receipt.to_dict(),
                    "working_memory_primed": True,
                }
            )
        return True


__all__ = [
    "HierarchicalChunk",
    "HierarchicalContextReceipt",
    "HierarchicalContextMemory",
]
