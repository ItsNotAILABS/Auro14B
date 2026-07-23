"""AURO governed near-300k context envelope.

The envelope accepts up to 294,912 tokens while keeping dense MESIE attention
bounded. It preserves deterministic chunk hashes, salience scores, retrieval
selection, and continuity receipts. This is an accepted-context surface, not a
claim that all 294,912 tokens are simultaneously processed by dense attention.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, List, Sequence

import numpy as np

ACCEPTED_CONTEXT_TOKENS = 294_912
DEFAULT_DENSE_WINDOW = 32_768
DEFAULT_CHUNK_SIZE = 4_096


def _hash_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


@dataclass(frozen=True)
class ContextChunk:
    index: int
    start: int
    end: int
    token_count: int
    sha256: str
    salience: float


@dataclass(frozen=True)
class ContextReceipt:
    schema: str
    accepted_tokens: int
    dense_tokens: int
    retrieved_tokens: int
    chunk_count: int
    envelope_sha256: str
    selected_chunk_indexes: List[int]
    truncated_input_tokens: int

    def to_dict(self):
        return asdict(self)


class ContextEnvelope:
    """Deterministically compress a long token stream into a bounded dense view."""

    def __init__(
        self,
        *,
        accepted_limit: int = ACCEPTED_CONTEXT_TOKENS,
        dense_window: int = DEFAULT_DENSE_WINDOW,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        retrieval_budget: int = 8_192,
    ) -> None:
        if accepted_limit < dense_window:
            raise ValueError("accepted_limit must be >= dense_window")
        if chunk_size <= 0 or dense_window <= 0 or retrieval_budget < 0:
            raise ValueError("context geometry must be positive")
        self.accepted_limit = int(accepted_limit)
        self.dense_window = int(dense_window)
        self.chunk_size = int(chunk_size)
        self.retrieval_budget = min(int(retrieval_budget), self.dense_window // 2)

    @staticmethod
    def _salience(chunk: np.ndarray, query_tail: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        unique_ratio = float(np.unique(chunk).size) / float(chunk.size)
        if query_tail.size:
            overlap = np.intersect1d(np.unique(chunk), np.unique(query_tail)).size
            overlap_ratio = float(overlap) / float(max(1, np.unique(query_tail).size))
        else:
            overlap_ratio = 0.0
        boundary = 1.0 if chunk.size and int(chunk[-1]) in {2, 3, 4, 10, 13} else 0.0
        return round(0.45 * unique_ratio + 0.50 * overlap_ratio + 0.05 * boundary, 8)

    def ingest(self, token_ids: Sequence[int] | np.ndarray) -> tuple[np.ndarray, ContextReceipt, List[ContextChunk]]:
        raw = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        truncated = max(0, int(raw.size) - self.accepted_limit)
        accepted = raw[-self.accepted_limit :]
        if accepted.size <= self.dense_window:
            receipt = ContextReceipt(
                schema="auro.context.envelope.v1",
                accepted_tokens=int(accepted.size),
                dense_tokens=int(accepted.size),
                retrieved_tokens=0,
                chunk_count=1 if accepted.size else 0,
                envelope_sha256=_hash_ids(accepted),
                selected_chunk_indexes=[],
                truncated_input_tokens=truncated,
            )
            chunks = [] if not accepted.size else [ContextChunk(0, 0, int(accepted.size), int(accepted.size), _hash_ids(accepted), 1.0)]
            return accepted.copy(), receipt, chunks

        recent_budget = self.dense_window - self.retrieval_budget
        recent = accepted[-recent_budget:]
        history = accepted[:-recent_budget]
        query_tail = recent[-min(1024, recent.size):]
        chunks: List[ContextChunk] = []
        arrays: List[np.ndarray] = []
        for index, start in enumerate(range(0, int(history.size), self.chunk_size)):
            arr = history[start : start + self.chunk_size]
            arrays.append(arr)
            chunks.append(ContextChunk(index, start, start + int(arr.size), int(arr.size), _hash_ids(arr), self._salience(arr, query_tail)))

        slots = max(0, self.retrieval_budget // self.chunk_size)
        ranked = sorted(chunks, key=lambda c: (-c.salience, -c.index))[:slots]
        selected_indexes = sorted(c.index for c in ranked)
        retrieved = np.concatenate([arrays[i] for i in selected_indexes]) if selected_indexes else np.empty(0, dtype=np.int64)
        if retrieved.size > self.retrieval_budget:
            retrieved = retrieved[-self.retrieval_budget:]
        dense = np.concatenate([retrieved, recent])[-self.dense_window:]
        receipt_payload = {
            "accepted_sha256": _hash_ids(accepted),
            "dense_sha256": _hash_ids(dense),
            "selected": selected_indexes,
            "geometry": [self.accepted_limit, self.dense_window, self.chunk_size, self.retrieval_budget],
        }
        envelope_hash = hashlib.sha256(json.dumps(receipt_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        receipt = ContextReceipt(
            schema="auro.context.envelope.v1",
            accepted_tokens=int(accepted.size),
            dense_tokens=int(dense.size),
            retrieved_tokens=int(retrieved.size),
            chunk_count=len(chunks),
            envelope_sha256=envelope_hash,
            selected_chunk_indexes=selected_indexes,
            truncated_input_tokens=truncated,
        )
        return dense, receipt, chunks
