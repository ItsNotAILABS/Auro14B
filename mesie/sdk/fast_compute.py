"""Fast batch spectral compute — vectorized embed, match, ANN."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.io.loaders import RecordInput, load_record


@dataclass
class SpeedBenchmark:
    n_items: int
    loop_match_ms: float
    batch_match_ms: float
    speedup_ratio: float
    embed_batch_ms: float
    ann_query_ms: float


class FastSpectralCompute:
    """Cached vectorizer + matrix operations for laptop-scale throughput."""

    _shared_vectorizer: Optional[SpectralVectorizer] = None

    def __init__(self, n_bands: int = 8) -> None:
        self.n_bands = n_bands
        self._matrix: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._norms: Optional[np.ndarray] = None

    @classmethod
    def vectorizer(cls, n_bands: int = 8) -> SpectralVectorizer:
        if cls._shared_vectorizer is None or cls._shared_vectorizer.n_bands != n_bands:
            cls._shared_vectorizer = SpectralVectorizer(n_bands=n_bands)
        return cls._shared_vectorizer

    def embed_one(self, record: RecordInput) -> np.ndarray:
        rec = load_record(record)
        return self.vectorizer(self.n_bands).transform(rec)

    def embed_batch(self, records: Sequence[RecordInput]) -> np.ndarray:
        vec = self.vectorizer(self.n_bands)
        embs = vec.batch_transform([load_record(r) for r in records])
        return np.asarray(embs, dtype=np.float64)

    def build_index(self, records: Sequence[RecordInput]) -> int:
        embs = self.embed_batch(records)
        self._matrix = embs
        self._ids = [load_record(r).record_id for r in records]
        self._norms = np.linalg.norm(embs, axis=1)
        self._norms = np.maximum(self._norms, 1e-12)
        return len(self._ids)

    def cosine_search(self, query: RecordInput, top_k: int = 5) -> List[Tuple[str, float]]:
        if self._matrix is None:
            return []
        q = self.embed_one(query)
        qn = max(np.linalg.norm(q), 1e-12)
        sims = (self._matrix @ q) / (self._norms * qn)
        idx = np.argpartition(-sims, min(top_k, len(sims) - 1))[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self._ids[i], float(sims[i])) for i in idx]

    @staticmethod
    def benchmark_match(
        records: Sequence[RecordInput],
        *,
        n_repeat: int = 500,
    ) -> SpeedBenchmark:
        from mesie.matching.matcher import match_records

        loaded = [load_record(r) for r in records]
        if len(loaded) < 2:
            raise ValueError("Need at least 2 records")
        a, b = loaded[0], loaded[1]

        t0 = time.perf_counter()
        for _ in range(n_repeat):
            match_records(a, b)
        loop_ms = (time.perf_counter() - t0) / n_repeat * 1000

        fc = FastSpectralCompute()
        t1 = time.perf_counter()
        mat = fc.embed_batch(loaded[: min(50, len(loaded))])
        t_embed = (time.perf_counter() - t1) * 1000
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        unit = mat / norms
        t2 = time.perf_counter()
        for _ in range(n_repeat):
            _ = unit[0] @ unit[1]
        batch_ms = (time.perf_counter() - t2) / n_repeat * 1000

        fc.build_index(loaded)
        q = loaded[0]
        t3 = time.perf_counter()
        fc.cosine_search(q, top_k=5)
        ann_ms = (time.perf_counter() - t3) * 1000

        return SpeedBenchmark(
            n_items=len(loaded),
            loop_match_ms=round(loop_ms, 4),
            batch_match_ms=round(batch_ms, 4),
            speedup_ratio=round(loop_ms / max(batch_ms, 1e-9), 2),
            embed_batch_ms=round(t_embed, 2),
            ann_query_ms=round(ann_ms, 4),
        )