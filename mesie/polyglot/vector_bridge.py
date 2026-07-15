"""Vector bridge — unifies embeddings, fingerprint ANN, and MAESI fast compute."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mesie.embeddings import SpectralFingerprintPipeline
from mesie.io.loaders import RecordInput, load_record
from mesie.sdk import FastSpectralCompute, MAESIClient
from mesie.polyglot.contract import record_to_dict


@dataclass
class VectorBridgeState:
    n_indexed: int = 0
    embed_dim: int = 0
    fingerprint_dim: int = 0
    has_fast_index: bool = False
    has_fingerprint_index: bool = False


@dataclass
class VectorQueryResult:
    record_id: str
    neighbors: List[Dict[str, Any]] = field(default_factory=list)
    fingerprint_hits: List[Dict[str, Any]] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    technical_hits: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


class AISVectorBridge:
    """Central vector layer for AIS polyglot — all runtimes route retrieval here."""

    def __init__(self, n_bands: int = 8, use_fingerprint: bool = True) -> None:
        self.fast = FastSpectralCompute(n_bands=n_bands)
        self.fingerprint = SpectralFingerprintPipeline() if use_fingerprint else None
        self._maesi = MAESIClient(fast=True, use_fingerprint=use_fingerprint)
        self._state = VectorBridgeState()

    @property
    def state(self) -> VectorBridgeState:
        return self._state

    def index(self, records: Sequence[RecordInput]) -> int:
        loaded = [load_record(r) for r in records]
        n = self.fast.build_index(loaded)
        self._state.n_indexed = n
        self._state.has_fast_index = n > 0
        if loaded:
            self._state.embed_dim = len(self.fast.embed_one(loaded[0]))
        if self.fingerprint:
            self.fingerprint.index_records(loaded)
            self._state.has_fingerprint_index = True
            if loaded:
                fp = self.fingerprint.process(loaded[0])
                self._state.fingerprint_dim = len(fp.combined_vector)
        self._maesi.index_corpus(loaded)
        return n

    def embed(self, record: RecordInput) -> np.ndarray:
        return self.fast.embed_one(record)

    def query(self, record: RecordInput, top_k: int = 5) -> VectorQueryResult:
        import time

        t0 = time.perf_counter()
        rec = load_record(record)
        emb = self.fast.embed_one(rec)
        neighbors = [
            {"record_id": rid, "similarity": round(sim, 4)}
            for rid, sim in self.fast.cosine_search(rec, top_k=top_k)
        ]
        fp_hits = []
        if self.fingerprint:
            fp_hits = [
                {"item_id": h.item_id, "similarity": round(h.similarity, 4)}
                for h in self.fingerprint.query(rec, top_k=top_k)
            ]
        maesi_q = self._maesi.query(rec, top_k=top_k)
        ms = (time.perf_counter() - t0) * 1000
        return VectorQueryResult(
            record_id=rec.record_id,
            neighbors=neighbors,
            fingerprint_hits=fp_hits,
            embedding=emb.tolist(),
            technical_hits=maesi_q.technical_hits,
            elapsed_ms=ms,
        )

    def export_contract_record(self, record: RecordInput) -> Dict[str, Any]:
        return record_to_dict(record)