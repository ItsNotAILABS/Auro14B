"""Polyglot engine — AISVectorPolyglot backend for Octopus EMBED/MATCH arms."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from mesie.engines.base import Engine
from mesie.engines.embedding_engine import EmbeddingEngine
from mesie.internal_api.messages import EngineResponse, MessageEnvelope
from mesie.io.loaders import load_record
from mesie.library.user_corpus import embed_paths, load_user_index
from mesie.polyglot.contract import AISVectorMessage, PolyglotAction, RuntimeId, SUITE_NAME
from mesie.polyglot.suite import AISVectorPolyglotSuite


class PolyglotEngine(Engine):
    """Default Octopus EMBED + MATCH arm engine — vector bridge + multi-runtime dispatch."""

    name = "polyglot"
    capabilities = [
        "validate",
        "match",
        "embed",
        "transform",
        "rank",
        "vector_query",
        "query",
        "parity",
        "health",
        "index",
        "load_user_library",
        "embed_user_paths",
        "user_library_status",
    ]

    def __init__(self, suite: Optional[AISVectorPolyglotSuite] = None) -> None:
        self._suite = suite or AISVectorPolyglotSuite()
        self._embedding = EmbeddingEngine()
        self._corpus_count = 0

    @property
    def suite(self) -> AISVectorPolyglotSuite:
        return self._suite

    def _runtime(self, message: MessageEnvelope, action: str) -> RuntimeId:
        if "runtime" in message.payload:
            return RuntimeId(message.payload["runtime"])
        return self._suite.routing.get(PolyglotAction(action), RuntimeId.PYTHON)

    def _index_user_records(self) -> int:
        corpus = self._embedding.user_corpus
        if corpus is None:
            return 0
        paths = corpus.record_paths()
        if not paths:
            return 0
        from mesie.library.user_corpus import _load_spectral_file

        records = [_load_spectral_file(p) for p in paths]
        n = self._suite.vector.index(records)
        self._corpus_count = corpus.count
        return n

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        if message.target not in (self.name, "*"):
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown action: {action}")

        try:
            if action == "health":
                h = self._suite.health()
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {"suite": SUITE_NAME, "runtimes": h.runtimes, "vector_indexed": h.vector_indexed},
                )

            if action == "parity":
                a = load_record(message.payload["record_a"])
                b = load_record(message.payload["record_b"])
                return EngineResponse(True, self.name, action, self._suite.parity_matrix(a, b))

            if action in ("embed", "transform"):
                rec = load_record(message.payload["record"])
                runtime = self._runtime(message, "embed")
                resp = self._suite.embed(rec, runtime)
                if not resp.ok:
                    return EngineResponse(False, self.name, action, error=resp.error)
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {
                        "embedding": resp.vector,
                        "record_id": rec.record_id,
                        "runtime": resp.runtime.value,
                        "mode": self._suite.adapter(runtime).mode,
                        "latency_ms": resp.latency_ms,
                    },
                )

            if action in ("vector_query", "query"):
                rec = load_record(message.payload["record"])
                top_k = int(message.payload.get("top_k", 5))
                q = self._suite.vector_query(rec, top_k=top_k)
                neighbors = q.neighbors or [
                    {"item_id": h["item_id"], "similarity": h["similarity"]}
                    for h in q.fingerprint_hits
                ]
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {
                        "neighbors": neighbors,
                        "fingerprint_hits": q.fingerprint_hits,
                        "technical_hits": q.technical_hits,
                        "embedding": q.embedding,
                        "elapsed_ms": q.elapsed_ms,
                        "engine": SUITE_NAME,
                    },
                )

            if action == "match":
                a = load_record(message.payload["record_a"])
                b = load_record(message.payload["record_b"])
                runtime = self._runtime(message, "match")
                resp = self._suite.match(a, b, runtime)
                if not resp.ok:
                    return EngineResponse(False, self.name, action, error=resp.error)
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {
                        **resp.data,
                        "reference_id": a.record_id,
                        "candidate_id": b.record_id,
                        "runtime": resp.runtime.value,
                        "mode": self._suite.adapter(runtime).mode,
                        "latency_ms": resp.latency_ms,
                    },
                )

            if action == "rank":
                runtime = self._runtime(message, "rank")
                msg = AISVectorMessage(
                    action=PolyglotAction.RANK,
                    runtime=runtime,
                    record=message.payload.get("record") or message.payload.get("query"),
                    candidates=message.payload.get("candidates", []),
                    top_k=int(message.payload.get("top_k", 5)),
                )
                resp = self._suite.dispatch(msg)
                if not resp.ok:
                    return EngineResponse(False, self.name, action, error=resp.error)
                return EngineResponse(True, self.name, action, resp.data)

            if action == "embed_user_paths":
                paths = message.payload.get("paths", [])
                save_to = message.payload.get("save_to")
                corpus = embed_paths(paths, save_to=save_to)
                self._embedding._user_corpus = corpus
                n = self._index_user_records()
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {
                        "embedded": corpus.count,
                        "index_path": str(corpus.index_path) if corpus.index_path else None,
                        "vector_indexed": n,
                        "engine": SUITE_NAME,
                    },
                )

            if action == "load_user_library":
                index_path = Path(message.payload["index_path"])
                corpus = load_user_index(index_path)
                self._embedding._user_corpus = corpus
                n = self._index_user_records()
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {
                        "loaded": corpus.count,
                        "indexed_records": n,
                        "index_path": str(index_path),
                        "engine": SUITE_NAME,
                    },
                )

            if action == "user_library_status":
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {
                        "user_entries": self._embedding.user_corpus.count if self._embedding.user_corpus else 0,
                        "vector_indexed": self._suite.vector.state.n_indexed,
                        "corpus_count": self._corpus_count,
                        "engine": SUITE_NAME,
                    },
                )

            if action == "index":
                records = [load_record(r) for r in message.payload["records"]]
                n = self._suite.vector.index(records)
                return EngineResponse(True, self.name, action, {"indexed": n, "engine": SUITE_NAME})

            if action == "validate":
                rec = load_record(message.payload["record"])
                runtime = self._runtime(message, "validate")
                resp = self._suite.validate(rec, runtime)
                if not resp.ok:
                    return EngineResponse(False, self.name, action, error=resp.error)
                return EngineResponse(True, self.name, action, resp.data)

            return EngineResponse(False, self.name, action, error="unhandled")
        except (KeyError, TypeError, ValueError) as exc:
            return EngineResponse(False, self.name, action, error=str(exc))