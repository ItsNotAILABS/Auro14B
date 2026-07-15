"""Embedding engine — vectorize and index spectra."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from pathlib import Path

from mesie.embeddings.retrieval import SpectralRetriever
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.engines.base import Engine
from mesie.internal_api.messages import EngineResponse, MessageEnvelope
from mesie.io.loaders import load_record, RecordInput
from mesie.library.user_corpus import UserSpectralCorpus, embed_paths, load_user_index


class EmbeddingEngine(Engine):
    name = "embedding"
    capabilities = [
        "transform",
        "batch_transform",
        "index",
        "query",
        "workflow_embed",
        "load_user_library",
        "embed_user_paths",
        "user_library_status",
    ]

    def __init__(self, n_bands: int = 8) -> None:
        self._vectorizer = SpectralVectorizer(n_bands=n_bands)
        self._retriever: Optional[SpectralRetriever] = None
        self._indexed_ids: List[str] = []
        self._user_corpus: Optional[UserSpectralCorpus] = None

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        if message.target != self.name and message.target != "*":
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown action: {action}")

        try:
            if action == "transform":
                rec = load_record(message.payload["record"])
                emb = self._vectorizer.transform(rec)
                return EngineResponse(True, self.name, action, {"embedding": emb.tolist(), "record_id": rec.record_id})

            if action == "batch_transform":
                records = [load_record(r) for r in message.payload["records"]]
                embs = self._vectorizer.batch_transform(records)
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {"embeddings": [e.tolist() for e in embs], "count": len(records)},
                )

            if action == "index":
                records = [load_record(r) for r in message.payload["records"]]
                self._retriever = SpectralRetriever(self._vectorizer)
                self._retriever.index(records)
                self._indexed_ids = [r.record_id for r in records]
                return EngineResponse(True, self.name, action, {"indexed": len(records)})

            if action == "query":
                if self._retriever is None:
                    return EngineResponse(False, self.name, action, error="Index not built; call index first")
                rec = load_record(message.payload["record"])
                top_k = int(message.payload.get("top_k", 3))
                hits = self._retriever.query(rec, top_k=top_k)
                return EngineResponse(True, self.name, action, {"neighbors": hits})

            if action == "workflow_embed":
                steps = message.payload.get("steps", [])
                vec = self._embed_workflow_state(steps)
                return EngineResponse(True, self.name, action, {"workflow_embedding": vec.tolist(), "step_count": len(steps)})

            if action == "embed_user_paths":
                paths = message.payload.get("paths", [])
                save_to = message.payload.get("save_to")
                corpus = embed_paths(paths, vectorizer=self._vectorizer, save_to=save_to)
                self._user_corpus = corpus
                self._ensure_retriever_with_user()
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {"embedded": corpus.count, "index_path": str(corpus.index_path) if corpus.index_path else None},
                )

            if action == "load_user_library":
                index_path = Path(message.payload["index_path"])
                corpus = load_user_index(index_path)
                self._user_corpus = corpus
                n = self._ensure_retriever_with_user()
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {"loaded": corpus.count, "indexed_records": n, "index_path": str(index_path)},
                )

            if action == "user_library_status":
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {
                        "user_entries": self._user_corpus.count if self._user_corpus else 0,
                        "indexed_ids": len(self._indexed_ids),
                        "has_retriever": self._retriever is not None,
                    },
                )

        except (KeyError, TypeError, ValueError) as exc:
            return EngineResponse(False, self.name, action, error=str(exc))

        return EngineResponse(False, self.name, action, error="Unhandled")

    def _embed_workflow_state(self, steps: List[Dict[str, Any]]) -> np.ndarray:
        """Compact workflow fingerprint for cross-arm logic."""
        flags = np.zeros(8, dtype=np.float64)
        for i, step in enumerate(steps[:8]):
            flags[i] = 1.0 if step.get("done") else 0.0
            flags[i] += 0.1 * float(step.get("priority", 0))
        return flags

    def _ensure_retriever_with_user(self) -> int:
        if self._user_corpus is None:
            return 0
        paths = self._user_corpus.record_paths()
        if not paths:
            return 0
        from mesie.library.user_corpus import _load_spectral_file

        records = [_load_spectral_file(p) for p in paths]
        self._retriever = SpectralRetriever(self._vectorizer)
        self._retriever.index(records)
        self._indexed_ids = [r.record_id for r in records]
        return len(records)

    @property
    def user_corpus(self) -> Optional[UserSpectralCorpus]:
        return self._user_corpus