"""Scriptural memory — doctrine-tagged embeddings that construct future behavior.

Memory is not a log side-channel. Records carry canon_id, article tags, and
vectors that are re-injected into generate / train residual streams.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_text(text: str, dim: int) -> np.ndarray:
    """Deterministic φ-ish bag hash embedding (MESIE-compatible, no torch)."""
    from auro_native_llm.model.phi_math import GOLDEN_ANGLE_RAD, PHI

    vec = np.zeros(dim, dtype=np.float64)
    tokens = text.lower().split() or [text]
    for i, tok in enumerate(tokens):
        h = hashlib.sha256(f"{i}:{tok}".encode()).digest()
        for j in range(dim):
            vec[j] += ((h[j % len(h)] / 255.0) - 0.5) * math.sin(j * GOLDEN_ANGLE_RAD + i / PHI)
    n = float(np.linalg.norm(vec)) or 1.0
    return vec / n


@dataclass
class MemoryRecord:
    record_id: str
    text: str
    embedding: List[float]
    canon_id: str
    model_id: str
    op: str
    article_ids: List[str] = field(default_factory=list)
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "text": self.text,
            "embedding_dim": len(self.embedding),
            "embedding": self.embedding,
            "canon_id": self.canon_id,
            "model_id": self.model_id,
            "op": self.op,
            "article_ids": list(self.article_ids),
            "importance": self.importance,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "content_sha256": _hash_text(self.text),
        }


class ScripturalMemory:
    """Bounded, doctrine-tagged memory with retrieval for residual fusion."""

    def __init__(
        self,
        capacity: int = 2048,
        embed_dim: int = 256,
        decay: float = 0.98,
        require_canon_tag: bool = True,
    ) -> None:
        self.capacity = capacity
        self.embed_dim = embed_dim
        self.decay = decay
        self.require_canon_tag = require_canon_tag
        self._records: List[MemoryRecord] = []
        self._prior_hash: str = "genesis"

    def __len__(self) -> int:
        return len(self._records)

    def write(
        self,
        text: str,
        *,
        canon_id: str,
        model_id: str,
        op: str,
        article_ids: Optional[Sequence[str]] = None,
        importance: float = 0.5,
        embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        if self.require_canon_tag and not canon_id:
            raise ValueError("scriptural memory write requires canon_id")
        emb = (
            list(embedding)
            if embedding is not None
            else _embed_text(text, self.embed_dim).tolist()
        )
        if len(emb) != self.embed_dim:
            # project/pad
            v = np.zeros(self.embed_dim, dtype=np.float64)
            a = np.asarray(emb, dtype=np.float64).ravel()
            v[: min(len(a), self.embed_dim)] = a[: self.embed_dim]
            n = float(np.linalg.norm(v)) or 1.0
            emb = (v / n).tolist()

        rec = MemoryRecord(
            record_id=f"sm-{uuid.uuid4().hex[:12]}",
            text=text,
            embedding=emb,
            canon_id=canon_id,
            model_id=model_id,
            op=op,
            article_ids=list(article_ids or []),
            importance=float(importance),
            metadata={
                **(metadata or {}),
                "prior_hash": self._prior_hash,
            },
        )
        self._records.append(rec)
        self._prior_hash = _hash_text(self._prior_hash + rec.record_id + text)
        self._decay_and_trim()
        return rec

    def _decay_and_trim(self) -> None:
        now = time.time()
        ttl_days = 30.0
        promote = 0.8
        kept: List[MemoryRecord] = []
        for r in self._records:
            r.importance *= self.decay
            age_days = (now - r.created_at) / 86400.0
            # MR1: expire after ttl unless promoted by importance
            if age_days > ttl_days and r.importance < promote:
                continue
            kept.append(r)
        self._records = kept
        self._records.sort(key=lambda r: r.importance, reverse=True)
        if len(self._records) > self.capacity:
            self._records = self._records[: self.capacity]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        query_embedding: Optional[Sequence[float]] = None,
    ) -> List[MemoryRecord]:
        if not self._records:
            return []
        q = (
            np.asarray(query_embedding, dtype=np.float64)
            if query_embedding is not None
            else _embed_text(query, self.embed_dim)
        )
        qn = float(np.linalg.norm(q)) or 1.0
        q = q / qn
        scored = []
        for r in self._records:
            v = np.asarray(r.embedding, dtype=np.float64)
            vn = float(np.linalg.norm(v)) or 1.0
            sim = float(np.dot(q, v / vn))
            scored.append((sim * (0.5 + 0.5 * r.importance), r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    def context_block(self, query: str, top_k: int = 3) -> str:
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return ""
        lines = ["[SCRIPTURAL_MEMORY]"]
        for h in hits:
            lines.append(f"- ({h.op}|{h.model_id}|imp={h.importance:.2f}) {h.text[:240]}")
        lines.append("[/SCRIPTURAL_MEMORY]")
        return "\n".join(lines)

    def fused_vector(self, query: str, dim: Optional[int] = None, top_k: int = 5) -> np.ndarray:
        d = dim or self.embed_dim
        hits = self.retrieve(query, top_k=top_k)
        acc = np.zeros(d, dtype=np.float64)
        if not hits:
            return acc
        wsum = 0.0
        for h in hits:
            v = np.asarray(h.embedding, dtype=np.float64)
            if v.size < d:
                pad = np.zeros(d)
                pad[: v.size] = v
                v = pad
            else:
                v = v[:d]
            w = max(h.importance, 0.01)
            acc += w * v
            wsum += w
        acc /= max(wsum, 1e-9)
        n = float(np.linalg.norm(acc)) or 1.0
        return acc / n

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "auro.scripture.memory.v1",
            "capacity": self.capacity,
            "embed_dim": self.embed_dim,
            "decay": self.decay,
            "prior_hash": self._prior_hash,
            "records": [r.to_dict() for r in self._records],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ScripturalMemory":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        mem = cls(
            capacity=int(data.get("capacity", 2048)),
            embed_dim=int(data.get("embed_dim", 256)),
            decay=float(data.get("decay", 0.98)),
        )
        mem._prior_hash = str(data.get("prior_hash", "genesis"))
        for r in data.get("records", []):
            mem._records.append(
                MemoryRecord(
                    record_id=r["record_id"],
                    text=r["text"],
                    embedding=list(r.get("embedding", [])),
                    canon_id=r["canon_id"],
                    model_id=r["model_id"],
                    op=r["op"],
                    article_ids=list(r.get("article_ids", [])),
                    importance=float(r.get("importance", 0.5)),
                    created_at=float(r.get("created_at", time.time())),
                    metadata=dict(r.get("metadata", {})),
                )
            )
        return mem

    def stats(self) -> Dict[str, Any]:
        return {
            "count": len(self._records),
            "capacity": self.capacity,
            "embed_dim": self.embed_dim,
            "prior_hash": self._prior_hash,
            "ops": sorted({r.op for r in self._records}),
        }
