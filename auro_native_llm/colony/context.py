"""500k-token logical context window — hierarchical, not single-pass attention.

Capacity is real storage + multi-scale summary + retrieval, so the colony can
*work with* ~500k tokens of history/skills/docs without one giant Softmax.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ~4 chars/token heuristic (English-ish)
CHARS_PER_TOKEN = 4
DEFAULT_TOKEN_BUDGET = 500_000


@dataclass
class ContextChunk:
    id: str
    text: str
    kind: str  # skill | doc | chat | system | germ
    tokens_est: int
    ts: float = field(default_factory=time.time)
    summary: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "tokens_est": self.tokens_est,
            "summary": self.summary[:200],
            "text_preview": self.text[:120],
            "meta": self.meta,
        }


def estimate_tokens(text: str) -> int:
    return max(1, (len(text or "") + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def summarize_chunk(text: str, max_chars: int = 240) -> str:
    t = " ".join((text or "").split())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


class Context500k:
    """Logical 500k-token context bank for the colony."""

    def __init__(self, token_budget: int = DEFAULT_TOKEN_BUDGET) -> None:
        self.token_budget = int(token_budget)
        self.chunks: Dict[str, ContextChunk] = {}
        self.order: List[str] = []  # oldest → newest
        self.scale_summaries: Dict[str, str] = {}  # coarse maps

    @property
    def tokens_used(self) -> int:
        return sum(c.tokens_est for c in self.chunks.values())

    def remaining(self) -> int:
        return max(0, self.token_budget - self.tokens_used)

    def _evict_if_needed(self) -> None:
        while self.tokens_used > self.token_budget and self.order:
            cid = self.order.pop(0)
            # fold into scale summary before drop
            ch = self.chunks.pop(cid, None)
            if ch:
                key = f"evicted:{ch.kind}"
                prev = self.scale_summaries.get(key, "")
                self.scale_summaries[key] = (prev + " | " + ch.summary)[-2000:]

    def ingest(
        self,
        text: str,
        kind: str = "doc",
        *,
        chunk_tokens: int = 1024,
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Split text into chunks and store under budget."""
        raw = text or ""
        if not raw.strip():
            return []
        max_chars = chunk_tokens * CHARS_PER_TOKEN
        ids: List[str] = []
        for i in range(0, len(raw), max_chars):
            piece = raw[i : i + max_chars]
            cid = "c-" + hashlib.sha1(f"{kind}:{i}:{piece[:64]}".encode()).hexdigest()[:12]
            ch = ContextChunk(
                id=cid,
                text=piece,
                kind=kind,
                tokens_est=estimate_tokens(piece),
                summary=summarize_chunk(piece),
                meta=meta or {},
            )
            if cid not in self.chunks:
                self.order.append(cid)
            self.chunks[cid] = ch
            ids.append(cid)
            self._evict_if_needed()
        self._rebuild_maps()
        return ids

    def _rebuild_maps(self) -> None:
        by_kind: Dict[str, List[str]] = {}
        for c in self.chunks.values():
            by_kind.setdefault(c.kind, []).append(c.summary)
        for k, sums in by_kind.items():
            self.scale_summaries[f"map:{k}"] = " · ".join(sums[:40])[:4000]

    def retrieve(self, query: str, top_k: int = 12, token_cap: int = 8000) -> str:
        """Retrieve most relevant chunk text under a local token cap."""
        q = (query or "").lower().split()
        scored = []
        for c in self.chunks.values():
            blob = (c.text + " " + c.summary + " " + c.kind).lower()
            s = sum(1 for t in q if t in blob)
            if s:
                scored.append((s, c.ts, c))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        parts: List[str] = []
        used = 0
        for _, _, c in scored[: top_k * 2]:
            t = c.text
            te = c.tokens_est
            if used + te > token_cap:
                continue
            parts.append(f"[{c.kind}/{c.id}] {t}")
            used += te
            if len(parts) >= top_k:
                break
        # always include coarse maps
        maps = "\n".join(f"{k}: {v[:300]}" for k, v in list(self.scale_summaries.items())[:8])
        body = "\n---\n".join(parts)
        if maps:
            body = f"[CONTEXT_MAPS tokens={self.tokens_used}/{self.token_budget}]\n{maps}\n\n{body}"
        return body

    def stats(self) -> Dict[str, Any]:
        by_kind: Dict[str, int] = {}
        for c in self.chunks.values():
            by_kind[c.kind] = by_kind.get(c.kind, 0) + c.tokens_est
        return {
            "schema": "auro.colony.context500k.v1",
            "token_budget": self.token_budget,
            "tokens_used": self.tokens_used,
            "utilization": self.tokens_used / max(self.token_budget, 1),
            "n_chunks": len(self.chunks),
            "by_kind_tokens": by_kind,
            "maps": list(self.scale_summaries.keys()),
        }
