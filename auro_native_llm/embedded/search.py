"""Embedded online + local search organ.

Online: DuckDuckGo instant answer / lite HTML (no API key).
Offline fallback: corpus keyword search over bundled docs.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str
    source: str  # online | local | cache

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
        }


class OnlineSearch:
    def __init__(self, timeout: float = 8.0, offline: bool = False) -> None:
        self.timeout = timeout
        self.offline = offline
        self.cache: Dict[str, List[SearchHit]] = {}
        self._local_docs: List[Dict[str, str]] = []
        self._load_local()

    def _load_local(self) -> None:
        root = Path(__file__).resolve().parents[2]
        for pattern in ("README.md", "docs/**/*.md", "native_llm/**/*.md", "examples/**/*.py"):
            for p in root.glob(pattern):
                if not p.is_file():
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")[:8000]
                except Exception:
                    continue
                self._local_docs.append({"path": str(p.relative_to(root)), "text": text})

    def _local_search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        q = query.lower().split()
        scored = []
        for doc in self._local_docs:
            text_l = doc["text"].lower()
            score = sum(1 for w in q if w in text_l)
            if score:
                # snippet around first hit
                idx = min((text_l.find(w) for w in q if w in text_l), default=0)
                snip = doc["text"][max(0, idx) : max(0, idx) + 240]
                scored.append((score, SearchHit(doc["path"], f"local://{doc['path']}", snip, "local")))
        scored.sort(key=lambda x: -x[0])
        return [h for _, h in scored[:top_k]]

    def _online_ddg(self, query: str, top_k: int = 5) -> List[SearchHit]:
        # Instant Answer API
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        )
        req = urllib.request.Request(url, headers={"User-Agent": "AuroMind/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        hits: List[SearchHit] = []
        if data.get("AbstractText"):
            hits.append(
                SearchHit(
                    title=data.get("Heading") or query,
                    url=data.get("AbstractURL") or "https://duckduckgo.com",
                    snippet=data.get("AbstractText", "")[:400],
                    source="online",
                )
            )
        for t in data.get("RelatedTopics") or []:
            if isinstance(t, dict) and t.get("Text"):
                hits.append(
                    SearchHit(
                        title=(t.get("Text") or "")[:80],
                        url=t.get("FirstURL") or "",
                        snippet=(t.get("Text") or "")[:300],
                        source="online",
                    )
                )
            if len(hits) >= top_k:
                break
        return hits[:top_k]

    def search(self, query: str, *, top_k: int = 5, online: bool = True) -> Dict[str, Any]:
        key = f"{query}|{top_k}|{online}"
        if key in self.cache:
            return {
                "ok": True,
                "action": "search",
                "query": query,
                "hits": [h.to_dict() for h in self.cache[key]],
                "cached": True,
            }
        hits: List[SearchHit] = []
        err = None
        if online and not self.offline:
            try:
                hits = self._online_ddg(query, top_k=top_k)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
        # MESIE multi-repo corpus (all GitHub + local monorepos)
        try:
            from auro_native_llm.corpus.bridge import get_index

            idx = get_index(include_github=False)
            for h in idx.search(query, top_k=top_k):
                hits.append(
                    SearchHit(
                        title=h.get("title", ""),
                        url=h.get("url", ""),
                        snippet=h.get("snippet", ""),
                        source="mesie_corpus",
                    )
                )
        except Exception:
            pass
        if len(hits) < top_k:
            local = self._local_search(query, top_k=top_k)
            seen = {h.url for h in hits}
            for h in local:
                if h.url not in seen:
                    hits.append(h)
                if len(hits) >= top_k * 2:
                    break
        # de-dupe by url
        dedup: List[SearchHit] = []
        seen_u = set()
        for h in hits:
            if h.url in seen_u:
                continue
            seen_u.add(h.url)
            dedup.append(h)
        hits = dedup[: max(top_k, 5)]
        self.cache[key] = hits
        return {
            "ok": True,
            "action": "search",
            "query": query,
            "hits": [h.to_dict() for h in hits],
            "online_error": err,
            "cached": False,
            "corpus": "mesie_multi_repo",
        }


class SearchOrgan:
    """Mind-facing search organ."""

    def __init__(self, offline: bool = False) -> None:
        self.engine = OnlineSearch(offline=offline)

    def search(self, query: str, **kw: Any) -> Dict[str, Any]:
        return self.engine.search(query, **kw)

    def teach_snippet(self) -> str:
        return (
            "SEARCH TOOL: ACTION: search query=\"...\" "
            "Returns web + local MESIE/Auro corpus hits. Prefer local for repo facts."
        )
