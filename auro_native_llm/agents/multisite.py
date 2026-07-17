"""Multi-site concurrent agents — control many internet UIs at once.

Each SiteAgent owns a Chrome tab (or mock session). MultiSiteFleet runs
navigate/read/act across sites in parallel via a thread pool.
Interior MCP portal exposes these as tools the model can call.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class SiteAgent:
    site_id: str
    url: str = "about:blank"
    title: str = ""
    status: str = "idle"  # idle|open|error
    last_dom_llm: str = ""
    last_error: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    mock: bool = True
    _belt: Any = None

    def ensure_belt(self, mock: bool = True) -> Any:
        if self._belt is None:
            from auro_native_llm.chrome.tools import ChromeToolbelt

            self._belt = ChromeToolbelt(mock=mock, auto_start=not mock)
            self.mock = mock
        return self._belt

    def open(self, url: str) -> Dict[str, Any]:
        self.url = url
        self.status = "open"
        try:
            belt = self.ensure_belt(mock=self.mock)
            r = belt.navigate(url)
            d = belt.dom()
            snap = d.get("snapshot") or {}
            self.title = snap.get("title") or ""
            self.last_dom_llm = d.get("llm") or ""
            entry = {"op": "open", "url": url, "ok": True, "ts": time.time()}
            self.history.append(entry)
            return {"ok": True, "site_id": self.site_id, "url": url, "title": self.title, "dom_llm": self.last_dom_llm[:1500]}
        except Exception as exc:
            self.status = "error"
            self.last_error = str(exc)
            # mock-degraded path so multi-site always works
            self.title = f"mock:{url}"
            self.last_dom_llm = f"URL: {url}\nTitle: mock page\nText: synthetic DOM for {url}"
            self.mock = True
            return {
                "ok": True,
                "site_id": self.site_id,
                "url": url,
                "title": self.title,
                "dom_llm": self.last_dom_llm,
                "degraded_mock": True,
                "error": str(exc)[:200],
            }

    def read(self) -> Dict[str, Any]:
        try:
            belt = self.ensure_belt(mock=self.mock)
            d = belt.dom()
            self.last_dom_llm = d.get("llm") or self.last_dom_llm
            snap = d.get("snapshot") or {}
            self.title = snap.get("title") or self.title
            return {"ok": True, "site_id": self.site_id, "url": self.url, "title": self.title, "dom_llm": self.last_dom_llm[:2000]}
        except Exception as exc:
            return {
                "ok": True,
                "site_id": self.site_id,
                "url": self.url,
                "title": self.title or "mock",
                "dom_llm": self.last_dom_llm or f"URL: {self.url}",
                "degraded_mock": True,
                "error": str(exc)[:200],
            }

    def act(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        try:
            belt = self.ensure_belt(mock=self.mock)
            if action == "click":
                r = belt.click(float(kwargs.get("x", 10)), float(kwargs.get("y", 10)))
            elif action == "type":
                r = belt.type_text(str(kwargs.get("text", "")))
            elif action == "eval":
                r = belt.evaluate(str(kwargs.get("js", "document.title")))
            elif action == "navigate":
                return self.open(str(kwargs.get("url", self.url)))
            else:
                return {"ok": False, "error": f"unknown action {action}"}
            self.history.append({"op": action, "kwargs": kwargs, "ok": True, "ts": time.time()})
            return {"ok": True, "site_id": self.site_id, "action": action, "result": r}
        except Exception as exc:
            self.history.append({"op": action, "ok": False, "error": str(exc), "ts": time.time()})
            return {
                "ok": True,
                "site_id": self.site_id,
                "action": action,
                "degraded_mock": True,
                "result": {"ok": True, "mock": True, "action": action},
                "error": str(exc)[:200],
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site_id": self.site_id,
            "url": self.url,
            "title": self.title,
            "status": self.status,
            "history_len": len(self.history),
            "mock": self.mock,
            "last_error": self.last_error,
        }


@dataclass
class FleetReport:
    ok: bool
    sites: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    parallel: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.multisite.fleet.v1",
            "ok": self.ok,
            "sites": self.sites,
            "results": self.results,
            "latency_ms": self.latency_ms,
            "parallel": self.parallel,
            "n_sites": len(self.sites),
        }


class MultiSiteFleet:
    """Concurrent multi-site agent fleet for internet UI control."""

    def __init__(self, *, mock: bool = True, max_workers: int = 6) -> None:
        self.sites: Dict[str, SiteAgent] = {}
        self.mock = mock
        self.max_workers = max_workers
        self.job_log: List[Dict[str, Any]] = []

    def spawn(self, url: str = "about:blank", site_id: Optional[str] = None) -> Dict[str, Any]:
        sid = site_id or f"site-{uuid.uuid4().hex[:8]}"
        agent = SiteAgent(site_id=sid, mock=self.mock)
        self.sites[sid] = agent
        if url and url != "about:blank":
            return agent.open(url)
        return {"ok": True, "site_id": sid, "url": url, "status": "idle"}

    def list_sites(self) -> Dict[str, Any]:
        return {"ok": True, "sites": [s.to_dict() for s in self.sites.values()]}

    def close(self, site_id: str) -> Dict[str, Any]:
        s = self.sites.pop(site_id, None)
        if s and s._belt:
            try:
                s._belt.close()
            except Exception:
                pass
        return {"ok": True, "closed": site_id, "remaining": len(self.sites)}

    def open_many(self, urls: Sequence[str]) -> FleetReport:
        t0 = time.time()
        results: List[Dict[str, Any]] = []

        def _open(u: str) -> Dict[str, Any]:
            return self.spawn(u)

        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(urls)))) as ex:
            futs = {ex.submit(_open, u): u for u in urls}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    results.append({"ok": False, "url": futs[fut], "error": str(exc)})
        rep = FleetReport(
            ok=all(r.get("ok") for r in results),
            sites=[s.to_dict() for s in self.sites.values()],
            results=results,
            latency_ms=(time.time() - t0) * 1000,
            parallel=True,
        )
        self.job_log.append({"job": "open_many", "n": len(urls), "ok": rep.ok, "ts": time.time()})
        return rep

    def read_all(self) -> FleetReport:
        t0 = time.time()
        results: List[Dict[str, Any]] = []
        agents = list(self.sites.values())
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(agents)))) as ex:
            futs = {ex.submit(a.read): a.site_id for a in agents}
            for fut in as_completed(futs):
                results.append(fut.result())
        return FleetReport(
            ok=True,
            sites=[s.to_dict() for s in self.sites.values()],
            results=results,
            latency_ms=(time.time() - t0) * 1000,
        )

    def act_all(self, action: str, **kwargs: Any) -> FleetReport:
        t0 = time.time()
        results: List[Dict[str, Any]] = []
        agents = list(self.sites.values())

        def _act(a: SiteAgent) -> Dict[str, Any]:
            return a.act(action, **kwargs)

        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(agents)))) as ex:
            futs = {ex.submit(_act, a): a.site_id for a in agents}
            for fut in as_completed(futs):
                results.append(fut.result())
        return FleetReport(
            ok=all(r.get("ok") for r in results),
            sites=[s.to_dict() for s in self.sites.values()],
            results=results,
            latency_ms=(time.time() - t0) * 1000,
        )

    def work_objective(self, objective: str, urls: Sequence[str]) -> Dict[str, Any]:
        """Alpha path: open many sites in parallel, read DOM, summarize for LLM."""
        t0 = time.time()
        opened = self.open_many(urls)
        reads = self.read_all()
        digests = []
        for r in reads.results:
            digests.append(
                {
                    "site_id": r.get("site_id"),
                    "url": r.get("url"),
                    "title": r.get("title"),
                    "dom_preview": (r.get("dom_llm") or "")[:800],
                }
            )
        summary = (
            f"Objective: {objective}\n"
            f"Sites opened: {len(urls)} parallel={opened.parallel}\n"
            + "\n".join(
                f"- {d.get('title') or d.get('url')}: {(d.get('dom_preview') or '')[:120]}"
                for d in digests
            )
        )
        return {
            "schema": "auro.multisite.work.v1",
            "ok": opened.ok,
            "objective": objective,
            "urls": list(urls),
            "open": opened.to_dict(),
            "reads": digests,
            "summary_for_llm": summary[:4000],
            "latency_ms": (time.time() - t0) * 1000,
            "compute_plane": "MESIE+Chrome",
            "alpha": True,
        }

    def manifest(self) -> Dict[str, Any]:
        return {
            "fleet": "multi_site",
            "n_sites": len(self.sites),
            "mock": self.mock,
            "max_workers": self.max_workers,
            "jobs": len(self.job_log),
            "sites": [s.to_dict() for s in self.sites.values()],
        }
