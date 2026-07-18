"""One virtual envelope: AI sandbox Google suite + optional real Chrome + collab bridge.

Everything Google-related the AI can use is routed through this envelope so:
  - actions are sandboxed / receipted
  - operator credentials are never silently used
  - collab projects share selected artifacts with the human
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from auro_native_llm.gworkspace.collab import CollabWorkspace, get_collab
from auro_native_llm.gworkspace.suite import GOOGLE_PROPERTIES, GoogleSuite

_HOME = Path.home()
_ENV_ROOT = _HOME / ".auro_workspace" / "google_sandbox"
_ENVELOPE: Optional["GoogleVirtualEnvelope"] = None


@dataclass
class EnvelopeReceipt:
    id: str
    action: str
    surface: str  # chrome | mail | drive | search | calendar | collab | sites
    payload: Dict[str, Any]
    ok: bool = True
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "surface": self.surface,
            "payload": self.payload,
            "ok": self.ok,
            "ts": self.ts,
        }


class GoogleVirtualEnvelope:
    """Single virtual Google workspace the AI owns + collab link to the human."""

    def __init__(
        self,
        mind: Any = None,
        *,
        chrome_mock: bool = True,
        root: Optional[Path] = None,
    ) -> None:
        self.mind = mind
        self.root = Path(root or _ENV_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)
        self.profile_dir = self.root / "chrome_profile"
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.suite = GoogleSuite(owner="auro-ai@sandbox.local")
        self.collab: CollabWorkspace = get_collab()
        self.receipts: List[EnvelopeReceipt] = []
        self.tabs: List[Dict[str, Any]] = []  # virtual + real tab log
        self.chrome_mock = chrome_mock
        self._chrome = None
        self.envelope_id = f"env-{uuid.uuid4().hex[:10]}"
        self.born_at = time.time()
        self._seed_sandbox_files()

    def _seed_sandbox_files(self) -> None:
        meta = {
            "envelope_id": self.envelope_id,
            "owner": self.suite.owner,
            "born_at": self.born_at,
            "claim_boundary": (
                "This is the AI sandbox Google envelope. "
                "It is NOT the operator's Google account. "
                "Real authenticated Google APIs require explicit OAuth + human approval."
            ),
        }
        (self.root / "ENVELOPE.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _receipt(self, action: str, surface: str, payload: Dict[str, Any], ok: bool = True) -> EnvelopeReceipt:
        r = EnvelopeReceipt(
            id=f"er-{uuid.uuid4().hex[:10]}",
            action=action,
            surface=surface,
            payload=payload,
            ok=ok,
        )
        self.receipts.append(r)
        if len(self.receipts) > 500:
            self.receipts = self.receipts[-500:]
        return r

    def _get_chrome(self) -> Any:
        if self._chrome is not None:
            return self._chrome
        from auro_native_llm.chrome.tools import ChromeToolbelt

        # dedicated sandbox profile via CDP user-data when real
        mock = self.chrome_mock
        if self.mind is not None and getattr(self.mind, "organs", None):
            existing = getattr(self.mind.organs, "chrome", None)
            if existing is not None and self.chrome_mock:
                self._chrome = existing
                return self._chrome
        self._chrome = ChromeToolbelt(mock=mock, auto_start=False, headless=True)
        return self._chrome

    # ------------------------------------------------------------------ catalog
    def surfaces(self) -> Dict[str, Any]:
        return {
            "schema": "auro.google_envelope.v1",
            "envelope_id": self.envelope_id,
            "surfaces": {
                "chrome": "Real CDP when available, else mock; AI profile dir isolated",
                "search": "Virtual search log + live Google Search URL via Chrome",
                "mail": "Virtual Gmail sandbox (AI-owned)",
                "drive": "Virtual Drive/Docs sandbox",
                "docs": "Virtual docs (drive kind=doc)",
                "calendar": "Virtual calendar",
                "sites": "Catalog of Google properties (navigate)",
                "collab": "Shared user+AI project workspace",
            },
            "google_properties": list(GOOGLE_PROPERTIES.keys()),
            "suite": self.suite.catalog(),
            "collab": self.collab.health(),
            "n_receipts": len(self.receipts),
            "n_tabs": len(self.tabs),
            "profile_dir": str(self.profile_dir),
            "claim_boundary": (
                "Sandbox envelope = AI's own virtual Google. "
                "Collab = shared with human. Real Gmail/Drive OAuth not enabled by default."
            ),
        }

    # ------------------------------------------------------------------ chrome / sites
    def open_property(self, name: str) -> Dict[str, Any]:
        key = name.lower().strip()
        prop = GOOGLE_PROPERTIES.get(key)
        if not prop:
            return {
                "ok": False,
                "error": f"unknown property {name}",
                "known": list(GOOGLE_PROPERTIES),
            }
        return self.navigate(prop["url"], surface="sites", label=prop["name"])

    def navigate(self, url: str, *, surface: str = "chrome", label: str = "") -> Dict[str, Any]:
        chrome = self._get_chrome()
        try:
            result = chrome.navigate(url)
            mode = "mock" if chrome.cdp.mock else "real_cdp"
            ok = bool(result.get("ok", True))
        except Exception as exc:
            # virtual tab fallback
            result = {"ok": True, "action": "navigate_virtual", "url": url, "error": str(exc)[:200]}
            mode = "virtual_only"
            ok = True
        tab = {
            "url": url,
            "label": label or url,
            "mode": mode,
            "ts": time.time(),
            "tab_id": f"tab-{uuid.uuid4().hex[:8]}",
        }
        self.tabs.append(tab)
        self.collab.track_url(url)
        rec = self._receipt("navigate", surface, {"url": url, "mode": mode, "label": label}, ok=ok)
        return {"ok": ok, "tab": tab, "chrome": result, "receipt": rec.to_dict()}

    def dom(self) -> Dict[str, Any]:
        chrome = self._get_chrome()
        try:
            snap = chrome.dom()
            rec = self._receipt("dom", "chrome", {"ok": True})
            return {"ok": True, "dom": snap, "receipt": rec.to_dict()}
        except Exception as exc:
            rec = self._receipt("dom", "chrome", {"error": str(exc)[:200]}, ok=False)
            last = self.tabs[-1] if self.tabs else {}
            return {
                "ok": True,
                "dom": {
                    "virtual": True,
                    "url": last.get("url"),
                    "note": "Mock/virtual DOM — grant real Chrome for live snapshot",
                },
                "receipt": rec.to_dict(),
            }

    def grant_real_chrome(self) -> Dict[str, Any]:
        """Try to open real Chrome CDP for the AI sandbox (still separate profile intent)."""
        self.chrome_mock = False
        chrome = self._get_chrome()
        chrome.cdp.mock = False
        out = chrome.grant_access(prefer_real=True)
        rec = self._receipt("grant_chrome", "chrome", out, ok=bool(out.get("ok")))
        return {"ok": out.get("ok"), "grant": out, "receipt": rec.to_dict()}

    # ------------------------------------------------------------------ suite ops
    def search(self, query: str, *, open_browser: bool = True) -> Dict[str, Any]:
        s = self.suite.search.query(query)
        nav = None
        if open_browser:
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            nav = self.navigate(url, surface="search", label=f"Search: {query}")
        rec = self._receipt("search", "search", {"q": query}, ok=True)
        return {"ok": True, "search": s, "browser": nav, "receipt": rec.to_dict()}

    def mail(self, action: str, **kw: Any) -> Dict[str, Any]:
        m = self.suite.mail
        if action == "list":
            out = {"ok": True, "messages": m.list_messages(kw.get("folder", "inbox"))}
        elif action == "read":
            out = m.read(str(kw.get("id") or kw.get("message_id") or ""))
        elif action == "compose":
            out = m.compose(
                str(kw.get("to") or "human@collab.local"),
                str(kw.get("subject") or "(no subject)"),
                str(kw.get("body") or ""),
                send=bool(kw.get("send", True)),
            )
            # mirror important mail into collab chat
            if out.get("ok") and kw.get("share_collab", True):
                self.collab.post(
                    f"[AI sandbox mail → {kw.get('to')}] {kw.get('subject')}\n{str(kw.get('body') or '')[:500]}",
                    author="ai",
                    meta={"surface": "mail"},
                )
        elif action == "search":
            out = {"ok": True, "messages": m.search(str(kw.get("query") or ""))}
        else:
            out = {"ok": False, "error": f"unknown mail action {action}"}
        rec = self._receipt(f"mail.{action}", "mail", {"action": action}, ok=bool(out.get("ok", True)))
        out["receipt"] = rec.to_dict()
        return out

    def drive(self, action: str, **kw: Any) -> Dict[str, Any]:
        d = self.suite.drive
        if action == "list":
            out = {"ok": True, "files": d.list_files(str(kw.get("parent_id") or "root"))}
        elif action == "create":
            out = d.create(
                str(kw.get("name") or "Untitled"),
                content=str(kw.get("content") or ""),
                kind=str(kw.get("kind") or "doc"),
            )
            if out.get("ok") and kw.get("share_collab", True):
                name = str(kw.get("name") or "Untitled")
                self.collab.write_doc(f"drive/{name}", str(kw.get("content") or ""))
        elif action == "read":
            out = d.read(str(kw.get("id") or kw.get("file_id") or ""))
        elif action == "write":
            out = d.write(str(kw.get("id") or ""), str(kw.get("content") or ""))
            if out.get("ok") and kw.get("share_collab"):
                f = (out.get("file") or {})
                self.collab.write_doc(f"drive/{f.get('name', 'doc')}", str(kw.get("content") or ""))
        elif action == "search":
            out = {"ok": True, "files": d.search(str(kw.get("query") or ""))}
        else:
            out = {"ok": False, "error": f"unknown drive action {action}"}
        rec = self._receipt(f"drive.{action}", "drive", {"action": action}, ok=bool(out.get("ok", True)))
        out["receipt"] = rec.to_dict()
        return out

    def calendar(self, action: str, **kw: Any) -> Dict[str, Any]:
        c = self.suite.calendar
        if action == "list":
            out = {"ok": True, "events": c.list_events()}
        elif action == "add":
            now = time.time()
            out = c.add(
                str(kw.get("title") or "Event"),
                float(kw.get("start_ts") or now + 3600),
                float(kw.get("end_ts") or now + 7200),
                description=str(kw.get("description") or ""),
            )
        else:
            out = {"ok": False, "error": f"unknown calendar action {action}"}
        rec = self._receipt(f"calendar.{action}", "calendar", {"action": action}, ok=bool(out.get("ok", True)))
        out["receipt"] = rec.to_dict()
        return out

    # ------------------------------------------------------------------ collab
    def collab_post(self, text: str, author: str = "user") -> Dict[str, Any]:
        if author == "user" and self.mind is not None:
            out = self.collab.ai_turn(self.mind, text)
        else:
            out = self.collab.post(text, author=author)
        rec = self._receipt("collab.post", "collab", {"author": author}, ok=bool(out.get("ok", True)))
        out["receipt"] = rec.to_dict()
        return out

    def collab_project(self, name: str, description: str = "") -> Dict[str, Any]:
        out = self.collab.create_project(name, description)
        rec = self._receipt("collab.project", "collab", {"name": name}, ok=True)
        out["receipt"] = rec.to_dict()
        return out

    # ------------------------------------------------------------------ unified dispatch
    def act(self, surface: str, action: str = "list", **kw: Any) -> Dict[str, Any]:
        """Unified entry: envelope.act('mail','compose', to=..., subject=..., body=...)."""
        s = surface.lower().strip()
        if s in ("chrome", "browser"):
            if action in ("navigate", "open"):
                return self.navigate(str(kw.get("url") or "https://www.google.com/"))
            if action == "dom":
                return self.dom()
            if action == "grant":
                return self.grant_real_chrome()
            if action == "open_property":
                return self.open_property(str(kw.get("name") or "search"))
            return self.navigate(str(kw.get("url") or "https://www.google.com/"))
        if s in ("sites", "property", "google"):
            return self.open_property(str(kw.get("name") or action or "search"))
        if s == "search":
            return self.search(str(kw.get("query") or kw.get("q") or action))
        if s in ("mail", "gmail", "email"):
            return self.mail(action, **kw)
        if s in ("drive", "docs", "sheets", "files"):
            return self.drive(action, **kw)
        if s in ("calendar", "cal"):
            return self.calendar(action, **kw)
        if s == "collab":
            if action in ("post", "chat", "message"):
                return self.collab_post(str(kw.get("text") or ""), author=str(kw.get("author") or "user"))
            if action in ("project", "create"):
                return self.collab_project(str(kw.get("name") or "Project"), str(kw.get("description") or ""))
            if action == "list":
                return {"ok": True, "projects": self.collab.list_projects()}
            if action == "status":
                return {"ok": True, "collab": self.collab.health()}
            return {"ok": False, "error": f"unknown collab action {action}"}
        if s in ("status", "health", "catalog"):
            return {"ok": True, **self.surfaces()}
        return {"ok": False, "error": f"unknown surface {surface}", "surfaces": list(self.surfaces()["surfaces"])}

    def health(self) -> Dict[str, Any]:
        chrome_h = {}
        try:
            chrome_h = self._get_chrome().health()
        except Exception as exc:
            chrome_h = {"error": str(exc)[:120]}
        return {
            "schema": "auro.google_envelope.health.v1",
            "envelope_id": self.envelope_id,
            "suite": self.suite.catalog(),
            "collab": self.collab.health(),
            "chrome": chrome_h,
            "tabs": self.tabs[-10:],
            "recent_receipts": [r.to_dict() for r in self.receipts[-10:]],
            "root": str(self.root),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.health()


def get_envelope(mind: Any = None, *, chrome_mock: bool = True, force: bool = False) -> GoogleVirtualEnvelope:
    global _ENVELOPE
    if _ENVELOPE is None or force:
        _ENVELOPE = GoogleVirtualEnvelope(mind=mind, chrome_mock=chrome_mock)
    elif mind is not None:
        _ENVELOPE.mind = mind
    return _ENVELOPE
