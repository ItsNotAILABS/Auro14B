"""Virtual Google suite services inside the AI sandbox."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Public Google property map (AI may open via Chrome when allowed)
GOOGLE_PROPERTIES: Dict[str, Dict[str, str]] = {
    "search": {"url": "https://www.google.com/", "name": "Google Search"},
    "mail": {"url": "https://mail.google.com/", "name": "Gmail"},
    "drive": {"url": "https://drive.google.com/", "name": "Google Drive"},
    "docs": {"url": "https://docs.google.com/document/u/0/", "name": "Google Docs"},
    "sheets": {"url": "https://docs.google.com/spreadsheets/u/0/", "name": "Google Sheets"},
    "slides": {"url": "https://docs.google.com/presentation/u/0/", "name": "Google Slides"},
    "calendar": {"url": "https://calendar.google.com/", "name": "Google Calendar"},
    "meet": {"url": "https://meet.google.com/", "name": "Google Meet"},
    "maps": {"url": "https://maps.google.com/", "name": "Google Maps"},
    "youtube": {"url": "https://www.youtube.com/", "name": "YouTube"},
    "news": {"url": "https://news.google.com/", "name": "Google News"},
    "scholar": {"url": "https://scholar.google.com/", "name": "Google Scholar"},
    "translate": {"url": "https://translate.google.com/", "name": "Google Translate"},
    "keep": {"url": "https://keep.google.com/", "name": "Google Keep"},
    "photos": {"url": "https://photos.google.com/", "name": "Google Photos"},
    "cloud": {"url": "https://console.cloud.google.com/", "name": "Google Cloud Console"},
    "gemini": {"url": "https://gemini.google.com/", "name": "Gemini"},
    "accounts": {"url": "https://myaccount.google.com/", "name": "Google Account"},
}


@dataclass
class VirtualMessage:
    id: str
    folder: str  # inbox | sent | drafts | trash
    from_addr: str
    to_addr: str
    subject: str
    body: str
    ts: float = field(default_factory=time.time)
    read: bool = False
    labels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "folder": self.folder,
            "from": self.from_addr,
            "to": self.to_addr,
            "subject": self.subject,
            "body": self.body[:4000],
            "ts": self.ts,
            "read": self.read,
            "labels": self.labels,
        }


@dataclass
class VirtualFile:
    id: str
    name: str
    kind: str  # doc | sheet | slide | folder | note | binary
    content: str
    parent_id: Optional[str] = None
    mime: str = "text/plain"
    ts: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "content_preview": self.content[:500],
            "content_len": len(self.content),
            "parent_id": self.parent_id,
            "mime": self.mime,
            "ts": self.ts,
            "meta": self.meta,
        }


@dataclass
class VirtualEvent:
    id: str
    title: str
    start_ts: float
    end_ts: float
    description: str = ""
    location: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "description": self.description,
            "location": self.location,
        }


class VirtualGmail:
    """AI-owned sandboxed Gmail — not the operator's real inbox."""

    def __init__(self, owner: str = "auro-ai@sandbox.local") -> None:
        self.owner = owner
        self.messages: Dict[str, VirtualMessage] = {}
        # seed welcome mail
        mid = self._id()
        self.messages[mid] = VirtualMessage(
            id=mid,
            folder="inbox",
            from_addr="system@auro.ghost",
            to_addr=owner,
            subject="Welcome to AI Gmail Sandbox",
            body=(
                "This is YOUR sandbox Gmail, not the operator's account.\n"
                "Compose, search, label freely. Real Gmail requires operator OAuth "
                "and is never accessed without explicit human approval."
            ),
            labels=["system"],
        )

    def _id(self) -> str:
        return f"msg-{uuid.uuid4().hex[:10]}"

    def list_messages(self, folder: str = "inbox", limit: int = 20) -> List[Dict[str, Any]]:
        rows = [m for m in self.messages.values() if m.folder == folder]
        rows.sort(key=lambda m: m.ts, reverse=True)
        return [m.to_dict() for m in rows[:limit]]

    def read(self, message_id: str) -> Dict[str, Any]:
        m = self.messages.get(message_id)
        if not m:
            return {"ok": False, "error": "not_found"}
        m.read = True
        return {"ok": True, "message": m.to_dict()}

    def compose(
        self,
        to: str,
        subject: str,
        body: str,
        *,
        send: bool = True,
    ) -> Dict[str, Any]:
        mid = self._id()
        folder = "sent" if send else "drafts"
        m = VirtualMessage(
            id=mid,
            folder=folder,
            from_addr=self.owner,
            to_addr=to,
            subject=subject,
            body=body,
            labels=["sandbox"],
        )
        self.messages[mid] = m
        # deliver copy to inbox if self-addressed or sandbox peer
        if send and ("sandbox" in to or to == self.owner):
            rid = self._id()
            self.messages[rid] = VirtualMessage(
                id=rid,
                folder="inbox",
                from_addr=self.owner,
                to_addr=to,
                subject=subject,
                body=body,
                labels=["delivered-sandbox"],
            )
        return {"ok": True, "message": m.to_dict(), "sent": send}

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        q = query.lower().split()
        hits = []
        for m in self.messages.values():
            blob = f"{m.subject} {m.body} {m.from_addr} {m.to_addr}".lower()
            score = sum(1 for t in q if t in blob)
            if score:
                hits.append((score, m))
        hits.sort(key=lambda x: -x[0])
        return [m.to_dict() for _, m in hits[:limit]]

    def stats(self) -> Dict[str, Any]:
        by: Dict[str, int] = {}
        for m in self.messages.values():
            by[m.folder] = by.get(m.folder, 0) + 1
        return {"owner": self.owner, "total": len(self.messages), "by_folder": by}


class VirtualDrive:
    """AI-owned sandboxed Drive/Docs."""

    def __init__(self) -> None:
        self.files: Dict[str, VirtualFile] = {}
        root = VirtualFile(id="root", name="My Drive", kind="folder", content="")
        self.files["root"] = root
        # starter doc
        did = f"doc-{uuid.uuid4().hex[:8]}"
        self.files[did] = VirtualFile(
            id=did,
            name="AI Sandbox Notebook.md",
            kind="doc",
            content="# AI Sandbox Notebook\n\nWrite project notes here. Shared to Collab when published.\n",
            parent_id="root",
            mime="text/markdown",
        )

    def list_files(self, parent_id: str = "root") -> List[Dict[str, Any]]:
        return [f.to_dict() for f in self.files.values() if f.parent_id == parent_id or (parent_id == "root" and f.id != "root")]

    def create(
        self,
        name: str,
        content: str = "",
        kind: str = "doc",
        parent_id: str = "root",
    ) -> Dict[str, Any]:
        fid = f"{kind}-{uuid.uuid4().hex[:8]}"
        f = VirtualFile(id=fid, name=name, kind=kind, content=content, parent_id=parent_id)
        self.files[fid] = f
        return {"ok": True, "file": f.to_dict()}

    def read(self, file_id: str) -> Dict[str, Any]:
        f = self.files.get(file_id)
        if not f:
            return {"ok": False, "error": "not_found"}
        d = f.to_dict()
        d["content"] = f.content
        return {"ok": True, "file": d}

    def write(self, file_id: str, content: str) -> Dict[str, Any]:
        f = self.files.get(file_id)
        if not f:
            return {"ok": False, "error": "not_found"}
        f.content = content
        f.ts = time.time()
        return {"ok": True, "file": f.to_dict()}

    def search(self, query: str) -> List[Dict[str, Any]]:
        q = query.lower()
        return [
            f.to_dict()
            for f in self.files.values()
            if q in f.name.lower() or q in f.content.lower()
        ]

    def stats(self) -> Dict[str, Any]:
        by: Dict[str, int] = {}
        for f in self.files.values():
            by[f.kind] = by.get(f.kind, 0) + 1
        return {"total": len(self.files), "by_kind": by}


class VirtualCalendar:
    def __init__(self) -> None:
        self.events: Dict[str, VirtualEvent] = {}

    def add(self, title: str, start_ts: float, end_ts: float, description: str = "") -> Dict[str, Any]:
        eid = f"evt-{uuid.uuid4().hex[:8]}"
        e = VirtualEvent(id=eid, title=title, start_ts=start_ts, end_ts=end_ts, description=description)
        self.events[eid] = e
        return {"ok": True, "event": e.to_dict()}

    def list_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        rows = sorted(self.events.values(), key=lambda e: e.start_ts)
        return [e.to_dict() for e in rows[:limit]]


class VirtualSearch:
    """Sandbox search log + optional live Chrome Google search."""

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def query(self, q: str, *, results: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        row = {
            "id": f"q-{uuid.uuid4().hex[:8]}",
            "q": q,
            "ts": time.time(),
            "results": results
            or [
                {"title": f"Sandbox hit for: {q}", "url": f"https://www.google.com/search?q={q.replace(' ', '+')}", "snippet": "Virtual result — open via Chrome for live SERP."},
            ],
        }
        self.history.append(row)
        return {"ok": True, "search": row}


class GoogleSuite:
    """Full virtual Google stack for the AI envelope."""

    def __init__(self, owner: str = "auro-ai@sandbox.local") -> None:
        self.owner = owner
        self.mail = VirtualGmail(owner=owner)
        self.drive = VirtualDrive()
        self.calendar = VirtualCalendar()
        self.search = VirtualSearch()
        self.properties = dict(GOOGLE_PROPERTIES)

    def catalog(self) -> Dict[str, Any]:
        return {
            "owner": self.owner,
            "properties": self.properties,
            "mail": self.mail.stats(),
            "drive": self.drive.stats(),
            "calendar_events": len(self.calendar.events),
            "search_history": len(self.search.history),
            "mode": "sandbox_virtual",
            "claim": (
                "Virtual services are AI-owned. Real Google account access "
                "requires operator OAuth and is never automatic."
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.catalog()
