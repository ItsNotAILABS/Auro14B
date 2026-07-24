"""Embedded Monaco code editor organ — server-side sessions + UI bridge.

Monaco runs in the browser (CDN); this organ owns sessions, edits, diffs,
and language modes so the mind can create/edit code programmatically.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MonacoSession:
    session_id: str
    language: str = "python"
    filename: str = "main.py"
    content: str = ""
    version: int = 1
    created_at: float = field(default_factory=time.time)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "language": self.language,
            "filename": self.filename,
            "content": self.content,
            "version": self.version,
            "lines": self.content.count("\n") + (1 if self.content else 0),
            "created_at": self.created_at,
            "edits": len(self.history),
        }


class MonacoOrgan:
    """Embedded code-editor organ for every Auro mind."""

    def __init__(self) -> None:
        self.sessions: Dict[str, MonacoSession] = {}

    def create(
        self,
        content: str = "",
        *,
        language: str = "python",
        filename: str = "main.py",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        sid = session_id or f"monaco-{uuid.uuid4().hex[:10]}"
        sess = MonacoSession(
            session_id=sid,
            language=language,
            filename=filename,
            content=content,
        )
        self.sessions[sid] = sess
        return {"ok": True, "action": "monaco.create", "session": sess.to_dict()}

    def get(self, session_id: str) -> Dict[str, Any]:
        sess = self.sessions.get(session_id)
        if not sess:
            return {"ok": False, "error": f"unknown session {session_id}"}
        return {"ok": True, "action": "monaco.get", "session": sess.to_dict()}

    def set_content(self, session_id: str, content: str) -> Dict[str, Any]:
        sess = self.sessions.get(session_id)
        if not sess:
            return {"ok": False, "error": f"unknown session {session_id}"}
        prev = sess.content
        sess.content = content
        sess.version += 1
        sess.history.append({"op": "set", "prev_len": len(prev), "new_len": len(content)})
        return {"ok": True, "action": "monaco.set", "session": sess.to_dict()}

    def insert(self, session_id: str, offset: int, text: str) -> Dict[str, Any]:
        sess = self.sessions.get(session_id)
        if not sess:
            return {"ok": False, "error": f"unknown session {session_id}"}
        offset = max(0, min(offset, len(sess.content)))
        sess.content = sess.content[:offset] + text + sess.content[offset:]
        sess.version += 1
        sess.history.append({"op": "insert", "offset": offset, "len": len(text)})
        return {"ok": True, "action": "monaco.insert", "session": sess.to_dict()}

    def replace(self, session_id: str, old: str, new: str, count: int = 1) -> Dict[str, Any]:
        sess = self.sessions.get(session_id)
        if not sess:
            return {"ok": False, "error": f"unknown session {session_id}"}
        if count <= 0:
            sess.content = sess.content.replace(old, new)
        else:
            sess.content = sess.content.replace(old, new, count)
        sess.version += 1
        sess.history.append({"op": "replace", "old": old[:80], "new": new[:80]})
        return {"ok": True, "action": "monaco.replace", "session": sess.to_dict()}

    def append(self, session_id: str, text: str) -> Dict[str, Any]:
        sess = self.sessions.get(session_id)
        if not sess:
            return {"ok": False, "error": f"unknown session {session_id}"}
        sess.content = sess.content + text
        sess.version += 1
        sess.history.append({"op": "append", "len": len(text)})
        return {"ok": True, "action": "monaco.append", "session": sess.to_dict()}

    def list_sessions(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "action": "monaco.list",
            "sessions": [s.to_dict() for s in self.sessions.values()],
        }

    def ui_payload(self, session_id: str) -> Dict[str, Any]:
        """Payload the frontend Monaco instance should load."""
        g = self.get(session_id)
        if not g.get("ok"):
            return g
        sess = g["session"]
        return {
            "ok": True,
            "monaco": {
                "session_id": sess["session_id"],
                "language": sess["language"],
                "filename": sess["filename"],
                "value": sess["content"],
                "version": sess["version"],
                "theme": "vs-dark",
            },
        }
