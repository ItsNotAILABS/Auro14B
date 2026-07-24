"""Collaborative project workspace — you + the AI working together."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_HOME = Path.home()
_DEFAULT_ROOT = _HOME / ".auro_workspace" / "collab"
_COLLAB: Optional["CollabWorkspace"] = None


@dataclass
class CollabMessage:
    id: str
    author: str  # user | ai | system
    text: str
    ts: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "author": self.author,
            "text": self.text,
            "ts": self.ts,
            "meta": self.meta,
        }


@dataclass
class CollabTask:
    id: str
    title: str
    status: str = "open"  # open | in_progress | done | blocked
    assignee: str = "ai"  # user | ai | both
    notes: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "assignee": self.assignee,
            "notes": self.notes,
            "ts": self.ts,
        }


@dataclass
class CollabProject:
    id: str
    name: str
    description: str = ""
    created_at: float = field(default_factory=time.time)
    messages: List[CollabMessage] = field(default_factory=list)
    tasks: List[CollabTask] = field(default_factory=list)
    docs: Dict[str, str] = field(default_factory=dict)  # name -> content
    browser_urls: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, full: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "n_messages": len(self.messages),
            "n_tasks": len(self.tasks),
            "n_docs": len(self.docs),
            "browser_urls": list(self.browser_urls)[-20:],
            "meta": self.meta,
        }
        if full:
            d["messages"] = [m.to_dict() for m in self.messages[-100:]]
            d["tasks"] = [t.to_dict() for t in self.tasks]
            d["docs"] = {k: v[:2000] for k, v in self.docs.items()}
        return d


class CollabWorkspace:
    """Shared place for human + AI project work."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root or _DEFAULT_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)
        self.projects: Dict[str, CollabProject] = {}
        self.active_project_id: Optional[str] = None
        self._load()
        if not self.projects:
            self.create_project(
                "Default Lab",
                "Shared workspace for you and Auro — projects, docs, tasks, browser.",
            )

    def _proj_path(self, pid: str) -> Path:
        return self.root / f"{pid}.json"

    def _load(self) -> None:
        for p in self.root.glob("*.json"):
            if p.name == "_meta.json":
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                proj = CollabProject(
                    id=data["id"],
                    name=data.get("name", p.stem),
                    description=data.get("description", ""),
                    created_at=float(data.get("created_at", time.time())),
                    docs=dict(data.get("docs") or {}),
                    browser_urls=list(data.get("browser_urls") or []),
                    meta=dict(data.get("meta") or {}),
                )
                for m in data.get("messages") or []:
                    proj.messages.append(
                        CollabMessage(
                            id=m["id"],
                            author=m.get("author", "user"),
                            text=m.get("text", ""),
                            ts=float(m.get("ts", time.time())),
                            meta=dict(m.get("meta") or {}),
                        )
                    )
                for t in data.get("tasks") or []:
                    proj.tasks.append(
                        CollabTask(
                            id=t["id"],
                            title=t.get("title", ""),
                            status=t.get("status", "open"),
                            assignee=t.get("assignee", "ai"),
                            notes=t.get("notes", ""),
                            ts=float(t.get("ts", time.time())),
                        )
                    )
                self.projects[proj.id] = proj
            except Exception:
                continue
        meta = self.root / "_meta.json"
        if meta.exists():
            try:
                self.active_project_id = json.loads(meta.read_text(encoding="utf-8")).get("active")
            except Exception:
                pass
        if not self.active_project_id and self.projects:
            self.active_project_id = next(iter(self.projects))

    def save(self, project_id: Optional[str] = None) -> None:
        pids = [project_id] if project_id else list(self.projects)
        for pid in pids:
            proj = self.projects.get(pid)
            if not proj:
                continue
            payload = {
                "id": proj.id,
                "name": proj.name,
                "description": proj.description,
                "created_at": proj.created_at,
                "messages": [m.to_dict() for m in proj.messages[-500:]],
                "tasks": [t.to_dict() for t in proj.tasks],
                "docs": proj.docs,
                "browser_urls": proj.browser_urls[-50:],
                "meta": proj.meta,
            }
            self._proj_path(pid).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (self.root / "_meta.json").write_text(
            json.dumps({"active": self.active_project_id, "ts": time.time()}),
            encoding="utf-8",
        )

    def create_project(self, name: str, description: str = "") -> Dict[str, Any]:
        pid = f"proj-{uuid.uuid4().hex[:10]}"
        proj = CollabProject(id=pid, name=name, description=description)
        proj.messages.append(
            CollabMessage(
                id=f"m-{uuid.uuid4().hex[:8]}",
                author="system",
                text=f"Project '{name}' created. You and the AI share this space.",
            )
        )
        proj.docs["README.md"] = f"# {name}\n\n{description}\n\n## Working together\n- User messages and AI replies land in chat\n- Tasks can be assigned to user, ai, or both\n- Docs are co-edited here\n- Browser URLs we open together are tracked\n"
        self.projects[pid] = proj
        self.active_project_id = pid
        self.save(pid)
        return {"ok": True, "project": proj.to_dict(full=True)}

    def list_projects(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.projects.values()]

    def set_active(self, project_id: str) -> Dict[str, Any]:
        if project_id not in self.projects:
            return {"ok": False, "error": "unknown project"}
        self.active_project_id = project_id
        self.save()
        return {"ok": True, "active": project_id}

    def active(self) -> Optional[CollabProject]:
        if self.active_project_id:
            return self.projects.get(self.active_project_id)
        return None

    def post(self, text: str, author: str = "user", meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        proj = self.active()
        if not proj:
            return {"ok": False, "error": "no active project"}
        msg = CollabMessage(
            id=f"m-{uuid.uuid4().hex[:8]}",
            author=author,
            text=text,
            meta=meta or {},
        )
        proj.messages.append(msg)
        self.save(proj.id)
        return {"ok": True, "message": msg.to_dict(), "project_id": proj.id}

    def add_task(self, title: str, assignee: str = "ai", notes: str = "") -> Dict[str, Any]:
        proj = self.active()
        if not proj:
            return {"ok": False, "error": "no active project"}
        t = CollabTask(id=f"t-{uuid.uuid4().hex[:8]}", title=title, assignee=assignee, notes=notes)
        proj.tasks.append(t)
        self.save(proj.id)
        return {"ok": True, "task": t.to_dict()}

    def update_task(self, task_id: str, **fields: Any) -> Dict[str, Any]:
        proj = self.active()
        if not proj:
            return {"ok": False, "error": "no active project"}
        for t in proj.tasks:
            if t.id == task_id:
                if "status" in fields:
                    t.status = str(fields["status"])
                if "notes" in fields:
                    t.notes = str(fields["notes"])
                if "assignee" in fields:
                    t.assignee = str(fields["assignee"])
                if "title" in fields:
                    t.title = str(fields["title"])
                self.save(proj.id)
                return {"ok": True, "task": t.to_dict()}
        return {"ok": False, "error": "task not found"}

    def write_doc(self, name: str, content: str) -> Dict[str, Any]:
        proj = self.active()
        if not proj:
            return {"ok": False, "error": "no active project"}
        proj.docs[name] = content
        self.save(proj.id)
        return {"ok": True, "name": name, "len": len(content)}

    def read_doc(self, name: str) -> Dict[str, Any]:
        proj = self.active()
        if not proj:
            return {"ok": False, "error": "no active project"}
        if name not in proj.docs:
            return {"ok": False, "error": "doc not found", "docs": list(proj.docs)}
        return {"ok": True, "name": name, "content": proj.docs[name]}

    def track_url(self, url: str) -> None:
        proj = self.active()
        if proj:
            proj.browser_urls.append(url)
            self.save(proj.id)

    def ai_turn(self, mind: Any, user_text: str) -> Dict[str, Any]:
        """User posts; AI replies into the same project thread."""
        self.post(user_text, author="user")
        proj = self.active()
        context = ""
        if proj:
            context = (
                f"Project: {proj.name}\n{proj.description}\n"
                f"Open tasks: {[t.title for t in proj.tasks if t.status != 'done'][:8]}\n"
                f"Docs: {list(proj.docs)[:10]}\n"
            )
        prompt = (
            f"[COLLAB WORKSPACE — reply as project partner AI]\n{context}\n"
            f"User: {user_text}\n"
            f"Give a concrete helpful reply and suggest next tasks if useful."
        )
        reply = ""
        # Prefer light LM path (no full mind.generate absorb/train stack)
        try:
            lang = getattr(mind, "language", None) if mind is not None else None
            if lang is not None and hasattr(lang, "generate"):
                r = lang.generate(prompt, max_new_tokens=48)
                reply = str(getattr(r, "text", None) or r)[:2000]
            elif mind is not None and hasattr(mind, "generate"):
                prev_train = getattr(mind, "train_every_act", True)
                mind.train_every_act = False
                try:
                    r = mind.generate(prompt, max_new_tokens=48)
                    if hasattr(r, "output"):
                        out = r.output
                        reply = str(getattr(out, "text", None) or out)[:2000]
                    else:
                        reply = str(getattr(r, "text", r))[:2000]
                finally:
                    mind.train_every_act = prev_train
            else:
                reply = (
                    f"Logged. Next: break '{user_text[:120]}' into tasks, "
                    f"open sandbox Drive note, and track browser URLs in collab."
                )
        except Exception as exc:
            reply = f"Received in collab (reply light-path error: {exc}). Continuing on project."
        self.post(reply, author="ai", meta={"kind": "collab_reply"})
        return {
            "ok": True,
            "user": user_text,
            "ai": reply,
            "project": (proj.to_dict(full=True) if proj else None),
        }

    def health(self) -> Dict[str, Any]:
        return {
            "schema": "auro.collab.v1",
            "root": str(self.root),
            "n_projects": len(self.projects),
            "active": self.active_project_id,
            "projects": self.list_projects(),
        }


def get_collab(root: Optional[Path] = None) -> CollabWorkspace:
    global _COLLAB
    if _COLLAB is None or (root and Path(root) != _COLLAB.root):
        _COLLAB = CollabWorkspace(root=root)
    return _COLLAB
