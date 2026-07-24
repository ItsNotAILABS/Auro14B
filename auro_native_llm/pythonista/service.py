"""JS-facing Pythonista service — App Service bridge + background scripts.

Mirrors potential-succotash:
  - AppServiceBridge (bidirectional messages)
  - BackgroundTaskScheduler (Python jobs under JS shell)

JS host:
  1. POSTs script / job definitions
  2. Polls UI tree + DB tables for render
  3. POSTs UI events (button taps) back into Python handlers
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from auro_native_llm.pythonista import ui as ui_mod
from auro_native_llm.pythonista.background import BackgroundScheduler
from auro_native_llm.pythonista.db import PythonistaDB
from auro_native_llm.pythonista.runtime import ScriptRuntime

_SERVICE: Optional["PythonistaService"] = None

_SCRIPTS_DIR = Path(__file__).with_name("scripts")


class PythonistaService:
    def __init__(self) -> None:
        self.db = PythonistaDB()
        self.runtime = ScriptRuntime(db=self.db)
        self.scheduler = BackgroundScheduler()
        self.last_run: Optional[Dict[str, Any]] = None
        self.message_log: List[Dict[str, Any]] = []
        self.started_at = time.time()

    def status(self) -> Dict[str, Any]:
        return {
            "schema": "auro.pythonista.service.v1",
            "ok": True,
            "uptime_s": time.time() - self.started_at,
            "pattern": "pythonista: python-bg + js-render-ui + db-tables",
            "source_inspiration": [
                "https://omz-software.com/pythonista/",
                "FreddyCreates/potential-succotash windows-runtime-sdk",
            ],
            "scheduler": self.scheduler.status(),
            "tables": self.db.list_tables(),
            "has_ui": ui_mod.ui_tree().get("type") != "empty" or bool(ui_mod.ui_tree().get("root")),
            "last_run_ok": None if not self.last_run else self.last_run.get("ok"),
            "scripts_dir": str(_SCRIPTS_DIR),
            "example_scripts": [p.name for p in _SCRIPTS_DIR.glob("*.py")] if _SCRIPTS_DIR.exists() else [],
        }

    def run_script(
        self,
        source: Optional[str] = None,
        *,
        script_name: Optional[str] = None,
        background: bool = False,
        interval_s: float = 0.0,
    ) -> Dict[str, Any]:
        if script_name and not source:
            path = _SCRIPTS_DIR / script_name
            if not path.exists():
                return {"ok": False, "error": f"script not found: {script_name}"}
            source = path.read_text(encoding="utf-8")
        if not source:
            return {"ok": False, "error": "source or script_name required"}

        if background:
            job = self.scheduler.submit(
                lambda: self.runtime.run(source or "", reset=True),
                name=script_name or "inline",
                interval_s=interval_s,
            )
            return {"ok": True, "background": True, "job": job.to_dict()}

        result = self.runtime.run(source, filename=script_name or "<inline>")
        self.last_run = result
        self._log("run_script", {"ok": result.get("ok"), "filename": script_name})
        return result

    def ui(self) -> Dict[str, Any]:
        tree = ui_mod.ui_tree()
        if self.last_run and self.last_run.get("ui"):
            # prefer last explicit present()
            tree = self.last_run["ui"] if tree.get("type") == "empty" else tree
        return tree

    def event(self, node_id: str, action: str = "action", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = ui_mod.dispatch_event(node_id, action, payload)
        # re-snapshot UI after handler may have mutated views/db
        out["ui"] = ui_mod.ui_tree()
        out["tables"] = [self.db.table_render(t) for t in self.db.list_tables()]
        self._log("event", {"node_id": node_id, "action": action, "ok": out.get("ok")})
        return out

    def table(self, name: str, limit: int = 200) -> Dict[str, Any]:
        return self.db.table_render(name, limit=limit)

    def tables(self) -> Dict[str, Any]:
        names = self.db.list_tables()
        return {
            "ok": True,
            "tables": names,
            "renders": [self.db.table_render(n, limit=50) for n in names],
        }

    def jobs(self) -> Dict[str, Any]:
        return {"ok": True, **self.scheduler.status()}

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        ok = self.scheduler.cancel(job_id)
        return {"ok": ok, "job_id": job_id}

    def bridge_send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """AppServiceBridge-style message into the service."""
        kind = payload.get("kind") or payload.get("type") or "ping"
        if kind == "ping":
            return {"ok": True, "pong": True, "ts": time.time()}
        if kind == "run":
            return self.run_script(
                payload.get("source"),
                script_name=payload.get("script_name"),
                background=bool(payload.get("background")),
                interval_s=float(payload.get("interval_s") or 0),
            )
        if kind == "event":
            return self.event(
                payload.get("node_id", ""),
                payload.get("action", "action"),
                payload.get("payload"),
            )
        if kind == "query_table":
            return self.table(payload.get("table", ""), int(payload.get("limit", 200)))
        return {"ok": False, "error": f"unknown bridge kind {kind}"}

    def _log(self, kind: str, data: Dict[str, Any]) -> None:
        self.message_log.append({"kind": kind, "data": data, "ts": time.time()})
        if len(self.message_log) > 200:
            self.message_log = self.message_log[-200:]


def get_service() -> PythonistaService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = PythonistaService()
    return _SERVICE
