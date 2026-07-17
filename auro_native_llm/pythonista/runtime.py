"""Run Pythonista-style scripts: capture UI tree, logs, DB, return envelope."""

from __future__ import annotations

import io
import sys
import time
import traceback
import types
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Dict, Optional

from auro_native_llm.pythonista import ui as ui_mod
from auro_native_llm.pythonista.db import PythonistaDB


class ScriptRuntime:
    """Execute a script string or file with Pythonista-like builtins injected."""

    def __init__(self, db: Optional[PythonistaDB] = None) -> None:
        self.db = db or PythonistaDB()

    def run(
        self,
        source: str,
        *,
        filename: str = "<pythonista>",
        globals_extra: Optional[Dict[str, Any]] = None,
        reset: bool = True,
    ) -> Dict[str, Any]:
        if reset:
            ui_mod.reset_ui()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        console = _Console()
        g: Dict[str, Any] = {
            "__name__": "__pythonista__",
            "__file__": filename,
            "ui": ui_mod,
            "db": self.db,
            "console": console,
        }
        if globals_extra:
            g.update(globals_extra)

        # `import ui` / `import db` work like Pythonista's built-in modules
        injected = {
            "ui": ui_mod,
            "db": self.db,
            "console": console,
        }
        saved = {k: sys.modules.get(k) for k in injected}
        for k, mod in injected.items():
            if not isinstance(mod, types.ModuleType):
                # wrap non-module (db instance) in a thin module
                m = types.ModuleType(k)
                if k == "db":
                    for name in (
                        "create_table",
                        "insert",
                        "query",
                        "table_render",
                        "list_tables",
                        "execute",
                    ):
                        setattr(m, name, getattr(mod, name))
                    m.db = mod  # type: ignore[attr-defined]
                elif k == "console":
                    for name in ("write", "clear", "hud_alert"):
                        setattr(m, name, getattr(mod, name))
                else:
                    m.obj = mod  # type: ignore[attr-defined]
                sys.modules[k] = m
            else:
                sys.modules[k] = mod

        t0 = time.time()
        ok = True
        err = None
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                code = compile(source, filename, "exec")
                exec(code, g, g)
        except Exception as exc:
            ok = False
            err = f"{exc}\n{traceback.format_exc()}"
        finally:
            for k, prev in saved.items():
                if prev is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = prev
        elapsed = time.time() - t0
        tree = ui_mod.ui_tree()
        tables = []
        try:
            for t in self.db.list_tables():
                tables.append(self.db.table_render(t, limit=50))
        except Exception:
            pass
        return {
            "schema": "auro.pythonista.run.v1",
            "ok": ok,
            "error": err,
            "filename": filename,
            "elapsed_s": elapsed,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "ui": tree,
            "tables": tables,
            "inspiration": {
                "app": "Pythonista (omz-software)",
                "pattern": "Python background + JS host renders ui/db",
                "succotash": "windows-runtime-sdk AppServiceBridge + BackgroundTaskScheduler",
            },
        }

    def run_file(self, path: str | Path, **kwargs: Any) -> Dict[str, Any]:
        p = Path(path)
        return self.run(p.read_text(encoding="utf-8"), filename=str(p), **kwargs)


class _Console:
    def write(self, s: str) -> None:
        print(s, end="")

    def clear(self) -> None:
        print("\n" * 2)

    def hud_alert(self, msg: str) -> None:
        print(f"[HUD] {msg}")


def run_script(source: str, **kwargs: Any) -> Dict[str, Any]:
    return ScriptRuntime().run(source, **kwargs)
