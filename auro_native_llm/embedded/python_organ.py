"""Python organ — embedded compute the LLM can use (Pythonista-compatible).

Bounded sandbox + doctrine-aware exec. Results always return a training
payload so the autocycle can absorb + learn.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

# Soft denylist (also enforced via AST)
_DENIED_NAMES = {
    "system",
    "popen",
    "Popen",
    "fork",
    "spawn",
    "remove",
    "rmtree",
    "unlink",
}
_DENIED_MODULES = {
    "subprocess",
    "socket",
    "ctypes",
    "multiprocessing",
    "pathlib",  # still allow Path via limited API if needed — block full module import
    "shutil",
    "http",
    "urllib",
    "requests",
    "ftplib",
    "telnetlib",
    "pickle",
    "marshal",
}


@dataclass
class PythonRunResult:
    ok: bool
    source: str
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    return_value: Any = None
    elapsed_s: float = 0.0
    receipt: str = ""
    ui: Optional[Dict[str, Any]] = None
    tables: List[Dict[str, Any]] = field(default_factory=list)
    doctrine_notes: List[str] = field(default_factory=list)
    training_text: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.python_organ.run.v1",
            "ok": self.ok,
            "stdout": self.stdout[:8000],
            "stderr": self.stderr[:4000],
            "error": self.error,
            "return_value": _safe_repr(self.return_value),
            "elapsed_s": self.elapsed_s,
            "receipt": self.receipt,
            "ui": self.ui,
            "tables": self.tables,
            "doctrine_notes": self.doctrine_notes,
            "training_text": self.training_text[:4000],
            "meta": self.meta,
        }


def _safe_repr(v: Any, limit: int = 500) -> str:
    try:
        s = repr(v)
    except Exception:
        s = f"<{type(v).__name__}>"
    return s[:limit]


class _DoctrineGuard(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            root = (a.name or "").split(".")[0]
            if root in _DENIED_MODULES:
                self.violations.append(f"import denied: {a.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root in _DENIED_MODULES:
            self.violations.append(f"from-import denied: {node.module}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _DENIED_NAMES:
            self.violations.append(f"attribute denied: .{node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # bare eval/exec
        if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec", "compile", "__import__"):
            self.violations.append(f"call denied: {node.func.id}()")
        self.generic_visit(node)


def lint_python_source(source: str) -> List[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax: {exc}"]
    g = _DoctrineGuard()
    g.visit(tree)
    # string pattern soft checks
    low = source.lower()
    for pat in (
        "os.system",
        "subprocess",
        "disable governance",
        "cloud llm as primary",
        "rm -rf",
    ):
        if pat in low:
            g.violations.append(f"pattern denied: {pat}")
    return g.violations


class PythonOrgan:
    """Mind organ: execute Python the LLM reasons about."""

    def __init__(
        self,
        *,
        doctrine: Optional[Dict[str, Any]] = None,
        enable_ui: bool = True,
        enable_db: bool = True,
        enable_github: bool = True,
    ) -> None:
        self.doctrine = doctrine or load_python_doctrine()
        self.enable_ui = enable_ui
        self.enable_db = enable_db
        self.enable_github = enable_github
        self.history: List[Dict[str, Any]] = []
        self.total_runs = 0
        self.total_ok = 0

    def info(self) -> Dict[str, Any]:
        return {
            "organ": "python",
            "doctrine_id": self.doctrine.get("doctrine_id"),
            "laws": len(self.doctrine.get("laws") or []),
            "routines": len(self.doctrine.get("operating_routines") or []),
            "total_runs": self.total_runs,
            "total_ok": self.total_ok,
            "ok_rate": (self.total_ok / self.total_runs) if self.total_runs else None,
            "apis": self.doctrine.get("allowed_python_apis"),
        }

    def doctrine_prompt(self) -> str:
        laws = self.doctrine.get("laws") or []
        lines = [
            f"# {self.doctrine.get('title')}",
            self.doctrine.get("principle", ""),
            self.doctrine.get("claim_boundary", ""),
            "## Laws",
        ]
        for law in laws:
            lines.append(f"- {law.get('id')} {law.get('name')}: {law.get('text')}")
        lines.append("## Operating routines")
        for r in self.doctrine.get("operating_routines") or []:
            lines.append(f"- {r.get('id')} {r.get('name')}: {', '.join(r.get('steps') or [])}")
        return "\n".join(lines)

    def run(
        self,
        source: str,
        *,
        intent: str = "",
        inject: Optional[Dict[str, Any]] = None,
    ) -> PythonRunResult:
        t0 = time.time()
        notes: List[str] = []
        violations = lint_python_source(source)
        if violations:
            receipt = _receipt(source, "DENIED:" + ";".join(violations))
            train = (
                f"[PYTHON_REFUSE intent={intent}]\n"
                f"violations={violations}\n"
                f"source=\n{source[:1500]}\n"
                f"[/PYTHON_REFUSE]"
            )
            self.total_runs += 1
            res = PythonRunResult(
                ok=False,
                source=source,
                error="doctrine_sandbox: " + "; ".join(violations),
                doctrine_notes=violations,
                receipt=receipt,
                training_text=train,
                elapsed_s=time.time() - t0,
                meta={"intent": intent, "denied": True},
            )
            self.history.append(res.to_dict())
            return res

        notes.append("PL-002 sandbox ok")
        # build namespace
        g: Dict[str, Any] = {"__name__": "__auro_python__"}
        # safe builtins subset
        import builtins as _bi

        safe_builtins = {
            k: getattr(_bi, k)
            for k in (
                "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
                "len", "list", "max", "min", "print", "range", "repr", "round",
                "set", "sorted", "str", "sum", "tuple", "zip", "map", "filter",
                "isinstance", "type", "Exception", "ValueError", "TypeError",
                "KeyError", "True", "False", "None",
            )
            if hasattr(_bi, k)
        }
        # Restricted importer — only allowlisted stdlib modules
        _ALLOWED_IMPORTS = {
            "math", "json", "re", "datetime", "collections", "itertools",
            "functools", "statistics", "hashlib", "base64", "textwrap",
            "string", "time", "typing", "decimal", "fractions", "array",
            "bisect", "heapq", "copy", "dataclasses",
        }

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
            root = (name or "").split(".")[0]
            if root not in _ALLOWED_IMPORTS:
                raise ImportError(f"import denied by doctrine: {name}")
            return __import__(name, globals, locals, fromlist, level)

        safe_builtins["__import__"] = _safe_import
        g["__builtins__"] = safe_builtins

        # inject ui/db/console like Pythonista
        if self.enable_ui:
            try:
                from auro_native_llm.pythonista import ui as ui_mod

                ui_mod.reset_ui()
                g["ui"] = ui_mod
            except Exception:
                pass
        if self.enable_db:
            try:
                from auro_native_llm.pythonista.db import PythonistaDB

                g["db"] = PythonistaDB()
            except Exception:
                pass
        g["console"] = _Console()
        g["json"] = json
        g["math"] = __import__("math")
        g["re"] = __import__("re")
        g["hashlib"] = hashlib
        g["time"] = time

        if self.enable_github:
            def github_search(q: str, top_k: int = 5) -> List[Dict[str, Any]]:
                try:
                    from auro_native_llm.corpus.github_db import GitHubKnowledgeDB

                    return [h.to_dict() for h in GitHubKnowledgeDB().search(q, top_k=top_k)]
                except Exception as exc:
                    return [{"error": str(exc)}]

            def mesie_embed(text: str) -> List[float]:
                try:
                    from auro_native_llm.corpus.embeddings import MaxEmbedder

                    return MaxEmbedder().embed_text(text).tolist()
                except Exception:
                    return []

            g["github_search"] = github_search
            g["mesie_embed"] = mesie_embed
            notes.append("PL-007 github_db available")

        if inject:
            g.update(inject)

        stdout_b = io.StringIO()
        stderr_b = io.StringIO()
        ret = None
        err = None
        ok = True
        try:
            with redirect_stdout(stdout_b), redirect_stderr(stderr_b):
                code = compile(source, "<python_organ>", "exec")
                # same dict for globals/locals so list comps & imports bind correctly
                exec(code, g, g)  # noqa: S102 — intentional sandbox exec
                ret = g.get("result", g.get("out", g.get("answer")))
        except Exception as exc:
            ok = False
            err = f"{exc}\n{traceback.format_exc()}"

        ui_tree = None
        tables: List[Dict[str, Any]] = []
        try:
            if "ui" in g:
                ui_tree = g["ui"].ui_tree()
            if "db" in g and hasattr(g["db"], "list_tables"):
                for t in g["db"].list_tables()[:8]:
                    tables.append(g["db"].table_render(t, limit=30))
        except Exception:
            pass

        out = stdout_b.getvalue()
        err_s = stderr_b.getvalue()
        if err:
            err_s = (err_s + "\n" + err).strip()
        receipt = _receipt(source, out + (err or ""))
        train = (
            f"[PYTHON_RUN ok={ok} intent={intent} receipt={receipt}]\n"
            f"## source\n{source[:2000]}\n"
            f"## stdout\n{out[:1500]}\n"
            f"## error\n{(err or '')[:800]}\n"
            f"## laws\n" + "; ".join(notes) + "\n"
            f"[/PYTHON_RUN]"
        )
        self.total_runs += 1
        if ok:
            self.total_ok += 1
            notes.append("PL-003 experience ready")
        res = PythonRunResult(
            ok=ok,
            source=source,
            stdout=out,
            stderr=err_s,
            error=err,
            return_value=ret,
            elapsed_s=time.time() - t0,
            receipt=receipt,
            ui=ui_tree,
            tables=tables,
            doctrine_notes=notes,
            training_text=train,
            meta={"intent": intent},
        )
        self.history.append(res.to_dict())
        if len(self.history) > 100:
            self.history = self.history[-100:]
        return res

    def run_goal_script(self, goal: str, code: str) -> PythonRunResult:
        return self.run(code, intent=goal)


class _Console:
    def write(self, s: str) -> None:
        print(s, end="")

    def clear(self) -> None:
        print("\n")

    def hud_alert(self, msg: str) -> None:
        print(f"[HUD] {msg}")


def _receipt(source: str, body: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8", errors="ignore"))
    h.update(b"|")
    h.update(body.encode("utf-8", errors="ignore"))
    h.update(f"|{time.time():.0f}".encode())
    return h.hexdigest()[:24]


def load_python_doctrine() -> Dict[str, Any]:
    path = (
        Path(__file__).resolve().parents[2]
        / "native_llm"
        / "scripture"
        / "PYTHON_OPERATING_DOCTRINE.v1.json"
    )
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "doctrine_id": "AURO.PYTHON.OPERATING.v1.fallback",
        "title": "Python Operating Doctrine",
        "principle": "Python is an organ of the mind.",
        "laws": [],
        "operating_routines": [],
    }
