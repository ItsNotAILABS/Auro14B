"""Auro backend HTTP API — work agents, generate, chrome, scripture.

Serves JSON API + simple frontend UI (backend → front).
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse, parse_qs

from auro_native_llm.work.agent import WorkAgent
from auro_native_llm.chrome.tools import ChromeToolbelt

_UI = Path(__file__).with_name("static") / "index.html"
_PORTAL_UI = Path(__file__).with_name("static") / "portal.html"
_WORKSPACE_UI = Path(__file__).with_name("static") / "workspace.html"

# shared agent (lite + mock chrome by default for safe local)
_AGENT = WorkAgent(lite=True, chrome_mock=True, use_scripture=True)
_CHROME = ChromeToolbelt(mock=True)
_MIND = None
_PORTAL = None
_ENVELOPE = None


def _get_mind():
    global _MIND
    if _MIND is None:
        try:
            from auro_native_llm.organism.family import build_mind

            _MIND = build_mind("Auro-2B", lite=True, chrome_mock=True)
        except Exception:
            _MIND = None
    return _MIND


def _get_envelope():
    global _ENVELOPE
    if _ENVELOPE is None:
        from auro_native_llm.gworkspace import get_envelope

        _ENVELOPE = get_envelope(_get_mind(), chrome_mock=True)
    return _ENVELOPE


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    n = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(n) if n else b"{}"
    try:
        data = json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[auro-api] {self.address_string()} {fmt % args}")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/ui", "/index.html"):
            html = _UI.read_text(encoding="utf-8") if _UI.exists() else "<h1>Auro UI missing</h1>"
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/portal", "/portal.html", "/ui/portal"):
            html = (
                _PORTAL_UI.read_text(encoding="utf-8")
                if _PORTAL_UI.exists()
                else "<h1>Portal UI missing</h1>"
            )
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/workspace", "/workspace.html", "/ui/workspace", "/google"):
            html = (
                _WORKSPACE_UI.read_text(encoding="utf-8")
                if _WORKSPACE_UI.exists()
                else "<h1>Workspace UI missing — collab + Google sandbox</h1>"
            )
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/v1/google", "/v1/google/health", "/v1/workspace"):
            try:
                env = _get_envelope()
                _json_response(self, 200, env.health())
            except Exception as exc:
                _json_response(self, 500, {"ok": False, "error": str(exc)})
            return
        if path in ("/v1/collab", "/v1/collab/projects"):
            try:
                env = _get_envelope()
                _json_response(self, 200, {"ok": True, "collab": env.collab.health()})
            except Exception as exc:
                _json_response(self, 500, {"ok": False, "error": str(exc)})
            return
        if path in ("/health", "/v1/health"):
            gstat = {}
            try:
                gstat = _get_envelope().surfaces()
            except Exception:
                gstat = {"ok": False}
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "service": "auro-work-api",
                    "compute_plane": "MESIE",
                    "native": True,
                    "chrome": _CHROME.health(),
                    "model_id": _AGENT.model_id,
                    "portal": "/portal",
                    "workspace": "/workspace",
                    "google_envelope": gstat.get("envelope_id") or gstat,
                    "alpha": [
                        "multi_site",
                        "interior_mcp",
                        "chaos_cuda",
                        "polyglot",
                        "google_envelope",
                        "collab_workspace",
                    ],
                },
            )
            return
        if path in ("/v1/models",):
            data = [
                {"id": m, "object": "model", "owned_by": "Auro/MESIE"}
                for m in ("Auro-2B", "Auro-4B", "Auro-8B", "Auro-14B", "Auro-100B")
            ]
            # also expose succotash catalogue size if present
            try:
                from auro_native_llm.succotash.registry import load_registry

                reg = load_registry(clone=False)
                data.append(
                    {
                        "id": "succotash-catalogue",
                        "object": "model_catalogue",
                        "owned_by": "FreddyCreates/potential-succotash",
                        "families": len(reg.model_families),
                        "engines": len(reg.engines),
                    }
                )
            except Exception:
                pass
            _json_response(self, 200, {"data": data})
            return
        # --- GitHub knowledge DB (max embeddings) ---
        if path.startswith("/v1/github"):
            from auro_native_llm.corpus.github_db import GitHubKnowledgeDB
            from urllib.parse import parse_qs, urlparse as _up

            gdb = GitHubKnowledgeDB()
            qs = parse_qs(_up(self.path).query)
            if path in ("/v1/github", "/v1/github/stats"):
                _json_response(self, 200, gdb.stats())
                return
            if path == "/v1/github/search":
                q = (qs.get("q") or qs.get("query") or [""])[0]
                k = int((qs.get("k") or qs.get("top_k") or ["8"])[0])
                hits = gdb.search(q, top_k=k)
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "query": q,
                        "hits": [h.to_dict() for h in hits],
                        "embedding_dim": gdb.stats().get("embedding_dim"),
                    },
                )
                return
            if path == "/v1/github/repos":
                _json_response(self, 200, {"ok": True, "repos": gdb.repo_counts()})
                return
        # --- Pythonista host (Python bg + JS render) ---
        if path.startswith("/v1/pythonista"):
            from auro_native_llm.pythonista.service import get_service

            svc = get_service()
            if path in ("/v1/pythonista", "/v1/pythonista/status"):
                _json_response(self, 200, svc.status())
                return
            if path == "/v1/pythonista/ui":
                _json_response(self, 200, svc.ui())
                return
            if path == "/v1/pythonista/tables":
                _json_response(self, 200, svc.tables())
                return
            if path.startswith("/v1/pythonista/table/"):
                name = path.rsplit("/", 1)[-1]
                _json_response(self, 200, svc.table(name))
                return
            if path == "/v1/pythonista/jobs":
                _json_response(self, 200, svc.jobs())
                return
            if path == "/v1/pythonista/host.js":
                js = (_UI.parent / "pythonista_host.js").read_text(encoding="utf-8") if (_UI.parent / "pythonista_host.js").exists() else "/* missing */"
                body = js.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        _json_response(self, 404, {"error": "not found", "path": path})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        body = _read_json(self)

        if path in ("/v1/work", "/work"):
            objective = body.get("objective") or body.get("prompt") or ""
            agent = WorkAgent(
                model_id=body.get("model", _AGENT.model_id),
                lite=bool(body.get("lite", True)),
                chrome_mock=bool(body.get("chrome_mock", True)),
                chrome_auto_start=bool(body.get("chrome_auto_start", False)),
                use_scripture=bool(body.get("scripture", True)),
            )
            result = agent.run(objective)
            _json_response(self, 200 if result.ok else 400, result.to_dict())
            return

        if path in ("/v1/generate", "/generate"):
            prompt = body.get("prompt") or ""
            mode = body.get("mode", "work")
            text = _AGENT.generate_text(prompt, mode=mode, max_new_tokens=int(body.get("max_tokens", 96)))
            _json_response(
                self,
                200,
                {"text": text, "mode": mode, "model": _AGENT.model_id, "compute_plane": "MESIE", "native": True},
            )
            return

        if path in ("/v1/reason", "/reason"):
            topic = body.get("topic") or body.get("prompt") or ""
            _json_response(self, 200, _AGENT.reason(topic))
            return

        if path in ("/v1/code", "/code"):
            task = body.get("task") or body.get("prompt") or ""
            _json_response(self, 200, _AGENT.code(task))
            return

        if path in ("/v1/chrome", "/chrome"):
            action = body.get("action", "dom")
            try:
                if action == "navigate":
                    out = _CHROME.navigate(body.get("url", "https://example.com"))
                elif action == "dom":
                    out = _CHROME.dom()
                elif action == "eval":
                    out = _CHROME.evaluate(body.get("js", "document.title"))
                elif action == "type":
                    out = _CHROME.type_text(body.get("text", ""))
                elif action == "click":
                    out = _CHROME.click(float(body.get("x", 0)), float(body.get("y", 0)))
                else:
                    out = {"ok": False, "error": f"unknown chrome action {action}"}
                _json_response(self, 200 if out.get("ok", True) else 400, out)
            except Exception as exc:
                _json_response(self, 500, {"ok": False, "error": str(exc)})
            return

        # Google virtual envelope + collab
        if path in ("/v1/google/act", "/google/act", "/v1/workspace/act"):
            try:
                env = _get_envelope()
                surface = body.get("surface") or "status"
                action = body.get("action") or "list"
                kw = {k: v for k, v in body.items() if k not in ("surface", "action")}
                out = env.act(surface, action, **kw)
                _json_response(self, 200 if out.get("ok", True) else 400, out)
            except Exception as exc:
                _json_response(self, 500, {"ok": False, "error": str(exc)})
            return

        if path in ("/v1/collab/post", "/collab/post"):
            try:
                mind = _get_mind()
                env = _get_envelope()
                text = body.get("text") or body.get("prompt") or ""
                if mind is not None:
                    out = env.collab_post(text, author=body.get("author", "user"))
                else:
                    out = env.collab.post(text, author=body.get("author", "user"))
                _json_response(self, 200 if out.get("ok", True) else 400, out)
            except Exception as exc:
                _json_response(self, 500, {"ok": False, "error": str(exc)})
            return

        if path in ("/v1/collab/project", "/collab/project"):
            try:
                env = _get_envelope()
                out = env.collab_project(
                    body.get("name") or "Project",
                    body.get("description") or "",
                )
                _json_response(self, 200, out)
            except Exception as exc:
                _json_response(self, 500, {"ok": False, "error": str(exc)})
            return

        if path in ("/v1/scripture/loop", "/scripture/loop"):
            from auro_native_llm.scripture import StructuredCognitiveLoop

            loop = StructuredCognitiveLoop(lite=True)
            r = loop.run(body.get("intent") or body.get("prompt") or "", model_id=body.get("model", "Auro-2B"))
            _json_response(self, 200 if r.ok else 400, r.to_dict())
            return

        if path in ("/v1/mind", "/mind"):
            from auro_native_llm.organism.family import build_mind

            global _MIND
            if _MIND is None:
                _MIND = build_mind(body.get("model", "Auro-2B"), lite=True, chrome_mock=True)
            mind = _MIND
            action = body.get("action", "info")
            args = body.get("args") or {}
            # allow flat kwargs on body for convenience
            for k in ("session_id", "notebook_id", "cell_id", "content", "text", "source",
                      "language", "filename", "title", "tool", "arguments", "query", "online",
                      "op", "objective", "old", "new", "offset", "cell_type", "js", "url"):
                if k in body and k not in args:
                    args[k] = body[k]
            if action == "info":
                _json_response(self, 200, mind.info())
            elif action == "teach":
                _json_response(self, 200, mind.teach().to_dict())
            elif action == "monaco":
                op = body.get("op", args.pop("op", "create"))
                _json_response(self, 200, mind.monaco(op, **args).to_dict())
            elif action == "jupyter":
                op = body.get("op", args.pop("op", "create"))
                _json_response(self, 200, mind.jupyter(op, **args).to_dict())
            elif action == "search":
                q = body.get("query") or args.get("query") or ""
                online = bool(body.get("online", args.get("online", False)))
                _json_response(self, 200, mind.search(q, online=online).to_dict())
            elif action == "mcp":
                op = body.get("op", args.pop("op", "spin_up"))
                _json_response(self, 200, mind.mcp(op, **args).to_dict())
            elif action == "work":
                _json_response(self, 200, mind.work(body.get("objective") or args.get("objective") or "").to_dict())
            elif action == "generate":
                _json_response(self, 200, mind.generate(body.get("prompt") or args.get("prompt") or "").to_dict())
            else:
                _json_response(self, 400, {"error": f"unknown mind action {action}"})
            return

        # --- Interior MCP portal + multi-site agents ---
        if path in ("/v1/portal", "/portal/api"):
            from auro_native_llm.organism.family import build_mind
            from auro_native_llm.embedded.portal import build_portal

            global _MIND, _PORTAL
            if _MIND is None:
                _MIND = build_mind(body.get("model", "Auro-2B"), lite=True, chrome_mock=True)
            if _PORTAL is None or body.get("action") == "spin":
                _PORTAL = build_portal(_MIND, chrome_mock=bool(body.get("chrome_mock", True)))
            portal = _PORTAL
            action = body.get("action", "manifest")
            if action == "spin":
                out = portal.spin()
                _json_response(self, 200, out)
                return
            if action == "manifest":
                _json_response(self, 200, portal.manifest())
                return
            if action == "list_tools":
                _json_response(self, 200, portal.list_tools())
                return
            if action == "call":
                tool = body.get("tool") or body.get("name")
                args = body.get("arguments") or body.get("args") or {}
                _json_response(self, 200, portal.call(str(tool), args))
                return
            if action == "multi_site":
                urls = body.get("urls") or ["https://example.com"]
                obj = body.get("objective") or "survey"
                _json_response(
                    self,
                    200,
                    _MIND.multi_site(obj, urls, chrome_mock=bool(body.get("chrome_mock", True))),
                )
                return
            _json_response(self, 400, {"error": f"unknown portal action {action}"})
            return

        # --- GitHub knowledge DB POST ops ---
        if path.startswith("/v1/github"):
            from auro_native_llm.corpus.github_db import GitHubKnowledgeDB

            gdb = GitHubKnowledgeDB()
            if path == "/v1/github/search":
                q = body.get("query") or body.get("q") or ""
                hits = gdb.search(q, top_k=int(body.get("top_k", 8)))
                _json_response(
                    self,
                    200,
                    {"ok": True, "query": q, "hits": [h.to_dict() for h in hits]},
                )
                return
            if path == "/v1/github/harvest":
                stats = gdb.harvest_and_ingest(
                    include_github=bool(body.get("include_github", True)),
                    include_succotash=bool(body.get("include_succotash", True)),
                    max_files=int(body.get("max_files", 2000)),
                    max_chars=int(body.get("max_chars", 6_000_000)),
                    reembed=bool(body.get("reembed", True)),
                )
                _json_response(self, 200, stats)
                return
            if path == "/v1/github/train":
                from auro_native_llm.corpus.continual import ContinualConfig, run_continual_training

                report = run_continual_training(
                    ContinualConfig(
                        model_id=body.get("model", "Auro-2B"),
                        rounds=int(body.get("rounds", 2)),
                        steps_per_round=int(body.get("steps", 10)),
                        resume_checkpoint=body.get("resume"),
                        expand_harvest=bool(body.get("expand", False)),
                        lite=bool(body.get("lite", True)),
                    )
                )
                slim = {
                    k: report.get(k)
                    for k in (
                        "ok",
                        "num_params_live",
                        "train_steps",
                        "elapsed_s",
                        "checkpoint",
                        "resumed",
                        "db_stats_final",
                    )
                }
                slim["history_tail"] = (report.get("history") or [])[-2:]
                _json_response(self, 200, slim)
                return

        # --- Pythonista: run scripts, events, bridge, background ---
        if path.startswith("/v1/pythonista"):
            from auro_native_llm.pythonista.service import get_service

            svc = get_service()
            if path in ("/v1/pythonista/run", "/v1/pythonista"):
                _json_response(
                    self,
                    200,
                    svc.run_script(
                        body.get("source"),
                        script_name=body.get("script_name") or body.get("script"),
                        background=bool(body.get("background")),
                        interval_s=float(body.get("interval_s") or 0),
                    ),
                )
                return
            if path == "/v1/pythonista/event":
                _json_response(
                    self,
                    200,
                    svc.event(
                        body.get("node_id") or body.get("id") or "",
                        body.get("action", "action"),
                        body.get("payload") or body.get("data"),
                    ),
                )
                return
            if path == "/v1/pythonista/bridge":
                _json_response(self, 200, svc.bridge_send(body))
                return
            if path == "/v1/pythonista/cancel":
                _json_response(self, 200, svc.cancel_job(body.get("job_id") or ""))
                return

        _json_response(self, 404, {"error": "not found", "path": path})


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Auro work API + UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--real-chrome", action="store_true", help="Use real Chrome CDP (not mock)")
    args = p.parse_args()

    global _CHROME, _AGENT
    if args.real_chrome:
        _CHROME = ChromeToolbelt(mock=False, auto_start=True)
        _AGENT = WorkAgent(lite=True, chrome_mock=False, chrome_auto_start=True, use_scripture=True)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {
                "status": "auro-work-api",
                "url": f"http://{args.host}:{args.port}/",
                "health": f"http://{args.host}:{args.port}/health",
                "compute_plane": "MESIE",
                "chrome_mock": not args.real_chrome,
            },
            indent=2,
        )
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutdown")
        httpd.server_close()


if __name__ == "__main__":
    main()
