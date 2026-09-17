"""Host OS control tools for Auro14B / RO14B models + external AIs (Claude/Grok).

Powers from POCKET research:
- Microsoft protocol (UIA / open apps / guarded shell)
- Fusion Sense / page symbols
- Desktop open
- Screenshot / sense
- Mesh dispatch (optional)

Primary path: call local POCKET host (http://127.0.0.1:8787) with desktop auth.
Fallback: best-effort local imports if pocket is on PYTHONPATH.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

POCKET_URL = (os.environ.get("POCKET_URL") or "http://127.0.0.1:8787").rstrip("/")
_token: Optional[str] = None


def tool_catalog() -> List[Dict[str, Any]]:
    """OpenAI-style tool definitions for Claude / Grok / Codex / Auro agents."""
    return [
        {
            "type": "function",
            "function": {
                "name": "host_status",
                "description": "Check POCKET host runtime + whether host control is available",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "host_open_app",
                "description": "Open an allowlisted Windows app (notepad, edge, explorer, code, terminal, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": "App id: edge|code|explorer|notepad|terminal|..."}
                    },
                    "required": ["app"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "host_sense_page",
                "description": "Fusion Sense: deep UIA+OCR+visual symbol graph of the current host UI (page_render)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Optional focus prompt"},
                        "max_ui": {"type": "integer", "description": "Max UI symbols", "default": 400},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "host_screenshot",
                "description": "Capture host screenshot via POCKET OCULUS/screenshot skill",
                "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "host_click",
                "description": "Click UI at coordinates or by symbol search (when POCKET ui_click available)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "query": {"type": "string", "description": "Optional text/symbol to find then click"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "host_dispatch",
                "description": "Dispatch @agent on POCKET mesh (DESIGN, ARCHON, OCULUS, FORGE_HEADLESS, ...)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["agent", "message"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "host_ms_protocol",
                "description": "Microsoft host protocol: status|open|sense|page|shell|screenshot",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "app": {"type": "string"},
                        "cmd": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["action"],
                },
            },
        },
    ]


def _http_json(method: str, path: str, body: Optional[dict] = None, token: Optional[str] = None) -> Dict[str, Any]:
    url = POCKET_URL + path
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["X-Pocket-Token"] = token
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {"ok": True}
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
            return {"ok": False, "error": err_body, "http": e.code}
        except Exception:
            return {"ok": False, "error": str(e), "http": e.code}
    except Exception as e:
        return {"ok": False, "error": str(e), "pocket_url": POCKET_URL}


def ensure_token() -> Optional[str]:
    global _token
    if _token:
        return _token
    r = _http_json("POST", "/v1/auth/desktop", {})
    if r.get("ok") and r.get("token"):
        _token = r["token"]
        return _token
    return None


def status() -> Dict[str, Any]:
    h = _http_json("GET", "/health")
    tok = ensure_token()
    return {
        "ok": bool(h.get("ok")),
        "pocket": h,
        "token": bool(tok),
        "tools": [t["function"]["name"] for t in tool_catalog()],
        "powers": [
            "open_app",
            "sense_page",
            "screenshot",
            "click",
            "mesh_dispatch",
            "ms_protocol",
        ],
        "note": "Host control via POCKET on 8787 — RO14B/Auro models + Claude/Grok use /v1/host/*",
    }


def open_app(app: str = "explorer") -> Dict[str, Any]:
    tok = ensure_token()
    if not tok:
        return {"ok": False, "error": "POCKET host not available for desktop auth"}
    # Prefer orchestrator-style message to PORTARIUS
    r = _http_json(
        "POST",
        "/v1/subagents/dispatch",
        {"name": "PORTARIUS", "message": f"open {app}"},
        token=tok,
    )
    if r.get("ok"):
        return r
    # fallback orchestrator
    return _http_json(
        "POST",
        "/v1/orchestrator/chat",
        {"prompt": f"open app {app}", "skill": "open_app"},
        token=tok,
    )


def sense_page(prompt: str = "", max_ui: int = 400) -> Dict[str, Any]:
    tok = ensure_token()
    if not tok:
        return {"ok": False, "error": "POCKET host not available"}
    # try vision page
    r = _http_json("GET", f"/v1/vision/page?max_ui={int(max_ui)}", token=tok)
    if r.get("ok") or r.get("symbols") or r.get("count"):
        return r
    return _http_json(
        "POST",
        "/v1/subagents/dispatch",
        {"name": "OCULUS", "message": prompt or "sense full page"},
        token=tok,
    )


def screenshot(prompt: str = "") -> Dict[str, Any]:
    tok = ensure_token()
    if not tok:
        return {"ok": False, "error": "POCKET host not available"}
    return _http_json(
        "POST",
        "/v1/subagents/dispatch",
        {"name": "OCULUS", "message": prompt or "screenshot"},
        token=tok,
    )


def click(x: float = 0, y: float = 0, query: str = "") -> Dict[str, Any]:
    tok = ensure_token()
    if not tok:
        return {"ok": False, "error": "POCKET host not available"}
    msg = f"click {query}" if query else f"click at {x},{y}"
    return _http_json(
        "POST",
        "/v1/subagents/dispatch",
        {"name": "ARCHON", "message": msg},
        token=tok,
    )


def dispatch(agent: str, message: str) -> Dict[str, Any]:
    tok = ensure_token()
    if not tok:
        return {"ok": False, "error": "POCKET host not available"}
    return _http_json(
        "POST",
        "/v1/subagents/dispatch",
        {"name": (agent or "ARCHON").upper(), "message": message or ""},
        token=tok,
    )


def ms_protocol(action: str = "status", **kwargs: Any) -> Dict[str, Any]:
    """Run Microsoft protocol actions via POCKET module if importable, else dispatch."""
    try:
        import sys
        from pathlib import Path

        pocket_src = Path(os.environ.get("POCKET_ROOT") or r"C:\Users\Medin\OneDrive\pocket-os") / "src"
        if pocket_src.is_dir() and str(pocket_src) not in sys.path:
            sys.path.insert(0, str(pocket_src))
        from pocket.ms_protocol import run as ms_run

        return ms_run(action, prompt=kwargs.get("prompt") or "", app=kwargs.get("app") or "explorer", cmd=kwargs.get("cmd") or "")
    except Exception:
        return dispatch("PORTARIUS", f"ms_protocol {action} {kwargs}")


def invoke(name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Dispatch a tool by name (for Claude/Grok tool_calls)."""
    args = arguments or {}
    n = (name or "").lower().replace("-", "_")
    if n in ("host_status", "status"):
        return status()
    if n in ("host_open_app", "open_app"):
        return open_app(str(args.get("app") or "explorer"))
    if n in ("host_sense_page", "sense_page", "page_render"):
        return sense_page(str(args.get("prompt") or ""), int(args.get("max_ui") or 400))
    if n in ("host_screenshot", "screenshot"):
        return screenshot(str(args.get("prompt") or ""))
    if n in ("host_click", "click"):
        return click(float(args.get("x") or 0), float(args.get("y") or 0), str(args.get("query") or ""))
    if n in ("host_dispatch", "dispatch"):
        return dispatch(str(args.get("agent") or "ARCHON"), str(args.get("message") or ""))
    if n in ("host_ms_protocol", "ms_protocol"):
        return ms_protocol(str(args.get("action") or "status"), **args)
    return {"ok": False, "error": f"unknown tool {name}", "tools": [t["function"]["name"] for t in tool_catalog()]}
