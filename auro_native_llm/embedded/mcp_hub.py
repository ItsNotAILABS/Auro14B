"""Self-spinning MCP hub — mind can start its own tool server and call tools.

Implements a minimal JSON-RPC MCP-like HTTP surface so Auro can:
  - spin up a local MCP server
  - list tools
  - call tools (monaco, jupyter, search, chrome, mind)
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional
from urllib import request as urlrequest

ToolHandler = Callable[[Dict[str, Any]], Dict[str, Any]]
InfoFn = Callable[[], Dict[str, Any]]


@dataclass
class MCPTool:
    name: str
    description: str
    handler: ToolHandler
    input_schema: Dict[str, Any] = field(default_factory=dict)

    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema or {"type": "object", "properties": {}},
        }


class MCPHub:
    """In-process MCP tool registry + optional HTTP server."""

    def __init__(self) -> None:
        self.tools: Dict[str, MCPTool] = {}
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.host = "127.0.0.1"
        self.port = 0
        self.server_id = f"mcp-{uuid.uuid4().hex[:8]}"
        self.call_log: List[Dict[str, Any]] = []

    def register(self, tool: MCPTool) -> None:
        self.tools[tool.name] = tool

    def list_tools(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        arguments = arguments or {}
        tool = self.tools.get(name)
        if not tool:
            return {"ok": False, "error": f"unknown tool {name}"}
        try:
            result = tool.handler(arguments)
            entry = {"tool": name, "ok": True, "ts": time.time()}
            self.call_log.append(entry)
            return {"ok": True, "tool": name, "result": result}
        except Exception as exc:
            self.call_log.append({"tool": name, "ok": False, "error": str(exc)})
            return {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}"}

    def spin_up(self, host: str = "127.0.0.1", port: int = 0) -> Dict[str, Any]:
        if self._server is not None:
            return {
                "ok": True,
                "already": True,
                "url": f"http://{self.host}:{self.port}",
                "server_id": self.server_id,
            }
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def _json(self, code: int, payload: Dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path in ("/health", "/mcp/health"):
                    self._json(200, {"ok": True, "server_id": hub.server_id, "tools": len(hub.tools)})
                    return
                if self.path in ("/tools", "/mcp/tools"):
                    self._json(200, {"tools": hub.list_tools()})
                    return
                self._json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n) if n else b"{}"
                try:
                    data = json.loads(raw.decode("utf-8") or "{}")
                except json.JSONDecodeError:
                    data = {}
                # JSON-RPC style
                if "method" in data:
                    method = data["method"]
                    params = data.get("params") or {}
                    if method in ("tools/list", "list_tools"):
                        self._json(200, {"jsonrpc": "2.0", "id": data.get("id"), "result": {"tools": hub.list_tools()}})
                        return
                    if method in ("tools/call", "call_tool"):
                        name = params.get("name") or params.get("tool")
                        args = params.get("arguments") or params.get("args") or {}
                        result = hub.call(str(name), args)
                        self._json(200, {"jsonrpc": "2.0", "id": data.get("id"), "result": result})
                        return
                # simple REST
                name = data.get("tool") or data.get("name")
                result = hub.call(str(name), data.get("arguments") or data.get("args") or {})
                self._json(200, result)

        self._server = ThreadingHTTPServer((host, port), Handler)
        self.host, self.port = self._server.server_address[0], self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return {
            "ok": True,
            "server_id": self.server_id,
            "url": f"http://{self.host}:{self.port}",
            "tools": self.list_tools(),
        }

    def shutdown(self) -> Dict[str, Any]:
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None
        return {"ok": True, "shutdown": True}

    def client_call(self, tool: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call via HTTP if server up, else in-process."""
        if self._server is None:
            return self.call(tool, arguments)
        payload = json.dumps({"tool": tool, "arguments": arguments or {}}).encode("utf-8")
        req = urlrequest.Request(
            f"http://{self.host}:{self.port}/mcp/call",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # REST path uses do_POST on any path
        req = urlrequest.Request(
            f"http://{self.host}:{self.port}/",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))


class MCPOrgan:
    """Mind organ: register embedded tools and spin MCP for itself."""

    def __init__(self, hub: Optional[MCPHub] = None) -> None:
        self.hub = hub or MCPHub()
        self._wired = False

    def wire_from_mind_organs(
        self,
        *,
        monaco: Any = None,
        jupyter: Any = None,
        search: Any = None,
        chrome: Any = None,
        mind_info: Optional[InfoFn] = None,
    ) -> None:
        if monaco is not None:
            self.hub.register(
                MCPTool(
                    "monaco.create",
                    "Create Monaco editor session",
                    lambda a: monaco.create(
                        a.get("content", ""),
                        language=a.get("language", "python"),
                        filename=a.get("filename", "main.py"),
                    ),
                    {"type": "object", "properties": {"content": {"type": "string"}, "language": {"type": "string"}}},
                )
            )
            self.hub.register(
                MCPTool(
                    "monaco.set",
                    "Set Monaco session content",
                    lambda a: monaco.set_content(a["session_id"], a.get("content", "")),
                    {"type": "object", "properties": {"session_id": {"type": "string"}, "content": {"type": "string"}}},
                )
            )
            self.hub.register(
                MCPTool(
                    "monaco.get",
                    "Get Monaco session",
                    lambda a: monaco.get(a["session_id"]),
                    {"type": "object", "properties": {"session_id": {"type": "string"}}},
                )
            )
        if jupyter is not None:
            self.hub.register(
                MCPTool(
                    "jupyter.create",
                    "Create notebook",
                    lambda a: jupyter.create(a.get("title", "Auro Notebook")),
                )
            )
            self.hub.register(
                MCPTool(
                    "jupyter.add_cell",
                    "Add notebook cell",
                    lambda a: jupyter.add_cell(a["notebook_id"], a.get("source", ""), cell_type=a.get("cell_type", "code")),
                )
            )
            self.hub.register(
                MCPTool(
                    "jupyter.execute",
                    "Execute notebook cell",
                    lambda a: jupyter.execute_cell(a["notebook_id"], a["cell_id"]),
                )
            )
            self.hub.register(
                MCPTool(
                    "jupyter.execute_all",
                    "Execute all code cells",
                    lambda a: jupyter.execute_all(a["notebook_id"]),
                )
            )
        if search is not None:
            self.hub.register(
                MCPTool(
                    "search",
                    "Online + local search",
                    lambda a: search.search(a.get("query", ""), top_k=int(a.get("top_k", 5)), online=bool(a.get("online", True))),
                )
            )
        if chrome is not None:
            self.hub.register(
                MCPTool("chrome.navigate", "Navigate Chrome", lambda a: chrome.navigate(a.get("url", "about:blank")))
            )
            self.hub.register(MCPTool("chrome.dom", "Chrome DOM snapshot", lambda a: chrome.dom()))
        if mind_info is not None:
            self.hub.register(MCPTool("mind.info", "Mind organ manifest", lambda a: mind_info()))
        self._wired = True

    def spin_up(self) -> Dict[str, Any]:
        if not self._wired:
            return {"ok": False, "error": "MCP not wired to organs yet"}
        return self.hub.spin_up()

    def call(self, tool: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.hub.call(tool, arguments)

    def list_tools(self) -> Dict[str, Any]:
        return {"ok": True, "tools": self.hub.list_tools()}
