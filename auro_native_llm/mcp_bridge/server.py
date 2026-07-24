"""Dependency-light AURO MCP/REST bridge server."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .core import AuroMCPBridge, ExecutionContext, HTTPServiceAdapter, LocalAdapter, RiskTier, ToolDefinition


def build_default_bridge(config: dict[str, Any] | None = None) -> AuroMCPBridge:
    config = config or {}
    bridge = AuroMCPBridge(receipt_path=config.get("receipt_path", "artifacts/mcp/receipts.jsonl"))
    local = LocalAdapter()
    local.register(
        ToolDefinition(
            name="auro.bridge.health",
            description="Return AURO MCP bridge health and receipt-chain status.",
            input_schema={"type": "object", "properties": {}},
            adapter="local",
            operation="health",
            risk_tier=RiskTier.READ_ONLY,
            tags=("auro", "health"),
        ),
        lambda _: {
            "ok": True,
            "server": "auro-mcp-bridge",
            "receipt_tip": bridge.receipts.tip,
            "tool_count": len(bridge.tools),
        },
    )
    local.register(
        ToolDefinition(
            name="auro.context.catalog",
            description="Return a compressed, query-ranked catalog of available AURO tools.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
            },
            adapter="local",
            operation="catalog",
            risk_tier=RiskTier.READ_ONLY,
            tags=("auro", "context", "meta-tool"),
        ),
        lambda args: {
            "catalog": bridge.context_catalog(
                str(args.get("query", "")), int(args.get("limit", 12))
            )
        },
    )
    bridge.add_adapter(local)
    for service in config.get("service_bridges", []):
        bridge.add_adapter(
            HTTPServiceAdapter(
                service,
                allowed_hosts=config.get("allowed_hosts", ("127.0.0.1", "localhost")),
            )
        )
    return bridge


class Handler(BaseHTTPRequestHandler):
    bridge: AuroMCPBridge
    api_token: str | None = None

    def _send(self, status: int, payload: Any) -> None:
        encoded = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _authorized(self) -> bool:
        return not self.api_token or self.headers.get("Authorization") == f"Bearer {self.api_token}"

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            self._send(401, {"ok": False, "error": "unauthorized"})
            return
        if self.path == "/health":
            self._send(
                200,
                {
                    "ok": True,
                    "uri": "mcp://localhost:8080",
                    "tools": len(self.bridge.tools),
                    "receipt_tip": self.bridge.receipts.tip,
                },
            )
        elif self.path == "/tools":
            self._send(200, {"tools": self.bridge.list_tools(compact=True)})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._send(401, {"ok": False, "error": "unauthorized"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"ok": False, "error": "invalid json"})
            return
        context = ExecutionContext(
            actor=self.headers.get("X-Auro-Actor", "auro-http"),
            confirmed=self.headers.get("X-Auro-Confirm", "false").lower() == "true",
            allow_isolated=self.headers.get("X-Auro-Isolated", "false").lower() == "true",
        )
        if self.path in {"/mcp", "/"}:
            self._send(200, self.bridge.handle_jsonrpc(body, context=context))
        elif self.path == "/invoke":
            try:
                self._send(
                    200,
                    self.bridge.call_tool(
                        body["name"], body.get("arguments") or {}, context=context
                    ),
                )
            except Exception as exc:
                self._send(400, {"ok": False, "error": str(exc)[:500]})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--token")
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    Handler.bridge = build_default_bridge(config)
    Handler.api_token = args.token
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {
                "ok": True,
                "mcp": f"mcp://{args.host}:{args.port}",
                "http": f"http://{args.host}:{args.port}/mcp",
                "tools": len(Handler.bridge.tools),
            }
        )
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
