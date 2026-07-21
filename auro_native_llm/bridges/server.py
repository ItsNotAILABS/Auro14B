"""Local AURO bridge control plane exposed at mcp://localhost:8080."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from .runtime import AuroCapabilityRegistry, JSServiceAdapter, MCPHTTPAdapter, ReceiptChain


def build_registry(config: Dict[str, Any], receipt_path: Path) -> AuroCapabilityRegistry:
    registry = AuroCapabilityRegistry(receipts=ReceiptChain(receipt_path))
    for item in config.get("mcp_servers", []):
        registry.register(
            MCPHTTPAdapter(
                item["name"],
                item["endpoint"],
                headers=item.get("headers"),
                timeout=float(item.get("timeout", 15)),
            )
        )
    for item in config.get("js_services", []):
        schema = item.get("schema") or json.loads(
            Path(item["schema_path"]).read_text(encoding="utf-8")
        )
        registry.register(
            JSServiceAdapter(
                item["name"],
                item["base_url"],
                schema,
                headers=item.get("headers"),
                timeout=float(item.get("timeout", 15)),
            )
        )
    registry.refresh()
    return registry


def make_handler(registry: AuroCapabilityRegistry):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, indent=2, default=str).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> Dict[str, Any]:
            length = int(self.headers.get("content-length", "0"))
            return json.loads(self.rfile.read(length).decode() or "{}")

        def do_GET(self):
            if self.path == "/health":
                self._send(200, registry.health())
            elif self.path.startswith("/tools"):
                self._send(200, {"tools": registry.compact_catalog("", max_tools=1000)})
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            payload = self._body()
            if self.path == "/refresh":
                self._send(200, registry.refresh())
            elif self.path == "/invoke":
                result = registry.invoke(
                    payload["tool"],
                    payload.get("arguments"),
                    confirmed=bool(payload.get("confirmed")),
                )
                status = 200 if result.get("ok") else 403 if result.get("denied") else 500
                self._send(status, result)
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def log_message(self, format, *args):
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/auro_bridge.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--receipts",
        type=Path,
        default=Path("artifacts/bridge/receipts.jsonl"),
    )
    args = parser.parse_args()
    config = (
        json.loads(args.config.read_text(encoding="utf-8"))
        if args.config.exists()
        else {"mcp_servers": [], "js_services": []}
    )
    registry = build_registry(config, args.receipts)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(registry))
    print(
        json.dumps(
            {
                "ok": True,
                "display": f"mcp://localhost:{args.port}",
                "http": f"http://{args.host}:{args.port}",
                "tools": len(registry.tools),
            }
        )
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
