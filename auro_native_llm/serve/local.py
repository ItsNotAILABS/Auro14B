"""Auro native local server — MESIE compute plane.

Validates the serving contract, optionally starts a stdlib HTTP server that
exposes the Auro family via MESIE-native generate/embed (no cloud LLM), plus a
read-only Browser-Brain archive for preserved HIM/AURO conversations.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from auro_native_llm.browser_brain.service import BrowserBrainService
from auro_native_llm.native_runtime import AuroNativeRuntime
from auro_native_llm.receipt import emit_receipt, load_json_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auro native serving — MESIE compute (scaffold + optional live server)"
    )
    parser.add_argument("--config", required=True, help="Path to serving contract JSON")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Start MESIE-native HTTP server (not just receipt validation)",
    )
    parser.add_argument("--host", default=None, help="Override host")
    parser.add_argument("--port", type=int, default=None, help="Override port")
    parser.add_argument("--parent", default="Auro-14B", help="Default parent model")
    parser.add_argument(
        "--repository-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root scanned by the read-only Browser-Brain archive",
    )
    args = parser.parse_args()

    config = load_json_config(args.config)
    required = ["schema", "server", "routes", "required_receipts"]
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"missing serving config keys: {', '.join(missing)}")

    config = dict(config)
    config.setdefault("compute_plane", "MESIE")
    config.setdefault("native", True)
    config.setdefault("cloud_llm", False)

    receipt = emit_receipt("serving_contract", args.config, config)

    if not args.live:
        return

    server_cfg = config.get("server", {})
    host = args.host or server_cfg.get("default_host", "127.0.0.1")
    port = int(args.port or server_cfg.get("default_port", 8090))
    runtime = AuroNativeRuntime(parent_model_id=args.parent)
    browser_brain = BrowserBrainService(args.repository_root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *a: Any) -> None:
            print(f"[auro-serve] {self.address_string()} {fmt % a}")

        def _json(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path in ("/health", "/v1/health"):
                health = runtime.health()
                health["browser_brain"] = {
                    "read_only": True,
                    "conversation_count": browser_brain.index()["conversation_count"],
                }
                self._json(200, health)
                return
            if path in ("/v1/models", "/models"):
                self._json(200, runtime.serve_models_payload())
                return
            if path in ("/v1/receipts/latest", "/receipts/latest"):
                self._json(200, receipt)
                return
            if path in ("/v1/browser-brain", "/browser-brain"):
                index = browser_brain.index()
                self._json(200, {
                    "schema": "auro.browser-brain.discovery.v1",
                    "read_only": True,
                    "conversation_count": index["conversation_count"],
                    "index_sha256": index["index_sha256"],
                    "routes": [
                        "/v1/browser-brain/conversations",
                        "/v1/browser-brain/conversations/{archive_id}",
                        "/v1/browser-brain/conversations/{archive_id}/timeline",
                        "/v1/browser-brain/conversations/{archive_id}/continuation",
                    ],
                })
                return
            if path == "/v1/browser-brain/conversations":
                needle = (query.get("q") or [""])[0]
                limit = int((query.get("limit") or ["100"])[0])
                self._json(200, browser_brain.list(needle, limit))
                return
            prefix = "/v1/browser-brain/conversations/"
            if path.startswith(prefix):
                suffix = path[len(prefix):].strip("/")
                parts = suffix.split("/") if suffix else []
                archive_id = parts[0] if parts else ""
                if not archive_id:
                    self._json(400, {"error": "archive_id required"})
                    return
                if len(parts) == 1:
                    payload = browser_brain.get(archive_id)
                elif parts[1] == "timeline":
                    payload = browser_brain.timeline(archive_id)
                elif parts[1] == "continuation":
                    maximum = int((query.get("max_characters") or ["24000"])[0])
                    payload = browser_brain.continuation_context(archive_id, maximum)
                else:
                    payload = None
                self._json(200, payload) if payload is not None else self._json(404, {"error": "conversation not found", "archive_id": archive_id})
                return
            self._json(404, {"error": "not found", "path": path})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            body = self._read_json()
            if path in ("/v1/completions", "/v1/chat/completions", "/v1/generate"):
                prompt = body.get("prompt") or body.get("input") or ""
                if not prompt and "messages" in body:
                    msgs = body.get("messages") or []
                    prompt = "\n".join(
                        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs
                    )
                model_id = body.get("model") or args.parent
                role = body.get("role")
                max_tokens = int(body.get("max_tokens", 256))
                try:
                    gen = runtime.generate(
                        prompt,
                        model_id=model_id,
                        role=role,
                        max_tokens=max_tokens,
                        spectral_context=body.get("spectral_context"),
                    )
                    self._json(
                        200,
                        {
                            "id": f"auro-native-{gen.model_id}",
                            "object": "text_completion",
                            "model": gen.model_id,
                            "compute_plane": "MESIE",
                            "native": True,
                            "choices": [{"text": gen.text, "index": 0, "finish_reason": "stop"}],
                            "auro": gen.to_dict(),
                        },
                    )
                except Exception as exc:
                    self._json(400, {"error": str(exc), "compute_plane": "MESIE"})
                return
            if path in ("/v1/dispatch", "/dispatch"):
                role = body.get("role", "plan")
                intent = body.get("intent") or body.get("prompt") or ""
                result = runtime.dispatch(role, intent, spectral_context=body.get("spectral_context"))
                self._json(200 if result.ok else 400, result.to_dict())
                return
            self._json(404, {"error": "not found", "path": path})

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(
        json.dumps(
            {
                "status": "auro-native-mesie-live",
                "host": host,
                "port": port,
                "compute_plane": "MESIE",
                "parent": args.parent,
                "models": runtime.family_models.list_models(),
                "browser_brain": {
                    "read_only": True,
                    "repository_root": str(Path(args.repository_root).resolve()),
                    "conversation_count": browser_brain.index()["conversation_count"],
                },
            },
            indent=2,
        )
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down Auro MESIE native server")
        httpd.server_close()


if __name__ == "__main__":
    main()
