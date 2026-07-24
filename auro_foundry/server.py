from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .generation import TextGenerator


class FoundryServer:
    def __init__(self, checkpoint: str | Path, *, device: str = "auto", static_dir: str | Path | None = None) -> None:
        self.generator = TextGenerator(checkpoint, device=device)
        self.static_dir = Path(static_dir or Path(__file__).with_name("web")).resolve()

    def handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "AuroFoundry/1.0"

            def _json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:
                self._json(200, {"ok": True})

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/health":
                    self._json(200, {"status": "ok", "model": outer.generator.metadata})
                    return
                if path == "/v1/models":
                    self._json(200, {"object": "list", "data": [outer.generator.metadata]})
                    return
                if path == "/":
                    path = "/index.html"
                candidate = (outer.static_dir / path.lstrip("/")).resolve()
                try:
                    candidate.relative_to(outer.static_dir)
                except ValueError:
                    self._json(403, {"error": "forbidden"})
                    return
                if not candidate.is_file():
                    self._json(404, {"error": "not found"})
                    return
                body = candidate.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                if path not in {"/v1/completions", "/v1/chat/completions"}:
                    self._json(404, {"error": "not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    request = json.loads(self.rfile.read(length) or b"{}")
                    prompt = request.get("prompt")
                    if prompt is None and isinstance(request.get("messages"), list):
                        prompt = "\n".join(
                            f"{item.get('role', 'user')}: {item.get('content', '')}"
                            for item in request["messages"]
                            if isinstance(item, dict)
                        )
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError("prompt or messages are required")
                    text = outer.generator.generate(
                        prompt,
                        max_new_tokens=int(request.get("max_tokens", request.get("max_new_tokens", 128))),
                        temperature=float(request.get("temperature", 0.8)),
                        top_k=int(request.get("top_k", 50)),
                        top_p=float(request.get("top_p", 0.95)),
                    )
                    if path.endswith("chat/completions"):
                        payload = {"object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
                    else:
                        payload = {"object": "text_completion", "choices": [{"index": 0, "text": text, "finish_reason": "stop"}]}
                    self._json(200, payload)
                except Exception as exc:
                    self._json(400, {"error": str(exc)})

            def log_message(self, format: str, *args) -> None:
                return

        return Handler


def serve(checkpoint: str | Path, *, host: str = "127.0.0.1", port: int = 8090, device: str = "auto", static_dir: str | Path | None = None) -> None:
    app = FoundryServer(checkpoint, device=device, static_dir=static_dir)
    server = ThreadingHTTPServer((host, port), app.handler())
    print(json.dumps({"status": "serving", "url": f"http://{host}:{port}", "checkpoint": str(checkpoint)}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="auro-foundry-serve")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--static-dir")
    args = parser.parse_args()
    serve(args.checkpoint, host=args.host, port=args.port, device=args.device, static_dir=args.static_dir)


if __name__ == "__main__":
    main()
