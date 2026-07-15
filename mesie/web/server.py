"""MESIE Web Server — standalone HTTP server for raw web deployment.

Uses only the Python standard library (http.server). No external
dependencies required. For production, use the WSGI interface with
gunicorn or similar.

Usage:
    python -m mesie.web                # default port 8080
    python -m mesie.web --port 9000    # custom port
    mesie-web --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from mesie.web.app import MESIEWebApp


class MESIERequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler bridging to MESIEWebApp."""

    app: MESIEWebApp  # set by serve()

    def do_OPTIONS(self) -> None:
        self._dispatch()

    def do_GET(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def do_PUT(self) -> None:
        self._dispatch()

    def do_DELETE(self) -> None:
        self._dispatch()

    def _dispatch(self) -> None:
        """Read request and forward to the app."""
        # Read body
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            content_length = 0
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # Collect headers as lowercase dict
        headers = {k.lower(): v for k, v in self.headers.items()}

        # Dispatch
        status, resp_headers, resp_body = self.app.handle_request(
            method=self.command,
            path=self.path,
            headers=headers,
            body=body,
        )

        # Send response
        self.send_response(status)
        for key, value in resp_headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(resp_body)

    def log_message(self, format: str, *args) -> None:
        """Structured log output."""
        sys.stderr.write(f"[mesie-web] {self.address_string()} - {format % args}\n")


def serve(
    host: str = "0.0.0.0",
    port: int = 8080,
    api_key: Optional[str] = None,
) -> None:
    """Start the MESIE web server.

    Args:
        host: Bind address.
        port: Listen port.
        api_key: Optional API key for authentication.
    """
    app = MESIEWebApp(api_key=api_key)
    MESIERequestHandler.app = app

    server = HTTPServer((host, port), MESIERequestHandler)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         MESIE Spectral Intelligence Engine — Web            ║
╠══════════════════════════════════════════════════════════════╣
║  Status:   ONLINE                                          ║
║  Address:  http://{host}:{port:<5}                            ║
║  Auth:     {"ENABLED" if api_key else "OPEN (no key)":50}║
║  Engines:  {len(app.router.registry.names()):<5} loaded                                    ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                ║
║    GET  /health              — health check                ║
║    GET  /v1/engines          — list engines                ║
║    POST /v1/engine/{{n}}/{{a}}   — call engine action         ║
║    POST /v1/match            — spectral matching           ║
║    POST /v1/validate         — record validation           ║
║    POST /v1/embed            — embeddings                  ║
║    POST /v1/intelligence/reason — AI reasoning             ║
║    POST /v1/tokenomics/score — cognitive scoring           ║
╚══════════════════════════════════════════════════════════════╝
""")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mesie-web] Shutting down.")
        server.shutdown()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mesie-web",
        description="Deploy MESIE Spectral Intelligence Engine to raw web",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Listen port (default: 8080)")
    parser.add_argument("--api-key", default=None, help="Require this API key for auth")
    args = parser.parse_args()
    serve(host=args.host, port=args.port, api_key=args.api_key)


if __name__ == "__main__":
    main()
