"""Production HTTP control plane for AURO independent harnesses."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import hmac
import json
import os
import uuid
from typing import Any

from .harness import IndependentHarnessFabric
from .harness_orchestrator import HarnessOrchestrator

API_VERSION = "3.0.0"
MAX_BODY = 1_048_576


def _authorized(header: str, expected: str) -> bool:
    return bool(expected and header.startswith("Bearer ") and hmac.compare_digest(header[7:], expected))


class HarnessHandler(BaseHTTPRequestHandler):
    fabric = IndependentHarnessFabric()
    orchestrator = HarnessOrchestrator(fabric)
    server_version = "AuroHarness/3.0"

    def do_GET(self):
        self.request_id = "hreq_" + uuid.uuid4().hex
        path = urlsplit(self.path).path
        if path == "/health":
            return self._json(200, {"ok": True, "service": "auro-harness", "version": API_VERSION})
        self._auth(False)
        if path == "/v1/harnesses":
            return self._json(200, {"harnesses": [x.to_dict() for x in self.fabric.store.list()], "fabric": self.fabric.manifest()})
        if path.startswith("/v1/harnesses/"):
            harness_id = path.split("/")[3]
            if not self.fabric.store.exists(harness_id):
                return self._error(404, "harness_not_found")
            return self._json(200, self.fabric.store.load(harness_id).to_dict())
        if path == "/v1/manifest":
            return self._json(200, self.fabric.manifest())
        return self._error(404, "not_found")

    def do_POST(self):
        self.request_id = "hreq_" + uuid.uuid4().hex
        path = urlsplit(self.path).path
        self._auth(True)
        body = self._body()
        if path == "/v1/harnesses":
            state = self.fabric.create_harness(
                str(body.get("objective") or ""),
                parent_id=body.get("parent_id"),
                model_id=str(body.get("model_id") or "Auro-2B"),
                agent_roster=body.get("agent_roster"),
                tasks=body.get("tasks"),
            )
            return self._json(201, state.to_dict())
        if path == "/v1/orchestrate":
            result = self.orchestrator.orchestrate(
                str(body.get("objective") or ""),
                model_id=str(body.get("model_id") or "Auro-2B"),
                max_children=int(body.get("max_children") or 6),
            )
            return self._json(201, result)
        if path.startswith("/v1/harnesses/"):
            parts = path.split("/")
            if len(parts) < 5:
                return self._error(404, "not_found")
            harness_id, action = parts[3], parts[4]
            if not self.fabric.store.exists(harness_id):
                return self._error(404, "harness_not_found")
            if action == "run":
                return self._json(200, self.fabric.run_until_blocked(harness_id, worker_id=str(body.get("worker_id") or "api"), max_cycles=int(body.get("max_cycles") or 16)))
            if action == "advance-tree":
                return self._json(200, self.orchestrator.advance_tree(harness_id, worker_id=str(body.get("worker_id") or "api"), cycles_per_child=int(body.get("cycles_per_child") or 8)))
            if action == "fanout":
                children = self.fabric.fan_out(harness_id, [str(x) for x in body.get("subproblems") or []], model_id=body.get("model_id"))
                return self._json(201, {"children": [x.to_dict() for x in children]})
            if action == "task":
                task = self.fabric.add_task(harness_id, str(body.get("objective") or ""), depends_on=list(body.get("depends_on") or []), max_attempts=int(body.get("max_attempts") or 3))
                return self._json(201, {"task": task.__dict__})
            if action == "pause":
                return self._json(200, self.fabric.pause(harness_id).to_dict())
            if action == "resume":
                return self._json(200, self.fabric.resume(harness_id).to_dict())
            if action == "cancel":
                return self._json(200, self.fabric.cancel(harness_id).to_dict())
            if action == "aggregate":
                return self._json(200, self.fabric.aggregate(harness_id))
        return self._error(404, "not_found")

    def _auth(self, write: bool):
        token = os.getenv("AURO_HARNESS_TOKEN") or os.getenv("AURO_API_TOKEN", "")
        if token and not _authorized(self.headers.get("authorization", ""), token):
            raise PermissionError("invalid harness token")
        if write and os.getenv("AURO_ENV", "").lower() == "production" and len(token) < 32:
            raise PermissionError("AURO_HARNESS_TOKEN must be at least 32 characters in production")

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or 0)
        if length <= 0 or length > MAX_BODY:
            raise ValueError("invalid body length")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON object required")
        return value

    def _json(self, status: int, payload: Any):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.send_header("cache-control", "no-store")
        self.send_header("x-request-id", getattr(self, "request_id", ""))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, code: str):
        return self._json(status, {"error": {"code": code, "request_id": getattr(self, "request_id", None)}})

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except PermissionError as exc:
            self._json(403, {"error": {"code": "forbidden", "message": str(exc)}})
        except Exception as exc:
            self._json(400, {"error": {"code": "invalid_request", "message": str(exc)[:500]}})


def serve(host: str = "127.0.0.1", port: int = 8092):
    server = ThreadingHTTPServer((host, int(port)), HarnessHandler)
    server.serve_forever()


if __name__ == "__main__":
    serve(os.getenv("AURO_HARNESS_HOST", "127.0.0.1"), int(os.getenv("AURO_HARNESS_PORT", "8092")))
