"""HTTP API for durable AURO multi-task orchestration."""
from __future__ import annotations

import argparse
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit
import uuid

from .task_service import TaskService


API_VERSION = "2026-08-25-task-v1"
MAX_REQUEST_BYTES = 2 * 1024 * 1024


def bearer_authorized(value: str, expected: str) -> bool:
    return bool(
        expected
        and value.startswith("Bearer ")
        and hmac.compare_digest(value[7:], expected)
    )


class TaskHandler(BaseHTTPRequestHandler):
    service: TaskService | None = None
    server_version = "AuroTaskOrchestrator/1.0"

    @classmethod
    def get_service(cls) -> TaskService:
        if cls.service is None:
            cls.service = TaskService()
        return cls.service

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except TaskApiError as exc:
            self._error(exc.status, exc.code, exc.message)
        except KeyError as exc:
            self._error(404, "not_found", str(exc).strip("'")[:500])
        except PermissionError as exc:
            self._error(403, "forbidden", str(exc)[:500])
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, "invalid_request", str(exc)[:500])
        except Exception as exc:
            message = str(exc)[:1000] if os.getenv("AURO_DEBUG") == "1" else "The task request could not be completed."
            self._error(500, "internal_error", message)

    def do_GET(self) -> None:
        self.request_id = self._request_id()
        path = self._path()
        if path in {"/health", "/v1/health/live"}:
            return self._json(200, {"ok": True, "service": "auro-task-orchestrator", "api_version": API_VERSION})
        self._require_api_auth()
        service = self.get_service()
        if path in {"/v1", "/v1/tasks/status"}:
            return self._json(200, service.status())
        scope = self._scope()
        query = parse_qs(urlsplit(self.path).query)
        if path == "/v1/tasks":
            return self._json(200, {"tasks": service.list(scope, limit=int(query.get("limit", ["100"])[0]))})
        run_id, suffix = self._task_route(path)
        if run_id is None:
            raise TaskApiError(404, "not_found", "The requested route does not exist.")
        if suffix == "":
            return self._json(200, service.get(run_id, scope))
        if suffix == "/events":
            return self._json(
                200,
                {
                    "events": service.events(
                        run_id,
                        scope,
                        after_event_id=int(query.get("after", ["0"])[0]),
                        limit=int(query.get("limit", ["500"])[0]),
                    )
                },
            )
        if suffix == "/artifacts":
            return self._json(200, {"artifacts": service.artifacts(run_id, scope)})
        match = suffix.removeprefix("/artifacts/").split("/") if suffix.startswith("/artifacts/") else []
        if len(match) == 2 and match[1] == "download":
            artifact_id = match[0]
            path_value = service.artifact_path(run_id, artifact_id, scope)
            artifact = next(item for item in service.artifacts(run_id, scope) if item["artifact_id"] == artifact_id)
            return self._download(path_value, artifact["media_type"], artifact["name"])
        raise TaskApiError(404, "not_found", "The requested task route does not exist.")

    def do_POST(self) -> None:
        self.request_id = self._request_id()
        self._require_api_auth()
        self._require_execution_auth()
        service = self.get_service()
        path = self._path()
        body = self._body()
        scope = self._scope(body.get("scope") if isinstance(body.get("scope"), dict) else None)

        if path == "/v1/tasks":
            request = body.get("task") if isinstance(body.get("task"), dict) else body
            result = service.submit(
                request,
                scope,
                idempotency_key=self.headers.get("idempotency-key") or body.get("idempotency_key"),
            )
            wait_seconds = float(body.get("wait_seconds", 0) or 0)
            if wait_seconds > 0:
                result = service.run_until_idle(
                    result["run_id"],
                    scope,
                    timeout_seconds=min(wait_seconds, 300.0),
                )
            return self._json(202 if result["status"] not in {"succeeded", "failed", "cancelled"} else 200, result)

        run_id, suffix = self._task_route(path)
        if run_id is None:
            raise TaskApiError(404, "not_found", "The requested route does not exist.")
        if suffix == "/pause":
            return self._json(200, service.pause(run_id, scope))
        if suffix == "/resume":
            return self._json(200, service.resume(run_id, scope))
        if suffix == "/cancel":
            return self._json(200, service.cancel(run_id, scope))
        if suffix == "/run":
            timeout = min(float(body.get("timeout_seconds", 300.0)), 3600.0)
            return self._json(200, service.run_until_idle(run_id, scope, timeout_seconds=timeout))
        if suffix == "/approve":
            step_id = str(body.get("step_id") or "").strip()
            if not step_id:
                raise TaskApiError(400, "step_id_required", "step_id is required for approval")
            return self._json(200, service.approve(run_id, step_id, scope))
        raise TaskApiError(404, "not_found", "The requested task route does not exist.")

    def do_OPTIONS(self) -> None:
        self.request_id = self._request_id()
        return self._bytes(204, "text/plain; charset=utf-8", b"")

    def _path(self) -> str:
        return urlsplit(self.path).path

    @staticmethod
    def _task_route(path: str) -> tuple[str | None, str]:
        prefix = "/v1/tasks/"
        if not path.startswith(prefix):
            return None, ""
        tail = path[len(prefix):]
        run_id, separator, rest = tail.partition("/")
        if not run_id:
            return None, ""
        return run_id, ("/" + rest if separator else "")

    def _scope(self, fallback: dict[str, Any] | None = None) -> dict[str, str]:
        source = dict(fallback or {})
        values = {
            "organization_id": self.headers.get("x-auro-organization-id") or source.get("organization_id"),
            "workspace_id": self.headers.get("x-auro-workspace-id") or source.get("workspace_id"),
            "operator_id": self.headers.get("x-auro-operator-id") or source.get("operator_id"),
        }
        return {key: str(value or "").strip() for key, value in values.items()}

    def _require_api_auth(self) -> None:
        expected = os.getenv("AURO_API_TOKEN", "")
        if expected and not bearer_authorized(self.headers.get("authorization", ""), expected):
            raise TaskApiError(401, "api_token_required", "A valid API bearer token is required.")

    def _require_execution_auth(self) -> None:
        expected = os.getenv("AURO_EXECUTION_TOKEN", "")
        header = self.headers.get("x-auro-execution-token", "")
        if not expected or not (
            hmac.compare_digest(header, expected)
            or bearer_authorized(self.headers.get("authorization", ""), expected)
        ):
            raise TaskApiError(403, "execution_token_required", "A valid execution token is required for task mutations.")

    def _body(self) -> dict[str, Any]:
        media = self.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media != "application/json":
            raise TaskApiError(415, "json_required", "Content-Type must be application/json.")
        try:
            length = int(self.headers.get("content-length", "0"))
        except ValueError as exc:
            raise TaskApiError(400, "invalid_content_length", "Content-Length must be an integer.") from exc
        if length <= 0:
            raise TaskApiError(400, "body_required", "A JSON request body is required.")
        if length > MAX_REQUEST_BYTES:
            raise TaskApiError(413, "body_too_large", "Task request body is too large.")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise TaskApiError(400, "object_required", "Task request must be a JSON object.")
        return value

    def _request_id(self) -> str:
        supplied = self.headers.get("x-request-id", "").strip()
        if supplied and len(supplied) <= 128 and supplied.replace("-", "").replace("_", "").isalnum():
            return supplied
        return "req_" + uuid.uuid4().hex

    def _error(self, status: int, code: str, message: str) -> None:
        if getattr(self, "wfile", None) is not None and not self.wfile.closed:
            self._json(status, {"error": {"code": code, "message": message, "request_id": getattr(self, "request_id", None)}})

    def _json(self, status: int, payload: Any) -> None:
        self._bytes(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _download(self, path: Path, media_type: str, name: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("content-type", media_type or "application/octet-stream")
        self.send_header("content-length", str(len(data)))
        self.send_header("content-disposition", f'attachment; filename="{name.replace(chr(34), "")}"')
        self.send_header("cache-control", "private, no-store")
        self.send_header("x-content-type-options", "nosniff")
        self.send_header("x-request-id", getattr(self, "request_id", ""))
        self.end_headers()
        self.wfile.write(data)

    def _bytes(self, status: int, media_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("content-type", media_type)
        self.send_header("content-length", str(len(data)))
        self.send_header("cache-control", "no-store")
        self.send_header("x-content-type-options", "nosniff")
        self.send_header("x-frame-options", "DENY")
        self.send_header("referrer-policy", "no-referrer")
        self.send_header("x-request-id", getattr(self, "request_id", ""))
        self.send_header("x-auro-task-api-version", API_VERSION)
        self.end_headers()
        if data:
            self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        return


class TaskApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status = int(status)
        self.code = str(code)
        self.message = str(message)
        super().__init__(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the AURO durable task orchestrator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and len(os.getenv("AURO_API_TOKEN", "")) < 32:
        raise SystemExit("non-loopback task server requires AURO_API_TOKEN with at least 32 characters")
    ThreadingHTTPServer((args.host, args.port), TaskHandler).serve_forever()


if __name__ == "__main__":
    main()
