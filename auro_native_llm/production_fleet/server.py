"""Production HTTP API and operator console for Auro14B · HIM.

Dependency-free by design. The service exposes a native receipt-rich contract
and an OpenAI-compatible chat subset while keeping execution separately gated.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .console import ASSETS
from .runtime import NovaRuntime

API_VERSION = "2026-07-18"
MAX_REQUEST_BYTES = 1_048_576
MAX_MESSAGE_CHARS = 12_000


def token_authorized(header: str, expected: str) -> bool:
    if not expected or not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header[7:], expected)


def extract_user_message(messages: Any) -> str:
    """Return the final user message from an OpenAI-style message array."""
    if not isinstance(messages, list):
        raise ValueError("messages_must_be_an_array")
    for item in reversed(messages):
        if isinstance(item, dict) and item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    raise ValueError("user_message_required")


def openai_completion(response: dict[str, Any], request_id: str) -> dict[str, Any]:
    """Adapt the native response without hiding its evidence extensions."""
    text = str(response.get("answer", ""))
    model = str((response.get("model") or {}).get("model", "auro-him"))
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "estimated": False},
        "auro": {
            "schema": response.get("schema"),
            "confidence": response.get("confidence"),
            "reasoning_summary": response.get("reasoning_summary", []),
            "agents": response.get("agents", []),
            "proposed_actions": response.get("proposed_actions", []),
            "executions": response.get("executions", []),
            "receipt": response.get("receipt"),
            "parameter_count_verified": (response.get("model") or {}).get("parameter_count_verified", False),
        },
    }


class Handler(BaseHTTPRequestHandler):
    runtime = NovaRuntime()
    server_version = "AuroHIM/1.0"

    def do_GET(self):
        self.request_id = self._request_id()
        path = self._path()
        if path in ASSETS:
            content_type, data = ASSETS[path]
            self._bytes(200, content_type, data)
        elif path in {"/health", "/v1/health/live"}:
            self._json(200, {"ok": True, "status": "live", "service": "auro-him-api", "api_version": API_VERSION})
        elif path == "/v1/health/ready":
            self._require_api_auth()
            endpoint = self.runtime.endpoint
            self._json(200, {
                "ok": True,
                "status": "ready",
                "service": "auro-him-api",
                "model_endpoint": {"id": endpoint.id, "model": endpoint.model, "base_url_configured": bool(endpoint.base_url)},
                "receipt_chain": self.runtime.capabilities.ledger.verify(),
            })
        elif path == "/v1":
            self._require_api_auth()
            self._json(200, self._discovery())
        elif path == "/v1/models":
            self._require_api_auth()
            self._json(200, {"object": "list", "data": [self._model()]})
        elif path.startswith("/v1/models/"):
            self._require_api_auth()
            model = self._model()
            if path.removeprefix("/v1/models/") != model["id"]:
                raise ApiError(404, "model_not_found", "The requested model is not configured.")
            self._json(200, model)
        elif path == "/v1/capabilities":
            self._require_api_auth()
            self._json(200, self.runtime.capabilities.manifest())
        elif path == "/v1/receipts/verify":
            self._require_api_auth()
            self._json(200, self.runtime.capabilities.ledger.verify())
        elif path == "/v1/receipts":
            self._require_api_auth()
            self._json(200, {"receipts": self.runtime.capabilities.ledger.tail(20)})
        elif path == "/v1/browser/tasks":
            self._require_api_auth(); self._json(200,{"tasks":self.runtime.capabilities.browser.list(50)})
        elif path == "/openapi.json":
            self._json(200, self._openapi())
        else:
            raise ApiError(404, "not_found", "The requested route does not exist.")

    def do_POST(self):
        self.request_id = self._request_id()
        path = self._path()
        self._require_api_auth()
        if path == "/v1/capabilities/call":
            body = self._body()
            approved = bool(body.get("approved", False))
            if approved:
                self._require_execution_auth()
            result = self.runtime.capabilities.call(
                str(body.get("name", "")), dict(body.get("arguments") or {}), approved=approved
            )
            self._json(200, result)
            return
        if path == "/v1/browser/tasks/claim":
            body=self._body();self._json(200,{"task":self.runtime.capabilities.browser.claim(str(body.get("worker_id") or "chrome"))});return
        if path.startswith("/v1/browser/tasks/") and path.endswith("/complete"):
            body=self._body();task_id=path.split("/")[4];self._json(200,self.runtime.capabilities.browser.complete(task_id,body.get("result"),body.get("error")));return
        if path in {"/v1/respond", "/v1/him/respond"}:
            body = self._body()
            message = self._message(body.get("message"))
            execute = bool(body.get("execute", False))
            if execute:
                self._require_execution_auth()
            result = self.runtime.respond(message, execute=execute)
            result["request_id"] = self.request_id
            self._json(200, result)
            return
        if path == "/v1/chat/completions":
            body = self._body()
            if body.get("stream"):
                raise ApiError(400, "streaming_not_supported", "Set stream=false; streaming is not implemented yet.")
            requested_model = str(body.get("model") or self.runtime.endpoint.model)
            if requested_model not in {self.runtime.endpoint.model, self.runtime.endpoint.id, "auro-him"}:
                raise ApiError(404, "model_not_found", "The requested model is not configured.")
            execute = bool(body.get("auro_execute", False))
            if execute:
                self._require_execution_auth()
            result = self.runtime.respond(self._message(extract_user_message(body.get("messages"))), execute=execute)
            self._json(200, openai_completion(result, self.request_id))
            return
        raise ApiError(404, "not_found", "The requested route does not exist.")

    def do_OPTIONS(self):
        self.request_id = self._request_id()
        self._bytes(204, "text/plain; charset=utf-8", b"")

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except ApiError as exc:
            self._error(exc.status, exc.code, exc.message)
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, "invalid_request", str(exc)[:300])
        except Exception:
            self._error(500, "internal_error", "The request could not be completed.")

    def log_message(self, format, *args):
        return

    def _path(self) -> str:
        return urlsplit(self.path).path

    def _request_id(self) -> str:
        supplied = self.headers.get("x-request-id", "").strip()
        if supplied and len(supplied) <= 128 and supplied.replace("-", "").replace("_", "").isalnum():
            return supplied
        return "req_" + uuid.uuid4().hex

    def _require_api_auth(self):
        expected = os.getenv("AURO_API_TOKEN", "")
        if expected and not token_authorized(self.headers.get("authorization", ""), expected):
            raise ApiError(401, "api_token_required", "A valid API bearer token is required.")

    def _require_execution_auth(self):
        expected = os.getenv("AURO_EXECUTION_TOKEN", "")
        header = self.headers.get("x-auro-execution-token", "")
        bearer = self.headers.get("authorization", "")
        if not expected or not (hmac.compare_digest(header, expected) or token_authorized(bearer, expected)):
            raise ApiError(403, "operator_token_required", "A valid execution token is required.")

    def _body(self) -> dict[str, Any]:
        content_type = self.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ApiError(415, "json_required", "Content-Type must be application/json.")
        try:
            length = int(self.headers.get("content-length", "0"))
        except ValueError as exc:
            raise ApiError(400, "invalid_content_length", "Content-Length must be an integer.") from exc
        if length <= 0:
            raise ApiError(400, "body_required", "A JSON request body is required.")
        if length > MAX_REQUEST_BYTES:
            raise ApiError(413, "request_body_too_large", "Request body exceeds 1 MiB.")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ApiError(400, "object_required", "The JSON body must be an object.")
        return value

    def _message(self, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ApiError(400, "message_required", "A non-empty user message is required.")
        message = value.strip()
        if len(message) > MAX_MESSAGE_CHARS:
            raise ApiError(413, "message_too_large", f"Message exceeds {MAX_MESSAGE_CHARS} characters.")
        return message

    def _model(self) -> dict[str, Any]:
        endpoint = self.runtime.endpoint
        return {
            "id": endpoint.model,
            "object": "model",
            "owned_by": "ItsNotAILABS",
            "auro_endpoint_id": endpoint.id,
            "role": endpoint.role,
            "parameter_count": endpoint.parameter_count,
            "parameter_count_verified": endpoint.parameter_count is not None,
            "agent_count_is_not_parameter_count": True,
        }

    def _discovery(self) -> dict[str, Any]:
        return {
            "service": "Auro14B · HIM",
            "api_version": API_VERSION,
            "native_response": "/v1/him/respond",
            "openai_compatible": "/v1/chat/completions",
            "models": "/v1/models",
            "capabilities": "/v1/capabilities",
            "receipts": "/v1/receipts",
            "openapi": "/openapi.json",
            "execution_header": "X-Auro-Execution-Token",
        }

    def _openapi(self) -> dict[str, Any]:
        return {
            "openapi": "3.1.0",
            "info": {"title": "Auro14B · HIM API", "version": API_VERSION},
            "servers": [{"url": "http://127.0.0.1:8090"}],
            "paths": {
                "/v1/health/live": {"get": {"summary": "Liveness"}},
                "/v1/health/ready": {"get": {"summary": "Runtime readiness"}},
                "/v1/models": {"get": {"summary": "Configured models"}},
                "/v1/him/respond": {"post": {"summary": "Native receipt-rich HIM response"}},
                "/v1/chat/completions": {"post": {"summary": "OpenAI-compatible chat completion"}},
                "/v1/capabilities": {"get": {"summary": "Native capability contracts"}},
                "/v1/capabilities/call": {"post": {"summary": "Call a governed native capability"}},
                "/v1/receipts": {"get": {"summary": "Recent receipts"}},
                "/v1/receipts/verify": {"get": {"summary": "Verify the receipt chain"}},
            },
        }

    def _error(self, status: int, code: str, message: str):
        if getattr(self, "wfile", None) is None or self.wfile.closed:
            return
        self._json(status, {"error": {"code": code, "message": message, "request_id": getattr(self, "request_id", None)}})

    def _json(self, status: int, payload: Any):
        self._bytes(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode())

    def _bytes(self, status: int, content_type: str, data: bytes):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.send_header("cache-control", "no-store")
        self.send_header("x-request-id", getattr(self, "request_id", ""))
        self.send_header("x-auro-api-version", API_VERSION)
        self.send_header("x-content-type-options", "nosniff")
        self.send_header("x-frame-options", "DENY")
        self.send_header("referrer-policy", "no-referrer")
        self.send_header("content-security-policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        self.end_headers()
        if data:
            self.wfile.write(data)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)


def main():
    parser = argparse.ArgumentParser(description="Serve Auro14B · HIM over HTTP")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
