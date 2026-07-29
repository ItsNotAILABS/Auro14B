"""Single-pass production HTTP API for AURO/HIM and the Auro-2B triad."""
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

API_VERSION = "2026-07-29"
MAX_REQUEST_BYTES = 1_048_576
MAX_MESSAGE_CHARS = 12_000
TRIAD_MODEL_ID = "auro-2b-triad-swarm"


def token_authorized(header: str, expected: str) -> bool:
    return bool(expected and header.startswith("Bearer ") and hmac.compare_digest(header[7:], expected))


def extract_user_message(messages: Any) -> str:
    if not isinstance(messages, list):
        raise ApiError(400, "messages_must_be_an_array", "messages must be an array")
    for item in reversed(messages):
        if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), str) and item["content"].strip():
            return item["content"].strip()
    raise ApiError(400, "user_message_required", "A non-empty user message is required.")


def openai_completion(response: dict[str, Any], request_id: str, *, model_override: str | None = None) -> dict[str, Any]:
    model = response.get("model") or {}
    text = str(response.get("answer") or response.get("text") or "")
    model_id = model_override or str(model.get("model") or response.get("model_id") or "auro-him")
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "estimated": False},
        "auro": {
            "schema": response.get("schema"),
            "confidence": response.get("confidence") or (response.get("structured_answer") or {}).get("confidence"),
            "reasoning_summary": response.get("reasoning_summary") or (response.get("structured_answer") or {}).get("reasoning_summary", []),
            "agents": response.get("agents", []),
            "proposed_actions": response.get("proposed_actions", []),
            "executions": response.get("executions", []),
            "receipt": response.get("receipt"),
            "runtime_receipt_sha256": response.get("runtime_receipt_sha256"),
            "promotion_ready": response.get("promotion_ready"),
            "blockers": response.get("blockers", []),
            "parameter_count_verified": model.get("parameter_count_verified", False),
            "agent_count_is_not_parameter_count": True,
        },
    }


class Handler(BaseHTTPRequestHandler):
    runtime: Any = None
    server_version = "AuroHIM/2.1"

    @classmethod
    def get_runtime(cls):
        if cls.runtime is None:
            from .runtime import NovaRuntime
            cls.runtime = NovaRuntime()
        return cls.runtime

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except ApiError as exc:
            self._error(exc.status, exc.code, exc.message)
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, "invalid_request", str(exc)[:300])
        except Exception as exc:
            self._error(500, "internal_error", str(exc)[:300] if os.getenv("AURO_DEBUG") == "1" else "The request could not be completed.")

    def do_GET(self):
        self.request_id = self._request_id()
        path = self._path()
        if path in ASSETS:
            content_type, data = ASSETS[path]
            return self._bytes(200, content_type, data)
        if path in {"/health", "/v1/health/live"}:
            return self._json(200, {"ok": True, "status": "live", "service": "auro-him-api", "api_version": API_VERSION})
        if path == "/openapi.json":
            return self._json(200, self._openapi())
        self._require_api_auth()
        runtime = self.get_runtime()
        if path == "/v1/health/ready":
            return self._json(200, {"ok": True, "status": "ready", "service": "auro-him-api", "receipt_chain": runtime.capabilities.ledger.verify(), "model_fleet": runtime.model_orchestrator.manifest(), "triad": runtime.triad_status()})
        if path == "/v1":
            return self._json(200, self._discovery())
        if path == "/v1/models":
            return self._json(200, {"object": "list", "data": self._models()})
        if path.startswith("/v1/models/"):
            requested = path.removeprefix("/v1/models/")
            model = next((item for item in self._models() if requested in {item["id"], item.get("auro_endpoint_id")}), None)
            if model is None:
                raise ApiError(404, "model_not_found", "The requested model is not configured.")
            return self._json(200, model)
        if path == "/v1/triad":
            return self._json(200, runtime.triad_status())
        if path == "/v1/capabilities":
            return self._json(200, runtime.capabilities.manifest())
        if path == "/v1/context":
            return self._json(200, runtime.context.stats())
        if path == "/v1/receipts/verify":
            return self._json(200, runtime.capabilities.ledger.verify())
        if path == "/v1/receipts":
            return self._json(200, {"receipts": runtime.capabilities.ledger.tail(20)})
        if path == "/v1/browser/tasks":
            return self._json(200, {"tasks": runtime.capabilities.browser.list(50)})
        if path.startswith("/v1/downloads/") and path.endswith(".zip"):
            artifact = runtime.capabilities.resolve_download(path.removeprefix("/v1/downloads/").removesuffix(".zip"))
            if artifact is None:
                raise ApiError(404, "artifact_not_found", "The requested artifact is not registered.")
            return self._bytes(200, "application/zip", artifact.read_bytes())
        raise ApiError(404, "not_found", "The requested route does not exist.")

    def do_POST(self):
        self.request_id = self._request_id()
        path = self._path()
        self._require_api_auth()
        runtime = self.get_runtime()
        body = self._body()
        if path == "/v1/capabilities/call":
            approved = bool(body.get("approved", False))
            if approved:
                self._require_execution_auth()
            return self._json(200, runtime.capabilities.call(str(body.get("name", "")), dict(body.get("arguments") or {}), approved=approved))
        if path == "/v1/context/query":
            pack = runtime.context.retrieve(self._message(body.get("query")), token_budget=int(body.get("token_budget") or runtime.context.default_budget), top_k=int(body.get("top_k") or 24))
            return self._json(200, pack.public())
        if path == "/v1/context/ingest":
            self._require_execution_auth()
            text = self._message(body.get("text"))
            return self._json(200, runtime.context.ingest(text, source=str(body.get("source") or "api"), kind=str(body.get("kind") or "document"), importance=float(body.get("importance", 0.5)), metadata=dict(body.get("metadata") or {}), chunk_tokens=int(body.get("chunk_tokens") or 900), allow_sensitive=bool(body.get("allow_sensitive", False))))
        if path == "/v1/browser/tasks/claim":
            return self._json(200, {"task": runtime.capabilities.browser.claim(str(body.get("worker_id") or "chrome"))})
        if path.startswith("/v1/browser/tasks/") and path.endswith("/complete"):
            task_id = path.split("/")[4]
            return self._json(200, runtime.capabilities.browser.complete(task_id, body.get("result"), body.get("error")))
        if path == "/v1/triad/respond":
            message = self._message(body.get("message"))
            context = str(body.get("context") or "")
            if len(context) > 100_000:
                raise ApiError(413, "context_too_large", "Triad continuation context exceeds 100,000 characters.")
            result = runtime.triad_respond(message, context=context)
            result["request_id"] = self.request_id
            return self._json(200, result)
        if path in {"/v1/respond", "/v1/him/respond"}:
            execute = bool(body.get("execute", False))
            if execute:
                self._require_execution_auth()
            result = runtime.respond(self._message(body.get("message")), execute=execute)
            result["request_id"] = self.request_id
            return self._json(200, result)
        if path == "/v1/chat/completions":
            if body.get("stream"):
                raise ApiError(400, "streaming_not_supported", "Set stream=false.")
            execute = bool(body.get("auro_execute", False))
            if execute:
                self._require_execution_auth()
            requested = str(body.get("model") or runtime.endpoint.model)
            allowed = {value for model in self._models() for value in (model["id"], model.get("auro_endpoint_id")) if value} | {"auro-him"}
            if requested not in allowed:
                raise ApiError(404, "model_not_found", "The requested model is not configured.")
            message = self._message(extract_user_message(body.get("messages")))
            if requested == TRIAD_MODEL_ID:
                result = runtime.triad_respond(message)
                return self._json(200, openai_completion(result, self.request_id, model_override=TRIAD_MODEL_ID))
            result = runtime.respond(message, execute=execute)
            return self._json(200, openai_completion(result, self.request_id))
        raise ApiError(404, "not_found", "The requested route does not exist.")

    def do_OPTIONS(self):
        self.request_id = self._request_id()
        return self._bytes(204, "text/plain; charset=utf-8", b"")

    def _path(self) -> str:
        return urlsplit(self.path).path

    def _request_id(self) -> str:
        supplied = self.headers.get("x-request-id", "").strip()
        return supplied if supplied and len(supplied) <= 128 and supplied.replace("-", "").replace("_", "").isalnum() else "req_" + uuid.uuid4().hex

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
        if self.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise ApiError(415, "json_required", "Content-Type must be application/json.")
        try:
            length = int(self.headers.get("content-length", "0"))
        except ValueError as exc:
            raise ApiError(400, "invalid_content_length", "Content-Length must be an integer.") from exc
        if length <= 0:
            raise ApiError(400, "body_required", "A JSON request body is required.")
        if length > MAX_REQUEST_BYTES:
            raise ApiError(413, "body_too_large", "Request body is too large.")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ApiError(400, "object_required", "The JSON body must be an object.")
        return value

    def _message(self, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ApiError(400, "message_required", "A non-empty message is required.")
        message = value.strip()
        if len(message) > MAX_MESSAGE_CHARS:
            raise ApiError(413, "message_too_large", f"Message exceeds {MAX_MESSAGE_CHARS} characters.")
        return message

    def _models(self) -> list[dict[str, Any]]:
        runtime = self.get_runtime()
        models = [{"id": item["model"], "object": "model", "owned_by": "ItsNotAILABS" if item["provider"] == "repository-native-open-weights" else item["provider"], "auro_endpoint_id": item["id"], "role": item["role"], "provider": item["provider"], "capabilities": item["capabilities"], "local": item["local"], "parameter_count": item["parameter_count"], "parameter_count_verified": item["parameter_count_verified"], "identity_verified": item["identity_verified"], "agent_count_is_not_parameter_count": True} for item in runtime.model_orchestrator.manifest()["models"]]
        triad = runtime.triad_status()
        models.append({"id": TRIAD_MODEL_ID, "object": "model", "owned_by": "ItsNotAILABS", "auro_endpoint_id": TRIAD_MODEL_ID, "role": "composed_hierarchical_runtime", "provider": "auro-composed-runtime", "capabilities": ["general", "code", "research", "creative", "tool", "mesie"], "local": True, "parameter_count": None, "parameter_count_verified": False, "identity_verified": bool(triad["enabled"]), "enabled": triad["enabled"], "claim": "Auro-2B plus three independently accounted 500M specialist lanes and atomic swarms; not one merged checkpoint", "agent_count_is_not_parameter_count": True})
        return models

    def _discovery(self) -> dict[str, Any]:
        return {"service": "Auro14B · HIM", "api_version": API_VERSION, "native_response": "/v1/him/respond", "triad_response": "/v1/triad/respond", "triad_status": "/v1/triad", "openai_compatible": "/v1/chat/completions", "models": "/v1/models", "capabilities": "/v1/capabilities", "receipts": "/v1/receipts", "openapi": "/openapi.json"}

    def _openapi(self) -> dict[str, Any]:
        return {"openapi": "3.1.0", "info": {"title": "Auro14B · HIM API", "version": API_VERSION}, "paths": {"/v1/health/live": {"get": {}}, "/v1/triad": {"get": {}}, "/v1/triad/respond": {"post": {}}, "/v1/chat/completions": {"post": {}}, "/v1/capabilities/call": {"post": {}}}}

    def _error(self, status: int, code: str, message: str):
        if getattr(self, "wfile", None) is not None and not self.wfile.closed:
            self._json(status, {"error": {"code": code, "message": message, "request_id": getattr(self, "request_id", None)}})

    def _json(self, status: int, payload: Any):
        return self._bytes(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode())

    def _bytes(self, status: int, content_type: str, data: bytes):
        self.send_response(status)
        for key, value in {"content-type": content_type, "content-length": str(len(data)), "cache-control": "no-store", "x-request-id": getattr(self, "request_id", ""), "x-auro-api-version": API_VERSION, "x-content-type-options": "nosniff", "x-frame-options": "DENY", "referrer-policy": "no-referrer"}.items():
            self.send_header(key, value)
        self.end_headers()
        if data:
            self.wfile.write(data)

    def log_message(self, format, *args):
        return


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)


def main():
    parser = argparse.ArgumentParser(description="Serve AURO/HIM over HTTP")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
