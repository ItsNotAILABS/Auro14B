"""Single-pass production HTTP API for Auro14B/HIM and the Auro-2B council."""
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

API_VERSION = "2026-08-25"
MAX_REQUEST_BYTES = 1_048_576
MAX_MESSAGE_CHARS = 12_000
MAX_PARENT_CONTEXT_CHARS = 48_000
MIN_PRODUCTION_SECRET_CHARS = 32


def token_authorized(header: str, expected: str) -> bool:
    return bool(
        expected
        and header.startswith("Bearer ")
        and hmac.compare_digest(header[7:], expected)
    )


def production_mode() -> bool:
    return (
        os.getenv("AURO_ENV", "").strip().lower() == "production"
        or os.getenv("AURO_PRODUCTION", "").strip().lower() in {"1", "true", "yes"}
    )


def production_security_status(host: str | None = None) -> dict[str, Any]:
    required = {
        "AURO_API_TOKEN": os.getenv("AURO_API_TOKEN", ""),
        "AURO_EXECUTION_TOKEN": os.getenv("AURO_EXECUTION_TOKEN", ""),
        "AURO_APPROVAL_HMAC_KEY": os.getenv("AURO_APPROVAL_HMAC_KEY", ""),
    }
    council_configured = bool(
        os.getenv("AURO_COUNCIL_CONFIG_JSON", "").strip()
        or os.getenv("AURO_COUNCIL_CONFIG_PATH", "").strip()
    )
    if council_configured:
        required["AURO_COUNCIL_RECEIPT_HMAC_KEY"] = os.getenv(
            "AURO_COUNCIL_RECEIPT_HMAC_KEY", ""
        )
    checks = {
        name: len(value) >= MIN_PRODUCTION_SECRET_CHARS
        for name, value in required.items()
    }
    distinct = len({value for value in required.values() if value}) == len(required)
    production = production_mode()
    ready = (not production) or (all(checks.values()) and distinct)
    return {
        "mode": "production" if production else "development",
        "ready": ready,
        "required_secret_min_chars": MIN_PRODUCTION_SECRET_CHARS,
        "secret_checks": checks,
        "secrets_distinct": distinct if production else None,
        "bind_host": host,
        "caller_boolean_approval": False,
        "signed_action_approval": True,
        "replay_protection": "one-time-per-action",
        "council_configured": council_configured,
        "council_signed_receipt_required_in_production": council_configured,
    }


def extract_user_message(messages: Any) -> str:
    if not isinstance(messages, list):
        raise ApiError(400, "messages_must_be_an_array", "messages must be an array")
    for item in reversed(messages):
        if (
            isinstance(item, dict)
            and item.get("role") == "user"
            and isinstance(item.get("content"), str)
            and item["content"].strip()
        ):
            return item["content"].strip()
    raise ApiError(400, "user_message_required", "A non-empty user message is required.")


def openai_completion(response: dict[str, Any], request_id: str) -> dict[str, Any]:
    model = response.get("model") or {}
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(model.get("model", "auro-him")),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": str(response.get("answer", "")),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "estimated": False,
        },
        "auro": {
            "schema": response.get("schema"),
            "confidence": response.get("confidence"),
            "reasoning_summary": response.get("reasoning_summary", []),
            "agents": response.get("agents", []),
            "proposed_actions": response.get("proposed_actions", []),
            "executions": response.get("executions", []),
            "receipt": response.get("receipt"),
            "parameter_count_verified": model.get("parameter_count_verified", False),
        },
    }


def openai_council_completion(response: dict[str, Any], request_id: str) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "auro-2b-council",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": str(response.get("text", "")),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "estimated": False,
        },
        "auro": {
            "schema": response.get("schema"),
            "evidence_class": response.get("evidence_class"),
            "release_evidence_ready": response.get("release_evidence_ready", False),
            "blockers": response.get("blockers", []),
            "atomic_agent_count": response.get("atomic_agent_count", 0),
            "model_backed_atomic_count": response.get("model_backed_atomic_count", 0),
            "estimated_text_reduction": response.get("estimated_text_reduction", 0.0),
            "runtime_receipt": response.get("runtime_receipt"),
            "composition_is_not_one_checkpoint": True,
        },
    }


class Handler(BaseHTTPRequestHandler):
    runtime: Any = None
    council_service: Any = None
    server_version = "AuroHIM/2.2"

    @classmethod
    def get_runtime(cls):
        if cls.runtime is None:
            from .runtime import NovaRuntime

            cls.runtime = NovaRuntime()
        return cls.runtime

    @classmethod
    def get_council_service(cls):
        if cls.council_service is None:
            from .council_service import CouncilService

            cls.council_service = CouncilService.from_env()
        return cls.council_service

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except ApiError as exc:
            self._error(exc.status, exc.code, exc.message)
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, "invalid_request", str(exc)[:300])
        except Exception as exc:
            self._error(
                500,
                "internal_error",
                str(exc)[:300]
                if os.getenv("AURO_DEBUG") == "1"
                else "The request could not be completed.",
            )

    def do_GET(self):
        self.request_id = self._request_id()
        path = self._path()
        if path in ASSETS:
            content_type, data = ASSETS[path]
            return self._bytes(200, content_type, data)
        if path in {"/health", "/v1/health/live"}:
            return self._json(
                200,
                {
                    "ok": True,
                    "status": "live",
                    "service": "auro-him-api",
                    "api_version": API_VERSION,
                },
            )
        if path == "/openapi.json":
            return self._json(200, self._openapi())

        self._require_api_auth()
        if path == "/v1/council":
            return self._json(200, self.get_council_service().status())

        runtime = self.get_runtime()
        if path == "/v1/health/ready":
            security = production_security_status()
            receipt_chain = runtime.capabilities.ledger.verify()
            neuro = runtime.capabilities.brain.snapshot().get(
                "neuromorphic_persistence", {}
            )
            ready = bool(security["ready"] and receipt_chain.get("valid", False))
            payload = {
                "ok": ready,
                "status": "ready" if ready else "not_ready",
                "service": "auro-him-api",
                "security": security,
                "receipt_chain": receipt_chain,
                "neuromorphic_persistence": neuro,
                "model_fleet": runtime.model_orchestrator.manifest(),
                "council": self.get_council_service().status(),
            }
            return self._json(200 if ready else 503, payload)
        if path == "/v1":
            return self._json(200, self._discovery())
        if path == "/v1/models":
            return self._json(200, {"object": "list", "data": self._models()})
        if path.startswith("/v1/models/"):
            requested = path.removeprefix("/v1/models/")
            model = next(
                (
                    item
                    for item in self._models()
                    if requested in {item["id"], item.get("auro_endpoint_id")}
                ),
                None,
            )
            if model is None:
                raise ApiError(
                    404,
                    "model_not_found",
                    "The requested model is not configured.",
                )
            return self._json(200, model)
        if path == "/v1/capabilities":
            return self._json(200, runtime.capabilities.manifest())
        if path == "/v1/context":
            return self._json(200, runtime.context.stats())
        if path == "/v1/receipts/verify":
            return self._json(200, runtime.capabilities.ledger.verify())
        if path == "/v1/receipts":
            return self._json(
                200,
                {"receipts": runtime.capabilities.ledger.tail(20)},
            )
        if path == "/v1/browser/tasks":
            return self._json(
                200,
                {"tasks": runtime.capabilities.browser.list(50)},
            )
        if path.startswith("/v1/downloads/") and path.endswith(".zip"):
            artifact = runtime.capabilities.resolve_download(
                path.removeprefix("/v1/downloads/").removesuffix(".zip")
            )
            if artifact is None:
                raise ApiError(
                    404,
                    "artifact_not_found",
                    "The requested artifact is not registered.",
                )
            return self._bytes(200, "application/zip", artifact.read_bytes())
        raise ApiError(404, "not_found", "The requested route does not exist.")

    def do_POST(self):
        self.request_id = self._request_id()
        path = self._path()
        self._require_api_auth()
        body = self._body()

        if path == "/v1/council/respond":
            service = self.get_council_service()
            if not service.configured:
                raise ApiError(
                    503,
                    "council_not_configured",
                    "Auro-2B council endpoints are not configured.",
                )
            result = service.respond(
                self._message(body.get("message")),
                full_parent_context=self._optional_context(
                    body.get("parent_context")
                ),
            )
            result["request_id"] = self.request_id
            return self._json(200, result)

        runtime = self.get_runtime()
        if path == "/v1/capabilities/call":
            name = str(body.get("name", ""))
            arguments = dict(body.get("arguments") or {})
            approval_grant = None
            if runtime.capabilities.requires_approval(name):
                self._require_execution_auth()
                from .capabilities import capability_action
                from .organ_sdk import build_server_approval

                action = capability_action(name, arguments)
                approval_grant = build_server_approval(
                    [action],
                    subject=f"http-api:{self.request_id}",
                    approval_id="cap_" + uuid.uuid4().hex,
                )
            return self._json(
                200,
                runtime.capabilities.call(
                    name,
                    arguments,
                    approval_grant=approval_grant,
                ),
            )
        if path == "/v1/context/query":
            pack = runtime.context.retrieve(
                self._message(body.get("query")),
                token_budget=int(
                    body.get("token_budget") or runtime.context.default_budget
                ),
                top_k=int(body.get("top_k") or 24),
            )
            return self._json(200, pack.public())
        if path == "/v1/context/ingest":
            self._require_execution_auth()
            text = self._message(body.get("text"))
            return self._json(
                200,
                runtime.context.ingest(
                    text,
                    source=str(body.get("source") or "api"),
                    kind=str(body.get("kind") or "document"),
                    importance=float(body.get("importance", 0.5)),
                    metadata=dict(body.get("metadata") or {}),
                    chunk_tokens=int(body.get("chunk_tokens") or 900),
                    allow_sensitive=bool(body.get("allow_sensitive", False)),
                ),
            )
        if path == "/v1/browser/tasks/claim":
            self._require_execution_auth()
            return self._json(
                200,
                {
                    "task": runtime.capabilities.browser.claim(
                        str(body.get("worker_id") or "chrome")
                    )
                },
            )
        if path.startswith("/v1/browser/tasks/") and path.endswith("/complete"):
            self._require_execution_auth()
            task_id = path.split("/")[4]
            return self._json(
                200,
                runtime.capabilities.browser.complete(
                    task_id,
                    body.get("result"),
                    body.get("error"),
                ),
            )
        if path in {"/v1/respond", "/v1/him/respond"}:
            execute = bool(body.get("execute", False))
            if execute:
                self._require_execution_auth()
            result = runtime.respond(
                self._message(body.get("message")),
                execute=execute,
                approval_grant=body.get("approval_grant"),
            )
            result["request_id"] = self.request_id
            return self._json(200, result)
        if path == "/v1/chat/completions":
            if body.get("stream"):
                raise ApiError(
                    400,
                    "streaming_not_supported",
                    "Set stream=false.",
                )
            requested = str(body.get("model") or runtime.endpoint.model)
            if requested == "auro-2b-council":
                service = self.get_council_service()
                if not service.configured:
                    raise ApiError(
                        503,
                        "council_not_configured",
                        "Auro-2B council endpoints are not configured.",
                    )
                result = service.respond(
                    self._message(extract_user_message(body.get("messages"))),
                    full_parent_context=self._optional_context(
                        body.get("auro_parent_context")
                    ),
                )
                return self._json(
                    200,
                    openai_council_completion(result, self.request_id),
                )

            execute = bool(body.get("auro_execute", False))
            if execute:
                self._require_execution_auth()
            allowed = {
                value
                for model in self._models()
                for value in (model["id"], model.get("auro_endpoint_id"))
                if value
            } | {"auro-him"}
            if requested not in allowed:
                raise ApiError(
                    404,
                    "model_not_found",
                    "The requested model is not configured.",
                )
            result = runtime.respond(
                self._message(extract_user_message(body.get("messages"))),
                execute=execute,
                approval_grant=body.get("auro_approval_grant"),
            )
            return self._json(
                200,
                openai_completion(result, self.request_id),
            )
        raise ApiError(404, "not_found", "The requested route does not exist.")

    def do_OPTIONS(self):
        self.request_id = self._request_id()
        return self._bytes(204, "text/plain; charset=utf-8", b"")

    def _path(self) -> str:
        return urlsplit(self.path).path

    def _request_id(self) -> str:
        supplied = self.headers.get("x-request-id", "").strip()
        if (
            supplied
            and len(supplied) <= 128
            and supplied.replace("-", "").replace("_", "").isalnum()
        ):
            return supplied
        return "req_" + uuid.uuid4().hex

    def _require_api_auth(self):
        expected = os.getenv("AURO_API_TOKEN", "")
        if expected and not token_authorized(
            self.headers.get("authorization", ""),
            expected,
        ):
            raise ApiError(
                401,
                "api_token_required",
                "A valid API bearer token is required.",
            )

    def _require_execution_auth(self):
        expected = os.getenv("AURO_EXECUTION_TOKEN", "")
        header = self.headers.get("x-auro-execution-token", "")
        bearer = self.headers.get("authorization", "")
        if not expected or not (
            hmac.compare_digest(header, expected)
            or token_authorized(bearer, expected)
        ):
            raise ApiError(
                403,
                "operator_token_required",
                "A valid execution token is required.",
            )

    def _body(self) -> dict[str, Any]:
        content_type = (
            self.headers.get("content-type", "")
            .split(";", 1)[0]
            .strip()
            .lower()
        )
        if content_type != "application/json":
            raise ApiError(
                415,
                "json_required",
                "Content-Type must be application/json.",
            )
        try:
            length = int(self.headers.get("content-length", "0"))
        except ValueError as exc:
            raise ApiError(
                400,
                "invalid_content_length",
                "Content-Length must be an integer.",
            ) from exc
        if length <= 0:
            raise ApiError(
                400,
                "body_required",
                "A JSON request body is required.",
            )
        if length > MAX_REQUEST_BYTES:
            raise ApiError(
                413,
                "body_too_large",
                "Request body is too large.",
            )
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ApiError(
                400,
                "object_required",
                "The JSON body must be an object.",
            )
        return value

    def _message(self, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ApiError(
                400,
                "message_required",
                "A non-empty message is required.",
            )
        message = value.strip()
        if len(message) > MAX_MESSAGE_CHARS:
            raise ApiError(
                413,
                "message_too_large",
                f"Message exceeds {MAX_MESSAGE_CHARS} characters.",
            )
        return message

    def _optional_context(self, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ApiError(
                400,
                "parent_context_must_be_string",
                "Parent context must be a string.",
            )
        if len(value) > MAX_PARENT_CONTEXT_CHARS:
            raise ApiError(
                413,
                "parent_context_too_large",
                f"Parent context exceeds {MAX_PARENT_CONTEXT_CHARS} characters.",
            )
        return value

    def _models(self) -> list[dict[str, Any]]:
        runtime = self.get_runtime()
        models = [
            {
                "id": item["model"],
                "object": "model",
                "owned_by": (
                    "ItsNotAILABS"
                    if item["provider"] == "repository-native-open-weights"
                    else item["provider"]
                ),
                "auro_endpoint_id": item["id"],
                "role": item["role"],
                "provider": item["provider"],
                "capabilities": item["capabilities"],
                "local": item["local"],
                "parameter_count": item["parameter_count"],
                "parameter_count_verified": item["parameter_count_verified"],
                "identity_verified": item["identity_verified"],
                "agent_count_is_not_parameter_count": True,
            }
            for item in runtime.model_orchestrator.manifest()["models"]
        ]
        council = self.get_council_service()
        if council.configured:
            models.append(
                {
                    "id": "auro-2b-council",
                    "object": "model",
                    "owned_by": "ItsNotAILABS",
                    "auro_endpoint_id": "auro-2b-council",
                    "role": "composed-hierarchical-runtime",
                    "provider": "explicit-endpoint-council",
                    "capabilities": [
                        "atomic-swarm",
                        "three-500m-specialists",
                        "MESIE-every-stage",
                        "consensus",
                        "python-wasm-fluidizer",
                    ],
                    "local": False,
                    "parameter_count": None,
                    "parameter_count_verified": False,
                    "identity_verified": False,
                    "agent_count_is_not_parameter_count": True,
                    "not_a_single_checkpoint": True,
                }
            )
        return models

    def _discovery(self) -> dict[str, Any]:
        return {
            "service": "Auro14B / HIM",
            "api_version": API_VERSION,
            "native_response": "/v1/him/respond",
            "council_status": "/v1/council",
            "council_response": "/v1/council/respond",
            "openai_compatible": "/v1/chat/completions",
            "models": "/v1/models",
            "capabilities": "/v1/capabilities",
            "receipts": "/v1/receipts",
            "openapi": "/openapi.json",
        }

    def _openapi(self) -> dict[str, Any]:
        return {
            "openapi": "3.1.0",
            "info": {
                "title": "Auro14B / HIM API",
                "version": API_VERSION,
            },
            "paths": {
                "/v1/health/live": {"get": {}},
                "/v1/health/ready": {"get": {}},
                "/v1/chat/completions": {"post": {}},
                "/v1/council": {"get": {}},
                "/v1/council/respond": {"post": {}},
                "/v1/capabilities/call": {"post": {}},
            },
        }

    def _error(self, status: int, code: str, message: str):
        if (
            getattr(self, "wfile", None) is not None
            and not self.wfile.closed
        ):
            self._json(
                status,
                {
                    "error": {
                        "code": code,
                        "message": message,
                        "request_id": getattr(self, "request_id", None),
                    }
                },
            )

    def _json(self, status: int, payload: Any):
        return self._bytes(
            status,
            "application/json; charset=utf-8",
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )

    def _bytes(self, status: int, content_type: str, data: bytes):
        self.send_response(status)
        headers = {
            "content-type": content_type,
            "content-length": str(len(data)),
            "cache-control": "no-store",
            "x-request-id": getattr(self, "request_id", ""),
            "x-auro-api-version": API_VERSION,
            "x-content-type-options": "nosniff",
            "x-frame-options": "DENY",
            "referrer-policy": "no-referrer",
        }
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if data:
            self.wfile.write(data)

    def log_message(self, format, *args):
        return


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message
        super().__init__(message)


def main():
    parser = argparse.ArgumentParser(description="Serve Auro14B/HIM over HTTP")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    security = production_security_status(args.host)
    if production_mode() and not security["ready"]:
        missing = [
            name
            for name, ok in security["secret_checks"].items()
            if not ok
        ]
        detail = (
            ", ".join(missing)
            if missing
            else "production secrets must be distinct"
        )
        raise SystemExit(
            f"AURO production security configuration invalid: {detail}"
        )
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
