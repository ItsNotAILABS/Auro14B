"""MESIE Web Application — WSGI-compatible HTTP handler.

Zero external dependencies: uses only the Python standard library
(http.server / json / urllib.parse). Bridges HTTP requests directly
to the MESIE internal engine bus so every engine is reachable from
raw web calls.
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from mesie.internal_api.router import InternalRouter
from mesie.internal_api.messages import EngineResponse


# =============================================================================
# CORS and response helpers
# =============================================================================

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-MESIE-Key",
    "Access-Control-Max-Age": "86400",
}


def _json_response(data: Any, status: int = 200) -> Tuple[int, Dict[str, str], bytes]:
    """Return a (status, headers, body) triple."""
    body = json.dumps(data, default=str).encode("utf-8")
    headers = {"Content-Type": "application/json", **CORS_HEADERS}
    return status, headers, body


def _error(message: str, status: int = 400) -> Tuple[int, Dict[str, str], bytes]:
    return _json_response({"error": message}, status)


# =============================================================================
# Route matching
# =============================================================================

Route = Tuple[str, str, Callable[..., Tuple[int, Dict[str, str], bytes]]]


class MESIEWebApp:
    """HTTP application exposing MESIE engines to the raw web.

    Can be used standalone via the built-in server or integrated with
    any WSGI-compatible server (gunicorn, uvicorn WSGI adapter, etc.).

    Attributes:
        router: Internal engine router.
        api_key: Optional API key for authentication (None = open).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        router: Optional[InternalRouter] = None,
    ) -> None:
        """Initialize the web app.

        Args:
            api_key: If set, requests must include Authorization or X-MESIE-Key header.
            router: Custom internal router. Creates default if None.
        """
        self.api_key = api_key
        self.router = router or InternalRouter()
        self._routes: List[Route] = self._build_routes()

    def _build_routes(self) -> List[Route]:
        """Register URL routes."""
        return [
            ("GET", "/", self._handle_root),
            ("GET", "/health", self._handle_health),
            ("GET", "/v1/engines", self._handle_list_engines),
            ("POST", "/v1/engine", self._handle_engine_call),
            ("POST", "/v1/match", self._handle_match),
            ("POST", "/v1/validate", self._handle_validate),
            ("POST", "/v1/embed", self._handle_embed),
            ("POST", "/v1/intelligence/reason", self._handle_reason),
            ("POST", "/v1/tokenomics/score", self._handle_tokenomics),
        ]

    # =========================================================================
    # Authentication
    # =========================================================================

    def _authenticate(self, headers: Dict[str, str]) -> bool:
        """Check authentication if api_key is configured."""
        if not self.api_key:
            return True
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
        else:
            token = headers.get("x-mesie-key", "")
        return token == self.api_key

    # =========================================================================
    # Request dispatching
    # =========================================================================

    def handle_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: bytes = b"",
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Dispatch an HTTP request and return (status, headers, body).

        Args:
            method: HTTP method (GET, POST, etc.).
            path: URL path.
            headers: Request headers (lowercase keys).
            body: Request body bytes.

        Returns:
            Tuple of (status_code, response_headers, response_body).
        """
        method = method.upper()

        # CORS preflight
        if method == "OPTIONS":
            return 204, CORS_HEADERS, b""

        # Strip query string and trailing slash
        path = path.split("?")[0]
        path = path.rstrip("/") or "/"

        # Authentication
        if not self._authenticate(headers):
            return _error("Unauthorized. Provide Authorization: ****** or X-MESIE-Key header.", 401)

        # Route matching
        # Handle dynamic engine route: /v1/engine/{name}/{action}
        if path.startswith("/v1/engine/") and method == "POST":
            parts = path.split("/")
            if len(parts) >= 5:
                engine_name = parts[3]
                action = "/".join(parts[4:])
                return self._handle_engine_call(headers, body, engine_name, action)

        for route_method, route_path, handler in self._routes:
            if method == route_method and path == route_path:
                return handler(headers, body)

        return _error(f"Not found: {method} {path}", 404)

    # =========================================================================
    # WSGI interface
    # =========================================================================

    def __call__(self, environ: Dict[str, Any], start_response: Callable) -> List[bytes]:
        """WSGI entry point for production servers."""
        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "/")
        content_length = int(environ.get("CONTENT_LENGTH") or 0)
        body = environ["wsgi.input"].read(content_length) if content_length > 0 else b""

        # Collect headers
        headers: Dict[str, str] = {}
        for key, value in environ.items():
            if key.startswith("HTTP_"):
                header_name = key[5:].replace("_", "-").lower()
                headers[header_name] = value
        if "CONTENT_TYPE" in environ:
            headers["content-type"] = environ["CONTENT_TYPE"]

        status, resp_headers, resp_body = self.handle_request(method, path, headers, body)

        status_line = f"{status} {'OK' if status < 400 else 'Error'}"
        response_headers = [(k, v) for k, v in resp_headers.items()]
        start_response(status_line, response_headers)
        return [resp_body]

    # =========================================================================
    # Route handlers
    # =========================================================================

    def _handle_root(self, headers: Dict[str, str], body: bytes) -> Tuple[int, Dict[str, str], bytes]:
        return _json_response({
            "service": "mesie-web",
            "version": "0.4.0",
            "status": "online",
            "engine": "python-native",
            "description": "Multi-Element Spectral Intelligence Engine — raw web deployment",
            "endpoints": [
                "GET  /health",
                "GET  /v1/engines",
                "POST /v1/engine/{name}/{action}",
                "POST /v1/match",
                "POST /v1/validate",
                "POST /v1/embed",
                "POST /v1/intelligence/reason",
                "POST /v1/tokenomics/score",
            ],
        })

    def _handle_health(self, headers: Dict[str, str], body: bytes) -> Tuple[int, Dict[str, str], bytes]:
        engines = self.router.registry.names()
        return _json_response({
            "status": "healthy",
            "engines_loaded": len(engines),
            "engine_names": engines,
        })

    def _handle_list_engines(self, headers: Dict[str, str], body: bytes) -> Tuple[int, Dict[str, str], bytes]:
        engines_info = []
        for eng in self.router.registry.all():
            engines_info.append({
                "name": eng.name,
                "capabilities": eng.capabilities,
            })
        return _json_response({"engines": engines_info})

    def _handle_engine_call(
        self,
        headers: Dict[str, str],
        body: bytes,
        engine_name: Optional[str] = None,
        action: Optional[str] = None,
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Generic engine call: POST /v1/engine/{name}/{action}."""
        payload = self._parse_body(body)
        if isinstance(payload, tuple):
            return payload  # error response

        if engine_name is None or action is None:
            return _error("Use /v1/engine/{engine_name}/{action}", 400)

        engine = self.router.registry.get(engine_name)
        if engine is None:
            return _error(f"Engine '{engine_name}' not found. Use GET /v1/engines to list.", 404)
        if not engine.supports(action):
            return _error(
                f"Engine '{engine_name}' does not support action '{action}'. "
                f"Capabilities: {engine.capabilities}",
                400,
            )

        try:
            response = self.router.call(engine_name, action, payload)
            return _json_response(response.to_dict(), 200 if response.ok else 422)
        except Exception as exc:
            return _error(f"Engine error: {exc}", 500)

    def _handle_match(self, headers: Dict[str, str], body: bytes) -> Tuple[int, Dict[str, str], bytes]:
        """Shortcut: POST /v1/match — spectral matching."""
        payload = self._parse_body(body)
        if isinstance(payload, tuple):
            return payload
        try:
            response = self.router.call("matching", "match", payload)
            return _json_response(response.to_dict(), 200 if response.ok else 422)
        except Exception as exc:
            return _error(f"Match error: {exc}", 500)

    def _handle_validate(self, headers: Dict[str, str], body: bytes) -> Tuple[int, Dict[str, str], bytes]:
        """Shortcut: POST /v1/validate — record validation."""
        payload = self._parse_body(body)
        if isinstance(payload, tuple):
            return payload
        try:
            response = self.router.call("validation", "validate", payload)
            return _json_response(response.to_dict(), 200 if response.ok else 422)
        except Exception as exc:
            return _error(f"Validation error: {exc}", 500)

    def _handle_embed(self, headers: Dict[str, str], body: bytes) -> Tuple[int, Dict[str, str], bytes]:
        """Shortcut: POST /v1/embed — generate embeddings."""
        payload = self._parse_body(body)
        if isinstance(payload, tuple):
            return payload
        try:
            response = self.router.call("embedding", "embed", payload)
            return _json_response(response.to_dict(), 200 if response.ok else 422)
        except Exception as exc:
            return _error(f"Embed error: {exc}", 500)

    def _handle_reason(self, headers: Dict[str, str], body: bytes) -> Tuple[int, Dict[str, str], bytes]:
        """Shortcut: POST /v1/intelligence/reason — reasoning over a record."""
        payload = self._parse_body(body)
        if isinstance(payload, tuple):
            return payload
        try:
            response = self.router.call("intelligence", "reason", payload)
            return _json_response(response.to_dict(), 200 if response.ok else 422)
        except Exception as exc:
            return _error(f"Reason error: {exc}", 500)

    def _handle_tokenomics(self, headers: Dict[str, str], body: bytes) -> Tuple[int, Dict[str, str], bytes]:
        """POST /v1/tokenomics/score — cognitive return scoring."""
        payload = self._parse_body(body)
        if isinstance(payload, tuple):
            return payload

        try:
            from mesie.cognitive.tokenomics import (
                CognitiveReturnMetrics,
                CognitiveReturnScores,
                TokenValueFunction,
                TokenScores,
                TokenValueWeights,
            )

            scores = CognitiveReturnScores(
                decision_quality=float(payload.get("decision_quality", 0)),
                actionability=float(payload.get("actionability", 0)),
                risk_control=float(payload.get("risk_control", 0)),
                reuse_value=float(payload.get("reuse_value", 0)),
                learning_gain=float(payload.get("learning_gain", 0)),
            )
            prompt_tokens = int(payload.get("prompt_tokens", 0))
            output_tokens = int(payload.get("output_tokens", 0))

            metrics = CognitiveReturnMetrics()
            cr = metrics.cognitive_return(scores)
            crpt = metrics.crpt(scores, prompt_tokens, output_tokens)

            # Token value if provided
            tv_result = None
            if "token_scores" in payload:
                tvf = TokenValueFunction()
                ts = payload["token_scores"]
                tv_scores = [
                    TokenScores(
                        decision_value=float(t.get("decision_value", 0)),
                        action_usefulness=float(t.get("action_usefulness", 0)),
                        risk_reduction=float(t.get("risk_reduction", 0)),
                        compression_contribution=float(t.get("compression_contribution", 0)),
                        memory_value=float(t.get("memory_value", 0)),
                        noise=float(t.get("noise", 0)),
                    )
                    for t in ts
                ]
                tv_result = {
                    "total_value": tvf.total_value(tv_scores),
                    "mean_value": tvf.mean_value(tv_scores),
                }

            result = {
                "cognitive_return": cr,
                "crpt": crpt,
                "scores": {
                    "decision_quality": scores.decision_quality,
                    "actionability": scores.actionability,
                    "risk_control": scores.risk_control,
                    "reuse_value": scores.reuse_value,
                    "learning_gain": scores.learning_gain,
                },
                "total_tokens": prompt_tokens + output_tokens,
            }
            if tv_result:
                result["token_value"] = tv_result

            return _json_response(result)
        except Exception as exc:
            return _error(f"Tokenomics error: {exc}", 500)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _parse_body(self, body: bytes) -> Any:
        """Parse JSON body or return error response."""
        if not body:
            return {}
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError) as exc:
            return _error(f"Invalid JSON body: {exc}", 400)
