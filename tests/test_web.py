"""Tests for MESIE Web — raw web deployment of intelligence engine."""

import json
import pytest

from mesie.web.app import MESIEWebApp


@pytest.fixture
def app():
    """Create a test app instance."""
    return MESIEWebApp()


@pytest.fixture
def auth_app():
    """Create a test app instance with API key."""
    return MESIEWebApp(api_key="test-secret-key")


# =============================================================================
# Basic endpoint tests
# =============================================================================


class TestRootAndHealth:
    """Test root and health endpoints."""

    def test_root_returns_service_info(self, app):
        status, headers, body = app.handle_request("GET", "/", {})
        assert status == 200
        data = json.loads(body)
        assert data["service"] == "mesie-web"
        assert data["status"] == "online"
        assert "endpoints" in data

    def test_health_check(self, app):
        status, headers, body = app.handle_request("GET", "/health", {})
        assert status == 200
        data = json.loads(body)
        assert data["status"] == "healthy"
        assert data["engines_loaded"] > 0
        assert isinstance(data["engine_names"], list)

    def test_list_engines(self, app):
        status, headers, body = app.handle_request("GET", "/v1/engines", {})
        assert status == 200
        data = json.loads(body)
        assert "engines" in data
        assert len(data["engines"]) > 0
        # Each engine has name and capabilities
        engine = data["engines"][0]
        assert "name" in engine
        assert "capabilities" in engine


# =============================================================================
# Authentication tests
# =============================================================================


class TestAuthentication:
    """Test API key authentication."""

    def test_open_app_no_auth_needed(self, app):
        status, _, _ = app.handle_request("GET", "/health", {})
        assert status == 200

    def test_auth_required_without_key(self, auth_app):
        status, _, body = auth_app.handle_request("GET", "/health", {})
        assert status == 401
        data = json.loads(body)
        assert "Unauthorized" in data["error"]

    def test_auth_with_bearer_token(self, auth_app):
        headers = {"authorization": "Bearer " + "test-secret-key"}
        status, _, _ = auth_app.handle_request("GET", "/health", headers)
        assert status == 200

    def test_auth_with_x_mesie_key(self, auth_app):
        headers = {"x-mesie-key": "test-secret-key"}
        status, _, _ = auth_app.handle_request("GET", "/health", headers)
        assert status == 200

    def test_auth_wrong_key(self, auth_app):
        headers = {"authorization": "Bearer " + "wrong-key"}
        status, _, _ = auth_app.handle_request("GET", "/health", headers)
        assert status == 401


# =============================================================================
# CORS tests
# =============================================================================


class TestCORS:
    """Test CORS headers."""

    def test_options_preflight(self, app):
        status, headers, body = app.handle_request("OPTIONS", "/v1/match", {})
        assert status == 204
        assert headers["Access-Control-Allow-Origin"] == "*"
        assert "POST" in headers["Access-Control-Allow-Methods"]

    def test_cors_headers_on_response(self, app):
        status, headers, _ = app.handle_request("GET", "/", {})
        assert headers["Access-Control-Allow-Origin"] == "*"


# =============================================================================
# Engine call tests
# =============================================================================


class TestEngineCalls:
    """Test dynamic engine invocation."""

    def test_engine_not_found(self, app):
        body = json.dumps({"record": {}}).encode()
        status, _, resp = app.handle_request(
            "POST", "/v1/engine/nonexistent/action", {}, body
        )
        assert status == 404
        data = json.loads(resp)
        assert "not found" in data["error"].lower()

    def test_unsupported_action(self, app):
        body = json.dumps({}).encode()
        status, _, resp = app.handle_request(
            "POST", "/v1/engine/validation/nonexistent_action", {}, body
        )
        assert status == 400
        data = json.loads(resp)
        assert "does not support" in data["error"]

    def test_invalid_json_body(self, app):
        status, _, resp = app.handle_request(
            "POST", "/v1/match", {}, b"not json{{"
        )
        assert status == 400
        data = json.loads(resp)
        assert "Invalid JSON" in data["error"]


# =============================================================================
# Tokenomics endpoint tests
# =============================================================================


class TestTokenomicsEndpoint:
    """Test the tokenomics scoring endpoint."""

    def test_basic_scoring(self, app):
        payload = {
            "decision_quality": 4.0,
            "actionability": 3.0,
            "risk_control": 2.0,
            "reuse_value": 3.0,
            "learning_gain": 1.0,
            "prompt_tokens": 100,
            "output_tokens": 200,
        }
        body = json.dumps(payload).encode()
        status, _, resp = app.handle_request("POST", "/v1/tokenomics/score", {}, body)
        assert status == 200
        data = json.loads(resp)
        assert data["cognitive_return"] == 13.0
        assert data["crpt"] == pytest.approx(13.0 / 300)
        assert data["total_tokens"] == 300

    def test_with_token_scores(self, app):
        payload = {
            "decision_quality": 5.0,
            "actionability": 5.0,
            "risk_control": 5.0,
            "reuse_value": 5.0,
            "learning_gain": 5.0,
            "prompt_tokens": 50,
            "output_tokens": 50,
            "token_scores": [
                {"decision_value": 3.0, "noise": 0.0},
                {"decision_value": 1.0, "noise": 5.0},
            ],
        }
        body = json.dumps(payload).encode()
        status, _, resp = app.handle_request("POST", "/v1/tokenomics/score", {}, body)
        assert status == 200
        data = json.loads(resp)
        assert data["cognitive_return"] == 25.0
        assert "token_value" in data
        assert data["token_value"]["total_value"] == pytest.approx(-1.0)

    def test_empty_body(self, app):
        status, _, resp = app.handle_request("POST", "/v1/tokenomics/score", {}, b"")
        assert status == 200
        data = json.loads(resp)
        assert data["cognitive_return"] == 0.0


# =============================================================================
# 404 and error handling
# =============================================================================


class TestErrorHandling:
    """Test error responses."""

    def test_404_unknown_path(self, app):
        status, _, resp = app.handle_request("GET", "/unknown/path", {})
        assert status == 404
        data = json.loads(resp)
        assert "Not found" in data["error"]

    def test_trailing_slash_stripped(self, app):
        status, _, body = app.handle_request("GET", "/health/", {})
        assert status == 200


# =============================================================================
# WSGI interface test
# =============================================================================


class TestWSGI:
    """Test WSGI compatibility."""

    def test_wsgi_call(self, app):
        from io import BytesIO

        environ = {
            "REQUEST_METHOD": "GET",
            "PATH_INFO": "/health",
            "CONTENT_LENGTH": "0",
            "wsgi.input": BytesIO(b""),
        }
        responses = []

        def start_response(status, headers):
            responses.append((status, headers))

        body_parts = app(environ, start_response)
        assert len(responses) == 1
        assert "200" in responses[0][0]
        data = json.loads(b"".join(body_parts))
        assert data["status"] == "healthy"
