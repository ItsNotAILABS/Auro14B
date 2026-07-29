from __future__ import annotations

import json
import os
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from auro_native_llm.production_fleet.server import Handler, TRIAD_MODEL_ID


class DummyOrchestrator:
    def manifest(self):
        return {
            "models": [
                {
                    "model": "fixture-main",
                    "provider": "repository-native-open-weights",
                    "id": "fixture-main",
                    "role": "general",
                    "capabilities": ["general"],
                    "local": True,
                    "parameter_count": 10,
                    "parameter_count_verified": True,
                    "identity_verified": True,
                }
            ]
        }


class DummyRuntime:
    endpoint = type("Endpoint", (), {"model": "fixture-main"})()
    model_orchestrator = DummyOrchestrator()

    def triad_status(self):
        return {"enabled": True, "configured": True, "error": None}

    def triad_respond(self, message, *, context=""):
        return {
            "schema": "auro.2b_triad_swarm.turn.v1",
            "text": f"triad:{message}",
            "structured_answer": {"confidence": 0.9, "reasoning_summary": ["fixture"]},
            "promotion_ready": False,
            "blockers": ["fixture"],
            "runtime_receipt_sha256": "a" * 64,
        }


def request_json(url, body=None, token="test-token"):
    headers = {"authorization": f"Bearer {token}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    request = Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    with urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode())


def test_triad_native_and_openai_routes_are_authenticated_and_single_pass(monkeypatch):
    monkeypatch.setenv("AURO_API_TOKEN", "test-token")
    original = Handler.runtime
    Handler.runtime = DummyRuntime()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        try:
            request_json(base + "/v1/triad", token="wrong")
        except HTTPError as exc:
            assert exc.code == 401
        else:
            raise AssertionError("unauthorized triad status was accepted")

        status, payload = request_json(base + "/v1/triad/respond", {"message": "hello"})
        assert status == 200
        assert payload["text"] == "triad:hello"
        assert payload["request_id"].startswith("req_")

        status, completion = request_json(
            base + "/v1/chat/completions",
            {"model": TRIAD_MODEL_ID, "messages": [{"role": "user", "content": "hello"}], "stream": False},
        )
        assert status == 200
        assert completion["model"] == TRIAD_MODEL_ID
        assert completion["choices"][0]["message"]["content"] == "triad:hello"
        assert completion["auro"]["agent_count_is_not_parameter_count"] is True
    finally:
        server.shutdown()
        server.server_close()
        Handler.runtime = original
        thread.join(timeout=5)
