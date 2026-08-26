import json
import os
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from auro_native_llm.production_fleet.server import Handler
from auro_native_llm.production_fleet.task_runtime import (
    DurableTaskRuntime,
    TaskRuntimeService,
)


def request(base, path, *, method="GET", body=None, execution=False, principal="user-1"):
    headers = {
        "authorization": "Bearer " + "a" * 32,
        "x-auro-principal-id": principal,
        "x-auro-organization-id": "org-1",
    }
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    if execution:
        headers["x-auro-execution-token"] = "b" * 32
    req = Request(base + path, data=data, headers=headers, method=method)
    with urlopen(req, timeout=10) as response:
        payload = response.read()
        return response.status, response.headers.get("content-type"), payload


def test_authenticated_task_api_lifecycle_delivers_bundle(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AURO_API_TOKEN", "a" * 32)
    monkeypatch.setenv("AURO_EXECUTION_TOKEN", "b" * 32)
    monkeypatch.setenv("AURO_APPROVAL_HMAC_KEY", "c" * 32)
    monkeypatch.setenv("AURO_TASK_RECEIPT_HMAC_KEY", "d" * 32)
    monkeypatch.setenv("AURO_TASK_RUNTIME_ENABLED", "1")

    runtime = DurableTaskRuntime(
        tmp_path / "tasks.sqlite3",
        tmp_path / "artifacts",
        signing_key="d" * 32,
    )
    Handler.task_service = TaskRuntimeService(runtime)
    Handler.runtime = None
    Handler.council_service = None

    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, _, raw = request(
            base,
            "/v1/task-runs",
            method="POST",
            body={
                "run_id": "api-run",
                "objective": "Produce one verified artifact",
                "quality_mode": "fast",
                "tasks": [
                    {
                        "step_id": "write",
                        "objective": "Write the deliverable",
                        "kind": "write",
                        "required_capabilities": ["write"],
                        "artifacts": [
                            {"name": "DELIVERABLE.md", "media_type": "text/markdown"}
                        ],
                    }
                ],
            },
        )
        assert status == 201
        created = json.loads(raw)
        assert created["run_id"] == "api-run"

        status, _, raw = request(
            base,
            "/v1/task-runs/api-run/claim",
            method="POST",
            execution=True,
            body={
                "worker_id": "worker-1",
                "capabilities": ["write"],
                "lease_seconds": 300,
            },
        )
        claimed = json.loads(raw)["step"]
        assert claimed["step_id"] == "write"
        assert claimed["lease_token"]

        request(
            base,
            "/v1/task-runs/api-run/steps/write/progress",
            method="POST",
            execution=True,
            body={
                "worker_id": "worker-1",
                "lease_token": claimed["lease_token"],
                "progress": {"percent": 50, "phase": "writing"},
            },
        )

        status, _, raw = request(
            base,
            "/v1/task-runs/api-run/steps/write/complete",
            method="POST",
            execution=True,
            body={
                "worker_id": "worker-1",
                "lease_token": claimed["lease_token"],
                "output": {"summary": "deliverable complete"},
                "artifacts": [
                    {
                        "name": "DELIVERABLE.md",
                        "media_type": "text/markdown",
                        "content": "# Deliverable\n\nComplete.\n",
                    }
                ],
                "validation": {"passed": True},
            },
        )
        completed_step = json.loads(raw)
        assert completed_step["status"] == "succeeded"

        status, _, raw = request(base, "/v1/task-runs/api-run")
        completed = json.loads(raw)
        assert completed["status"] == "succeeded"
        assert completed["result"]["evidence_class"] == "E4-signed-receipt"
        assert completed["artifacts"][0]["name"] == "DELIVERABLE.md"

        status, content_type, bundle = request(
            base,
            "/v1/task-runs/api-run/bundle.zip",
            execution=True,
        )
        assert status == 200
        assert content_type == "application/zip"
        assert bundle.startswith(b"PK")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        Handler.task_service = None
        Handler.runtime = None
        Handler.council_service = None


def test_task_routes_require_principal_and_execution_authority(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AURO_API_TOKEN", "a" * 32)
    monkeypatch.setenv("AURO_EXECUTION_TOKEN", "b" * 32)
    monkeypatch.setenv("AURO_TASK_RUNTIME_ENABLED", "1")
    Handler.task_service = TaskRuntimeService(
        DurableTaskRuntime(tmp_path / "tasks.sqlite3", tmp_path / "artifacts")
    )

    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        req = Request(
            base + "/v1/task-runs",
            headers={"authorization": "Bearer " + "a" * 32},
        )
        try:
            urlopen(req, timeout=10)
            raise AssertionError("principal-less task request should fail")
        except HTTPError as exc:
            assert exc.code == 400

        request(
            base,
            "/v1/task-runs",
            method="POST",
            body={
                "run_id": "auth-run",
                "objective": "Auth test",
                "quality_mode": "fast",
                "tasks": [{"step_id": "a", "objective": "A"}],
            },
        )
        try:
            request(
                base,
                "/v1/task-runs/auth-run/claim",
                method="POST",
                body={"worker_id": "worker", "capabilities": []},
            )
            raise AssertionError("claim without execution authority should fail")
        except HTTPError as exc:
            assert exc.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        Handler.task_service = None
