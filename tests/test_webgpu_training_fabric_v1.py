import threading
import time
from pathlib import Path

import numpy as np
import pytest

from auro_native_llm.webgpu_cluster.coordinator import (
    Cluster,
    decode_f32,
    encode_f32,
    serve,
)


def _multiply(a, b, m, k, n):
    return [
        sum(a[row * k + inner] * b[inner * n + col] for inner in range(k))
        for row in range(m)
        for col in range(n)
    ]


def _claim_and_complete(cluster, worker_id="fixture-browser-webgpu"):
    job = cluster.claim(
        worker_id,
        wait_seconds=3,
        capabilities={"webgpu": True, "shaderSha256": "a" * 64},
        user_agent="test-browser",
        origin="http://127.0.0.1:9000",
    )
    assert job is not None
    a = decode_f32(job["a"]["base64"])
    b = decode_f32(job["b"]["base64"])
    m, k = job["a"]["shape"]
    _, n = job["b"]["shape"]
    values = _multiply(a, b, m, k, n)
    receipt = cluster.complete(
        job["job_id"],
        worker_id,
        job["lease_token"],
        result_base64=encode_f32(values),
        shape=[m, n],
        elapsed_ms=1.25,
        backend="browser-webgpu",
        worker_evidence={"secureContext": True},
    )
    return job, receipt


def test_cluster_dispatches_matrix_job_and_emits_signed_receipt():
    cluster = Cluster(
        default_timeout=5,
        receipt_signing_key="k" * 32,
        signer_id="test-coordinator",
    )
    receipts = []

    def worker():
        receipts.append(_claim_and_complete(cluster)[1])

    thread = threading.Thread(target=worker)
    thread.start()
    result = cluster.matmul(
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8], [9, 10], [11, 12]],
    )
    thread.join(timeout=5)

    assert result == [[58.0, 64.0], [139.0, 154.0]]
    assert receipts[0]["status"] == "completed"
    assert receipts[0]["backend_reported"] == "browser-webgpu"
    assert receipts[0]["signature"]
    assert receipts[0]["custody"] == "local-signed"
    assert cluster.get_receipt(receipts[0]["receipt_sha256"])["job_id"]
    assert cluster.status()["jobs"]["completed"] == 1
    cluster.close()


def test_lease_token_is_one_time_and_bound_to_worker():
    cluster = Cluster()
    job_id = cluster.enqueue_encoded(encode_f32([2.0]), encode_f32([3.0]), 1, 1, 1)
    job = cluster.claim("worker-a", wait_seconds=0, capabilities={"webgpu": True})
    assert job["job_id"] == job_id

    with pytest.raises(ValueError, match="another worker"):
        cluster.complete(
            job_id,
            "worker-b",
            job["lease_token"],
            result_base64=encode_f32([6.0]),
            shape=[1, 1],
        )

    cluster.complete(
        job_id,
        "worker-a",
        job["lease_token"],
        result_base64=encode_f32([6.0]),
        shape=[1, 1],
    )
    with pytest.raises(ValueError, match="not leased"):
        cluster.complete(
            job_id,
            "worker-a",
            job["lease_token"],
            result_base64=encode_f32([6.0]),
            shape=[1, 1],
        )
    cluster.close()


def test_expired_lease_is_reissued_with_new_token():
    cluster = Cluster(max_attempts=2)
    cluster.lease_seconds = 0.02
    job_id = cluster.enqueue_encoded(encode_f32([2.0]), encode_f32([3.0]), 1, 1, 1)
    first = cluster.claim("worker-a", wait_seconds=0, capabilities={"webgpu": True})
    assert first["job_id"] == job_id
    time.sleep(0.03)
    second = cluster.claim("worker-b", wait_seconds=0, capabilities={"webgpu": True})
    assert second["job_id"] == job_id
    assert second["lease_token"] != first["lease_token"]
    assert second["attempt"] == 2
    with pytest.raises(ValueError):
        cluster.complete(
            job_id,
            "worker-a",
            first["lease_token"],
            result_base64=encode_f32([6.0]),
            shape=[1, 1],
        )
    cluster.close()


def test_sqlite_wal_preserves_queue_result_and_receipt_across_restart(tmp_path):
    database = tmp_path / "cluster.sqlite"
    first = Cluster(database, receipt_signing_key="z" * 32)
    job_id = first.enqueue_encoded(
        encode_f32([1.0, 2.0]),
        encode_f32([3.0, 4.0]),
        1,
        2,
        1,
    )
    first.close()

    second = Cluster(database, receipt_signing_key="z" * 32)
    job, receipt = _claim_and_complete(second, "persistent-browser")
    assert job["job_id"] == job_id
    receipt_sha = receipt["receipt_sha256"]
    second.close()

    third = Cluster(database, receipt_signing_key="z" * 32)
    status = third.get_job(job_id, include_result=True)
    assert status["status"] == "completed"
    assert decode_f32(status["result_base64"]) == [11.0]
    assert third.get_receipt(receipt_sha)["signature"]
    assert third.status()["persistent"] is True
    third.close()


def test_http_client_and_training_plane_use_browser_cluster(monkeypatch):
    cluster = Cluster(default_timeout=5)
    server = serve(
        "127.0.0.1",
        0,
        token="cluster-secret-which-is-long-enough",
        cluster=cluster,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    stop = threading.Event()

    def worker():
        while not stop.is_set():
            try:
                _claim_and_complete(cluster, "training-browser-node")
            except AssertionError:
                continue
            except Exception:
                if not stop.is_set():
                    raise

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    deadline = time.time() + 2
    while cluster.status()["ready_workers"] == 0 and time.time() < deadline:
        time.sleep(0.01)

    monkeypatch.setenv(
        "AURO_WEBGPU_CLUSTER_URL",
        f"http://127.0.0.1:{server.server_port}",
    )
    monkeypatch.setenv(
        "AURO_WEBGPU_CLUSTER_TOKEN",
        "cluster-secret-which-is-long-enough",
    )
    monkeypatch.setenv("AURO_WEBGPU_CLUSTER_REQUIRE_WORKER", "1")
    try:
        from auro_native_llm.polyglot.cuda_plane import get_cuda_plane

        plane = get_cuda_plane(refresh=True)
        assert plane.backend == "webgpu_cluster"
        output = plane.matmul(
            np.array([[1, 2]], dtype=np.float32),
            np.array([[3], [4]], dtype=np.float32),
        )
        np.testing.assert_allclose(output, np.array([[11.0]]), rtol=0, atol=1e-6)

        step = plane.train_step_linear(
            np.array([[0.5, -0.25]], dtype=np.float32),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[1.0, 2.0]], dtype=np.float32),
            lr=0.01,
        )
        assert step["ok"] is True
        assert step["backend"] == "webgpu_cluster"
        assert np.isfinite(step["loss"])
        assert cluster.status()["jobs"]["completed"] >= 3
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        worker_thread.join(timeout=1)
        cluster.close()


def test_non_loopback_binding_requires_strong_token():
    with pytest.raises(ValueError, match="32\+ character token"):
        serve("0.0.0.0", 0, token="short")


def test_browser_node_uses_webgpu_lease_renewal_and_avoids_query_token():
    source = Path("browser-brain/training/node.js").read_text(encoding="utf-8")
    assert "navigator.gpu.requestAdapter" in source
    assert "@compute @workgroup_size(8,8,1)" in source
    assert "if(row<dims.m && aCol<dims.k)" in source
    assert 'backend: "browser-webgpu"' in source
    assert 'request("/lease/renew"' in source
    assert "lease_token: job.lease_token" in source
    assert 'fragment.get("token")' in source
    assert 'query.get("token")' not in source
    assert "history.replaceState" in source


def test_cluster_status_keeps_training_claim_boundary_explicit():
    cluster = Cluster()
    status = cluster.status()
    assert "matrix-compute substrate" in status["training_backend_claim"]
    assert status["ready_workers"] == 0
    cluster.close()
