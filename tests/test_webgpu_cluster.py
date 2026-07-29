import threading
import time
from pathlib import Path

from auro_native_llm.webgpu_cluster.coordinator import Cluster, decode_f32, encode_f32


def multiply(a, b, m, k, n):
    return [sum(a[row * k + inner] * b[inner * n + col] for inner in range(k)) for row in range(m) for col in range(n)]


def test_cluster_dispatches_real_matrix_job_and_reassembles_result():
    cluster = Cluster(default_timeout=5, lease_seconds=2)
    receipts = []

    def worker():
        job = cluster.claim("fixture-browser-webgpu", wait_seconds=2, capabilities={"webgpu": True})
        assert job is not None
        a = decode_f32(job["a"]["base64"])
        b = decode_f32(job["b"]["base64"])
        m, k = job["a"]["shape"]
        _, n = job["b"]["shape"]
        result = multiply(a, b, m, k, n)
        receipts.append(cluster.complete(job["job_id"], "fixture-browser-webgpu", result_base64=encode_f32(result), shape=[m, n], elapsed_ms=1.25, backend="browser-webgpu"))

    thread = threading.Thread(target=worker)
    thread.start()
    result = cluster.matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])
    thread.join(timeout=5)
    assert result == [[58.0, 64.0], [139.0, 154.0]]
    assert receipts[0]["status"] == "completed"
    assert receipts[0]["backend"] == "browser-webgpu"
    assert len(receipts[0]["receipt_sha256"]) == 64
    assert cluster.status()["jobs"]["completed"] == 1


def test_expired_worker_lease_is_reissued():
    cluster = Cluster(lease_seconds=0.02)
    job = cluster.enqueue_encoded(encode_f32([2.0]), encode_f32([3.0]), 1, 1, 1)
    first = cluster.claim("worker-a", wait_seconds=0)
    assert first["job_id"] == job.job_id
    time.sleep(0.03)
    second = cluster.claim("worker-b", wait_seconds=0)
    assert second["job_id"] == job.job_id


def test_browser_node_contains_webgpu_shader_bounds_checks_and_receipt_backend():
    source = Path("browser-brain/training/node.js").read_text(encoding="utf-8")
    assert "navigator.gpu.requestAdapter" in source
    assert "@compute @workgroup_size(8,8,1)" in source
    assert "if(row<dims.m&&aCol<dims.k)" in source
    assert "backend:'browser-webgpu'" in source
    assert "await submit(job,null,error.message)" in source
