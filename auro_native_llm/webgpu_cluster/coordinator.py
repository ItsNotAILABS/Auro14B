"""Dependency-free local WebGPU training coordinator.

Browser tabs poll for bounded float32 matrix jobs, execute them on WebGPU, and
return results. The coordinator binds to loopback by default and can require a
shared token. It is a compute transport; it does not claim browser hardware was
used unless a completed job receipt names a WebGPU worker.
"""
from __future__ import annotations

from array import array
import base64
from collections import deque
from dataclasses import asdict, dataclass, field
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import secrets
import sys
import threading
import time
from typing import Any, Sequence
from urllib.parse import parse_qs, urlsplit


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def encode_f32(values: Sequence[float]) -> str:
    data = array("f", (float(value) for value in values))
    if data.itemsize != 4:
        raise RuntimeError("platform float array is not 32-bit")
    if sys.byteorder != "little":
        data.byteswap()
    return base64.b64encode(data.tobytes()).decode("ascii")


def decode_f32(value: str) -> list[float]:
    raw = base64.b64decode(value.encode("ascii"), validate=True)
    if len(raw) % 4:
        raise ValueError("float32 payload byte length must be divisible by four")
    data = array("f")
    data.frombytes(raw)
    if sys.byteorder != "little":
        data.byteswap()
    return [float(item) for item in data]


def flatten_matrix(value: Sequence[Sequence[float]] | Sequence[float], rows: int | None = None, cols: int | None = None) -> tuple[list[float], int, int]:
    seq = list(value)
    if not seq:
        raise ValueError("matrix cannot be empty")
    if isinstance(seq[0], (list, tuple, array)):
        matrix = [list(row) for row in seq]  # type: ignore[arg-type]
        width = len(matrix[0])
        if width <= 0 or any(len(row) != width for row in matrix):
            raise ValueError("matrix rows must have equal nonzero width")
        return [float(item) for row in matrix for item in row], len(matrix), width
    if rows is None or cols is None or int(rows) * int(cols) != len(seq):
        raise ValueError("flat matrices require matching rows and cols")
    return [float(item) for item in seq], int(rows), int(cols)


@dataclass
class MatmulJob:
    job_id: str
    m: int
    k: int
    n: int
    a_base64: str
    b_base64: str
    created_at: float
    status: str = "queued"
    worker_id: str | None = None
    lease_expires_at: float | None = None
    result_base64: str | None = None
    result_shape: tuple[int, int] | None = None
    error: str | None = None
    elapsed_ms: float | None = None
    completed_at: float | None = None
    result_receipt_sha256: str | None = None
    event: threading.Event = field(default_factory=threading.Event, repr=False)

    def public_job(self) -> dict[str, Any]:
        return {
            "schema": "auro.webgpu.matmul.job.v1",
            "job_id": self.job_id,
            "kind": "matmul_f32",
            "a": {"shape": [self.m, self.k], "base64": self.a_base64, "dtype": "float32-le"},
            "b": {"shape": [self.k, self.n], "base64": self.b_base64, "dtype": "float32-le"},
            "output_shape": [self.m, self.n],
            "lease_expires_at": self.lease_expires_at,
        }


class Cluster:
    def __init__(self, *, max_matrix_elements: int = 16_777_216, lease_seconds: float = 30.0, default_timeout: float = 120.0) -> None:
        self.max_matrix_elements = int(max_matrix_elements)
        self.lease_seconds = float(lease_seconds)
        self.default_timeout = float(default_timeout)
        self._jobs: dict[str, MatmulJob] = {}
        self._queue: deque[str] = deque()
        self._condition = threading.Condition()
        self._workers: dict[str, dict[str, Any]] = {}

    def matmul(self, a: Sequence[Sequence[float]] | Sequence[float], b: Sequence[Sequence[float]] | Sequence[float], *, a_shape: tuple[int, int] | None = None, b_shape: tuple[int, int] | None = None, timeout: float | None = None) -> list[list[float]]:
        af, m, k = flatten_matrix(a, *(a_shape or (None, None)))
        bf, bk, n = flatten_matrix(b, *(b_shape or (None, None)))
        if k != bk:
            raise ValueError(f"incompatible matmul shapes {(m, k)} and {(bk, n)}")
        job = self.enqueue_encoded(encode_f32(af), encode_f32(bf), m, k, n)
        if not job.event.wait(timeout if timeout is not None else self.default_timeout):
            with self._condition:
                job.status = "timeout"
                job.error = "coordinator wait timeout"
            raise TimeoutError(f"WebGPU job {job.job_id} timed out")
        if job.status != "completed" or not job.result_base64:
            raise RuntimeError(job.error or f"WebGPU job ended as {job.status}")
        values = decode_f32(job.result_base64)
        return [values[row * n : (row + 1) * n] for row in range(m)]

    def enqueue_encoded(self, a_base64: str, b_base64: str, m: int, k: int, n: int) -> MatmulJob:
        m, k, n = int(m), int(k), int(n)
        if min(m, k, n) <= 0:
            raise ValueError("matrix dimensions must be positive")
        if max(m * k, k * n, m * n) > self.max_matrix_elements:
            raise ValueError("matrix exceeds coordinator element limit")
        if len(decode_f32(a_base64)) != m * k or len(decode_f32(b_base64)) != k * n:
            raise ValueError("matrix payload length does not match shape")
        job = MatmulJob("wgpu_" + secrets.token_hex(12), m, k, n, a_base64, b_base64, time.time())
        with self._condition:
            self._jobs[job.job_id] = job
            self._queue.append(job.job_id)
            self._condition.notify_all()
        return job

    def claim(self, worker_id: str, *, wait_seconds: float = 10.0, capabilities: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not worker_id or len(worker_id) > 160:
            raise ValueError("worker_id must contain 1..160 characters")
        deadline = time.time() + max(0.0, min(float(wait_seconds), 30.0))
        with self._condition:
            self._workers[worker_id] = {"last_seen": time.time(), "capabilities": dict(capabilities or {})}
            while True:
                self._requeue_expired_locked()
                while self._queue:
                    job_id = self._queue.popleft()
                    job = self._jobs.get(job_id)
                    if job is None or job.status != "queued":
                        continue
                    job.status = "leased"
                    job.worker_id = worker_id
                    job.lease_expires_at = time.time() + self.lease_seconds
                    return job.public_job()
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def complete(self, job_id: str, worker_id: str, *, result_base64: str | None = None, shape: Sequence[int] | None = None, elapsed_ms: float | None = None, error: str | None = None, backend: str = "webgpu") -> dict[str, Any]:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                raise ValueError("unknown job")
            if job.status != "leased" or job.worker_id != worker_id:
                raise ValueError("job is not leased to this worker")
            if job.lease_expires_at is not None and job.lease_expires_at < time.time():
                raise ValueError("job lease expired")
            if error:
                job.status, job.error = "failed", str(error)[:500]
            else:
                expected_shape = (job.m, job.n)
                actual_shape = tuple(int(item) for item in (shape or ()))
                if actual_shape != expected_shape:
                    raise ValueError(f"result shape {actual_shape} != {expected_shape}")
                values = decode_f32(str(result_base64 or ""))
                if len(values) != job.m * job.n or not all(math.isfinite(item) for item in values):
                    raise ValueError("invalid result values")
                job.result_base64 = str(result_base64)
                job.result_shape = expected_shape
                job.status = "completed"
            job.elapsed_ms = float(elapsed_ms) if elapsed_ms is not None else None
            job.completed_at = time.time()
            receipt = {
                "schema": "auro.webgpu.matmul.receipt.v1",
                "job_id": job.job_id,
                "worker_id": worker_id,
                "backend": backend,
                "shape": [job.m, job.k, job.n],
                "status": job.status,
                "a_sha256": hashlib.sha256(base64.b64decode(job.a_base64)).hexdigest(),
                "b_sha256": hashlib.sha256(base64.b64decode(job.b_base64)).hexdigest(),
                "result_sha256": hashlib.sha256(base64.b64decode(job.result_base64)).hexdigest() if job.result_base64 else None,
                "elapsed_ms": job.elapsed_ms,
            }
            job.result_receipt_sha256 = _sha(receipt)
            job.event.set()
            self._condition.notify_all()
            return {**receipt, "receipt_sha256": job.result_receipt_sha256}

    def status(self) -> dict[str, Any]:
        with self._condition:
            self._requeue_expired_locked()
            counts: dict[str, int] = {}
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
            workers = {worker_id: {**value, "age_seconds": round(time.time() - value["last_seen"], 3)} for worker_id, value in self._workers.items()}
            return {
                "schema": "auro.webgpu.cluster.status.v1",
                "jobs": counts,
                "workers": workers,
                "queue_depth": len(self._queue),
                "max_matrix_elements": self.max_matrix_elements,
                "training_backend_claim": "configured browser WebGPU transport; hardware use requires completed browser-webgpu worker receipts",
            }

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            row = {key: value for key, value in asdict(job).items() if key != "event"}
            row.pop("a_base64", None)
            row.pop("b_base64", None)
            row.pop("result_base64", None)
            return row

    def _requeue_expired_locked(self) -> None:
        now = time.time()
        for job in self._jobs.values():
            if job.status == "leased" and job.lease_expires_at is not None and job.lease_expires_at < now:
                job.status = "queued"
                job.worker_id = None
                job.lease_expires_at = None
                self._queue.append(job.job_id)


class CoordinatorHandler(BaseHTTPRequestHandler):
    cluster = Cluster()
    token = ""
    server_version = "AuroWebGPUCoordinator/1.1"

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        if not self._authorized():
            return self._json(401, {"error": "cluster token required"})
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/status":
            return self._json(200, self.cluster.status())
        if parsed.path == "/job":
            worker_id = str(query.get("worker_id", [""])[0]).strip()
            if not worker_id:
                return self._json(400, {"error": "worker_id required"})
            capabilities: dict[str, Any] = {}
            raw_caps = query.get("capabilities", [""])[0]
            if raw_caps:
                try:
                    value = json.loads(raw_caps)
                    if isinstance(value, dict):
                        capabilities = value
                except json.JSONDecodeError:
                    pass
            try:
                job = self.cluster.claim(worker_id, wait_seconds=float(query.get("wait", ["10"])[0]), capabilities=capabilities)
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, {"job": job})
        if parsed.path.startswith("/job/"):
            row = self.cluster.get_job(parsed.path.removeprefix("/job/"))
            return self._json(200 if row else 404, row or {"error": "not found"})
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            return self._json(401, {"error": "cluster token required"})
        try:
            body = self._body()
            path = urlsplit(self.path).path
            if path == "/matmul":
                a = body["a"]
                b = body["b"]
                m, k = (int(item) for item in a["shape"])
                bk, n = (int(item) for item in b["shape"])
                if k != bk:
                    raise ValueError("incompatible matrix shapes")
                result = self.cluster.matmul(decode_f32(a["base64"]), decode_f32(b["base64"]), a_shape=(m, k), b_shape=(bk, n), timeout=float(body.get("timeout", self.cluster.default_timeout)))
                flat = [item for row in result for item in row]
                return self._json(200, {"result": {"shape": [m, n], "base64": encode_f32(flat), "dtype": "float32-le"}})
            if path == "/result":
                receipt = self.cluster.complete(str(body["job_id"]), str(body["worker_id"]), result_base64=body.get("result", {}).get("base64"), shape=body.get("result", {}).get("shape"), elapsed_ms=body.get("elapsed_ms"), error=body.get("error"), backend=str(body.get("backend") or "webgpu"))
                return self._json(200, {"ok": True, "receipt": receipt})
            return self._json(404, {"error": "not found"})
        except (ValueError, KeyError, TypeError, TimeoutError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)[:500]})

    def _authorized(self) -> bool:
        return not self.token or secrets.compare_digest(self.headers.get("x-auro-cluster-token", ""), self.token)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length <= 0 or length > 128 * 1024 * 1024:
            raise ValueError("invalid request size")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON object required")
        return value

    def _cors(self) -> None:
        origin = self.headers.get("origin", "")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme in {"http", "https"} and parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
                self.send_header("access-control-allow-origin", origin)
                self.send_header("vary", "origin")
        self.send_header("access-control-allow-methods", "GET,POST,OPTIONS")
        self.send_header("access-control-allow-headers", "content-type,x-auro-cluster-token")
        self.send_header("cache-control", "no-store")
        self.send_header("x-content-type-options", "nosniff")

    def _json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self._cors()
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8765, *, token: str = "", cluster: Cluster | None = None) -> ThreadingHTTPServer:
    class BoundHandler(CoordinatorHandler):
        pass

    BoundHandler.cluster = cluster or Cluster()
    BoundHandler.token = token
    return ThreadingHTTPServer((host, int(port)), BoundHandler)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Serve the AURO browser WebGPU training coordinator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=os.getenv("AURO_WEBGPU_CLUSTER_TOKEN", ""))
    args = parser.parse_args()
    print(json.dumps({"listening": f"http://{args.host}:{args.port}", "token_required": bool(args.token)}))
    serve(args.host, args.port, token=args.token).serve_forever()
