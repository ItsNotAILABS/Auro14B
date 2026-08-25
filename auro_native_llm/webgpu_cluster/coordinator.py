"""Persistent, dependency-free coordinator for AURO browser WebGPU nodes.

Browser tabs lease bounded float32 matrix jobs, execute them on WebGPU, renew
leases while working, and return results. SQLite/WAL preserves queue, replay,
worker, and receipt state across coordinator restarts.

A completed receipt proves that a registered worker returned a result under the
browser-WebGPU protocol. It does not independently attest GPU hardware,
benchmark quality, training convergence, or checkpoint promotion.
"""
from __future__ import annotations

from array import array
import argparse
import base64
from collections import Counter
from dataclasses import dataclass
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import sqlite3
import sys
import threading
import time
from typing import Any, Sequence
from urllib.parse import parse_qs, urlsplit


SCHEMA_VERSION = "auro.webgpu-cluster.v2"
DEFAULT_MAX_REQUEST_BYTES = 128 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_bytes(value: bytes | None) -> str | None:
    return hashlib.sha256(value).hexdigest() if value is not None else None


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


def _decode_f32_bytes(value: str) -> bytes:
    raw = base64.b64decode(value.encode("ascii"), validate=True)
    if len(raw) % 4:
        raise ValueError("float32 payload byte length must be divisible by four")
    return raw


def _encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def flatten_matrix(
    value: Sequence[Sequence[float]] | Sequence[float],
    rows: int | None = None,
    cols: int | None = None,
) -> tuple[list[float], int, int]:
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


def _is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class JobLease:
    job_id: str
    lease_token: str
    worker_id: str
    lease_expires_at: float
    attempt: int


class Cluster:
    """SQLite-backed queue and receipt store for browser matrix workers."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        max_matrix_elements: int = 16_777_216,
        max_queued_jobs: int = 128,
        lease_seconds: float = 60.0,
        default_timeout: float = 180.0,
        max_attempts: int = 3,
        worker_ttl_seconds: float = 90.0,
        retention_seconds: float = 86_400.0,
        receipt_signing_key: str | bytes | None = None,
        signer_id: str = "auro-webgpu-coordinator",
    ) -> None:
        self.db_path = str(db_path)
        self.max_matrix_elements = max(1, int(max_matrix_elements))
        self.max_queued_jobs = max(1, int(max_queued_jobs))
        self.lease_seconds = max(5.0, float(lease_seconds))
        self.default_timeout = max(1.0, float(default_timeout))
        self.max_attempts = max(1, min(int(max_attempts), 10))
        self.worker_ttl_seconds = max(5.0, float(worker_ttl_seconds))
        self.retention_seconds = max(60.0, float(retention_seconds))
        raw_key = receipt_signing_key or b""
        self.receipt_signing_key = (
            raw_key.encode("utf-8") if isinstance(raw_key, str) else raw_key
        )
        self.signer_id = signer_id
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,
            timeout=30.0,
        )
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    @property
    def persistent(self) -> bool:
        return self.db_path != ":memory:"

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _initialize(self) -> None:
        with self._lock:
            if self.persistent:
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    m INTEGER NOT NULL,
                    k INTEGER NOT NULL,
                    n INTEGER NOT NULL,
                    a BLOB NOT NULL,
                    b BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    worker_id TEXT,
                    lease_token TEXT,
                    lease_expires_at REAL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    result BLOB,
                    result_rows INTEGER,
                    result_cols INTEGER,
                    error TEXT,
                    elapsed_ms REAL,
                    backend TEXT,
                    completed_at REAL,
                    receipt_sha256 TEXT,
                    receipt_signature TEXT,
                    signer_id TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status_created
                    ON jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    last_seen REAL NOT NULL,
                    capabilities_json TEXT NOT NULL,
                    user_agent TEXT,
                    origin TEXT,
                    completed_jobs INTEGER NOT NULL DEFAULT 0,
                    failed_jobs INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS receipts (
                    receipt_sha256 TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )
            self._requeue_expired_locked(time.time())

    def matmul(
        self,
        a: Sequence[Sequence[float]] | Sequence[float],
        b: Sequence[Sequence[float]] | Sequence[float],
        *,
        a_shape: tuple[int, int] | None = None,
        b_shape: tuple[int, int] | None = None,
        timeout: float | None = None,
    ) -> list[list[float]]:
        af, m, k = flatten_matrix(a, *(a_shape or (None, None)))
        bf, bk, n = flatten_matrix(b, *(b_shape or (None, None)))
        if k != bk:
            raise ValueError(f"incompatible matmul shapes {(m, k)} and {(bk, n)}")
        job_id = self.enqueue_encoded(encode_f32(af), encode_f32(bf), m, k, n)
        row = self.wait(job_id, timeout=timeout)
        if row["status"] != "completed" or not row.get("result_base64"):
            raise RuntimeError(row.get("error") or f"WebGPU job ended as {row['status']}")
        values = decode_f32(str(row["result_base64"]))
        return [values[index * n : (index + 1) * n] for index in range(m)]

    def enqueue_encoded(
        self,
        a_base64: str,
        b_base64: str,
        m: int,
        k: int,
        n: int,
    ) -> str:
        m, k, n = int(m), int(k), int(n)
        if min(m, k, n) <= 0:
            raise ValueError("matrix dimensions must be positive")
        if max(m * k, k * n, m * n) > self.max_matrix_elements:
            raise ValueError("matrix exceeds coordinator element limit")
        a_bytes = _decode_f32_bytes(a_base64)
        b_bytes = _decode_f32_bytes(b_base64)
        if len(a_bytes) != m * k * 4 or len(b_bytes) != k * n * 4:
            raise ValueError("matrix payload length does not match shape")
        now = time.time()
        with self._condition:
            self._prune_locked(now)
            queued = self._connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','leased')"
            ).fetchone()[0]
            if int(queued) >= self.max_queued_jobs:
                raise RuntimeError("WebGPU cluster queue is full")
            job_id = "wgpu_" + secrets.token_hex(12)
            self._connection.execute(
                """
                INSERT INTO jobs (
                    job_id, kind, m, k, n, a, b, created_at, updated_at,
                    status, max_attempts
                ) VALUES (?, 'matmul_f32', ?, ?, ?, ?, ?, ?, ?, 'queued', ?)
                """,
                (job_id, m, k, n, a_bytes, b_bytes, now, now, self.max_attempts),
            )
            self._condition.notify_all()
            return job_id

    def wait(self, job_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        deadline = time.time() + (
            self.default_timeout if timeout is None else max(0.01, float(timeout))
        )
        with self._condition:
            while True:
                row = self._job_locked(job_id, include_payload=False)
                if row is None:
                    raise ValueError("unknown job")
                if row["status"] in {"completed", "failed", "cancelled"}:
                    return self._job_locked(job_id, include_payload=True) or row
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(f"WebGPU job {job_id} timed out while still {row['status']}")
                self._condition.wait(min(remaining, 1.0))
                self._requeue_expired_locked(time.time())

    def claim(
        self,
        worker_id: str,
        *,
        wait_seconds: float = 10.0,
        capabilities: dict[str, Any] | None = None,
        user_agent: str = "",
        origin: str = "",
    ) -> dict[str, Any] | None:
        worker_id = str(worker_id).strip()
        if not worker_id or len(worker_id) > 160:
            raise ValueError("worker_id must contain 1..160 characters")
        caps = dict(capabilities or {})
        if caps.get("webgpu") is not True:
            raise ValueError("worker must declare webgpu=true")
        deadline = time.time() + max(0.0, min(float(wait_seconds), 30.0))
        with self._condition:
            while True:
                now = time.time()
                self._upsert_worker_locked(worker_id, caps, user_agent, origin, now)
                self._requeue_expired_locked(now)
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    row = self._connection.execute(
                        """
                        SELECT * FROM jobs
                        WHERE status='queued' AND attempt < max_attempts
                        ORDER BY created_at, job_id
                        LIMIT 1
                        """
                    ).fetchone()
                    if row is not None:
                        lease_token = secrets.token_urlsafe(32)
                        lease_expires = now + self.lease_seconds
                        updated = self._connection.execute(
                            """
                            UPDATE jobs
                            SET status='leased', worker_id=?, lease_token=?,
                                lease_expires_at=?, attempt=attempt+1, updated_at=?
                            WHERE job_id=? AND status='queued'
                            """,
                            (
                                worker_id,
                                lease_token,
                                lease_expires,
                                now,
                                row["job_id"],
                            ),
                        ).rowcount
                        if updated == 1:
                            self._connection.execute("COMMIT")
                            leased = self._connection.execute(
                                "SELECT * FROM jobs WHERE job_id=?",
                                (row["job_id"],),
                            ).fetchone()
                            return self._public_job(leased)
                    self._connection.execute("COMMIT")
                except Exception:
                    self._connection.execute("ROLLBACK")
                    raise
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._condition.wait(min(remaining, 1.0))

    def renew(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
    ) -> dict[str, Any]:
        now = time.time()
        with self._condition:
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            self._validate_lease(row, worker_id, lease_token, now)
            lease_expires = now + self.lease_seconds
            self._connection.execute(
                "UPDATE jobs SET lease_expires_at=?, updated_at=? WHERE job_id=?",
                (lease_expires, now, job_id),
            )
            return {
                "schema": "auro.webgpu.lease-renewal.v1",
                "job_id": job_id,
                "worker_id": worker_id,
                "lease_expires_at": lease_expires,
            }

    def complete(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        *,
        result_base64: str | None = None,
        shape: Sequence[int] | None = None,
        elapsed_ms: float | None = None,
        error: str | None = None,
        backend: str = "browser-webgpu",
        worker_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        with self._condition:
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            self._validate_lease(row, worker_id, lease_token, now)
            result_bytes: bytes | None = None
            status = "failed" if error else "completed"
            normalized_error = str(error)[:1000] if error else None
            if not error:
                expected_shape = (int(row["m"]), int(row["n"]))
                actual_shape = tuple(int(item) for item in (shape or ()))
                if actual_shape != expected_shape:
                    raise ValueError(f"result shape {actual_shape} != {expected_shape}")
                result_bytes = _decode_f32_bytes(str(result_base64 or ""))
                values = array("f")
                values.frombytes(result_bytes)
                if sys.byteorder != "little":
                    values.byteswap()
                if len(values) != expected_shape[0] * expected_shape[1]:
                    raise ValueError("result value count does not match shape")
                if not all(math.isfinite(float(item)) for item in values):
                    raise ValueError("result contains non-finite values")
            self._connection.execute(
                """
                UPDATE jobs
                SET status=?, result=?, result_rows=?, result_cols=?, error=?,
                    elapsed_ms=?, backend=?, completed_at=?, updated_at=?,
                    lease_token=NULL, lease_expires_at=NULL
                WHERE job_id=?
                """,
                (
                    status,
                    result_bytes,
                    int(row["m"]) if result_bytes is not None else None,
                    int(row["n"]) if result_bytes is not None else None,
                    normalized_error,
                    float(elapsed_ms) if elapsed_ms is not None else None,
                    str(backend)[:80],
                    now,
                    now,
                    job_id,
                ),
            )
            worker = self._connection.execute(
                "SELECT capabilities_json FROM workers WHERE worker_id=?",
                (worker_id,),
            ).fetchone()
            receipt = {
                "schema": "auro.webgpu.matmul-receipt.v2",
                "job_id": job_id,
                "worker_id": worker_id,
                "backend_reported": str(backend)[:80],
                "attempt": int(row["attempt"]),
                "shape": [int(row["m"]), int(row["k"]), int(row["n"])],
                "status": status,
                "a_sha256": _sha_bytes(bytes(row["a"])),
                "b_sha256": _sha_bytes(bytes(row["b"])),
                "result_sha256": _sha_bytes(result_bytes),
                "elapsed_ms": float(elapsed_ms) if elapsed_ms is not None else None,
                "worker_capabilities_sha256": _sha(
                    json.loads(worker["capabilities_json"]) if worker else {}
                ),
                "worker_evidence": dict(worker_evidence or {}),
                "completed_at": now,
                "claim_boundary": (
                    "worker-reported browser WebGPU execution; independent hardware "
                    "attestation and model-training evidence are separate"
                ),
            }
            receipt_hash = _sha(receipt)
            signature = None
            if self.receipt_signing_key:
                signature = hmac.new(
                    self.receipt_signing_key,
                    receipt_hash.encode("ascii"),
                    hashlib.sha256,
                ).hexdigest()
            receipt.update(
                {
                    "receipt_sha256": receipt_hash,
                    "signature": signature,
                    "signer_id": self.signer_id if signature else None,
                    "custody": "local-signed" if signature else "local-unsigned",
                }
            )
            self._connection.execute(
                """
                UPDATE jobs
                SET receipt_sha256=?, receipt_signature=?, signer_id=?
                WHERE job_id=?
                """,
                (receipt_hash, signature, receipt.get("signer_id"), job_id),
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO receipts VALUES (?, ?, ?)",
                (receipt_hash, json.dumps(receipt, sort_keys=True), now),
            )
            counter = "completed_jobs" if status == "completed" else "failed_jobs"
            self._connection.execute(
                f"UPDATE workers SET {counter}={counter}+1, last_seen=? WHERE worker_id=?",
                (now, worker_id),
            )
            self._condition.notify_all()
            return receipt

    def cancel(self, job_id: str, reason: str = "operator_cancelled") -> dict[str, Any]:
        now = time.time()
        with self._condition:
            row = self._connection.execute(
                "SELECT status FROM jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise ValueError("unknown job")
            if row["status"] in {"completed", "failed", "cancelled"}:
                return self.get_job(job_id) or {}
            self._connection.execute(
                """
                UPDATE jobs SET status='cancelled', error=?, updated_at=?,
                    completed_at=?, lease_token=NULL, lease_expires_at=NULL
                WHERE job_id=?
                """,
                (str(reason)[:500], now, now, job_id),
            )
            self._condition.notify_all()
            return self.get_job(job_id) or {}

    def status(self) -> dict[str, Any]:
        now = time.time()
        with self._condition:
            self._requeue_expired_locked(now)
            self._prune_locked(now)
            counts = Counter(
                row["status"]
                for row in self._connection.execute("SELECT status FROM jobs")
            )
            workers: dict[str, Any] = {}
            for row in self._connection.execute(
                "SELECT * FROM workers ORDER BY worker_id"
            ):
                age = now - float(row["last_seen"])
                workers[row["worker_id"]] = {
                    "last_seen": float(row["last_seen"]),
                    "age_seconds": round(age, 3),
                    "ready": age <= self.worker_ttl_seconds,
                    "capabilities": json.loads(row["capabilities_json"]),
                    "completed_jobs": int(row["completed_jobs"]),
                    "failed_jobs": int(row["failed_jobs"]),
                }
            ready_workers = sum(1 for value in workers.values() if value["ready"])
            receipt_count = int(
                self._connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
            )
            return {
                "schema": "auro.webgpu.cluster-status.v2",
                "persistent": self.persistent,
                "database": "sqlite-wal" if self.persistent else "sqlite-memory",
                "jobs": dict(counts),
                "queue_depth": int(counts.get("queued", 0)),
                "workers": workers,
                "ready_workers": ready_workers,
                "receipt_count": receipt_count,
                "receipt_signing_configured": bool(self.receipt_signing_key),
                "limits": {
                    "max_matrix_elements": self.max_matrix_elements,
                    "max_queued_jobs": self.max_queued_jobs,
                    "lease_seconds": self.lease_seconds,
                    "max_attempts": self.max_attempts,
                    "retention_seconds": self.retention_seconds,
                },
                "training_backend_claim": (
                    "matrix-compute substrate only; complete browser-WebGPU receipts, "
                    "training logs, checkpoint hashes, and evaluations are required for "
                    "a model-training claim"
                ),
            }

    def get_job(self, job_id: str, *, include_result: bool = False) -> dict[str, Any] | None:
        with self._lock:
            return self._job_locked(job_id, include_payload=include_result)

    def get_receipt(self, receipt_sha256: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json FROM receipts WHERE receipt_sha256=?",
                (receipt_sha256,),
            ).fetchone()
            return json.loads(row["payload_json"]) if row else None

    def _job_locked(self, job_id: str, *, include_payload: bool) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        value = {
            "schema": "auro.webgpu.job-status.v2",
            "job_id": row["job_id"],
            "kind": row["kind"],
            "shape": [int(row["m"]), int(row["k"]), int(row["n"])],
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
            "status": row["status"],
            "worker_id": row["worker_id"],
            "lease_expires_at": row["lease_expires_at"],
            "attempt": int(row["attempt"]),
            "max_attempts": int(row["max_attempts"]),
            "error": row["error"],
            "elapsed_ms": row["elapsed_ms"],
            "backend": row["backend"],
            "completed_at": row["completed_at"],
            "receipt_sha256": row["receipt_sha256"],
            "receipt_signature": row["receipt_signature"],
            "signer_id": row["signer_id"],
        }
        if include_payload and row["result"] is not None:
            value["result_shape"] = [int(row["result_rows"]), int(row["result_cols"])]
            value["result_base64"] = _encode_bytes(bytes(row["result"]))
        return value

    def _public_job(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "schema": "auro.webgpu.matmul-job.v2",
            "job_id": row["job_id"],
            "kind": row["kind"],
            "a": {
                "shape": [int(row["m"]), int(row["k"])],
                "base64": _encode_bytes(bytes(row["a"])),
                "dtype": "float32-le",
            },
            "b": {
                "shape": [int(row["k"]), int(row["n"])],
                "base64": _encode_bytes(bytes(row["b"])),
                "dtype": "float32-le",
            },
            "output_shape": [int(row["m"]), int(row["n"])],
            "worker_id": row["worker_id"],
            "lease_token": row["lease_token"],
            "lease_expires_at": float(row["lease_expires_at"]),
            "attempt": int(row["attempt"]),
            "max_attempts": int(row["max_attempts"]),
        }

    def _validate_lease(
        self,
        row: sqlite3.Row | None,
        worker_id: str,
        lease_token: str,
        now: float,
    ) -> None:
        if row is None:
            raise ValueError("unknown job")
        if row["status"] != "leased":
            raise ValueError("job is not leased")
        if not hmac.compare_digest(str(row["worker_id"] or ""), str(worker_id)):
            raise ValueError("job is leased to another worker")
        if not hmac.compare_digest(str(row["lease_token"] or ""), str(lease_token)):
            raise ValueError("invalid or replayed lease token")
        if float(row["lease_expires_at"] or 0.0) < now:
            raise ValueError("job lease expired")

    def _upsert_worker_locked(
        self,
        worker_id: str,
        capabilities: dict[str, Any],
        user_agent: str,
        origin: str,
        now: float,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO workers (
                worker_id, last_seen, capabilities_json, user_agent, origin
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
                last_seen=excluded.last_seen,
                capabilities_json=excluded.capabilities_json,
                user_agent=excluded.user_agent,
                origin=excluded.origin
            """,
            (
                worker_id,
                now,
                json.dumps(capabilities, sort_keys=True),
                str(user_agent)[:500],
                str(origin)[:500],
            ),
        )

    def _requeue_expired_locked(self, now: float) -> None:
        rows = self._connection.execute(
            """
            SELECT job_id, attempt, max_attempts
            FROM jobs
            WHERE status='leased' AND lease_expires_at < ?
            """,
            (now,),
        ).fetchall()
        for row in rows:
            if int(row["attempt"]) >= int(row["max_attempts"]):
                self._connection.execute(
                    """
                    UPDATE jobs SET status='failed', error='lease attempts exhausted',
                        completed_at=?, updated_at=?, worker_id=NULL,
                        lease_token=NULL, lease_expires_at=NULL
                    WHERE job_id=?
                    """,
                    (now, now, row["job_id"]),
                )
            else:
                self._connection.execute(
                    """
                    UPDATE jobs SET status='queued', worker_id=NULL,
                        lease_token=NULL, lease_expires_at=NULL, updated_at=?
                    WHERE job_id=?
                    """,
                    (now, row["job_id"]),
                )
        if rows:
            self._condition.notify_all()

    def _prune_locked(self, now: float) -> None:
        threshold = now - self.retention_seconds
        self._connection.execute(
            "DELETE FROM jobs WHERE status IN ('completed','failed','cancelled') AND completed_at < ?",
            (threshold,),
        )
        self._connection.execute(
            "DELETE FROM workers WHERE last_seen < ?",
            (threshold,),
        )
        self._connection.execute(
            "DELETE FROM receipts WHERE created_at < ?",
            (threshold,),
        )


class CoordinatorHandler(BaseHTTPRequestHandler):
    cluster = Cluster()
    token = ""
    allowed_origins: set[str] = set()
    max_request_bytes = DEFAULT_MAX_REQUEST_BYTES
    server_version = "AuroWebGPUCoordinator/2.0"

    def do_OPTIONS(self) -> None:
        if not self._origin_allowed():
            return self._json(403, {"error": "origin not allowed"})
        self.send_response(204)
        self._cors()
        self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
        self.send_header(
            "access-control-allow-headers",
            "content-type, x-auro-cluster-token",
        )
        self.send_header("access-control-max-age", "600")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._authorized():
            return self._json(401, {"error": "cluster token required"})
        if not self._origin_allowed():
            return self._json(403, {"error": "origin not allowed"})
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/status":
            return self._json(200, self.cluster.status())
        if parsed.path == "/job":
            worker_id = str(query.get("worker_id", [""])[0]).strip()
            raw_caps = query.get("capabilities", ["{}"]) [0]
            try:
                capabilities = json.loads(raw_caps)
            except json.JSONDecodeError:
                return self._json(400, {"error": "capabilities must be JSON"})
            if not isinstance(capabilities, dict):
                return self._json(400, {"error": "capabilities must be an object"})
            try:
                job = self.cluster.claim(
                    worker_id,
                    wait_seconds=float(query.get("wait", ["10"])[0]),
                    capabilities=capabilities,
                    user_agent=self.headers.get("user-agent", ""),
                    origin=self.headers.get("origin", ""),
                )
            except (ValueError, RuntimeError) as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, {"job": job})
        if parsed.path.startswith("/job/"):
            row = self.cluster.get_job(parsed.path.removeprefix("/job/"))
            return self._json(200 if row else 404, row or {"error": "not found"})
        if parsed.path.startswith("/receipt/"):
            row = self.cluster.get_receipt(parsed.path.removeprefix("/receipt/"))
            return self._json(200 if row else 404, row or {"error": "not found"})
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            return self._json(401, {"error": "cluster token required"})
        if not self._origin_allowed():
            return self._json(403, {"error": "origin not allowed"})
        try:
            body = self._body()
            path = urlsplit(self.path).path
            if path in {"/jobs", "/matmul"}:
                a = body["a"]
                b = body["b"]
                m, k = (int(item) for item in a["shape"])
                bk, n = (int(item) for item in b["shape"])
                if k != bk:
                    raise ValueError("incompatible matrix shapes")
                job_id = self.cluster.enqueue_encoded(
                    str(a["base64"]),
                    str(b["base64"]),
                    m,
                    k,
                    n,
                )
                if path == "/jobs":
                    return self._json(
                        202,
                        {
                            "job_id": job_id,
                            "status_url": f"/job/{job_id}",
                        },
                    )
                row = self.cluster.wait(
                    job_id,
                    timeout=float(body.get("timeout", self.cluster.default_timeout)),
                )
                if row["status"] != "completed":
                    raise RuntimeError(row.get("error") or row["status"])
                return self._json(
                    200,
                    {
                        "job_id": job_id,
                        "result": {
                            "shape": row["result_shape"],
                            "base64": row["result_base64"],
                            "dtype": "float32-le",
                        },
                        "receipt_sha256": row["receipt_sha256"],
                    },
                )
            if path == "/lease/renew":
                return self._json(
                    200,
                    self.cluster.renew(
                        str(body["job_id"]),
                        str(body["worker_id"]),
                        str(body["lease_token"]),
                    ),
                )
            if path == "/result":
                receipt = self.cluster.complete(
                    str(body["job_id"]),
                    str(body["worker_id"]),
                    str(body["lease_token"]),
                    result_base64=(body.get("result") or {}).get("base64"),
                    shape=(body.get("result") or {}).get("shape"),
                    elapsed_ms=body.get("elapsed_ms"),
                    error=body.get("error"),
                    backend=str(body.get("backend") or "browser-webgpu"),
                    worker_evidence=dict(body.get("worker_evidence") or {}),
                )
                return self._json(200, {"ok": True, "receipt": receipt})
            if path == "/cancel":
                return self._json(
                    200,
                    self.cluster.cancel(
                        str(body["job_id"]),
                        str(body.get("reason") or "operator_cancelled"),
                    ),
                )
            return self._json(404, {"error": "not found"})
        except TimeoutError as exc:
            return self._json(504, {"error": str(exc)[:500]})
        except (ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)[:500]})

    def _authorized(self) -> bool:
        if not self.token:
            return True
        return hmac.compare_digest(
            self.headers.get("x-auro-cluster-token", ""),
            self.token,
        )

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("origin", "").strip()
        if not origin:
            return True
        if origin in self.allowed_origins:
            return True
        parsed = urlsplit(origin)
        return parsed.scheme in {"http", "https"} and _is_loopback(
            str(parsed.hostname or "")
        )

    def _body(self) -> dict[str, Any]:
        content_type = self.headers.get("content-type", "").split(";", 1)[0].strip()
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json")
        length = int(self.headers.get("content-length", "0"))
        if length <= 0 or length > self.max_request_bytes:
            raise ValueError("invalid request size")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON object required")
        return value

    def _cors(self) -> None:
        origin = self.headers.get("origin", "").strip()
        if origin and self._origin_allowed():
            self.send_header("access-control-allow-origin", origin)
            self.send_header("vary", "origin")

    def _json(self, status: int, value: Any) -> None:
        raw = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(raw)))
        self.send_header("cache-control", "no-store")
        self.send_header("x-content-type-options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    token: str = "",
    db_path: str | Path = ":memory:",
    cluster: Cluster | None = None,
    allowed_origins: Sequence[str] = (),
    receipt_signing_key: str | bytes | None = None,
) -> ThreadingHTTPServer:
    if not _is_loopback(host) and len(token) < 32:
        raise ValueError("non-loopback coordinator binding requires a 32+ character token")
    active_cluster = cluster or Cluster(
        db_path,
        receipt_signing_key=receipt_signing_key,
    )
    handler = type(
        "BoundCoordinatorHandler",
        (CoordinatorHandler,),
        {
            "cluster": active_cluster,
            "token": token,
            "allowed_origins": set(allowed_origins),
        },
    )
    return ThreadingHTTPServer((host, int(port)), handler)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the AURO browser-WebGPU cluster")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--db",
        default="state/auro-webgpu-cluster.sqlite",
        help="SQLite/WAL database path, or :memory: for an ephemeral coordinator.",
    )
    parser.add_argument(
        "--allowed-origin",
        action="append",
        default=[],
        help="Additional exact browser Origin allowed by CORS.",
    )
    args = parser.parse_args()

    token = secrets.token_urlsafe(32) if False else ""
    token = __import__("os").getenv("AURO_WEBGPU_CLUSTER_TOKEN", token)
    signing_key = __import__("os").getenv("AURO_WEBGPU_RECEIPT_HMAC_KEY", "") or None
    db_path = Path(args.db)
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    server = serve(
        args.host,
        args.port,
        token=token,
        db_path=str(db_path),
        allowed_origins=args.allowed_origin,
        receipt_signing_key=signing_key,
    )
    print(
        json.dumps(
            {
                "service": "auro-webgpu-cluster",
                "url": f"http://{args.host}:{server.server_port}",
                "database": str(db_path),
                "token_required": bool(token),
                "receipt_signing": bool(signing_key),
            },
            indent=2,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        server.RequestHandlerClass.cluster.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
