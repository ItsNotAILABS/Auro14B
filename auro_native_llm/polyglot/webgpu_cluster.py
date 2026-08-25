"""Client adapter for AURO's persistent browser-WebGPU training cluster."""
from __future__ import annotations

import base64
import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np


class WebGPUClusterError(RuntimeError):
    pass


class WebGPUClusterClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("AURO_WEBGPU_CLUSTER_URL", "http://127.0.0.1:8765")
        ).rstrip("/")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("AURO_WEBGPU_CLUSTER_URL must be an absolute HTTP(S) URL")
        self.token = (
            token
            if token is not None
            else os.getenv("AURO_WEBGPU_CLUSTER_TOKEN", "")
        )
        self.timeout = max(1.0, float(timeout))

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.token:
            headers["x-auro-cluster-token"] = self.token
        return headers

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        data = None
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            headers=self._headers(),
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout or self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise WebGPUClusterError(
                f"cluster HTTP {exc.code} for {path}: {detail}"
            ) from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise WebGPUClusterError(f"cluster request failed for {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise WebGPUClusterError("cluster response must be a JSON object")
        return value

    def status(self) -> dict[str, Any]:
        return self._request("/status", timeout=min(self.timeout, 10.0))

    def ready(self, *, require_worker: bool = True) -> bool:
        status = self.status()
        if status.get("schema") != "auro.webgpu.cluster-status.v2":
            return False
        return not require_worker or int(status.get("ready_workers", 0)) > 0

    @staticmethod
    def _matrix_payload(value: np.ndarray) -> dict[str, Any]:
        matrix = np.asarray(value, dtype="<f4", order="C")
        if matrix.ndim != 2:
            raise ValueError("WebGPU matrices must be rank 2")
        return {
            "shape": list(matrix.shape),
            "base64": base64.b64encode(matrix.tobytes()).decode("ascii"),
            "dtype": "float32-le",
        }

    def submit_matmul(self, a: np.ndarray, b: np.ndarray) -> str:
        left = np.asarray(a, dtype="<f4", order="C")
        right = np.asarray(b, dtype="<f4", order="C")
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError(
                f"incompatible matmul shapes {left.shape} and {right.shape}"
            )
        response = self._request(
            "/jobs",
            method="POST",
            payload={
                "a": self._matrix_payload(left),
                "b": self._matrix_payload(right),
            },
        )
        job_id = str(response.get("job_id") or "")
        if not job_id:
            raise WebGPUClusterError("cluster did not return a job_id")
        return job_id

    def job(self, job_id: str) -> dict[str, Any]:
        if not job_id or "/" in job_id:
            raise ValueError("invalid job_id")
        return self._request(f"/job/{job_id}")

    def wait(
        self,
        job_id: str,
        *,
        timeout: float | None = None,
        poll_seconds: float = 0.05,
    ) -> dict[str, Any]:
        deadline = time.time() + (self.timeout if timeout is None else float(timeout))
        while True:
            row = self.job(job_id)
            if row.get("status") in {"completed", "failed", "cancelled"}:
                return row
            if time.time() >= deadline:
                raise TimeoutError(f"WebGPU job {job_id} timed out")
            time.sleep(max(0.01, min(float(poll_seconds), 1.0)))

    def receipt(self, receipt_sha256: str) -> dict[str, Any]:
        if len(receipt_sha256) != 64:
            raise ValueError("receipt_sha256 must contain 64 characters")
        return self._request(f"/receipt/{receipt_sha256}")

    def cancel(self, job_id: str, reason: str = "client_cancelled") -> dict[str, Any]:
        return self._request(
            "/cancel",
            method="POST",
            payload={"job_id": job_id, "reason": reason},
        )

    def matmul(
        self,
        a: np.ndarray,
        b: np.ndarray,
        *,
        timeout: float | None = None,
    ) -> np.ndarray:
        left = np.asarray(a, dtype="<f4", order="C")
        right = np.asarray(b, dtype="<f4", order="C")
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError(
                f"incompatible matmul shapes {left.shape} and {right.shape}"
            )
        response = self._request(
            "/matmul",
            method="POST",
            payload={
                "a": self._matrix_payload(left),
                "b": self._matrix_payload(right),
                "timeout": self.timeout if timeout is None else float(timeout),
            },
            timeout=(self.timeout if timeout is None else float(timeout)) + 5.0,
        )
        result = response.get("result") or {}
        shape = tuple(int(item) for item in result.get("shape") or ())
        if shape != (left.shape[0], right.shape[1]):
            raise WebGPUClusterError(
                f"cluster returned shape {shape}; expected {(left.shape[0], right.shape[1])}"
            )
        try:
            raw = base64.b64decode(str(result["base64"]), validate=True)
        except (KeyError, ValueError) as exc:
            raise WebGPUClusterError("cluster returned invalid float32 payload") from exc
        expected_bytes = shape[0] * shape[1] * 4
        if len(raw) != expected_bytes:
            raise WebGPUClusterError(
                f"cluster returned {len(raw)} bytes; expected {expected_bytes}"
            )
        values = np.frombuffer(raw, dtype="<f4").copy().reshape(shape)
        if not np.isfinite(values).all():
            raise WebGPUClusterError("cluster returned non-finite values")
        return values.astype(np.float64)
