"""Client adapter for AURO's local browser-WebGPU training cluster."""
from __future__ import annotations

import base64
import json
import os
from urllib.request import Request, urlopen
from typing import Any

import numpy as np


class WebGPUClusterClient:
    def __init__(self, base_url: str | None = None, token: str | None = None, timeout: float = 120.0):
        self.base_url = (base_url or os.getenv("AURO_WEBGPU_CLUSTER_URL", "http://127.0.0.1:8765")).rstrip("/")
        self.token = token if token is not None else os.getenv("AURO_WEBGPU_CLUSTER_TOKEN", "")
        self.timeout = float(timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.token:
            headers["x-auro-cluster-token"] = self.token
        return headers

    def status(self) -> dict[str, Any]:
        request = Request(self.base_url + "/status", headers=self._headers())
        with urlopen(request, timeout=min(self.timeout, 10.0)) as response:
            return json.loads(response.read().decode())

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        left = np.asarray(a, dtype="<f4", order="C")
        right = np.asarray(b, dtype="<f4", order="C")
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError(f"incompatible matmul shapes {left.shape} and {right.shape}")
        payload = {
            "a": {"shape": list(left.shape), "base64": base64.b64encode(left.tobytes()).decode("ascii"), "dtype": "float32-le"},
            "b": {"shape": list(right.shape), "base64": base64.b64encode(right.tobytes()).decode("ascii"), "dtype": "float32-le"},
            "timeout": self.timeout,
        }
        request = Request(
            self.base_url + "/matmul",
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers=self._headers(),
            method="POST",
        )
        with urlopen(request, timeout=self.timeout + 5.0) as response:
            body = json.loads(response.read().decode())
        result = body["result"]
        shape = tuple(int(item) for item in result["shape"])
        raw = base64.b64decode(result["base64"])
        values = np.frombuffer(raw, dtype="<f4").copy().reshape(shape)
        return values.astype(np.float64)
