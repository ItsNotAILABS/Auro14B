"""Accelerated compute plane for AURO training.

Detection order when explicitly configured:

1. Persistent browser-WebGPU cluster
2. PyTorch CUDA or Apple MPS
3. CuPy CUDA
4. ChaosCUDA
5. NumPy BLAS

The stable ``matmul`` API is used by ``train_step_linear`` and by AURO training
code. Selecting the WebGPU backend therefore moves real gradient matrix
products, not a disconnected visual demo. Complete model-training claims still
require dataset, optimizer, loss, checkpoint, hash, evaluation, and promotion
receipts in addition to browser matrix receipts.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

_PLANE: Optional["CudaPlane"] = None


def _truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class CudaPlane:
    backend: str = "numpy"
    device: str = "cpu"
    torch = None
    cupy = None
    cuda_available: bool = False
    accelerated_available: bool = False
    device_name: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def detect(cls) -> "CudaPlane":
        plane = cls()

        cluster_url = os.getenv("AURO_WEBGPU_CLUSTER_URL", "").strip()
        if cluster_url:
            try:
                from auro_native_llm.polyglot.webgpu_cluster import (
                    WebGPUClusterClient,
                )

                timeout = float(os.getenv("AURO_WEBGPU_CLUSTER_TIMEOUT", "180"))
                client = WebGPUClusterClient(
                    cluster_url,
                    os.getenv("AURO_WEBGPU_CLUSTER_TOKEN", ""),
                    timeout=timeout,
                )
                status = client.status()
                require_worker = _truthy(
                    "AURO_WEBGPU_CLUSTER_REQUIRE_WORKER",
                    True,
                )
                if status.get("schema") != "auro.webgpu.cluster-status.v2":
                    raise RuntimeError("unexpected WebGPU cluster status schema")
                if require_worker and int(status.get("ready_workers", 0)) <= 0:
                    raise RuntimeError("WebGPU cluster has no ready browser workers")
                plane.backend = "webgpu_cluster"
                plane.device = "browser-webgpu-mesh"
                plane.accelerated_available = True
                plane.device_name = "AURO persistent browser WebGPU cluster"
                plane.meta["webgpu_cluster"] = status
                plane.meta["webgpu_cluster_url"] = cluster_url
                plane.meta["require_ready_worker"] = require_worker
                plane.meta["hardware_claim_boundary"] = (
                    "worker-reported browser-WebGPU receipts are required; "
                    "independent hardware attestation is separate"
                )
                plane.meta["training_claim_boundary"] = (
                    "matrix transport is not a complete model-training claim"
                )
                plane._webgpu = client  # type: ignore[attr-defined]
                return plane
            except Exception as exc:
                plane.meta["webgpu_cluster_error"] = str(exc)[:500]
                if _truthy("AURO_WEBGPU_CLUSTER_REQUIRED", False):
                    raise RuntimeError(
                        f"required WebGPU cluster is unavailable: {exc}"
                    ) from exc

        try:
            import torch

            plane.torch = torch
            if torch.cuda.is_available():
                plane.backend = "torch_cuda"
                plane.device = "cuda"
                plane.cuda_available = True
                plane.accelerated_available = True
                plane.device_name = torch.cuda.get_device_name(0)
                plane.meta["torch"] = torch.__version__
                plane.meta["cuda_version"] = getattr(torch.version, "cuda", None)
                return plane
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                plane.backend = "torch_mps"
                plane.device = "mps"
                plane.accelerated_available = True
                plane.device_name = "Apple MPS"
                plane.meta["torch"] = torch.__version__
                return plane
            plane.meta["torch"] = torch.__version__
            plane.meta["torch_note"] = "torch present without CUDA or MPS"
        except Exception as exc:
            plane.meta["torch_error"] = str(exc)[:200]

        try:
            import cupy as cp

            plane.cupy = cp
            count = int(cp.cuda.runtime.getDeviceCount())
            if count <= 0:
                raise RuntimeError("CuPy reported no CUDA devices")
            plane.backend = "cupy_cuda"
            plane.device = "cuda"
            plane.cuda_available = True
            plane.accelerated_available = True
            plane.device_name = "cupy-cuda"
            plane.meta["device_count"] = count
            return plane
        except Exception as exc:
            plane.meta["cupy_error"] = str(exc)[:200]

        try:
            from auro_native_llm.chaos_cuda.plane import get_chaos_cuda

            chaos = get_chaos_cuda()
            plane.backend = "chaos_cuda"
            plane.device = chaos.device
            plane.accelerated_available = True
            plane.device_name = chaos.device_name
            plane.meta["chaos"] = chaos.info()
            plane.meta["lab"] = "Novel Chaos Labs"
            plane._chaos = chaos  # type: ignore[attr-defined]
            return plane
        except Exception as exc:
            plane.meta["chaos_error"] = str(exc)[:200]

        plane.backend = "numpy"
        plane.device = "cpu"
        plane.device_name = "NumPy BLAS"
        plane.meta["blas_threads"] = (
            os.environ.get("OMP_NUM_THREADS")
            or os.environ.get("OPENBLAS_NUM_THREADS")
        )
        return plane

    def info(self) -> Dict[str, Any]:
        return {
            "schema": "auro.accelerated-compute-plane.v3",
            "backend": self.backend,
            "device": self.device,
            "cuda_available": self.cuda_available,
            "accelerated_available": self.accelerated_available,
            "device_name": self.device_name,
            "meta": self.meta,
        }

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Execute one matrix product on the selected backend.

        A selected browser cluster fails visibly. It is not silently replaced by
        local NumPy after a job has been accepted, because doing so would corrupt
        backend evidence and reproducibility.
        """
        left = np.asarray(a, dtype=np.float32)
        right = np.asarray(b, dtype=np.float32)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
            raise ValueError(
                f"incompatible matmul shapes {left.shape} and {right.shape}"
            )

        webgpu = getattr(self, "_webgpu", None)
        if webgpu is not None and self.backend == "webgpu_cluster":
            return webgpu.matmul(left, right)

        chaos = getattr(self, "_chaos", None)
        if chaos is not None and self.backend == "chaos_cuda":
            return chaos.matmul(left, right)

        if self.torch is not None and self.backend.startswith("torch"):
            torch = self.torch
            try:
                device = torch.device(
                    "cuda"
                    if self.cuda_available and self.device == "cuda"
                    else ("mps" if self.device == "mps" else "cpu")
                )
                output = (
                    torch.as_tensor(left, device=device)
                    @ torch.as_tensor(right, device=device)
                )
                return output.detach().cpu().numpy().astype(np.float64)
            except Exception as exc:
                self.meta["torch_matmul_error"] = str(exc)[:300]

        if self.cupy is not None and self.backend == "cupy_cuda":
            try:
                return self.cupy.asnumpy(
                    self.cupy.asarray(left) @ self.cupy.asarray(right)
                ).astype(np.float64)
            except Exception as exc:
                self.meta["cupy_matmul_error"] = str(exc)[:300]

        return left.astype(np.float64) @ right.astype(np.float64)

    def train_step_linear(
        self,
        W: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        lr: float = 1e-3,
    ) -> Dict[str, Any]:
        """Run one MSE linear training step using the selected matrix backend.

        ``W`` is [out, in], ``X`` is [in, batch], and ``Y`` is [out, batch].
        This verifies training-path integration; it is not an LLM training claim.
        """
        chaos = getattr(self, "_chaos", None)
        if chaos is not None and self.backend == "chaos_cuda":
            return chaos.train_step_linear(W, X, Y, lr=lr)

        started = time.time()
        weights = np.asarray(W, dtype=np.float32)
        inputs = np.asarray(X, dtype=np.float32)
        targets = np.asarray(Y, dtype=np.float32)
        if weights.ndim != 2 or inputs.ndim != 2 or targets.ndim != 2:
            raise ValueError("W, X, and Y must be rank-2 arrays")
        if weights.shape[1] != inputs.shape[0]:
            raise ValueError("W and X shapes are incompatible")
        if (weights.shape[0], inputs.shape[1]) != targets.shape:
            raise ValueError("prediction and target shapes are incompatible")

        if self.torch is not None and self.backend.startswith("torch"):
            torch = self.torch
            try:
                device = torch.device(
                    "cuda"
                    if self.cuda_available and self.device == "cuda"
                    else ("mps" if self.device == "mps" else "cpu")
                )
                weight = torch.nn.Parameter(torch.as_tensor(weights, device=device))
                x = torch.as_tensor(inputs, device=device)
                y = torch.as_tensor(targets, device=device)
                prediction = weight @ x
                loss = torch.mean((prediction - y) ** 2)
                loss.backward()
                with torch.no_grad():
                    weight -= lr * weight.grad
                return {
                    "ok": True,
                    "backend": self.backend,
                    "loss": float(loss.detach().cpu()),
                    "W": weight.detach().cpu().numpy().astype(np.float64),
                    "sec": time.time() - started,
                    "training_claim_boundary": "single linear step only",
                }
            except Exception as exc:
                self.meta["torch_train_step_error"] = str(exc)[:300]

        prediction = self.matmul(weights, inputs)
        error = prediction - targets
        loss = float(np.mean(error ** 2))
        gradient = (2.0 / max(inputs.shape[1], 1)) * self.matmul(
            error,
            inputs.T,
        )
        updated = weights - float(lr) * gradient
        return {
            "ok": True,
            "backend": self.backend,
            "loss": loss,
            "W": np.asarray(updated, dtype=np.float64),
            "sec": time.time() - started,
            "training_claim_boundary": "single linear step only",
        }

    def spectral_batch_energy(self, batch: np.ndarray) -> np.ndarray:
        """Compute per-row spectral energy; WebGPU matmul does not imply FFT support."""
        values = np.asarray(batch, dtype=np.float32)
        if self.torch is not None and self.cuda_available and self.device == "cuda":
            torch = self.torch
            try:
                spectrum = torch.abs(
                    torch.fft.rfft(torch.as_tensor(values, device="cuda"), dim=-1)
                )
                return spectrum.sum(dim=-1).detach().cpu().numpy().astype(np.float64)
            except Exception as exc:
                self.meta["torch_fft_error"] = str(exc)[:300]
        return np.abs(np.fft.rfft(values, axis=-1)).sum(axis=-1).astype(np.float64)


def get_cuda_plane(refresh: bool = False) -> CudaPlane:
    global _PLANE
    if _PLANE is None or refresh:
        _PLANE = CudaPlane.detect()
    return _PLANE
