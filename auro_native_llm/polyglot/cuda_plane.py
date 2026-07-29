"""Accelerated compute plane for AURO training.

An explicitly configured browser-WebGPU cluster is preferred, followed by
vendor CUDA/MPS, CuPy, ChaosCUDA and NumPy. The stable matmul API is consumed by
AuroLanguageModel.train_step, so the selected backend accelerates real gradient
matrix products rather than a disconnected demo.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

_PLANE: Optional["CudaPlane"] = None


@dataclass
class CudaPlane:
    backend: str = "numpy"
    device: str = "cpu"
    torch = None
    cupy = None
    cuda_available: bool = False
    device_name: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def detect(cls) -> "CudaPlane":
        plane = cls()
        cluster_url = os.getenv("AURO_WEBGPU_CLUSTER_URL", "").strip()
        if cluster_url:
            try:
                from auro_native_llm.polyglot.webgpu_cluster import WebGPUClusterClient

                client = WebGPUClusterClient(cluster_url, os.getenv("AURO_WEBGPU_CLUSTER_TOKEN", ""), timeout=float(os.getenv("AURO_WEBGPU_CLUSTER_TIMEOUT", "120")))
                status = client.status()
                plane.backend = "webgpu_cluster"
                plane.device = "browser-webgpu-mesh"
                plane.device_name = "AURO browser WebGPU cluster"
                plane.meta["webgpu_cluster"] = status
                plane.meta["hardware_claim_boundary"] = "completed browser-webgpu job receipts are required"
                plane._webgpu = client  # type: ignore[attr-defined]
                return plane
            except Exception as exc:
                plane.meta["webgpu_cluster_error"] = str(exc)[:300]

        try:
            import torch

            plane.torch = torch
            if torch.cuda.is_available():
                plane.backend = "torch_cuda"
                plane.device = "cuda"
                plane.cuda_available = True
                plane.device_name = torch.cuda.get_device_name(0)
                plane.meta["torch"] = torch.__version__
                plane.meta["cuda_version"] = getattr(torch.version, "cuda", None)
                return plane
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                plane.backend = "torch_mps"
                plane.device = "mps"
                plane.device_name = "Apple MPS"
                plane.meta["torch"] = torch.__version__
                return plane
            plane.meta["torch"] = torch.__version__
            plane.meta["torch_note"] = "torch present but no CUDA"
        except Exception as exc:
            plane.meta["torch_error"] = str(exc)[:200]

        try:
            import cupy as cp

            plane.cupy = cp
            _ = cp.cuda.runtime.getDeviceCount()
            plane.backend = "cupy_cuda"
            plane.device = "cuda"
            plane.cuda_available = True
            plane.device_name = "cupy-cuda"
            return plane
        except Exception as exc:
            plane.meta["cupy_error"] = str(exc)[:200]

        try:
            from auro_native_llm.chaos_cuda.plane import get_chaos_cuda

            chaos = get_chaos_cuda()
            plane.backend = "chaos_cuda"
            plane.device = chaos.device
            plane.cuda_available = True
            plane.device_name = chaos.device_name
            plane.meta["chaos"] = chaos.info()
            plane.meta["lab"] = "Novel Chaos Labs"
            plane._chaos = chaos  # type: ignore[attr-defined]
            return plane
        except Exception as exc:
            plane.meta["chaos_error"] = str(exc)[:200]

        plane.backend = "numpy"
        plane.device = "cpu"
        plane.device_name = "numpy"
        plane.meta["blas_threads"] = os.environ.get("OMP_NUM_THREADS") or os.environ.get("OPENBLAS_NUM_THREADS")
        return plane

    def info(self) -> Dict[str, Any]:
        return {"schema": "auro.cuda_plane.v2", "backend": self.backend, "device": self.device, "cuda_available": self.cuda_available, "device_name": self.device_name, "meta": self.meta}

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        webgpu = getattr(self, "_webgpu", None)
        if webgpu is not None and self.backend == "webgpu_cluster":
            return webgpu.matmul(a, b)
        chaos = getattr(self, "_chaos", None)
        if chaos is not None and self.backend == "chaos_cuda":
            return chaos.matmul(a, b)
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if self.torch is not None and self.backend.startswith("torch"):
            t = self.torch
            try:
                dev = t.device("cuda" if self.cuda_available and self.device == "cuda" else ("mps" if self.device == "mps" else "cpu"))
                out = t.as_tensor(a, device=dev) @ t.as_tensor(b, device=dev)
                return out.detach().cpu().numpy().astype(np.float64)
            except Exception:
                pass
        if self.cupy is not None and self.backend == "cupy_cuda":
            try:
                return self.cupy.asnumpy(self.cupy.asarray(a) @ self.cupy.asarray(b)).astype(np.float64)
            except Exception:
                pass
        return a.astype(np.float64) @ b.astype(np.float64)

    def train_step_linear(self, W: np.ndarray, X: np.ndarray, Y: np.ndarray, *, lr: float = 1e-3) -> Dict[str, Any]:
        chaos = getattr(self, "_chaos", None)
        if chaos is not None and self.backend == "chaos_cuda":
            return chaos.train_step_linear(W, X, Y, lr=lr)
        t0 = time.time()
        W = np.asarray(W, dtype=np.float32)
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        if self.torch is not None and self.backend.startswith("torch"):
            t = self.torch
            try:
                dev = t.device("cuda" if self.cuda_available and self.device == "cuda" else ("mps" if self.device == "mps" else "cpu"))
                w = t.nn.Parameter(t.as_tensor(W, device=dev))
                x = t.as_tensor(X, device=dev)
                y = t.as_tensor(Y, device=dev)
                pred = w @ x
                loss = t.mean((pred - y) ** 2)
                loss.backward()
                with t.no_grad():
                    w -= lr * w.grad
                return {"ok": True, "backend": self.backend, "loss": float(loss.detach().cpu()), "W": w.detach().cpu().numpy().astype(np.float64), "sec": time.time() - t0}
            except Exception:
                pass
        pred = self.matmul(W, X)
        err = pred - Y
        loss = float(np.mean(err ** 2))
        grad = (2.0 / max(X.shape[1], 1)) * self.matmul(err, X.T)
        return {"ok": True, "backend": self.backend, "loss": loss, "W": np.asarray(W - lr * grad, dtype=np.float64), "sec": time.time() - t0}

    def spectral_batch_energy(self, batch: np.ndarray) -> np.ndarray:
        x = np.asarray(batch, dtype=np.float32)
        if self.torch is not None and self.cuda_available and self.device == "cuda":
            t = self.torch
            try:
                spec = t.abs(t.fft.rfft(t.as_tensor(x, device="cuda"), dim=-1))
                return spec.sum(dim=-1).detach().cpu().numpy().astype(np.float64)
            except Exception:
                pass
        return np.abs(np.fft.rfft(x, axis=-1)).sum(axis=-1).astype(np.float64)


def get_cuda_plane(refresh: bool = False) -> CudaPlane:
    global _PLANE
    if _PLANE is None or refresh:
        _PLANE = CudaPlane.detect()
    return _PLANE
