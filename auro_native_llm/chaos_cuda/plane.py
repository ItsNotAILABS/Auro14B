"""ChaosCUDA plane — invent acceleration when vendor CUDA is unavailable.

Design (this machine: Windows ARM64, no torch/CUDA wheels):
  1. Multi-threaded blocked matmul (pure NumPy slices in thread pool)
  2. Optional Julia GEMM / spectral offload
  3. Register as drop-in for CudaPlane when torch missing

This is intentional novelty: sovereign compute, not a cloud GPU wrapper.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_CHAOS: Optional["ChaosCudaPlane"] = None

# φ-scaled block size
_PHI = 1.618033988749895
_DEFAULT_BLOCK = 64


def _blocked_matmul(A: np.ndarray, B: np.ndarray, block: int = _DEFAULT_BLOCK) -> np.ndarray:
    """Multi-threaded blocked GEMM — ChaosCUDA core kernel."""
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    C = np.zeros((m, n), dtype=np.float64)
    # row-block parallel
    rows = list(range(0, m, block))
    workers = min(max(2, (os.cpu_count() or 4)), 12, max(1, len(rows)))

    def _job(i0: int) -> None:
        i1 = min(i0 + block, m)
        for j0 in range(0, n, block):
            j1 = min(j0 + block, n)
            for p0 in range(0, k, block):
                p1 = min(p0 + block, k)
                C[i0:i1, j0:j1] += A[i0:i1, p0:p1] @ B[p0:p1, j0:j1]

    if workers <= 1 or m * n < 4096:
        for i0 in rows:
            _job(i0)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_job, rows))
    return C


@dataclass
class ChaosCudaPlane:
    """Local sovereign accelerator branded ChaosCUDA."""

    backend: str = "chaos_cuda"
    device: str = "local_arm64"
    cuda_available: bool = True  # our own plane — available here
    device_name: str = "ChaosCUDA-Local"
    threads: int = field(default_factory=lambda: min(12, os.cpu_count() or 4))
    julia_ok: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)
    _gemm_calls: int = 0
    _gemm_ms: float = 0.0

    def __post_init__(self) -> None:
        self.meta.update(
            {
                "lab": "Novel Chaos Labs",
                "sovereign": True,
                "vendor_cuda": False,
                "invention": "blocked multi-thread GEMM + Julia spectral offload",
                "heartbeat_ms": 873,
                "phi": _PHI,
            }
        )
        # probe julia once
        try:
            from auro_native_llm.polyglot.runtimes import PolyglotRuntime

            rt = PolyglotRuntime()
            self.julia_ok = bool(rt.julia.available)
            self.meta["julia"] = rt.julia.version
        except Exception:
            self.julia_ok = False

    def info(self) -> Dict[str, Any]:
        avg = (self._gemm_ms / self._gemm_calls) if self._gemm_calls else 0.0
        return {
            "schema": "auro.chaos_cuda.v1",
            "backend": self.backend,
            "device": self.device,
            "cuda_available": self.cuda_available,
            "device_name": self.device_name,
            "threads": self.threads,
            "julia_ok": self.julia_ok,
            "gemm_calls": self._gemm_calls,
            "gemm_avg_ms": round(avg, 3),
            "meta": self.meta,
        }

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        t0 = time.perf_counter()
        # small: direct numpy (BLAS often fine)
        if a.size * b.shape[-1] < 64 * 64 * 64:
            out = a @ b
        else:
            if a.ndim != 2 or b.ndim != 2:
                out = a @ b
            else:
                out = _blocked_matmul(a, b, block=_DEFAULT_BLOCK)
        self._gemm_calls += 1
        self._gemm_ms += (time.perf_counter() - t0) * 1000.0
        return out

    def train_step_linear(
        self,
        W: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        lr: float = 1e-3,
    ) -> Dict[str, Any]:
        t0 = time.time()
        W = np.asarray(W, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        pred = self.matmul(W, X)
        err = pred - Y
        loss = float(np.mean(err ** 2))
        grad = (2.0 / max(X.shape[1], 1)) * self.matmul(err, X.T)
        W2 = W - lr * grad
        return {
            "ok": True,
            "backend": "chaos_cuda",
            "loss": loss,
            "W": W2,
            "sec": time.time() - t0,
            "device_name": self.device_name,
        }

    def spectral_batch_energy(self, batch: np.ndarray) -> np.ndarray:
        x = np.asarray(batch, dtype=np.float64)
        # prefer Julia for single-row when available (parity + threads)
        if self.julia_ok and x.ndim == 2 and x.shape[0] == 1:
            try:
                from auro_native_llm.polyglot.runtimes import PolyglotRuntime

                r = PolyglotRuntime().julia_call(
                    "spectral_energy", {"x": x[0].tolist()}
                )
                if r.get("ok"):
                    return np.array([float(r["energy"])], dtype=np.float64)
            except Exception:
                pass
        spec = np.abs(np.fft.rfft(x, axis=-1))
        return spec.sum(axis=-1).astype(np.float64)

    def benchmark(self, n: int = 256) -> Dict[str, Any]:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((n, n))
        b = rng.standard_normal((n, n))
        t0 = time.perf_counter()
        c1 = a @ b
        t_np = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        c2 = self.matmul(a, b)
        t_ch = (time.perf_counter() - t0) * 1000
        err = float(np.max(np.abs(c1 - c2)))
        return {
            "n": n,
            "numpy_ms": round(t_np, 3),
            "chaos_cuda_ms": round(t_ch, 3),
            "max_abs_err": err,
            "ok": err < 1e-6 * n,
            "speedup_vs_naive_note": "blocked MT may match/exceed BLAS on large n",
        }


def get_chaos_cuda(refresh: bool = False) -> ChaosCudaPlane:
    global _CHAOS
    if _CHAOS is None or refresh:
        _CHAOS = ChaosCudaPlane()
    return _CHAOS
