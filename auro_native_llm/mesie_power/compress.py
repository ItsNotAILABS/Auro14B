"""MESIE-style compression for multi-embeddings and experience banks.

Modes:
  - svd     : truncated SVD / PCA projection (dense compression)
  - topk    : keep top-k magnitude coords (sparse) — like grad top-k
  - phi     : φ-strided subsample + residual
  - hybrid  : SVD then top-k on residual
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from auro_native_llm.model.phi_math import PHI


@dataclass
class CompressedBank:
    """Compressed vector bank for fast retrieval + train sampling."""

    codes: np.ndarray  # [N, k]
    ids: List[str]
    method: str
    original_dim: int
    compressed_dim: int
    components: Optional[np.ndarray] = None  # SVD basis [k, D]
    mean: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            codes=self.codes,
            ids=np.array(self.ids, dtype=object),
            method=self.method,
            original_dim=self.original_dim,
            compressed_dim=self.compressed_dim,
            components=self.components if self.components is not None else np.zeros(0),
            mean=self.mean if self.mean is not None else np.zeros(0),
            meta_json=json.dumps(self.meta),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CompressedBank":
        data = np.load(path, allow_pickle=True)
        comps = data["components"]
        mean = data["mean"]
        return cls(
            codes=data["codes"],
            ids=[str(x) for x in data["ids"].tolist()],
            method=str(data["method"]),
            original_dim=int(data["original_dim"]),
            compressed_dim=int(data["compressed_dim"]),
            components=comps if comps.size else None,
            mean=mean if mean.size else None,
            meta=json.loads(str(data["meta_json"])),
        )

    def reconstruct(self, i: int) -> np.ndarray:
        code = self.codes[i]
        if self.method in ("svd", "hybrid") and self.components is not None:
            mean = self.mean if self.mean is not None else 0.0
            return mean + code @ self.components
        # topk / phi stored as dense already or sparse-filled
        return code

    def info(self) -> Dict[str, Any]:
        n, k = self.codes.shape if self.codes.ndim == 2 else (0, 0)
        ratio = (k / self.original_dim) if self.original_dim else 0.0
        return {
            "n": n,
            "compressed_dim": k,
            "original_dim": self.original_dim,
            "ratio": ratio,
            "method": self.method,
            "bytes": int(self.codes.nbytes),
            "meta": self.meta,
        }


class MesieCompressor:
    """Compress multi-embeddings with MESIE-aligned strategies."""

    def __init__(self, method: str = "hybrid", target_dim: int = 256, topk: float = 0.15) -> None:
        self.method = method
        self.target_dim = target_dim
        self.topk = topk
        self._components: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, matrix: np.ndarray) -> "MesieCompressor":
        X = np.asarray(matrix, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            self._fitted = True
            return self
        self._mean = X.mean(axis=0)
        Xc = X - self._mean
        k = min(self.target_dim, Xc.shape[0], Xc.shape[1])
        # economy SVD
        try:
            # Xc ≈ U S Vt ; components = Vt[:k]
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
            self._components = vt[:k]
        except Exception:
            # fallback: random φ-orthogonal-ish projection
            rng = np.random.default_rng(42)
            proj = rng.standard_normal((k, Xc.shape[1]))
            proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12
            self._components = proj
        self._fitted = True
        return self

    def transform(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float64).ravel()
        if self.method == "topk":
            return self._topk(v)
        if self.method == "phi":
            return self._phi_subsample(v)
        if not self._fitted or self._components is None:
            return v[: self.target_dim] if v.size > self.target_dim else v
        mean = self._mean if self._mean is not None else 0.0
        code = (v - mean) @ self._components.T
        if self.method == "hybrid":
            # residual top-k energy as extra signal folded in
            recon = mean + code @ self._components
            resid = v - recon
            sparse = self._topk(resid)
            # blend: code is primary
            if sparse.size >= code.size:
                code = 0.85 * code + 0.15 * sparse[: code.size]
        return code.astype(np.float64)

    def transform_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return np.stack([self.transform(row) for row in matrix], axis=0)

    def compress_bank(
        self,
        matrix: np.ndarray,
        ids: Sequence[str],
        *,
        fit: bool = True,
    ) -> CompressedBank:
        X = np.asarray(matrix, dtype=np.float64)
        if fit:
            self.fit(X)
        codes = self.transform_matrix(X)
        return CompressedBank(
            codes=codes,
            ids=list(ids),
            method=self.method,
            original_dim=int(X.shape[1]) if X.ndim == 2 else 0,
            compressed_dim=int(codes.shape[1]) if codes.ndim == 2 else 0,
            components=self._components,
            mean=self._mean,
            meta={
                "topk": self.topk,
                "target_dim": self.target_dim,
                "ts": time.time(),
                "compression_ratio": (
                    codes.shape[1] / X.shape[1] if X.ndim == 2 and X.shape[1] else 0
                ),
            },
        )

    def _topk(self, v: np.ndarray) -> np.ndarray:
        k = max(1, int(len(v) * self.topk))
        k = min(k, self.target_dim, len(v))
        out = np.zeros(self.target_dim, dtype=np.float64)
        idx = np.argpartition(np.abs(v), -k)[-k:]
        # pack into fixed target_dim by sorting indices
        idx = np.sort(idx)
        vals = v[idx]
        # store as interleaved or just pad values
        n = min(len(vals), self.target_dim)
        out[:n] = vals[:n]
        return out

    def _phi_subsample(self, v: np.ndarray) -> np.ndarray:
        n = len(v)
        k = min(self.target_dim, n)
        # golden-ratio stride through vector
        step = max(PHI - 1.0, 0.1)
        idx = []
        x = 0.0
        while len(idx) < k:
            i = int(x) % n
            if i not in idx:
                idx.append(i)
            x += step * n / k
            if len(idx) > n:
                break
        while len(idx) < k:
            idx.append(len(idx) % n)
        return v[np.array(idx[:k])]

    def info(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "target_dim": self.target_dim,
            "topk": self.topk,
            "fitted": self._fitted,
            "components": None if self._components is None else list(self._components.shape),
        }
