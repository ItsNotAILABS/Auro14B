"""Locality-sensitive hashing for compact spectral fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class LSHSignature:
    """Compact bit signature for fast bucket lookup."""

    bits: Tuple[int, ...]
    bucket_key: str
    hyperplane_count: int

    def to_hex(self) -> str:
        n = len(self.bits)
        val = 0
        for i, b in enumerate(self.bits):
            if b:
                val |= 1 << (i % 64)
        return f"{val:016x}_{n}"


class LSHHasher:
    """Random hyperplane LSH (cosine similarity)."""

    def __init__(
        self,
        dim: int,
        n_planes: int = 16,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.n_planes = n_planes
        rng = np.random.default_rng(seed)
        self._planes = rng.standard_normal((n_planes, dim))
        norms = np.linalg.norm(self._planes, axis=1, keepdims=True)
        self._planes /= np.maximum(norms, 1e-12)

    def hash(self, vector: np.ndarray) -> LSHSignature:
        v = np.asarray(vector, dtype=np.float64).ravel()
        if v.shape[0] != self.dim:
            v = np.resize(v, self.dim)
        projections = self._planes @ v
        bits = tuple(1 if p >= 0 else 0 for p in projections)
        bucket_key = "".join(str(b) for b in bits)
        return LSHSignature(bits=bits, bucket_key=bucket_key, hyperplane_count=self.n_planes)


class LSHIndex:
    """Bucket index keyed by LSH signatures."""

    def __init__(self, hasher: LSHHasher) -> None:
        self.hasher = hasher
        self._buckets: Dict[str, List[str]] = {}
        self._vectors: Dict[str, np.ndarray] = {}
        self._signatures: Dict[str, LSHSignature] = {}

    def add(self, item_id: str, vector: np.ndarray) -> LSHSignature:
        sig = self.hasher.hash(vector)
        self._buckets.setdefault(sig.bucket_key, []).append(item_id)
        self._vectors[item_id] = np.asarray(vector, dtype=np.float64).copy()
        self._signatures[item_id] = sig
        return sig

    def candidates_for(self, vector: np.ndarray, *, max_buckets: int = 4) -> List[str]:
        """Return IDs in same bucket; optionally probe flipped bits."""
        sig = self.hasher.hash(vector)
        ids = list(self._buckets.get(sig.bucket_key, []))
        if len(ids) >= 10:
            return ids
        for i in range(min(sig.hyperplane_count, max_buckets)):
            flipped = list(sig.bits)
            flipped[i] = 1 - flipped[i]
            key = "".join(str(b) for b in flipped)
            ids.extend(self._buckets.get(key, []))
        return list(dict.fromkeys(ids))

    @property
    def size(self) -> int:
        return len(self._vectors)