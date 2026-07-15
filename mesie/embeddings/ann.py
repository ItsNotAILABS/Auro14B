"""Approximate nearest neighbor search over spectral fingerprints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mesie.embeddings.lsh import LSHHasher, LSHIndex, LSHSignature


@dataclass
class ANNHit:
    item_id: str
    distance: float
    similarity: float
    lsh_bucket: Optional[str] = None


class ANNIndex:
    """Vector store with LSH pre-filter and exact re-ranking."""

    def __init__(
        self,
        metric: str = "cosine",
        use_lsh: bool = True,
        lsh_planes: int = 16,
    ) -> None:
        self.metric = metric
        self.use_lsh = use_lsh
        self.lsh_planes = lsh_planes
        self._ids: List[str] = []
        self._vectors: Dict[str, np.ndarray] = {}
        self._lsh: Optional[LSHIndex] = None

    def add(self, item_id: str, vector: np.ndarray) -> Optional[LSHSignature]:
        v = np.asarray(vector, dtype=np.float64).ravel()
        self._vectors[item_id] = v
        if item_id not in self._ids:
            self._ids.append(item_id)
        if self.use_lsh:
            if self._lsh is None:
                hasher = LSHHasher(dim=v.shape[0], n_planes=self.lsh_planes)
                self._lsh = LSHIndex(hasher)
            return self._lsh.add(item_id, v)
        return None

    def add_batch(self, items: Sequence[Tuple[str, np.ndarray]]) -> None:
        for item_id, vec in items:
            self.add(item_id, vec)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "cosine":
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12:
                return 1.0
            return 1.0 - float(np.dot(a, b) / (na * nb))
        return float(np.linalg.norm(a - b))

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        *,
        probe_exact: bool = False,
    ) -> List[ANNHit]:
        """Approximate nearest neighbors: LSH candidates then exact rerank."""
        v = np.asarray(vector, dtype=np.float64).ravel()
        if not self._vectors:
            return []

        if probe_exact or not self.use_lsh or self._lsh is None:
            candidate_ids = list(self._vectors.keys())
        else:
            candidate_ids = self._lsh.candidates_for(v)
            if len(candidate_ids) < top_k * 2:
                candidate_ids = list(self._vectors.keys())

        hits: List[ANNHit] = []
        for iid in candidate_ids:
            if iid not in self._vectors:
                continue
            d = self._distance(v, self._vectors[iid])
            hits.append(
                ANNHit(
                    item_id=iid,
                    distance=d,
                    similarity=1.0 - d if self.metric == "cosine" else 1.0 / (1.0 + d),
                    lsh_bucket=self._lsh._signatures[iid].bucket_key if self._lsh and iid in self._lsh._signatures else None,
                )
            )
        hits.sort(key=lambda h: h.distance)
        return hits[: max(top_k, 1)]

    @property
    def size(self) -> int:
        return len(self._vectors)