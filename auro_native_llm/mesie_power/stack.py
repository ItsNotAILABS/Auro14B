"""Unified MESIE power stack for the autocycle."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from auro_native_llm.mesie_power.compress import MesieCompressor, CompressedBank
from auro_native_llm.mesie_power.multi_embed import MultiMesieEmbedder

_STACK: Optional["MesiePowerStack"] = None


class MesiePowerStack:
    """Everything-MESIE plane used by SENSE / ABSORB / TRAIN."""

    def __init__(
        self,
        *,
        compress_method: str = "hybrid",
        compress_dim: int = 256,
    ) -> None:
        self.embedder = MultiMesieEmbedder()
        self.compressor = MesieCompressor(method=compress_method, target_dim=compress_dim)
        self.bank: Optional[CompressedBank] = None
        self._ann = None
        try:
            from mesie.embeddings.ann import ANNIndex

            self._ann = ANNIndex(metric="cosine", use_lsh=True, lsh_planes=24)
        except Exception:
            self._ann = None

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.embed_text(text)

    def embed_views(self, text: str) -> Dict[str, np.ndarray]:
        return self.embedder.embed_views(text)

    def compress(self, vec: np.ndarray) -> np.ndarray:
        return self.compressor.transform(vec)

    def index_texts(self, texts: List[str], ids: Optional[List[str]] = None) -> Dict[str, Any]:
        ids = ids or [f"t{i}" for i in range(len(texts))]
        mat = self.embedder.embed_batch([t[:4000] for t in texts])
        self.bank = self.compressor.compress_bank(mat, ids, fit=True)
        if self._ann is not None:
            for i, t_id in enumerate(ids):
                self._ann.add(t_id, mat[i])
        return {
            "n": len(texts),
            "embed_dim": self.embedder.dim,
            "compress": self.bank.info() if self.bank else None,
            "ann": self._ann is not None,
        }

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        q = self.embed(query)
        hits: List[Dict[str, Any]] = []
        ann_n = 0
        if self._ann is not None:
            ann_n = self._ann.size() if callable(getattr(self._ann, "size", None)) else int(getattr(self._ann, "size", 0) or 0)
            if not ann_n and hasattr(self._ann, "_ids"):
                ann_n = len(self._ann._ids)
        if self._ann is not None and ann_n > 0:
            for h in self._ann.query(q, top_k=top_k):
                hits.append(
                    {
                        "id": h.item_id,
                        "similarity": h.similarity,
                        "distance": h.distance,
                        "source": "mesie.ann+lsh",
                    }
                )
            return hits
        # fallback: compressed bank cosine
        if self.bank is not None and self.bank.codes.size:
            qc = self.compress(q)
            codes = self.bank.codes
            # normalize
            qn = qc / (np.linalg.norm(qc) + 1e-12)
            cn = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
            scores = cn @ qn
            order = np.argsort(-scores)[:top_k]
            for i in order:
                hits.append(
                    {
                        "id": self.bank.ids[int(i)],
                        "similarity": float(scores[int(i)]),
                        "source": "compressed_bank",
                    }
                )
        return hits

    def absorb_payload(self, text: str, *, kind: str = "power") -> Dict[str, Any]:
        """Pack multi-view + compressed code for trainer Experience.embedding."""
        views = self.embed_views(text)
        fused = views["fused"]
        code = self.compress(fused)
        return {
            "embedding": code.tolist(),
            "fused_dim": int(fused.size),
            "compressed_dim": int(code.size),
            "view_norms": {k: float(np.linalg.norm(v)) for k, v in views.items() if k != "fused"},
            "kind": kind,
            "mesie": True,
        }

    def info(self) -> Dict[str, Any]:
        return {
            "schema": "auro.mesie.power_stack.v1",
            "embedder": self.embedder.info(),
            "compressor": self.compressor.info(),
            "bank": self.bank.info() if self.bank else None,
            "ann": bool(self._ann is not None and (
                (self._ann.size() if callable(getattr(self._ann, "size", None)) else int(getattr(self._ann, "size", 0) or 0))
                or len(getattr(self._ann, "_ids", []))
            )),
            "arsenal": [
                "SpectralVectorizer×2",
                "SpectralFeatureEncoder",
                "Helix",
                "LSH/ANN",
                "MultiMeaning",
                "φ-FFT",
                "SVD hybrid compress",
                "SpectralGPT MoE stack",
            ],
        }


def get_power_stack(**kwargs: Any) -> MesiePowerStack:
    global _STACK
    if _STACK is None:
        _STACK = MesiePowerStack(**kwargs)
    return _STACK
