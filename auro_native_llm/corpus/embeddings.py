"""Max embeddings for GitHub / MESIE corpus.

Stacks every available local embed path so retrieval + training see dense
structure, not a single tiny projection:

  - SpectralVectorizer (max bands)
  - multi-scale FFT text spectra
  - φ-harmonic hash features
  - character / token n-gram bag
  - optional Helix geometry slice
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from auro_native_llm.model.phi_math import PHI, PHI_INV, GOLDEN_ANGLE_RAD


class MaxEmbedder:
    """High-dimensional local embedder (no cloud). Default ~512–1024 dims."""

    def __init__(
        self,
        *,
        n_bands: int = 128,
        n_fft_scales: int = 4,
        n_phi: int = 128,
        n_ngram: int = 256,
        target_dim: int = 0,  # 0 = concat full
        seed: int = 42,
    ) -> None:
        self.n_bands = n_bands
        self.n_fft_scales = n_fft_scales
        self.n_phi = n_phi
        self.n_ngram = n_ngram
        self.target_dim = target_dim
        self.seed = seed
        self._vectorizer = None
        self._spectral_dim = 0
        try:
            from mesie.embeddings.vectorizers import SpectralVectorizer

            self._vectorizer = SpectralVectorizer(n_bands=n_bands)
            self._spectral_dim = int(getattr(self._vectorizer, "embedding_dim", n_bands + 9))
        except Exception:
            self._spectral_dim = n_bands + 9
        # dims: spectral + fft scales * (bands+4) + phi + ngram
        self.raw_dim = (
            self._spectral_dim
            + self.n_fft_scales * (64 + 4)
            + self.n_phi
            + self.n_ngram
        )
        self.dim = self.target_dim if self.target_dim > 0 else self.raw_dim
        rng = np.random.default_rng(seed)
        if self.target_dim > 0 and self.target_dim != self.raw_dim:
            self._proj = rng.standard_normal((self.target_dim, self.raw_dim)).astype(np.float64)
            self._proj *= (2.0 / max(self.raw_dim, 1)) ** 0.5
        else:
            self._proj = None

    def embed_text(self, text: str) -> np.ndarray:
        parts = [
            self._spectral_text(text),
            self._multi_fft(text),
            self._phi_features(text),
            self._ngram_bag(text),
        ]
        vec = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts])
        if vec.size < self.raw_dim:
            vec = np.pad(vec, (0, self.raw_dim - vec.size))
        elif vec.size > self.raw_dim:
            vec = vec[: self.raw_dim]
        if self._proj is not None:
            vec = self._proj @ vec
        n = float(np.linalg.norm(vec) + 1e-12)
        return (vec / n).astype(np.float64)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.stack([self.embed_text(t) for t in texts], axis=0)

    def _spectral_text(self, text: str) -> np.ndarray:
        raw = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8).astype(np.float64)
        if raw.size < 16:
            raw = np.pad(raw, (0, 16 - raw.size))
        # try MESIE SpectralVectorizer via synthetic record
        if self._vectorizer is not None:
            try:
                from mesie.core.records import MultiElementRecord, SpectralComponent

                freq = np.linspace(1.0, 200.0, min(raw.size, 512))
                amp = raw[: freq.size]
                if amp.size < freq.size:
                    amp = np.pad(amp, (0, freq.size - amp.size))
                comp = SpectralComponent(name="text", frequency=freq, amplitude=amp.astype(float))
                rec = MultiElementRecord(record_id="txt", components=[comp])
                v = np.asarray(self._vectorizer.transform(rec), dtype=np.float64).ravel()
                if v.size:
                    if v.size < self._spectral_dim:
                        v = np.pad(v, (0, self._spectral_dim - v.size))
                    return v[: self._spectral_dim]
            except Exception:
                pass
        # fallback spectral stats
        spec = np.abs(np.fft.rfft(raw[:4096]))
        if spec.size < self._spectral_dim:
            spec = np.pad(spec, (0, self._spectral_dim - spec.size))
        return spec[: self._spectral_dim]

    def _multi_fft(self, text: str) -> np.ndarray:
        raw = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8).astype(np.float64)
        if raw.size < 8:
            raw = np.pad(raw, (0, 8 - raw.size))
        chunks = []
        n = raw.size
        for i in range(self.n_fft_scales):
            # golden-ratio window sizes
            win = max(32, int(n / (PHI ** i)))
            seg = raw[:win] if win <= n else np.pad(raw, (0, win - n))
            spec = np.abs(np.fft.rfft(seg))
            # compress to 64 bins
            bins = 64
            if spec.size >= bins:
                idx = np.linspace(0, spec.size - 1, bins).astype(int)
                b = spec[idx]
            else:
                b = np.pad(spec, (0, bins - spec.size))
            # stats
            energy = float(np.sum(b) + 1e-12)
            ent = float(-np.sum((b / energy) * np.log(b / energy + 1e-12)))
            centroid = float(np.sum(np.arange(bins) * b) / energy)
            flat = float(np.exp(np.mean(np.log(b + 1e-12))) / (np.mean(b) + 1e-12))
            chunks.append(np.concatenate([b, np.array([energy, ent, centroid, flat])]))
        return np.concatenate(chunks)

    def _phi_features(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
        out = np.zeros(self.n_phi, dtype=np.float64)
        for i in range(self.n_phi):
            b = h[i % len(h)]
            phase = (i * GOLDEN_ANGLE_RAD) + (b / 255.0) * math.tau
            out[i] = math.sin(phase) * (PHI_INV ** (i % 8)) + math.cos(phase * PHI) * 0.5
        # mix in length / diversity
        out[0] += math.log1p(len(text)) * 0.01
        uniq = len(set(text.lower())) / max(len(text), 1)
        out[1] += uniq
        return out

    def _ngram_bag(self, text: str) -> np.ndarray:
        t = text.lower()
        bag = np.zeros(self.n_ngram, dtype=np.float64)
        # char 3-grams hashed into bag
        for i in range(max(0, len(t) - 2)):
            g = t[i : i + 3]
            hv = int(hashlib.md5(g.encode()).hexdigest(), 16) % self.n_ngram
            bag[hv] += 1.0
        # word unigrams
        for w in t.split():
            if len(w) < 2:
                continue
            hv = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.n_ngram
            bag[hv] += 1.5
        s = float(bag.sum() + 1e-12)
        return bag / s

    def info(self) -> Dict[str, Any]:
        return {
            "schema": "auro.max_embedder.v1",
            "dim": self.dim,
            "raw_dim": self.raw_dim,
            "n_bands": self.n_bands,
            "n_fft_scales": self.n_fft_scales,
            "n_phi": self.n_phi,
            "n_ngram": self.n_ngram,
            "spectral_dim": self._spectral_dim,
            "mesie_vectorizer": self._vectorizer is not None,
            "projected": self._proj is not None,
        }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
