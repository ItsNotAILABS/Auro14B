"""Multi-embedding — every MESIE embed path stacked.

Views:
  1. SpectralVectorizer @ 256 bands (max MESIE spectral)
  2. SpectralFeatureEncoder stats (when record-shaped)
  3. Multi-scale FFT (φ window cascade)
  4. φ-harmonic lattice
  5. n-gram bag
  6. Helix geometry slice (optional)
  7. Meaning residual (Latin/Sanskrit/Nahuatl) when available
  8. LSH signature bits as soft features
  9. MultiMesie second vectorizer @ mid bands (diversity)
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from auro_native_llm.model.phi_math import GOLDEN_ANGLE_RAD, PHI, PHI_INV


class MultiMesieEmbedder:
    """Stack all MESIE-local embedders into one multi-view vector."""

    def __init__(
        self,
        *,
        bands_hi: int = 256,
        bands_mid: int = 64,
        n_fft_scales: int = 5,
        n_phi: int = 160,
        n_ngram: int = 320,
        n_helix: int = 64,
        n_meaning: int = 64,
        n_lsh: int = 32,
        seed: int = 42,
    ) -> None:
        self.bands_hi = bands_hi
        self.bands_mid = bands_mid
        self.n_fft_scales = n_fft_scales
        self.n_phi = n_phi
        self.n_ngram = n_ngram
        self.n_helix = n_helix
        self.n_meaning = n_meaning
        self.n_lsh = n_lsh
        self.seed = seed

        self._vec_hi = None
        self._vec_mid = None
        self._encoder = None
        self._helix = None
        self._meaning = None
        self._lsh = None
        self._views: Dict[str, int] = {}

        try:
            from mesie.embeddings.vectorizers import SpectralVectorizer

            self._vec_hi = SpectralVectorizer(n_bands=bands_hi)
            self._vec_mid = SpectralVectorizer(n_bands=bands_mid)
            self._views["spectral_hi"] = int(self._vec_hi.embedding_dim)
            self._views["spectral_mid"] = int(self._vec_mid.embedding_dim)
        except Exception:
            self._views["spectral_hi"] = bands_hi + 9
            self._views["spectral_mid"] = bands_mid + 9

        try:
            from mesie.embeddings.encoders import SpectralFeatureEncoder

            self._encoder = SpectralFeatureEncoder()
            self._views["encoder"] = 16  # soft upper bound; padded
        except Exception:
            self._views["encoder"] = 16

        self._views["fft"] = n_fft_scales * (64 + 4)
        self._views["phi"] = n_phi
        self._views["ngram"] = n_ngram
        self._views["helix"] = n_helix
        self._views["meaning"] = n_meaning
        self._views["lsh"] = n_lsh

        try:
            from mesie.helix.vector_helix import HelixConfig, VectorHelix

            self._helix = VectorHelix(HelixConfig(embedding_bands=8, max_nodes=128))
        except Exception:
            self._helix = None

        try:
            from auro_native_llm.model.meaning import MultiMeaningField

            self._meaning = MultiMeaningField(n_meaning)
        except Exception:
            self._meaning = None

        try:
            from mesie.embeddings.lsh import LSHHasher

            # dim for LSH set after first full vector; use n_phi as probe dim
            self._lsh = LSHHasher(dim=max(n_phi, 32), n_planes=n_lsh)
        except Exception:
            self._lsh = None

        self.raw_dim = int(sum(self._views.values()))
        self.dim = self.raw_dim

    def embed_text(self, text: str) -> np.ndarray:
        parts = [
            self._spectral_text(text, self._vec_hi, self._views["spectral_hi"]),
            self._spectral_text(text, self._vec_mid, self._views["spectral_mid"]),
            self._encoder_text(text),
            self._multi_fft(text),
            self._phi_features(text),
            self._ngram_bag(text),
            self._helix_slice(text),
            self._meaning_vec(text),
            self._lsh_bits(text),
        ]
        vec = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts])
        if vec.size < self.raw_dim:
            vec = np.pad(vec, (0, self.raw_dim - vec.size))
        elif vec.size > self.raw_dim:
            vec = vec[: self.raw_dim]
        n = float(np.linalg.norm(vec) + 1e-12)
        return (vec / n).astype(np.float64)

    def embed_views(self, text: str) -> Dict[str, np.ndarray]:
        """Named multi-views for training / diagnostics."""
        return {
            "spectral_hi": self._spectral_text(text, self._vec_hi, self._views["spectral_hi"]),
            "spectral_mid": self._spectral_text(text, self._vec_mid, self._views["spectral_mid"]),
            "encoder": self._encoder_text(text),
            "fft": self._multi_fft(text),
            "phi": self._phi_features(text),
            "ngram": self._ngram_bag(text),
            "helix": self._helix_slice(text),
            "meaning": self._meaning_vec(text),
            "lsh": self._lsh_bits(text),
            "fused": self.embed_text(text),
        }

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.stack([self.embed_text(t) for t in texts], axis=0)

    # ---- views ----
    def _text_to_amp(self, text: str, n: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        raw = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8).astype(np.float64)
        if raw.size < 16:
            raw = np.pad(raw, (0, 16 - raw.size))
        if raw.size > n:
            # φ-strided downsample
            idx = np.linspace(0, raw.size - 1, n).astype(int)
            raw = raw[idx]
        freq = np.linspace(1.0, 250.0, raw.size)
        return freq, raw

    def _spectral_text(self, text: str, vectorizer: Any, dim: int) -> np.ndarray:
        freq, amp = self._text_to_amp(text)
        if vectorizer is not None:
            try:
                from mesie.core.records import MultiElementRecord, SpectralComponent

                comp = SpectralComponent(
                    name="text",
                    frequency=freq,
                    amplitude=amp.astype(float),
                )
                rec = MultiElementRecord(record_id="txt", components=[comp])
                v = np.asarray(vectorizer.transform(rec), dtype=np.float64).ravel()
                if v.size < dim:
                    v = np.pad(v, (0, dim - v.size))
                return v[:dim]
            except Exception:
                pass
        spec = np.abs(np.fft.rfft(amp))
        if spec.size < dim:
            spec = np.pad(spec, (0, dim - spec.size))
        return spec[:dim]

    def _encoder_text(self, text: str) -> np.ndarray:
        dim = self._views["encoder"]
        out = np.zeros(dim, dtype=np.float64)
        if self._encoder is None:
            out[0] = len(text) / 1000.0
            return out
        try:
            from mesie.core.records import MultiElementRecord, SpectralComponent

            freq, amp = self._text_to_amp(text, 256)
            rec = MultiElementRecord(
                record_id="enc",
                components=[
                    SpectralComponent(name="t", frequency=freq, amplitude=amp.astype(float))
                ],
            )
            feats = self._encoder.encode(rec)
            vals = list(feats.values()) if isinstance(feats, dict) else list(feats)
            arr = np.asarray(vals, dtype=np.float64).ravel()
            if arr.size < dim:
                arr = np.pad(arr, (0, dim - arr.size))
            return arr[:dim]
        except Exception:
            out[0] = len(text) / 1000.0
            return out

    def _multi_fft(self, text: str) -> np.ndarray:
        raw = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8).astype(np.float64)
        if raw.size < 8:
            raw = np.pad(raw, (0, 8 - raw.size))
        chunks = []
        n = raw.size
        for i in range(self.n_fft_scales):
            win = max(32, int(n / (PHI ** i)))
            seg = raw[:win] if win <= n else np.pad(raw, (0, win - n))
            spec = np.abs(np.fft.rfft(seg))
            bins = 64
            if spec.size >= bins:
                idx = np.linspace(0, spec.size - 1, bins).astype(int)
                b = spec[idx]
            else:
                b = np.pad(spec, (0, bins - spec.size))
            energy = float(np.sum(b) + 1e-12)
            p = b / energy
            ent = float(-np.sum(p * np.log(p + 1e-12)))
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
        out[0] += math.log1p(len(text)) * 0.01
        out[1] += len(set(text.lower())) / max(len(text), 1)
        return out

    def _ngram_bag(self, text: str) -> np.ndarray:
        t = text.lower()
        bag = np.zeros(self.n_ngram, dtype=np.float64)
        for i in range(max(0, len(t) - 2)):
            g = t[i : i + 3]
            hv = int(hashlib.md5(g.encode()).hexdigest(), 16) % self.n_ngram
            bag[hv] += 1.0
        for w in t.split():
            if len(w) < 2:
                continue
            hv = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.n_ngram
            bag[hv] += 1.5
        return bag / float(bag.sum() + 1e-12)

    def _helix_slice(self, text: str) -> np.ndarray:
        out = np.zeros(self.n_helix, dtype=np.float64)
        if self._helix is None:
            # geometric fallback spiral
            for i in range(self.n_helix):
                out[i] = math.sin(i * GOLDEN_ANGLE_RAD + len(text) * PHI_INV)
            return out
        try:
            # store text as node energy
            vec = self._phi_features(text)[:8]
            if hasattr(self._helix, "embed") or hasattr(self._helix, "project"):
                fn = getattr(self._helix, "embed", None) or getattr(self._helix, "project", None)
                if fn:
                    h = np.asarray(fn(vec), dtype=np.float64).ravel()
                    if h.size < self.n_helix:
                        h = np.pad(h, (0, self.n_helix - h.size))
                    return h[: self.n_helix]
            for i in range(self.n_helix):
                out[i] = math.sin(i * GOLDEN_ANGLE_RAD + vec[i % len(vec)])
            return out
        except Exception:
            for i in range(self.n_helix):
                out[i] = math.sin(i * GOLDEN_ANGLE_RAD)
            return out

    def _meaning_vec(self, text: str) -> np.ndarray:
        if self._meaning is None:
            return np.zeros(self.n_meaning, dtype=np.float64)
        try:
            v = np.asarray(self._meaning.embed(text), dtype=np.float64).ravel()
            if v.size < self.n_meaning:
                v = np.pad(v, (0, self.n_meaning - v.size))
            return v[: self.n_meaning]
        except Exception:
            return np.zeros(self.n_meaning, dtype=np.float64)

    def _lsh_bits(self, text: str) -> np.ndarray:
        out = np.zeros(self.n_lsh, dtype=np.float64)
        probe = self._phi_features(text)
        if self._lsh is not None:
            try:
                # ensure dim match
                if probe.size != getattr(self._lsh, "dim", probe.size):
                    d = int(getattr(self._lsh, "dim", probe.size))
                    if probe.size < d:
                        probe = np.pad(probe, (0, d - probe.size))
                    else:
                        probe = probe[:d]
                sig = self._lsh.hash(probe) if hasattr(self._lsh, "hash") else None
                if sig is not None:
                    if hasattr(sig, "bits"):
                        bits = np.asarray(sig.bits, dtype=np.float64).ravel()
                    elif hasattr(sig, "to_hex"):
                        hx = sig.to_hex()
                        bits = np.array(
                            [int(hx[i % len(hx)], 16) / 15.0 for i in range(self.n_lsh)],
                            dtype=np.float64,
                        )
                    else:
                        bits = probe[: self.n_lsh]
                    if bits.size < self.n_lsh:
                        bits = np.pad(bits, (0, self.n_lsh - bits.size))
                    return bits[: self.n_lsh]
            except Exception:
                pass
        # soft hash bits
        h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
        for i in range(self.n_lsh):
            out[i] = ((h[i % len(h)] >> (i % 8)) & 1)
        return out

    def info(self) -> Dict[str, Any]:
        return {
            "schema": "auro.mesie.multi_embed.v1",
            "dim": self.dim,
            "raw_dim": self.raw_dim,
            "views": dict(self._views),
            "mesie_vectorizer_hi": self._vec_hi is not None,
            "mesie_vectorizer_mid": self._vec_mid is not None,
            "encoder": self._encoder is not None,
            "helix": self._helix is not None,
            "meaning": self._meaning is not None,
            "lsh": self._lsh is not None,
            "bands_hi": self.bands_hi,
            "bands_mid": self.bands_mid,
        }
