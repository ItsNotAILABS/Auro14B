"""First-class MESIE vector fusion into Auro text hidden states.

Uses SpectralVectorizer + optional Helix geometry so spectral memory is not
bolted on after generation — it participates in the forward residual stream.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from auro_native_llm.model.phi_math import phi_init


class MesieSpectralFuser:
    """Project MESIE embeddings into LM hidden dim and residual-add."""

    def __init__(self, hidden_dim: int, seed: int = 42) -> None:
        self.hidden_dim = hidden_dim
        self._vectorizer = None
        self._helix = None
        try:
            from mesie.embeddings.vectorizers import SpectralVectorizer

            # max local spectral bands (was 16; 128 → ~137-D MESIE vectors)
            self._vectorizer = SpectralVectorizer(n_bands=128)
            src_dim = int(self._vectorizer.embedding_dim)
        except Exception:
            src_dim = 32
        self.proj = phi_init((hidden_dim, src_dim), seed=seed, layer=3) * 0.5
        self.bias = np.zeros(hidden_dim, dtype=np.float64)
        try:
            from mesie.helix.vector_helix import HelixConfig, VectorHelix

            self._helix = VectorHelix(HelixConfig(embedding_bands=8, max_nodes=256))
        except Exception:
            self._helix = None

    def embed_record(self, record: Any) -> np.ndarray:
        if self._vectorizer is None or record is None:
            return np.zeros(self.hidden_dim, dtype=np.float64)
        try:
            vec = np.asarray(self._vectorizer.transform(record), dtype=np.float64).ravel()
        except Exception:
            return np.zeros(self.hidden_dim, dtype=np.float64)
        return self._project(vec)

    def embed_signal(self, frequency: Sequence[float], amplitude: Sequence[float]) -> np.ndarray:
        """Build a minimal MultiElementRecord-like embedding from raw arrays."""
        try:
            from mesie.core.records import MultiElementRecord, SpectralComponent

            comp = SpectralComponent(
                name="signal",
                frequency=np.asarray(frequency, dtype=float),
                amplitude=np.asarray(amplitude, dtype=float),
            )
            rec = MultiElementRecord(record_id="inline", components=[comp])
            return self.embed_record(rec)
        except Exception:
            # FFT features fallback
            amp = np.asarray(amplitude, dtype=np.float64).ravel()
            if amp.size < 4:
                amp = np.pad(amp, (0, 4 - amp.size))
            spec = np.abs(np.fft.rfft(amp))
            return self._project(spec)

    def embed_text_as_spectrum(self, text: str) -> np.ndarray:
        """Physics spectrum embed: dispersion lattice + Wiener energy + Landau fixed point."""
        try:
            from auro_native_llm.physics import get_physics_engine

            eng = get_physics_engine()
            vec = eng.embed_physics(text, self.hidden_dim)
            # residual through learned proj when vectorizer path unavailable
            return vec.astype(np.float64)
        except Exception:
            raw = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8).astype(np.float64)
            if raw.size < 8:
                raw = np.pad(raw, (0, 8 - raw.size))
            freq = np.linspace(1.0, 100.0, raw.size)
            return self.embed_signal(freq, raw)

    def _project(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float64).ravel()
        src = self.proj.shape[1]
        if v.size < src:
            pad = np.zeros(src, dtype=np.float64)
            pad[: v.size] = v
            v = pad
        elif v.size > src:
            # pool
            idx = np.linspace(0, v.size, src, endpoint=False).astype(int)
            v = v[idx]
        out = self.proj @ v + self.bias
        n = float(np.linalg.norm(out)) or 1.0
        return out / n

    def fuse_hidden(
        self,
        hidden: np.ndarray,
        spectral_vec: np.ndarray,
        blend: float = 0.1,
    ) -> np.ndarray:
        """Physics residual fuse: Landau force + spectral proj (not flat add)."""
        h = np.asarray(hidden, dtype=np.float64)
        s = np.asarray(spectral_vec, dtype=np.float64).ravel()
        if s.size != h.shape[-1]:
            if s.size < h.shape[-1]:
                pad = np.zeros(h.shape[-1])
                pad[: s.size] = s
                s = pad
            else:
                s = s[: h.shape[-1]]
        try:
            from auro_native_llm.physics.formulas import landau_field_force, phi_schrodinger_step

            out = h.copy()
            # last-token order parameter pulled toward spectral external field
            tail = out[..., -1, :]
            # batch loop
            if tail.ndim == 1:
                force = landau_field_force(tail / (np.linalg.norm(tail) + 1e-12), s)
                out[..., -1, :] = tail + blend * force * (np.linalg.norm(tail) + 1e-12)
                out[..., -1, :] = phi_schrodinger_step(out[..., -1, :], dt=0.03)
            else:
                for b in range(tail.shape[0]):
                    tb = tail[b]
                    n = float(np.linalg.norm(tb)) or 1.0
                    force = landau_field_force(tb / n, s)
                    out[b, -1, :] = tb + blend * force * n
                    out[b, -1, :] = phi_schrodinger_step(out[b, -1, :], dt=0.03)
            out += s * (blend * 0.15)
            return out
        except Exception:
            add = np.zeros_like(h)
            add[..., -1, :] = s * blend
            add[..., :, :] += s * (blend * 0.25)
            return h + add
