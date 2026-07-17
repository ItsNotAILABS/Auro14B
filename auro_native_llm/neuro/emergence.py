"""NeuroEmergence Core — fused into Auro LM as cognitive substrate.

Lineage: FreddyCreates/BRAIN-AI- NeuroEmergence + mesie.cognitive.SpectralNeuroCore
Heartbeat 873ms, multi-scale attention, TAURUS working/long memory, harmonic peaks.

This is not a wrapper API — it sits in the residual stream of generation/train.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from auro_native_llm.model.phi_math import PHI, GOLDEN_ANGLE_RAD

HEARTBEAT_MS = 873.0  # BRAIN-AI / SOLUS sovereign heartbeat


@dataclass
class EmergenceState:
    coherence: float = 0.0
    pulse: int = 0
    last_ms: float = 0.0
    attention_entropy: float = 0.0
    harmonic_peaks: List[float] = field(default_factory=list)
    memory_hits: int = 0
    core_id: str = "auro_neuro_0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coherence": self.coherence,
            "pulse": self.pulse,
            "last_ms": self.last_ms,
            "attention_entropy": self.attention_entropy,
            "harmonic_peaks": self.harmonic_peaks[:8],
            "memory_hits": self.memory_hits,
            "core_id": self.core_id,
            "heartbeat_ms": HEARTBEAT_MS,
            "phi": PHI,
            "lineage": [
                "FreddyCreates/BRAIN-AI- NeuroEmergence Core",
                "mesie.cognitive.SpectralNeuroCore",
                "Auro LM residual bridge",
            ],
        }


class NeuroEmergenceCore:
    """Local NeuroEmergence unit — SpectralNeuroCore when importable, else pure phi-core."""

    def __init__(self, d_model: int = 256, n_heads: int = 8) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.state = EmergenceState(core_id=f"neuro_{d_model}d")
        self._core = None
        self._proj = None
        try:
            from mesie.cognitive.neurocores import NeuroCoreConfig, SpectralNeuroCore

            cfg = NeuroCoreConfig(
                core_id=self.state.core_id,
                d_model=min(d_model, 256),  # keep TAURUS light
                n_attention_heads=min(n_heads, 8),
                memory_capacity=256,
                working_memory_slots=7,
                multi_scale_levels=4,
                enable_cross_band=True,
                enable_harmonics=True,
            )
            self._core = SpectralNeuroCore(cfg)
            # project neuro d_model → LM hidden
            rng = np.random.default_rng(42)
            self._proj = rng.standard_normal((d_model, cfg.d_model)).astype(np.float64) * 0.02
            self.state.core_id = cfg.core_id
        except Exception:
            # pure local emergence without TAURUS deps
            rng = np.random.default_rng(7)
            self._W = rng.standard_normal((n_heads, d_model, d_model)).astype(np.float64) * 0.02
            self._mem: List[np.ndarray] = []

        self._t0 = time.perf_counter()
        self._pulse = 0

    def pulse(self) -> EmergenceState:
        self._pulse += 1
        now = (time.perf_counter() - self._t0) * 1000.0
        # coherence from pulse regularity vs 873ms target
        target = HEARTBEAT_MS
        err = abs((now / max(self._pulse, 1)) - target) / target
        self.state.pulse = self._pulse
        self.state.last_ms = now
        self.state.coherence = float(1.0 / (1.0 + err))
        return self.state

    def process_hidden(
        self,
        hidden: np.ndarray,
        *,
        text: str = "",
        store: bool = True,
    ) -> Dict[str, Any]:
        """Fuse NeuroEmergence into last-token (or mean) hidden residual.

        hidden: [B, T, D] or [T, D]
        """
        h = np.asarray(hidden, dtype=np.float64)
        if h.ndim == 2:
            h = h[np.newaxis, ...]
        B, T, D = h.shape
        # spectrum from last token + text energy
        last = h[:, -1, :]  # [B, D]
        spectrum = last[0]
        if text:
            raw = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8).astype(
                np.float64
            )
            if raw.size:
                # mix text spectrum into first dims
                n = min(raw.size, D)
                spectrum = spectrum.copy()
                spectrum[:n] = 0.7 * spectrum[:n] + 0.3 * (raw[:n] / 255.0)

        emb_vec = None
        attn_ent = 0.0
        peaks: List[float] = []
        mem_hits = 0

        if self._core is not None:
            try:
                res = self._core.process(
                    spectrum,
                    context={"tag": text[:80], "source": "auro_lm"},
                    store_in_memory=store,
                )
                emb_vec = np.asarray(res.embedding, dtype=np.float64).ravel()
                attn_ent = float(res.attention_analysis.get("entropy", 0.0)) if res.attention_analysis else 0.0
                peaks = list(res.harmonic_peaks or [])[:8]
                mem_hits = len(res.memory_matches or [])
                if self._proj is not None and emb_vec.size:
                    # proj: [D_lm, d_neuro] @ emb
                    if emb_vec.size < self._proj.shape[1]:
                        emb_vec = np.pad(emb_vec, (0, self._proj.shape[1] - emb_vec.size))
                    residual = self._proj @ emb_vec[: self._proj.shape[1]]
                else:
                    residual = emb_vec[:D] if emb_vec.size >= D else np.pad(emb_vec, (0, D - emb_vec.size))
            except Exception:
                residual = self._local_process(spectrum, D)
        else:
            residual = self._local_process(spectrum, D)

        # φ-scaled residual blend into last token
        blend = 0.15 * (1.0 / PHI)
        h2 = h.copy()
        h2[:, -1, :] = h2[:, -1, :] + blend * residual[:D]
        # normalize lightly
        n = np.linalg.norm(h2[:, -1, :], axis=-1, keepdims=True) + 1e-12
        h2[:, -1, :] = h2[:, -1, :] * (np.linalg.norm(h[:, -1, :], axis=-1, keepdims=True) + 1e-12) / n

        self.pulse()
        self.state.attention_entropy = attn_ent
        self.state.harmonic_peaks = peaks
        self.state.memory_hits = mem_hits
        # coherence from residual energy stability
        e = float(np.linalg.norm(residual))
        self.state.coherence = float(0.5 * self.state.coherence + 0.5 / (1.0 + abs(e - 1.0)))

        return {
            "hidden": h2 if hidden.ndim == 3 else h2[0],
            "residual": residual[:D],
            "emergence": self.state.to_dict(),
            "blend": blend,
        }

    def _local_process(self, spectrum: np.ndarray, D: int) -> np.ndarray:
        s = spectrum.ravel()
        if s.size < D:
            s = np.pad(s, (0, D - s.size))
        else:
            s = s[:D]
        if hasattr(self, "_W"):
            outs = []
            for h in range(self.n_heads):
                outs.append(s @ self._W[h])
            out = np.mean(np.stack(outs, axis=0), axis=0)
        else:
            # phi harmonic filter
            idx = np.arange(D, dtype=np.float64)
            out = s * np.cos(idx * GOLDEN_ANGLE_RAD) + np.roll(s, 1) * (1.0 / PHI)
        if not hasattr(self, "_mem"):
            self._mem = []
        self._mem.append(out)
        if len(self._mem) > 64:
            self._mem.pop(0)
        if len(self._mem) > 1:
            out = 0.7 * out + 0.3 * np.mean(np.stack(self._mem[-5:], axis=0), axis=0)
        return out

    def info(self) -> Dict[str, Any]:
        return {
            "schema": "auro.neuro.emergence.v1",
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "spectral_neurocore": self._core is not None,
            "state": self.state.to_dict(),
            "heartbeat_ms": HEARTBEAT_MS,
        }


class NeuroBridge:
    """Attach NeuroEmergence to an AuroLanguageModel instance."""

    def __init__(self, language: Any) -> None:
        d = int(getattr(language.config, "hidden_dim", 256))
        heads = int(getattr(language.config, "num_heads", 8))
        self.language = language
        self.core = NeuroEmergenceCore(d_model=d, n_heads=heads)
        language._neuro = self  # type: ignore[attr-defined]

    def fuse_forward_outputs(self, outputs: Dict[str, Any], text: str = "") -> Dict[str, Any]:
        hidden = outputs.get("last_hidden_state")
        if hidden is None:
            return outputs
        fused = self.core.process_hidden(hidden, text=text, store=True)
        h2 = fused["hidden"]
        outputs["last_hidden_state"] = h2
        # recompute logits
        try:
            outputs["logits"] = np.einsum("...d,dv->...v", h2, self.language.core.lm_head_weight)
        except Exception:
            pass
        outputs["neuro_emergence"] = fused["emergence"]
        outputs["neuro_blend"] = fused["blend"]
        return outputs
