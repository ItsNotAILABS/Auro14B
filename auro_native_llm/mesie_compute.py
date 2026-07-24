"""MESIE compute plane for Auro native models.

All Auro lanes run compute through MESIE — not external cloud LLM APIs.
Backends (in preference order when available):

1. ``mesie.compute`` torch spectral modules (GPU/CPU autograd graph)
2. ``mesie.foundation`` spectral transformer blocks (NumPy)
3. ``phantom_native.SovereignNeuroCore`` (zero-dep MESIE native)
4. Deterministic spectral-FFT fallback (always)

Training/scheduling hooks use ``mesie.training_fabric``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class ComputeBackendKind(str, Enum):
    """Which MESIE compute path executed the work."""

    TORCH_SPECTRAL = "mesie.compute.torch_spectral"
    FOUNDATION_TRANSFORMER = "mesie.foundation.transformer"
    SOVEREIGN_NEUROCORE = "phantom_native.neurocore"
    SPECTRAL_FFT = "mesie.spectral_fft_fallback"


@dataclass
class MesieComputeProfile:
    """Runnable MESIE profile derived from an Auro architecture target.

    ``parameter_target`` is the family claim (e.g. 14B). Local MESIE compute
    uses ``d_model`` / ``n_layers`` / ``n_heads`` so the model is *native* and
    executable without inventing a multi-billion-parameter checkpoint.
    """

    model_id: str
    parameter_target: int
    tier: str
    d_model: int
    n_layers: int
    n_heads: int
    kv_heads: int
    context_tokens: int
    vocab_hint: int
    bands: int = 16

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "parameter_target": self.parameter_target,
            "tier": self.tier,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "kv_heads": self.kv_heads,
            "context_tokens": self.context_tokens,
            "vocab_hint": self.vocab_hint,
            "bands": self.bands,
            "compute_plane": "MESIE",
            "weight_mode": "mesie-native-runtime-not-trained-checkpoint",
        }


# Scale architecture targets into runnable MESIE profiles (capacity ladder).
_TIER_RUNTIME: Dict[str, Tuple[int, int, int, int]] = {
    # tier: (d_model, n_layers, n_heads, context_cap_local)
    "edge": (256, 4, 8, 2048),
    "specialist": (384, 6, 8, 4096),
    "general": (512, 8, 8, 4096),
    "orchestrator": (640, 10, 10, 8192),
    "frontier": (768, 12, 12, 8192),
}


def profile_from_lane(
    model_id: str,
    parameter_target: int,
    tier: str,
    architecture: Optional[Dict[str, Any]] = None,
) -> MesieComputeProfile:
    """Build a MESIE compute profile from a family lane."""
    arch = architecture or {}
    d_rt, layers_rt, heads_rt, ctx_cap = _TIER_RUNTIME.get(
        tier, (512, 8, 8, 4096)
    )
    # Prefer runtime-scaled dims; keep arch metadata as hints
    d_model = int(arch.get("runtime_d_model", d_rt))
    n_layers = int(arch.get("runtime_layers", layers_rt))
    n_heads = int(arch.get("runtime_heads", heads_rt))
    kv_heads = int(arch.get("kv_heads", max(1, n_heads // 4)))
    if n_heads % kv_heads != 0:
        kv_heads = n_heads
    ctx_target = int(arch.get("context_window_tokens_target", ctx_cap))
    context_tokens = min(ctx_target, ctx_cap)
    vocab_hint = int(arch.get("vocab_size_target", 128000))
    return MesieComputeProfile(
        model_id=model_id,
        parameter_target=parameter_target,
        tier=tier,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        kv_heads=kv_heads,
        context_tokens=context_tokens,
        vocab_hint=vocab_hint,
    )


@dataclass
class MesieForwardResult:
    """Result of one MESIE native forward pass."""

    hidden: np.ndarray
    embedding: List[float]
    spectral_metrics: Dict[str, float]
    backend: ComputeBackendKind
    latency_ms: float
    tokens_used: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hidden_shape": list(self.hidden.shape),
            "embedding_dim": len(self.embedding),
            "spectral_metrics": self.spectral_metrics,
            "backend": self.backend.value,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "metadata": self.metadata,
            "compute_plane": "MESIE",
        }


def text_to_signal(text: str, length: int) -> np.ndarray:
    """Deterministic character/byte signal for MESIE native compute."""
    if length <= 0:
        return np.zeros(1, dtype=np.float64)
    raw = (text or " ").encode("utf-8", errors="replace")
    if not raw:
        raw = b" "
    signal = np.zeros(length, dtype=np.float64)
    for i, b in enumerate(raw[: length * 4]):
        signal[i % length] += (b / 255.0) - 0.5
    # mild high-pass for spectral content
    if length > 2:
        signal = signal - np.mean(signal)
    return signal


def spectral_fft_metrics(signal: np.ndarray, bands: int = 16) -> Dict[str, float]:
    """NumPy spectral metrics (always-on MESIE-compatible diagnostics)."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size < 4:
        x = np.pad(x, (0, 4 - x.size))
    window = np.hanning(x.size)
    spectrum = np.fft.rfft(x * window)
    power = np.abs(spectrum) ** 2
    total = float(power.sum()) + 1e-12
    dist = power / total
    entropy = float(-(dist * np.log(dist + 1e-12)).sum() / math.log(len(dist)))
    freqs = np.linspace(0.0, 1.0, len(power))
    centroid = float((dist * freqs).sum())
    flatness = float(np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-12))
    high = float(power[freqs >= 0.25].sum() / total)
    # band energy
    edges = np.linspace(0.0, 1.0, bands + 1) ** 2
    band_energy = []
    for i in range(bands):
        sel = (freqs >= edges[i]) & (freqs <= edges[i + 1] if i == bands - 1 else freqs < edges[i + 1])
        if not np.any(sel):
            band_energy.append(0.0)
        else:
            band_energy.append(float(power[sel].mean()))
    be = np.array(band_energy, dtype=np.float64)
    be = be / (be.sum() + 1e-12)
    return {
        "spectral_entropy": entropy,
        "spectral_centroid": centroid,
        "spectral_flatness": flatness,
        "high_frequency_ratio": high,
        "band_energy_peak": float(be.max()),
        "band_energy_argmax": float(int(be.argmax())),
    }


class MESIEComputePlane:
    """Unified MESIE compute plane used by every Auro native model.

    Instantiated once per process (or per parent orchestrator). Stateless
    enough to share; NeuroCore instances are cached per profile key.
    """

    def __init__(self) -> None:
        self._cores: Dict[str, Any] = {}
        self._foundation_blocks: Dict[str, Any] = {}
        self._torch_ok: Optional[bool] = None
        self._capabilities = self.probe_capabilities()

    def probe_capabilities(self) -> Dict[str, bool]:
        caps = {
            "sovereign_neurocore": False,
            "foundation_transformer": False,
            "spectral_gpt": False,
            "torch_spectral": False,
            "training_fabric": False,
            "spectral_vectorizer": False,
            "helix": False,
            "intelligence_protocols": False,
            "connectome": False,
            "pretraining": False,
            "miniverse": False,
            "match_validate": False,
            "generation_psd_fas": False,
            "agentic": False,
            "mesie_package": False,
        }
        try:
            import mesie  # noqa: F401

            caps["mesie_package"] = True
        except Exception:
            pass
        try:
            from phantom_native.neurocore import SovereignNeuroCore  # noqa: F401

            caps["sovereign_neurocore"] = True
        except Exception:
            pass
        try:
            from mesie.foundation.models.transformer_blocks import (  # noqa: F401
                SpectralMultiHeadAttention,
                RMSNorm,
            )

            caps["foundation_transformer"] = True
        except Exception:
            pass
        try:
            from mesie.foundation import SpectralGPT  # noqa: F401

            caps["spectral_gpt"] = True
        except Exception:
            pass
        try:
            import torch  # noqa: F401
            from mesie.compute.torch_spectral import MESIESpectralProjector  # noqa: F401

            caps["torch_spectral"] = True
            self._torch_ok = True
        except Exception:
            self._torch_ok = False
        try:
            from mesie.training_fabric import NodeRegistry  # noqa: F401

            caps["training_fabric"] = True
        except Exception:
            pass
        try:
            from mesie.embeddings import SpectralVectorizer  # noqa: F401

            caps["spectral_vectorizer"] = True
        except Exception:
            pass
        try:
            from mesie.helix import HelixEncoder  # noqa: F401

            caps["helix"] = True
        except Exception:
            pass
        try:
            from mesie import IntelligenceProtocol  # noqa: F401

            caps["intelligence_protocols"] = True
        except Exception:
            pass
        try:
            from mesie.connectome import build_default_connectome  # noqa: F401

            caps["connectome"] = True
        except Exception:
            pass
        try:
            from mesie.pretraining import MaskedSpectralModeling  # noqa: F401

            caps["pretraining"] = True
        except Exception:
            pass
        try:
            from mesie.cognitive.miniverse import RecursiveMemoryContainer  # noqa: F401

            caps["miniverse"] = True
        except Exception:
            pass
        try:
            from mesie import load_record, validate_record, match_records  # noqa: F401

            caps["match_validate"] = True
        except Exception:
            pass
        try:
            from mesie.generation import generate_psd, generate_fas  # noqa: F401

            caps["generation_psd_fas"] = True
        except Exception:
            pass
        try:
            from mesie.agentic import AgentSpawner  # noqa: F401

            caps["agentic"] = True
        except Exception:
            pass
        return caps

    @property
    def capabilities(self) -> Dict[str, bool]:
        return dict(self._capabilities)

    def _core_key(self, profile: MesieComputeProfile) -> str:
        return f"{profile.model_id}:{profile.d_model}:{profile.n_heads}"

    def _get_neurocore(self, profile: MesieComputeProfile) -> Any:
        key = self._core_key(profile)
        if key not in self._cores:
            try:
                from phantom_native.neurocore import SovereignNeuroCore

                self._cores[key] = SovereignNeuroCore(
                    {
                        "d_model": profile.d_model,
                        "n_heads": profile.n_heads,
                        "memory_capacity": 32 + profile.n_layers * 4,
                    }
                )
            except Exception:
                self._cores[key] = None
        return self._cores[key]

    def embed_text(self, text: str, profile: MesieComputeProfile) -> List[float]:
        """MESIE-native embedding (installed helix / spectral, not cloud)."""
        # Prefer bound MesieRuntimeStack (pip-installed mesie transformers stack)
        try:
            from auro_native_llm.mesie_runtime import get_mesie_runtime

            rt = get_mesie_runtime(profile.model_id, lite=True)
            if rt.helix_encoder is not None or rt.vectorizer is not None:
                return rt.embed_text(text, dim=profile.d_model)
        except Exception:
            pass
        signal = text_to_signal(text, profile.d_model)
        # Direct HelixEncoder from installed mesie
        if self._capabilities.get("helix"):
            try:
                from mesie.helix import HelixEncoder

                proj = HelixEncoder().encode(signal)
                flat = getattr(proj, "flat_embedding", None)
                if flat is not None:
                    vec = np.asarray(flat, dtype=np.float64).ravel()
                    if vec.size < profile.d_model:
                        pad = np.zeros(profile.d_model, dtype=np.float64)
                        pad[: vec.size] = vec
                        vec = pad
                    else:
                        vec = vec[: profile.d_model]
                    norm = float(np.linalg.norm(vec)) or 1.0
                    return (vec / norm).tolist()
            except Exception:
                pass
        core = self._get_neurocore(profile)
        if core is not None:
            try:
                from phantom_native.sovereign_tensor import SovereignTensor

                out = core.forward(SovereignTensor(data=signal.tolist(), shape=(profile.d_model,)))
                data = list(out.data) if hasattr(out, "data") else signal.tolist()
                vec = np.asarray(data, dtype=np.float64)
                norm = float(np.linalg.norm(vec)) or 1.0
                return (vec / norm).tolist()
            except Exception:
                pass
        # Helix-style fallback
        emb = np.zeros(profile.d_model, dtype=np.float64)
        for i, val in enumerate(signal):
            phase = math.sin(i * 0.1) * math.cos((i % profile.d_model) * 0.1)
            emb[i % profile.d_model] += float(val) * phase
        norm = float(np.linalg.norm(emb)) or 1.0
        return (emb / norm).tolist()

    def forward(
        self,
        text: str,
        profile: MesieComputeProfile,
        *,
        prefer_torch: bool = True,
    ) -> MesieForwardResult:
        """Run a native MESIE forward pass for this Auro lane profile."""
        t0 = time.perf_counter()
        tokens_used = min(len(text or " "), profile.context_tokens)
        signal = text_to_signal(text, profile.d_model)

        # 1) Torch spectral path
        if prefer_torch and self._capabilities.get("torch_spectral"):
            try:
                result = self._forward_torch(signal, profile, tokens_used, t0)
                if result is not None:
                    return result
            except Exception:
                pass

        # 2) Foundation transformer blocks (NumPy)
        if self._capabilities.get("foundation_transformer"):
            try:
                result = self._forward_foundation(signal, text, profile, tokens_used, t0)
                if result is not None:
                    return result
            except Exception:
                pass

        # 3) Sovereign NeuroCore
        if self._capabilities.get("sovereign_neurocore"):
            try:
                result = self._forward_sovereign(signal, profile, tokens_used, t0)
                if result is not None:
                    return result
            except Exception:
                pass

        # 4) Always-on FFT fallback
        metrics = spectral_fft_metrics(signal, bands=profile.bands)
        hidden = signal.reshape(1, -1)
        emb = self.embed_text(text, profile)
        return MesieForwardResult(
            hidden=hidden,
            embedding=emb,
            spectral_metrics=metrics,
            backend=ComputeBackendKind.SPECTRAL_FFT,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            tokens_used=tokens_used,
            metadata={"layers_applied": 0, "plane": "MESIE"},
        )

    def _forward_sovereign(
        self,
        signal: np.ndarray,
        profile: MesieComputeProfile,
        tokens_used: int,
        t0: float,
    ) -> Optional[MesieForwardResult]:
        from phantom_native.sovereign_tensor import SovereignTensor

        core = self._get_neurocore(profile)
        if core is None:
            return None
        data = signal.tolist()
        # Stack layers by repeated resonance passes (MESIE-native depth)
        for _ in range(max(1, profile.n_layers)):
            out = core.forward(SovereignTensor(data=data, shape=(profile.d_model,)))
            data = list(out.data) if hasattr(out, "data") else data
        hidden = np.asarray(data, dtype=np.float64).reshape(1, -1)
        metrics = spectral_fft_metrics(hidden.ravel(), bands=profile.bands)
        emb = hidden.ravel().copy()
        norm = float(np.linalg.norm(emb)) or 1.0
        emb = (emb / norm).tolist()
        return MesieForwardResult(
            hidden=hidden,
            embedding=emb,
            spectral_metrics=metrics,
            backend=ComputeBackendKind.SOVEREIGN_NEUROCORE,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            tokens_used=tokens_used,
            metadata={"layers_applied": profile.n_layers, "plane": "MESIE"},
        )

    def _forward_foundation(
        self,
        signal: np.ndarray,
        text: str,
        profile: MesieComputeProfile,
        tokens_used: int,
        t0: float,
    ) -> Optional[MesieForwardResult]:
        from mesie.foundation.models.transformer_blocks import RMSNorm

        d = profile.d_model
        # Sequence of patches from text signal
        seq_len = min(32, max(4, d // 8))
        patch = np.zeros((seq_len, d), dtype=np.float64)
        base = text_to_signal(text, seq_len * d)
        for i in range(seq_len):
            patch[i] = base[i * d : (i + 1) * d] if (i + 1) * d <= base.size else signal

        norm = RMSNorm(d)
        x = norm.forward(patch)
        # Lightweight multi-layer residual mixing (MESIE foundation primitives)
        for layer in range(profile.n_layers):
            # frequency-aware residual: mix + phase shift
            phase = np.sin(np.arange(d) * (0.05 + layer * 0.01))
            attn = x @ (x.T / math.sqrt(d))
            attn = np.exp(attn - attn.max(axis=-1, keepdims=True))
            attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-12)
            x = norm.forward(x + (attn @ x) * phase)

        hidden = x.mean(axis=0, keepdims=True)
        metrics = spectral_fft_metrics(hidden.ravel(), bands=profile.bands)
        emb = hidden.ravel().copy()
        nrm = float(np.linalg.norm(emb)) or 1.0
        return MesieForwardResult(
            hidden=hidden,
            embedding=(emb / nrm).tolist(),
            spectral_metrics=metrics,
            backend=ComputeBackendKind.FOUNDATION_TRANSFORMER,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            tokens_used=tokens_used,
            metadata={"layers_applied": profile.n_layers, "seq_len": seq_len, "plane": "MESIE"},
        )

    def _forward_torch(
        self,
        signal: np.ndarray,
        profile: MesieComputeProfile,
        tokens_used: int,
        t0: float,
    ) -> Optional[MesieForwardResult]:
        import torch
        from mesie.compute.torch_spectral import MESIESpectralProjector

        # Fake sequence: [1, seq, hidden] from tiled signal
        seq = min(64, max(8, profile.d_model // 4))
        hidden = torch.zeros(1, seq, profile.d_model, dtype=torch.float32)
        sig = torch.tensor(signal, dtype=torch.float32)
        for i in range(seq):
            shift = torch.roll(sig, shifts=i * 3)
            if shift.numel() >= profile.d_model:
                hidden[0, i] = shift[: profile.d_model]
            else:
                pad = torch.zeros(profile.d_model)
                pad[: shift.numel()] = shift
                hidden[0, i] = pad
        projector = MESIESpectralProjector(bands=profile.bands)
        state = projector(hidden)
        metrics = state.detached_metrics()
        pooled = state.pooled_features.detach().cpu().numpy().ravel()
        if pooled.size < profile.d_model:
            emb = np.zeros(profile.d_model, dtype=np.float64)
            emb[: pooled.size] = pooled
        else:
            emb = pooled[: profile.d_model].astype(np.float64)
        nrm = float(np.linalg.norm(emb)) or 1.0
        emb = (emb / nrm).tolist()
        return MesieForwardResult(
            hidden=hidden.detach().cpu().numpy().mean(axis=1),
            embedding=emb,
            spectral_metrics=metrics,
            backend=ComputeBackendKind.TORCH_SPECTRAL,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            tokens_used=tokens_used,
            metadata={"layers_applied": 1, "seq_len": seq, "plane": "MESIE"},
        )

    def discover_node(self, root: str = ".") -> Dict[str, Any]:
        """Discover local MESIE compute node via training fabric."""
        if not self._capabilities.get("training_fabric"):
            return {
                "ok": False,
                "error": "mesie.training_fabric unavailable",
                "compute_plane": "MESIE",
            }
        try:
            from mesie.training_fabric.discovery import discover_compute_node

            node, meta = discover_compute_node(root)
            return {
                "ok": True,
                "node": node.to_dict(),
                "meta": meta,
                "compute_plane": "MESIE",
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "compute_plane": "MESIE"}

    def health(self) -> Dict[str, Any]:
        return {
            "compute_plane": "MESIE",
            "capabilities": self.capabilities,
            "native": True,
            "cloud_llm": False,
        }


# Process-wide shared plane
_PLANE: Optional[MESIEComputePlane] = None


def get_compute_plane() -> MESIEComputePlane:
    global _PLANE
    if _PLANE is None:
        _PLANE = MESIEComputePlane()
    return _PLANE
