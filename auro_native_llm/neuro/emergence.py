"""NeuroEmergence Core — fused into AURO as an internal cognitive substrate.

Lineage: FreddyCreates/BRAIN-AI- NeuroEmergence + mesie.cognitive.SpectralNeuroCore.
The bridge now has two neuromorphic lanes:

1. Internal transformer modulation: each MESIE transformer block's proposed
   residual update is regulated before MoE and before the next transformer layer.
2. Output NeuroEmergence residual: the established last-token NeuroEmergence
   residual remains available after the transformer stack.

Both lanes report normalized activity/energy proxies only. They do not claim
biological equivalence or measured physical-energy efficiency.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np

from auro_native_llm.model.phi_math import PHI, GOLDEN_ANGLE_RAD
from auro_native_llm.neuro.internal_modulation import (
    InternalNeuromorphicConfig,
    InternalNeuromorphicModulator,
)
from auro_native_llm.neuro.spiking_gate import SpikingGateConfig, SpikingResidualGate

HEARTBEAT_MS = 873.0


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
                "Auro internal transformer + residual bridge",
            ],
        }


class NeuroEmergenceCore:
    """Local NeuroEmergence unit — SpectralNeuroCore when importable, else phi-core."""

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
                d_model=min(d_model, 256),
                n_attention_heads=min(n_heads, 8),
                memory_capacity=256,
                working_memory_slots=7,
                multi_scale_levels=4,
                enable_cross_band=True,
                enable_harmonics=True,
            )
            self._core = SpectralNeuroCore(cfg)
            rng = np.random.default_rng(42)
            self._proj = rng.standard_normal((d_model, cfg.d_model)).astype(np.float64) * 0.02
            self.state.core_id = cfg.core_id
        except Exception:
            rng = np.random.default_rng(7)
            self._W = rng.standard_normal((n_heads, d_model, d_model)).astype(np.float64) * 0.02
            self._mem: List[np.ndarray] = []

        self._t0 = time.perf_counter()
        self._pulse = 0

    def pulse(self) -> EmergenceState:
        self._pulse += 1
        now = (time.perf_counter() - self._t0) * 1000.0
        err = abs((now / max(self._pulse, 1)) - HEARTBEAT_MS) / HEARTBEAT_MS
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
        h = np.asarray(hidden, dtype=np.float64)
        if h.ndim == 2:
            h = h[np.newaxis, ...]
        _, _, d = h.shape
        spectrum = h[:, -1, :][0]
        if text:
            raw = np.frombuffer(text.encode("utf-8", errors="ignore"), dtype=np.uint8).astype(np.float64)
            if raw.size:
                n = min(raw.size, d)
                spectrum = spectrum.copy()
                spectrum[:n] = 0.7 * spectrum[:n] + 0.3 * (raw[:n] / 255.0)

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
                    if emb_vec.size < self._proj.shape[1]:
                        emb_vec = np.pad(emb_vec, (0, self._proj.shape[1] - emb_vec.size))
                    residual = self._proj @ emb_vec[: self._proj.shape[1]]
                else:
                    residual = emb_vec[:d] if emb_vec.size >= d else np.pad(emb_vec, (0, d - emb_vec.size))
            except Exception:
                residual = self._local_process(spectrum, d)
        else:
            residual = self._local_process(spectrum, d)

        blend = 0.15 * (1.0 / PHI)
        h2 = h.copy()
        h2[:, -1, :] = h2[:, -1, :] + blend * residual[:d]
        norm = np.linalg.norm(h2[:, -1, :], axis=-1, keepdims=True) + 1e-12
        h2[:, -1, :] = h2[:, -1, :] * (
            np.linalg.norm(h[:, -1, :], axis=-1, keepdims=True) + 1e-12
        ) / norm

        self.pulse()
        self.state.attention_entropy = attn_ent
        self.state.harmonic_peaks = peaks
        self.state.memory_hits = mem_hits
        energy = float(np.linalg.norm(residual))
        self.state.coherence = float(
            0.5 * self.state.coherence + 0.5 / (1.0 + abs(energy - 1.0))
        )
        return {
            "hidden": h2 if hidden.ndim == 3 else h2[0],
            "residual": residual[:d],
            "emergence": self.state.to_dict(),
            "blend": blend,
        }

    def _local_process(self, spectrum: np.ndarray, d: int) -> np.ndarray:
        s = spectrum.ravel()
        if s.size < d:
            s = np.pad(s, (0, d - s.size))
        else:
            s = s[:d]
        if hasattr(self, "_W"):
            out = np.mean(
                np.stack([s @ self._W[head] for head in range(self.n_heads)], axis=0),
                axis=0,
            )
        else:
            idx = np.arange(d, dtype=np.float64)
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
            "schema": "auro.neuro.emergence.v3",
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "spectral_neurocore": self._core is not None,
            "state": self.state.to_dict(),
            "heartbeat_ms": HEARTBEAT_MS,
        }


class NeuroBridge:
    """Attach internal neuromorphic regulation and NeuroEmergence to AURO."""

    def __init__(self, language: Any) -> None:
        d = int(getattr(language.config, "hidden_dim", 256))
        heads = int(getattr(language.config, "num_heads", 8))
        self.language = language
        self.core = NeuroEmergenceCore(d_model=d, n_heads=heads)
        extra = dict(getattr(language.config, "extra", {}) or {})
        self.use_spiking_gate = bool(extra.get("use_neuromorphic_residual", False))
        self.spiking_gate = None
        self.internal_modulator = None
        self.internal_transformer_enabled = bool(
            extra.get("use_internal_neuromorphic_transformer", self.use_spiking_gate)
        )

        if self.use_spiking_gate:
            self.spiking_gate = SpikingResidualGate(
                SpikingGateConfig(
                    target_activity=float(extra.get("neuromorphic_target_activity", 0.18)),
                    threshold_quantile=float(extra.get("neuromorphic_threshold_quantile", 0.82)),
                    minimum_gate=float(extra.get("neuromorphic_minimum_gate", 0.35)),
                    energy_penalty_weight=float(extra.get("neuromorphic_energy_penalty_weight", 0.01)),
                    activity_penalty_weight=float(extra.get("neuromorphic_activity_penalty_weight", 0.02)),
                    inhibitory_gain=float(extra.get("neuromorphic_inhibitory_gain", 0.35)),
                )
            )

        if self.internal_transformer_enabled:
            self.internal_modulator = InternalNeuromorphicModulator(
                d,
                InternalNeuromorphicConfig(
                    target_activity=float(extra.get("neuromorphic_target_activity", 0.18)),
                    threshold_quantile=float(extra.get("neuromorphic_threshold_quantile", 0.82)),
                    minimum_gate=float(extra.get("neuromorphic_internal_minimum_gate", 0.45)),
                    inhibitory_gain=float(extra.get("neuromorphic_inhibitory_gain", 0.35)),
                    working_memory_threshold_gain=float(
                        extra.get("neuromorphic_working_memory_threshold_gain", 0.18)
                    ),
                    membrane_decay=float(extra.get("neuromorphic_membrane_decay", 0.88)),
                    residual_mix=float(extra.get("neuromorphic_internal_residual_mix", 1.0)),
                ),
            )
            self._install_internal_transformer_modulation()

        language._neuro = self  # type: ignore[attr-defined]

    def _install_internal_transformer_modulation(self) -> None:
        """Wrap each MESIE transformer block without replacing its implementation."""
        if self.internal_modulator is None:
            return
        for layer_dict in getattr(self.language.core, "layers", []):
            block = layer_dict.get("transformer")
            layer_idx = int(layer_dict.get("idx", getattr(block, "layer_idx", 0)))
            if block is None or getattr(block, "_auro_neuromorphic_wrapped", False):
                continue
            original = block.forward
            block._auro_original_forward = original  # type: ignore[attr-defined]

            def wrapped_forward(*args, __original=original, __layer_idx=layer_idx, **kwargs):
                if not args:
                    return __original(*args, **kwargs)
                block_input = np.asarray(args[0])
                result = __original(*args, **kwargs)
                if not isinstance(result, tuple) or not result:
                    return result
                block_output = np.asarray(result[0])
                modulated, receipt = self.internal_modulator.modulate(
                    block_input,
                    block_output,
                    layer_idx=__layer_idx,
                )
                layer_dict_ref = getattr(self, "_internal_layer_receipts", None)
                if layer_dict_ref is None:
                    self._internal_layer_receipts = {}
                self._internal_layer_receipts[__layer_idx] = asdict(receipt)
                return (modulated, *result[1:])

            block.forward = wrapped_forward  # type: ignore[method-assign]
            block._auro_neuromorphic_wrapped = True  # type: ignore[attr-defined]
            block._auro_neuromorphic_layer_idx = layer_idx  # type: ignore[attr-defined]

    def reset_internal_state(self) -> None:
        if self.internal_modulator is not None:
            self.internal_modulator.reset()
        self._internal_layer_receipts = {}

    def fuse_forward_outputs(self, outputs: Dict[str, Any], text: str = "") -> Dict[str, Any]:
        hidden = outputs.get("last_hidden_state")
        if hidden is None:
            return outputs
        base_hidden = np.asarray(hidden, dtype=np.float64)
        fused = self.core.process_hidden(base_hidden, text=text, store=True)
        h2 = np.asarray(fused["hidden"], dtype=np.float64)

        gate_receipt = None
        if self.spiking_gate is not None:
            base3 = base_hidden if base_hidden.ndim == 3 else base_hidden[np.newaxis, ...]
            fused3 = h2 if h2.ndim == 3 else h2[np.newaxis, ...]
            delta = fused3[:, -1, :] - base3[:, -1, :]
            delta_vector = delta.mean(axis=0)
            gated_delta, gate_receipt = self.spiking_gate.apply(base_hidden, delta_vector)
            gated3 = base3.copy()
            gated3[:, -1, :] = base3[:, -1, :] + gated_delta
            h2 = gated3 if base_hidden.ndim == 3 else gated3[0]

        outputs["last_hidden_state"] = h2
        try:
            outputs["logits"] = np.einsum("...d,dv->...v", h2, self.language.core.lm_head_weight)
        except Exception:
            pass
        outputs["neuro_emergence"] = fused["emergence"]
        outputs["neuro_blend"] = fused["blend"]
        outputs["neuromorphic_residual_enabled"] = self.spiking_gate is not None
        outputs["internal_neuromorphic_transformer_enabled"] = self.internal_modulator is not None
        outputs["internal_neuromorphic_transformer"] = (
            self.internal_modulator.snapshot() if self.internal_modulator is not None else None
        )
        if gate_receipt is not None:
            outputs["neuromorphic_residual"] = asdict(gate_receipt)
            outputs["neuromorphic_regularizer"] = float(gate_receipt.regularizer)
        else:
            outputs["neuromorphic_residual"] = None
            outputs["neuromorphic_regularizer"] = 0.0
        return outputs

    def info(self) -> Dict[str, Any]:
        return {
            "schema": "auro.neuro.bridge.v3",
            "emergence": self.core.info(),
            "neuromorphic_residual_enabled": self.spiking_gate is not None,
            "spiking_gate": self.spiking_gate.manifest() if self.spiking_gate is not None else None,
            "internal_transformer_enabled": self.internal_modulator is not None,
            "internal_transformer": (
                self.internal_modulator.snapshot() if self.internal_modulator is not None else None
            ),
            "internal_position": "after_transformer_block_before_moe_and_next_layer",
            "checkpoint_quality_verified": False,
            "physical_energy_efficiency_verified": False,
        }
