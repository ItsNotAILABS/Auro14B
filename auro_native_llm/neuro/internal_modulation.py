"""Internal neuromorphic modulation for AURO transformer blocks.

Unlike the output-side NeuroEmergence residual, this controller acts between
MESIE transformer layers. It observes the proposed residual update of each
transformer block, maintains a leaky membrane state per layer, derives sparse
event activity, applies inhibitory homeostasis, and gates the block residual
before MoE and before the next transformer layer see it.

Working-memory compute pressure lowers effective thresholds modestly so unresolved
surprise can recruit more internal activity. Metrics are normalized proxies only;
there is no physical-energy or biological-equivalence claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InternalNeuromorphicConfig:
    membrane_decay: float = 0.88
    target_activity: float = 0.18
    threshold_quantile: float = 0.82
    minimum_gate: float = 0.45
    maximum_gate: float = 1.0
    inhibitory_gain: float = 0.40
    working_memory_threshold_gain: float = 0.18
    residual_mix: float = 1.0
    eps: float = 1e-8


@dataclass(frozen=True)
class InternalNeuromorphicReceipt:
    layer_idx: int
    activity_rate: float
    sparsity: float
    threshold: float
    effective_threshold: float
    inhibitory_tone: float
    gate_mean: float
    residual_rms: float
    membrane_rms: float
    working_memory_pressure: float
    changed_hidden_stream: bool
    physical_energy_claim: bool = False
    biological_equivalence_claim: bool = False


class InternalNeuromorphicModulator:
    """Stateful spiking controller over transformer block residuals."""

    schema = "auro.neuro.internal-transformer.v1"

    def __init__(self, hidden_dim: int, config: InternalNeuromorphicConfig | None = None):
        self.hidden_dim = int(hidden_dim)
        self.config = config or InternalNeuromorphicConfig()
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self._membrane: dict[int, np.ndarray] = {}
        self._receipts: dict[int, InternalNeuromorphicReceipt] = {}
        self.total_calls = 0

    def reset(self) -> None:
        self._membrane.clear()
        self._receipts.clear()
        self.total_calls = 0

    def _working_memory_pressure(self) -> float:
        try:
            from auro_native_llm.model.working_memory import current_compute_pressure

            return float(np.clip(current_compute_pressure(), 0.0, 1.0))
        except Exception:
            return 0.0

    def modulate(
        self,
        block_input: np.ndarray,
        block_output: np.ndarray,
        *,
        layer_idx: int,
    ) -> tuple[np.ndarray, InternalNeuromorphicReceipt]:
        x = np.asarray(block_input, dtype=np.float64)
        y = np.asarray(block_output, dtype=np.float64)
        if x.shape != y.shape or x.shape[-1] != self.hidden_dim:
            raise ValueError("block input/output shape mismatch")

        residual = y - x
        reduce_axes = tuple(range(residual.ndim - 1))
        drive = np.sqrt(np.mean(np.square(residual), axis=reduce_axes))
        membrane = self._membrane.get(int(layer_idx))
        if membrane is None or membrane.shape != drive.shape:
            membrane = np.zeros_like(drive)
        cfg = self.config
        membrane = cfg.membrane_decay * membrane + (1.0 - cfg.membrane_decay) * drive
        self._membrane[int(layer_idx)] = membrane

        raw_rms = float(np.sqrt(np.mean(membrane * membrane)))
        scale = max(raw_rms, cfg.eps)
        normalized = membrane / scale
        if raw_rms <= cfg.eps:
            threshold = 0.0
            events = np.zeros_like(normalized, dtype=bool)
        else:
            threshold = float(np.quantile(normalized, cfg.threshold_quantile))
            pressure = self._working_memory_pressure()
            effective_threshold = threshold * (1.0 - cfg.working_memory_threshold_gain * pressure)
            events = normalized >= effective_threshold
        pressure = self._working_memory_pressure()
        effective_threshold = threshold * (1.0 - cfg.working_memory_threshold_gain * pressure)

        activity_rate = float(events.mean()) if events.size else 0.0
        overload = max(0.0, activity_rate - cfg.target_activity)
        inhibitory_tone = min(
            1.0,
            overload * cfg.inhibitory_gain / max(cfg.target_activity, cfg.eps),
        )
        channel_gate = np.where(events, 1.0 - inhibitory_tone, cfg.minimum_gate)
        channel_gate = np.clip(channel_gate, cfg.minimum_gate, cfg.maximum_gate)
        gate_shape = (1,) * (residual.ndim - 1) + (self.hidden_dim,)
        gated_residual = residual * channel_gate.reshape(gate_shape)
        mix = float(np.clip(cfg.residual_mix, 0.0, 1.0))
        modulated = x + (1.0 - mix) * residual + mix * gated_residual

        residual_rms = float(np.sqrt(np.mean(residual * residual)))
        changed = bool(np.any(np.abs(modulated - y) > 1e-12))
        receipt = InternalNeuromorphicReceipt(
            layer_idx=int(layer_idx),
            activity_rate=round(activity_rate, 8),
            sparsity=round(1.0 - activity_rate, 8),
            threshold=round(float(threshold), 8),
            effective_threshold=round(float(effective_threshold), 8),
            inhibitory_tone=round(float(inhibitory_tone), 8),
            gate_mean=round(float(channel_gate.mean()), 8),
            residual_rms=round(residual_rms, 8),
            membrane_rms=round(raw_rms, 8),
            working_memory_pressure=round(pressure, 8),
            changed_hidden_stream=changed,
        )
        self._receipts[int(layer_idx)] = receipt
        self.total_calls += 1
        return modulated.astype(np.asarray(block_output).dtype, copy=False), receipt

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "hidden_dim": self.hidden_dim,
            "config": asdict(self.config),
            "total_calls": int(self.total_calls),
            "layers_seen": sorted(self._receipts),
            "layer_receipts": {
                str(idx): asdict(receipt) for idx, receipt in sorted(self._receipts.items())
            },
            "acts_between_transformer_layers": True,
            "affects_pre_moe_hidden_state": True,
            "physical_energy_claim": False,
            "biological_equivalence_claim": False,
        }


__all__ = [
    "InternalNeuromorphicConfig",
    "InternalNeuromorphicReceipt",
    "InternalNeuromorphicModulator",
]
