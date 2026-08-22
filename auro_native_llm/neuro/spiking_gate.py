"""Model-internal sparse spiking gate for AURO NeuroEmergence residuals.

The gate derives event activity from hidden/residual state and returns a bounded
multiplicative residual gate plus explicit regularization metrics. It does not
claim biological equivalence or physical energy efficiency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpikingGateConfig:
    target_activity: float = 0.18
    threshold_quantile: float = 0.82
    minimum_gate: float = 0.35
    maximum_gate: float = 1.0
    energy_penalty_weight: float = 0.01
    activity_penalty_weight: float = 0.02
    inhibitory_gain: float = 0.35
    eps: float = 1e-8


@dataclass(frozen=True)
class SpikingGateReceipt:
    activity_rate: float
    sparsity: float
    threshold: float
    inhibitory_tone: float
    energy_proxy: float
    activity_penalty: float
    energy_penalty: float
    regularizer: float
    gate_mean: float
    physical_energy_claim: bool = False
    biological_equivalence_claim: bool = False


class SpikingResidualGate:
    """Deterministic hidden-state activity gate with no trainable parameters."""

    def __init__(self, config: SpikingGateConfig | None = None):
        self.config = config or SpikingGateConfig()
        if not 0.0 <= self.config.target_activity <= 1.0:
            raise ValueError("target_activity must be in [0, 1]")
        if not 0.0 < self.config.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
        if not 0.0 <= self.config.minimum_gate <= self.config.maximum_gate <= 1.0:
            raise ValueError("gate bounds must satisfy 0 <= min <= max <= 1")

    def apply(self, hidden: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, SpikingGateReceipt]:
        h = np.asarray(hidden, dtype=np.float64)
        r = np.asarray(residual, dtype=np.float64)
        if h.ndim not in (2, 3):
            raise ValueError("hidden must be [T,D] or [B,T,D]")
        if r.ndim == 1:
            residual_vector = r
        elif r.ndim == 2 and r.shape[0] == 1:
            residual_vector = r[0]
        else:
            raise ValueError("residual must be [D] or [1,D]")
        d = h.shape[-1]
        if residual_vector.shape[-1] != d:
            raise ValueError("hidden and residual dimensions must match")

        last = h[-1] if h.ndim == 2 else h[:, -1, :].mean(axis=0)
        magnitude = np.abs(last)
        raw_rms = float(np.sqrt(np.mean(magnitude * magnitude)))
        scale = max(raw_rms, self.config.eps)
        normalized = magnitude / scale
        if raw_rms <= self.config.eps:
            threshold = 0.0
            events = np.zeros_like(normalized, dtype=bool)
        else:
            threshold = float(np.quantile(normalized, self.config.threshold_quantile))
            events = (normalized >= threshold) & (magnitude > self.config.eps)
        activity_rate = float(events.mean())
        sparsity = 1.0 - activity_rate

        overload = max(0.0, activity_rate - self.config.target_activity)
        inhibitory_tone = min(1.0, overload * self.config.inhibitory_gain / max(self.config.target_activity, self.config.eps))
        active_gain = np.where(events, 1.0 - inhibitory_tone, self.config.minimum_gate)
        active_gain = np.clip(active_gain, self.config.minimum_gate, self.config.maximum_gate)

        residual_rms = float(np.sqrt(np.mean(residual_vector * residual_vector)))
        energy_proxy = residual_rms / scale if raw_rms > self.config.eps else 0.0
        activity_penalty = (activity_rate - self.config.target_activity) ** 2
        energy_penalty = energy_proxy * activity_rate
        regularizer = (
            self.config.activity_penalty_weight * activity_penalty
            + self.config.energy_penalty_weight * energy_penalty
        )

        gated = residual_vector * active_gain
        receipt = SpikingGateReceipt(
            activity_rate=round(activity_rate, 8),
            sparsity=round(sparsity, 8),
            threshold=round(threshold, 8),
            inhibitory_tone=round(inhibitory_tone, 8),
            energy_proxy=round(energy_proxy, 8),
            activity_penalty=round(activity_penalty, 8),
            energy_penalty=round(energy_penalty, 8),
            regularizer=round(float(regularizer), 8),
            gate_mean=round(float(active_gain.mean()), 8),
        )
        return gated, receipt

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "auro.neuro.spiking-residual-gate.v1",
            "config": asdict(self.config),
            "trainable_parameters": 0,
            "physical_energy_claim": False,
            "biological_equivalence_claim": False,
        }
