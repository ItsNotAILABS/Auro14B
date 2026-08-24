"""Active fast/slow working memory for AURO.

This controller is intentionally on the model compute path rather than a passive
store. A compact recurrent state survives batch-size-one inference calls,
produces a difficulty signal before the next transformer pass, then updates from
new hidden states after the pass. Fast state tracks immediate context; slow state
consolidates persistent high-surprise structure.

The controller also implements a bounded local predictive-plasticity update for
its vector gates. This is not a claim of end-to-end checkpoint training; it is a
real online self-supervised adaptation mechanism over working-memory parameters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WorkingMemoryConfig:
    fast_decay: float = 0.72
    slow_decay: float = 0.985
    input_gain: float = 0.28
    fast_read_gain: float = 0.08
    slow_read_gain: float = 0.06
    surprise_threshold: float = 0.12
    consolidation_gain: float = 0.10
    pressure_decay: float = 0.90
    local_plasticity_lr: float = 2e-4
    max_parameter_delta: float = 0.02
    epsilon: float = 1e-8


class ActiveWorkingMemory:
    """Persistent active working memory with fast/slow timescales.

    State is persistent for a single live sequence. Callers should use isolated
    controllers for parallel samples; AuroLanguageModel handles that policy.
    """

    schema = "auro.active-working-memory.v1"

    def __init__(self, hidden_dim: int, config: WorkingMemoryConfig | None = None):
        self.hidden_dim = int(hidden_dim)
        self.config = config or WorkingMemoryConfig()
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        # Trainable/local-plastic vector gates. Start close to identity behavior.
        self.input_gate = np.ones(self.hidden_dim, dtype=np.float64)
        self.fast_gate = np.ones(self.hidden_dim, dtype=np.float64)
        self.slow_gate = np.ones(self.hidden_dim, dtype=np.float64)
        self.read_gate = np.ones(self.hidden_dim, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.fast = np.zeros(self.hidden_dim, dtype=np.float64)
        self.slow = np.zeros(self.hidden_dim, dtype=np.float64)
        self.last_hidden = np.zeros(self.hidden_dim, dtype=np.float64)
        self.initialized = False
        self.compute_pressure = 0.0
        self.tokens_seen = 0
        self.consolidations = 0
        self.plasticity_updates = 0
        self.mean_surprise = 0.0
        self.last_surprise = 0.0
        self.last_prediction_error = 0.0

    def _norm(self, x: np.ndarray) -> float:
        return float(np.linalg.norm(x) + self.config.epsilon)

    def _unit(self, x: np.ndarray) -> np.ndarray:
        return x / self._norm(x)

    def prior_difficulty(self, token_shape: tuple[int, ...] | list[int] | int) -> np.ndarray:
        """Difficulty signal used by adaptive MoE before the next core pass."""
        if isinstance(token_shape, int):
            shape = (token_shape,)
        else:
            shape = tuple(int(v) for v in token_shape)
        pressure = float(np.clip(self.compute_pressure, 0.0, 1.0))
        if not shape:
            return np.asarray(pressure, dtype=np.float64)
        # Slightly prioritize later tokens while preserving the recurrent pressure.
        if len(shape) == 1:
            ramp = np.linspace(0.85, 1.0, max(1, shape[0]), dtype=np.float64)
            return np.clip(pressure * ramp, 0.0, 1.0)
        seq = shape[-1]
        ramp = np.linspace(0.85, 1.0, max(1, seq), dtype=np.float64)
        return np.clip(np.broadcast_to(ramp, shape) * pressure, 0.0, 1.0)

    def _local_plasticity(self, hidden: np.ndarray, prediction_error: np.ndarray, surprise: float) -> None:
        """Bounded self-supervised update of memory gates from predictive error."""
        lr = float(self.config.local_plasticity_lr)
        if lr <= 0.0:
            return
        h = self._unit(hidden)
        e = self._unit(prediction_error)
        scale = lr * float(np.clip(surprise, 0.0, 2.0))
        delta = np.clip(scale * h * e, -self.config.max_parameter_delta, self.config.max_parameter_delta)
        self.input_gate = np.clip(self.input_gate + delta, 0.5, 1.5)
        self.fast_gate = np.clip(self.fast_gate + 0.5 * delta, 0.5, 1.5)
        if surprise >= self.config.surprise_threshold:
            self.slow_gate = np.clip(self.slow_gate + 0.25 * delta, 0.5, 1.5)
        self.read_gate = np.clip(self.read_gate + 0.25 * np.abs(delta), 0.5, 1.5)
        self.plasticity_updates += 1

    def step(self, hidden: np.ndarray) -> tuple[np.ndarray, float]:
        x = np.asarray(hidden, dtype=np.float64).reshape(self.hidden_dim)
        prediction = self.fast + self.slow
        error = x - prediction if self.initialized else x
        surprise = self._norm(error) / max(self._norm(x), self.config.epsilon)
        surprise = float(np.clip(surprise, 0.0, 2.0))

        cfg = self.config
        input_term = cfg.input_gain * self.input_gate * self._unit(x)
        self.fast = np.tanh(cfg.fast_decay * self.fast_gate * self.fast + input_term)

        if surprise >= cfg.surprise_threshold:
            gate = cfg.consolidation_gain * min(1.0, surprise)
            candidate = np.tanh(self.slow_gate * self.fast)
            self.slow = cfg.slow_decay * self.slow + gate * candidate
            self.consolidations += 1
        else:
            self.slow *= cfg.slow_decay

        read = (
            cfg.fast_read_gain * self.fast
            + cfg.slow_read_gain * self.slow
        ) * self.read_gate
        read_norm = self._norm(read)
        if read_norm > 1.0:
            read = read / read_norm

        self.compute_pressure = float(np.clip(
            cfg.pressure_decay * self.compute_pressure
            + (1.0 - cfg.pressure_decay) * min(1.0, surprise),
            0.0,
            1.0,
        ))
        self.tokens_seen += 1
        self.mean_surprise += (surprise - self.mean_surprise) / self.tokens_seen
        self.last_surprise = surprise
        self.last_prediction_error = self._norm(error)
        self.last_hidden = x.copy()
        self.initialized = True
        self._local_plasticity(x, error, surprise)
        return read, surprise

    def fuse(self, hidden: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        values = np.asarray(hidden)
        if values.shape[-1] != self.hidden_dim:
            raise ValueError("hidden dimension mismatch")
        fused = values.copy()
        flat = fused.reshape(-1, self.hidden_dim)
        for idx in range(len(flat)):
            read, _ = self.step(flat[idx])
            flat[idx] = flat[idx] + read.astype(flat.dtype, copy=False)
        return fused, self.snapshot()

    def context_vector(self) -> np.ndarray:
        value = self.fast + self.slow
        norm = self._norm(value)
        return value / norm if norm > 1.0 else value.copy()

    def parameter_arrays(self) -> dict[str, np.ndarray]:
        return {
            "working_memory.input_gate": self.input_gate,
            "working_memory.fast_gate": self.fast_gate,
            "working_memory.slow_gate": self.slow_gate,
            "working_memory.read_gate": self.read_gate,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tokens_seen": self.tokens_seen,
            "consolidations": self.consolidations,
            "plasticity_updates": self.plasticity_updates,
            "mean_surprise": float(self.mean_surprise),
            "last_surprise": float(self.last_surprise),
            "last_prediction_error": float(self.last_prediction_error),
            "compute_pressure": float(self.compute_pressure),
            "fast_norm": self._norm(self.fast),
            "slow_norm": self._norm(self.slow),
            "context_norm": self._norm(self.context_vector()),
            "config": asdict(self.config),
            "active": True,
            "fast_slow_timescales": True,
            "online_local_plasticity": True,
        }


__all__ = ["WorkingMemoryConfig", "ActiveWorkingMemory"]
