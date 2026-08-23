"""Recurrent surprise-written memory for AURO sequence processing.

This module adds a compact persistent state lane that sits inside the AURO
language-model forward path. It writes only informative hidden-state changes,
retrieves content against the current token representation, and returns a
bounded residual for reinjection before downstream meaning/spectral/neuro
planes.

The design is intentionally checkpoint-compatible at the AuroLanguageModel
surface: memory state is runtime state, not a replacement for transformer
weights. It can be reset between independent sequences or preserved across
turns for recurrent sessions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RecurrentMemoryConfig:
    max_slots: int = 256
    surprise_threshold: float = 0.10
    write_strength: float = 0.35
    read_strength: float = 0.10
    decay: float = 0.995
    top_k: int = 8
    recency_bias: float = 0.04
    norm_epsilon: float = 1e-8


@dataclass
class RecurrentMemoryReceipt:
    tokens_seen: int = 0
    writes: int = 0
    skipped_writes: int = 0
    reads: int = 0
    slots: int = 0
    mean_surprise: float = 0.0
    max_surprise: float = 0.0
    mean_read_gate: float = 0.0
    state_norm: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "schema": "auro.recurrent-surprise-memory.receipt.v1",
            "persistent_runtime_state": True,
            "changes_checkpoint_weights": False,
        }


class RecurrentSurpriseMemory:
    """Bounded content-addressed recurrent memory over hidden states.

    Each token is compared to an exponentially-smoothed recurrent state. The
    normalized prediction error becomes a surprise score. High-surprise states
    are normalized and stored with a scalar strength. Reads use cosine
    similarity plus a small recency prior and return a bounded residual.
    """

    schema = "auro.recurrent-surprise-memory.v1"

    def __init__(self, hidden_dim: int, config: RecurrentMemoryConfig | None = None):
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.config = config or RecurrentMemoryConfig()
        if self.config.max_slots <= 0:
            raise ValueError("max_slots must be positive")
        if self.config.top_k <= 0:
            raise ValueError("top_k must be positive")
        self.reset()

    def reset(self) -> None:
        self._keys = np.empty((0, self.hidden_dim), dtype=np.float64)
        self._values = np.empty((0, self.hidden_dim), dtype=np.float64)
        self._strength = np.empty((0,), dtype=np.float64)
        self._age = np.empty((0,), dtype=np.float64)
        self._state = np.zeros((self.hidden_dim,), dtype=np.float64)
        self._initialized = False
        self.receipt = RecurrentMemoryReceipt()

    @staticmethod
    def _safe_norm(x: np.ndarray, eps: float) -> float:
        return float(np.sqrt(np.sum(np.square(x))) + eps)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return x / self._safe_norm(x, self.config.norm_epsilon)

    def _observe_one(self, value: np.ndarray) -> float:
        value = np.asarray(value, dtype=np.float64).reshape(self.hidden_dim)
        if not self._initialized:
            surprise = 1.0
            prediction_error = value
            self._state = value.copy()
            self._initialized = True
        else:
            prediction_error = value - self._state
            surprise = self._safe_norm(prediction_error, self.config.norm_epsilon) / max(
                self._safe_norm(value, self.config.norm_epsilon),
                self.config.norm_epsilon,
            )
            # Slow recurrent state: stable enough to detect semantic transitions.
            self._state = 0.90 * self._state + 0.10 * value

        self.receipt.tokens_seen += 1
        old_mean = self.receipt.mean_surprise
        n = self.receipt.tokens_seen
        self.receipt.mean_surprise = old_mean + (float(surprise) - old_mean) / n
        self.receipt.max_surprise = max(self.receipt.max_surprise, float(surprise))
        self.receipt.state_norm = self._safe_norm(self._state, self.config.norm_epsilon)

        if surprise >= self.config.surprise_threshold:
            key = self._normalize(value)
            error_unit = self._normalize(prediction_error)
            value_memory = self._normalize(
                (1.0 - self.config.write_strength) * key
                + self.config.write_strength * error_unit
            )
            self._keys = np.concatenate([self._keys, key[None, :]], axis=0)[-self.config.max_slots :]
            self._values = np.concatenate([self._values, value_memory[None, :]], axis=0)[-self.config.max_slots :]
            self._strength = np.concatenate([
                self._strength,
                np.asarray([min(2.0, max(0.0, float(surprise)))], dtype=np.float64),
            ])[-self.config.max_slots :]
            self._age = np.concatenate([self._age + 1.0, np.asarray([0.0])])[-self.config.max_slots :]
            self.receipt.writes += 1
        else:
            if len(self._age):
                self._age += 1.0
            self.receipt.skipped_writes += 1

        if len(self._strength):
            self._strength *= self.config.decay
        self.receipt.slots = int(len(self._keys))
        return float(surprise)

    def observe(self, hidden: np.ndarray) -> np.ndarray:
        values = np.asarray(hidden, dtype=np.float64).reshape(-1, self.hidden_dim)
        surprises = np.zeros((len(values),), dtype=np.float64)
        for idx, value in enumerate(values):
            surprises[idx] = self._observe_one(value)
        return surprises

    def read(self, query: np.ndarray) -> tuple[np.ndarray, float]:
        q = np.asarray(query, dtype=np.float64).reshape(self.hidden_dim)
        if not len(self._keys):
            return np.zeros_like(q), 0.0
        qn = self._normalize(q)
        similarity = self._keys @ qn
        recency = 1.0 / (1.0 + self._age)
        scores = similarity + self.config.recency_bias * recency
        top_k = min(int(self.config.top_k), len(scores))
        indices = np.argpartition(scores, -top_k)[-top_k:]
        selected = scores[indices]
        selected = selected - np.max(selected)
        weights = np.exp(np.clip(selected, -30.0, 0.0)) * np.maximum(self._strength[indices], 1e-6)
        weights = weights / (np.sum(weights) + self.config.norm_epsilon)
        memory = weights @ self._values[indices]
        # Read gate depends on query-to-memory agreement; never exceeds configured strength.
        agreement = float(np.clip(np.dot(qn, self._normalize(memory)), -1.0, 1.0))
        gate = self.config.read_strength * max(0.0, agreement)
        self.receipt.reads += 1
        n = self.receipt.reads
        self.receipt.mean_read_gate += (gate - self.receipt.mean_read_gate) / n
        return memory, float(gate)

    def fuse(self, hidden: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        values = np.asarray(hidden)
        if values.shape[-1] != self.hidden_dim:
            raise ValueError("hidden dimension mismatch")
        fused = values.copy()
        flat = fused.reshape(-1, self.hidden_dim)
        # Observe/read causally in token order: current token can retrieve earlier writes,
        # then becomes eligible to affect future tokens.
        surprises: list[float] = []
        gates: list[float] = []
        for idx in range(len(flat)):
            memory, gate = self.read(flat[idx]) if len(self._keys) else (np.zeros(self.hidden_dim), 0.0)
            if gate > 0.0:
                flat[idx] = flat[idx] + gate * memory.astype(flat.dtype, copy=False)
            gates.append(float(gate))
            surprises.append(self._observe_one(flat[idx]))
        receipt = self.receipt.to_dict()
        receipt["last_surprise"] = surprises[-1] if surprises else 0.0
        receipt["last_read_gate"] = gates[-1] if gates else 0.0
        receipt["max_slots"] = self.config.max_slots
        receipt["top_k"] = self.config.top_k
        return fused, receipt

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "hidden_dim": self.hidden_dim,
            "slots": int(len(self._keys)),
            "max_slots": int(self.config.max_slots),
            "state_norm": self._safe_norm(self._state, self.config.norm_epsilon),
            "receipt": self.receipt.to_dict(),
            "config": asdict(self.config),
            "claim_boundary": {
                "persistent_runtime_state": True,
                "checkpoint_weights_changed": False,
                "trained_memory_quality_verified": False,
            },
        }


__all__ = ["RecurrentMemoryConfig", "RecurrentMemoryReceipt", "RecurrentSurpriseMemory"]
