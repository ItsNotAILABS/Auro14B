"""Delta-state attention, recurrent surprise memory, and native multi-sense residuals.

This module is AURO's bounded recurrent memory plane inside the language-model
forward path. It combines two complementary mechanisms:

1. Delta memory: stores normalized hidden-state transitions selected by novelty.
2. Surprise memory: maintains a recurrent state, writes high prediction-error
   states, and retrieves content against the current token representation.

The hybrid result is reinjected before downstream meaning/spectral/physics/neuro
planes. Batch-size-one inference preserves recurrent state across forward calls;
multi-sample batches use isolated ephemeral banks to prevent cross-sample state
contamination.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hashlib
import numpy as np

from .recurrent_memory import RecurrentMemoryConfig, RecurrentSurpriseMemory


@dataclass
class DeltaAttentionReceipt:
    tokens_seen: int = 0
    deltas_kept: int = 0
    deltas_skipped: int = 0
    memory_slots: int = 0
    dense_score_ops: int = 0
    delta_score_ops: int = 0
    novelty_mean: float = 0.0

    def to_dict(self):
        ratio = self.delta_score_ops / max(1, self.dense_score_ops)
        return {
            **self.__dict__,
            "estimated_score_op_ratio": ratio,
            "measured_wall_speedup": False,
        }


class DeltaAttentionEngine:
    """Hybrid recurrent memory over residual transitions and surprise states."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        max_slots: int = 256,
        novelty_threshold: float = 0.08,
        blend: float = 0.05,
        surprise_max_slots: int | None = None,
        surprise_threshold: float = 0.10,
        surprise_blend: float = 0.10,
        surprise_top_k: int = 8,
    ):
        self.hidden_dim = hidden_dim
        self.max_slots = max_slots
        self.novelty_threshold = novelty_threshold
        self.blend = blend
        self.surprise_memory = RecurrentSurpriseMemory(
            hidden_dim,
            RecurrentMemoryConfig(
                max_slots=int(surprise_max_slots or max_slots),
                surprise_threshold=float(surprise_threshold),
                read_strength=float(surprise_blend),
                top_k=int(surprise_top_k),
            ),
        )
        self.reset()

    def reset(self):
        self._slots = np.empty((0, self.hidden_dim), dtype=np.float64)
        self._last = None
        self.receipt = DeltaAttentionReceipt()
        if hasattr(self, "surprise_memory"):
            self.surprise_memory.reset()

    def observe(self, hidden: np.ndarray) -> DeltaAttentionReceipt:
        values = np.asarray(hidden, dtype=np.float64).reshape(-1, self.hidden_dim)
        previous = self._last
        novelties = []
        selected = []
        for value in values:
            delta = value if previous is None else value - previous
            scale = float(np.linalg.norm(value)) + 1e-9
            novelty = float(np.linalg.norm(delta) / scale)
            novelties.append(novelty)
            if previous is None or novelty >= self.novelty_threshold:
                selected.append(delta)
            else:
                self.receipt.deltas_skipped += 1
            previous = value
        self._last = previous.copy() if previous is not None else self._last
        if selected:
            new = np.asarray(selected)
            norms = np.linalg.norm(new, axis=1, keepdims=True) + 1e-9
            self._slots = np.concatenate([self._slots, new / norms], axis=0)[-self.max_slots:]
            self.receipt.deltas_kept += len(selected)
        self.receipt.tokens_seen += len(values)
        self.receipt.memory_slots = len(self._slots)
        self.receipt.dense_score_ops += len(values) * max(1, self.receipt.tokens_seen) * self.hidden_dim
        self.receipt.delta_score_ops += len(values) * max(1, len(self._slots)) * self.hidden_dim
        self.receipt.novelty_mean = float(np.mean(novelties)) if novelties else 0.0
        return self.receipt

    def residual(self, query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=np.float64).reshape(-1, self.hidden_dim)
        if not len(self._slots):
            return np.zeros_like(q)
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        scores = np.clip(qn @ self._slots.T, -30, 30)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True) + 1e-9
        return weights @ self._slots

    def _spawn_ephemeral(self) -> "DeltaAttentionEngine":
        cfg = self.surprise_memory.config
        return DeltaAttentionEngine(
            self.hidden_dim,
            max_slots=self.max_slots,
            novelty_threshold=self.novelty_threshold,
            blend=self.blend,
            surprise_max_slots=cfg.max_slots,
            surprise_threshold=cfg.surprise_threshold,
            surprise_blend=cfg.read_strength,
            surprise_top_k=cfg.top_k,
        )

    def fuse(self, hidden: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Fuse transition memory and recurrent surprise memory causally."""
        values = np.asarray(hidden)
        if values.ndim >= 3 and values.shape[0] > 1:
            outputs = []
            sample_receipts = []
            for sample in values:
                bank = self._spawn_ephemeral()
                sample_output, sample_receipt = bank.fuse(sample)
                outputs.append(sample_output)
                sample_receipts.append(sample_receipt)
            return np.stack(outputs, axis=0), {
                "schema": "auro.delta-attention.hybrid-memory.batch-isolated.v1",
                "batch_isolation": True,
                "batch_size": int(values.shape[0]),
                "persistent_runtime_state_used": False,
                "sample_receipts": sample_receipts,
                "checkpoint_weights_changed": False,
            }

        self.observe(values)
        fused = values.copy()
        delta = self.residual(values[..., -1, :]).reshape(fused[..., -1, :].shape)
        fused[..., -1, :] += self.blend * delta.astype(fused.dtype, copy=False)
        fused, surprise_receipt = self.surprise_memory.fuse(fused)
        receipt = self.receipt.to_dict()
        receipt["schema"] = "auro.delta-attention.hybrid-memory.v2"
        receipt["delta_memory"] = {
            "slots": int(len(self._slots)),
            "max_slots": int(self.max_slots),
            "blend": float(self.blend),
        }
        receipt["surprise_memory"] = surprise_receipt
        receipt["persistent_across_forward_calls"] = True
        receipt["batch_isolation"] = False
        receipt["reset_boundary_explicit"] = True
        receipt["checkpoint_weights_changed"] = False
        return fused, receipt

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "auro.delta-attention.hybrid-memory.v2",
            "delta": self.receipt.to_dict(),
            "delta_slots": int(len(self._slots)),
            "surprise": self.surprise_memory.snapshot(),
            "persistent_across_forward_calls": True,
            "batch_policy": "persistent_for_batch_1_ephemeral_isolated_for_batch_gt_1",
        }


class MultiSenseAdapter:
    """Project text/image/audio/sensor arrays into one MESIE spectral residual."""
    MODALITIES = ("text", "image", "audio", "sensor", "video", "document", "code", "event")

    def __init__(self, hidden_dim: int, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.seed = seed

    def encode(self, modality: str, value: Any) -> np.ndarray:
        if modality not in self.MODALITIES:
            raise ValueError(f"unsupported modality: {modality}")
        if isinstance(value, str):
            raw = np.frombuffer(value.encode("utf-8"), dtype=np.uint8).astype(float)
        else:
            raw = np.asarray(value, dtype=float).reshape(-1)
        if not raw.size:
            return np.zeros(self.hidden_dim)
        raw = (raw - raw.mean()) / (raw.std() + 1e-9)
        spectrum = np.abs(np.fft.rfft(raw, n=max(2, self.hidden_dim * 2)))[:self.hidden_dim]
        digest = hashlib.sha256(f"{self.seed}:{modality}".encode()).digest()
        phase = np.frombuffer(digest, dtype=np.uint8).astype(float)
        phase = np.resize(phase, self.hidden_dim) / 255.0
        vector = spectrum * np.cos(2 * np.pi * phase)
        return vector / (np.linalg.norm(vector) + 1e-9)

    def fuse(self, senses: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        vectors = [self.encode(name, value) for name, value in senses.items()]
        if not vectors:
            return np.zeros(self.hidden_dim), {"modalities": [], "native": True}
        fused = np.mean(vectors, axis=0)
        fused /= np.linalg.norm(fused) + 1e-9
        return fused, {
            "modalities": sorted(senses),
            "native": True,
            "spectral": True,
            "hidden_dim": self.hidden_dim,
        }
