"""Movement engine — spectral trajectory and phase progression (robotics metaphor)."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from mesie.engines.base import Engine
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.internal_api.messages import EngineResponse, MessageEnvelope
from mesie.io.loaders import load_record, RecordInput


class MovementEngine(Engine):
    name = "movement"
    capabilities = ["position", "advance", "trajectory", "velocity_signature"]

    def __init__(self) -> None:
        self._position: float = 0.0
        self._trajectory: List[np.ndarray] = []
        self._vectorizer = SpectralVectorizer()

    def handle(self, message: MessageEnvelope) -> Optional[EngineResponse]:
        if message.target not in (self.name, "*"):
            return None
        action = message.action
        if action not in self.capabilities:
            return EngineResponse(False, self.name, action, error=f"Unknown action: {action}")

        try:
            if action == "position":
                return EngineResponse(True, self.name, action, {"position": self._position, "steps": len(self._trajectory)})

            if action == "advance":
                delta = float(message.payload.get("delta", 0.1))
                record = message.payload.get("record")
                self._position = float(np.clip(self._position + delta, 0.0, 1.0))
                if record is not None:
                    rec = load_record(record)
                    emb = self._vectorizer.transform(rec)
                    shifted = self._phase_shift_embedding(emb, self._position)
                    self._trajectory.append(shifted)
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {"position": self._position, "trajectory_len": len(self._trajectory)},
                )

            if action == "trajectory":
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {
                        "positions": [float(i / max(len(self._trajectory) - 1, 1)) for i in range(len(self._trajectory))],
                        "embeddings": [e.tolist() for e in self._trajectory[-20:]],
                    },
                )

            if action == "velocity_signature":
                rec = load_record(message.payload["record"])
                amp = np.asarray(rec.components[0].amplitude, dtype=np.float64)
                vel = np.gradient(amp)
                return EngineResponse(
                    True,
                    self.name,
                    action,
                    {
                        "mean_velocity": float(np.mean(np.abs(vel))),
                        "peak_velocity": float(np.max(np.abs(vel))),
                    },
                )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            return EngineResponse(False, self.name, action, error=str(exc))

        return EngineResponse(False, self.name, action, error="Unhandled")

    @staticmethod
    def _phase_shift_embedding(embedding: np.ndarray, phase: float) -> np.ndarray:
        rot = np.roll(embedding, int(phase * len(embedding)) % max(len(embedding), 1))
        return rot * (0.5 + 0.5 * phase)