"""Mini Brain — local autonomous reasoning core (zero 3rd-party inference)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from mesie.sdk.solus.constants import LOCAL_ENGINE, PHI


@dataclass
class BrainThought:
    conclusion: str
    confidence: float
    evidence: List[str]
    engine: str = LOCAL_ENGINE


class MiniBrain:
    """Compact local reasoner — caretaker cognition inside the SDK organism."""

    def __init__(self, domain: str) -> None:
        self.domain = domain
        self._memory: List[BrainThought] = []

    def reason(self, context: Dict[str, Any]) -> BrainThought:
        score = float(context.get("score", context.get("confidence", 0.5)))
        metric = float(context.get("metric", score))
        complexity = float(context.get("complexity", 0.3))

        evidence = []
        if score >= 0.8:
            conclusion = f"{self.domain}: strong local signal — caretaker approves"
            confidence = min(0.98, score * PHI / (PHI + 1))
            evidence.append("high composite confidence")
        elif score >= 0.5:
            conclusion = f"{self.domain}: moderate signal — monitor via mini heart"
            confidence = score * 0.9
            evidence.append("mid-range confidence band")
        else:
            conclusion = f"{self.domain}: weak signal — invoke helper caretaker"
            confidence = max(0.2, score)
            evidence.append("low confidence; escalate locally")

        if complexity > 0.7:
            evidence.append("high complexity — phi-scaled caution")
            confidence *= 0.92

        thought = BrainThought(conclusion=conclusion, confidence=round(confidence, 4), evidence=evidence)
        self._memory.append(thought)
        if len(self._memory) > 128:
            self._memory.pop(0)
        return thought

    def embed_context(self, values: List[float]) -> np.ndarray:
        """Local context vector — no external model."""
        arr = np.asarray(values[:32], dtype=np.float64)
        if arr.size == 0:
            arr = np.zeros(8)
        pad = np.zeros(max(0, 16 - arr.size))
        vec = np.concatenate([arr[:16], pad])[:16]
        vec[0] = self.domain.__hash__() % 1000 / 1000.0
        vec[1] = PHI
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-12)