"""Mini Heart — SDK organism vitality pulse (fully local, no 3rd party)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from mesie.sdk.solus.constants import HEARTBEAT_MS, PHI, SOLUS_BRAND


@dataclass
class VitalsSnapshot:
    bpm: float
    pulse_count: int
    coherence: float
    sdk_health: str
    caretaker: str
    elapsed_ms: float
    sovereign: bool = True


class MiniHeart:
    """Rhythm caretaker inside the SDK organism — monitors vitality locally."""

    def __init__(self, caretaker_name: str) -> None:
        self.caretaker_name = caretaker_name
        self._start = time.perf_counter()
        self._pulse_count = 0
        self._last_pulse = self._start
        self._history: List[float] = []

    def pulse(self, sdk_metric: float = 1.0) -> VitalsSnapshot:
        now = time.perf_counter()
        self._pulse_count += 1
        interval_ms = (now - self._last_pulse) * 1000
        self._last_pulse = now
        self._history.append(sdk_metric)
        if len(self._history) > 64:
            self._history.pop(0)

        coherence = 1.0
        if len(self._history) >= 2:
            import numpy as np
            h = np.asarray(self._history, dtype=np.float64)
            coherence = float(1.0 / (1.0 + np.std(h) / (np.mean(h) + 1e-12)))

        bpm = 60000.0 / max(interval_ms, HEARTBEAT_MS * 0.1)
        health = "thriving" if coherence > 0.85 else "stable" if coherence > 0.6 else "watch"

        return VitalsSnapshot(
            bpm=round(bpm, 2),
            pulse_count=self._pulse_count,
            coherence=round(coherence, 4),
            sdk_health=health,
            caretaker=self.caretaker_name,
            elapsed_ms=round((now - self._start) * 1000, 2),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brand": SOLUS_BRAND,
            "caretaker": self.caretaker_name,
            "pulse_count": self._pulse_count,
            "heartbeat_ms": HEARTBEAT_MS,
            "phi": PHI,
            "sovereign": True,
        }