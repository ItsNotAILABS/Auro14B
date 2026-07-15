"""SDK Solus Organism — hosts math caretakers inside MAESI (for SDK + external users)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from mesie.sdk.solus.constants import HEARTBEAT_MS, LOCAL_ENGINE, SOLUS_BRAND
from mesie.sdk.solus.logic_prover import SolusLogicProver
from mesie.sdk.solus.pattern_forge import SolusPatternForge


@dataclass
class OrganismVitals:
    organism: str
    caretakers: List[str]
    heartbeats: int
    sdk_health: str
    sovereign: bool
    uptime_ms: float


@dataclass
class OrganismCaretakerResult:
    caretaker: str
    action: str
    ok: bool
    data: Dict[str, Any]
    heart: Dict[str, Any]
    brain: Dict[str, Any]


@dataclass
class SDKSolusOrganism:
    """Living SDK organism — Logic Prover + Pattern Forge as autonomous local caretakers."""

    logic: SolusLogicProver = field(default_factory=SolusLogicProver)
    pattern: SolusPatternForge = field(default_factory=SolusPatternForge)
    _start: float = field(default_factory=time.perf_counter)
    _heartbeats: int = 0

    @property
    def caretaker_names(self) -> List[str]:
        return [self.logic.caretaker_id, self.pattern.caretaker_id]

    def pulse(self) -> OrganismVitals:
        self._heartbeats += 1
        lv = self.logic.heart.pulse(0.9)
        pv = self.pattern.heart.pulse(0.85)
        health = "thriving" if lv.coherence > 0.8 and pv.coherence > 0.8 else "stable"
        return OrganismVitals(
            organism="maesi-sdk-organism",
            caretakers=self.caretaker_names,
            heartbeats=self._heartbeats,
            sdk_health=health,
            sovereign=True,
            uptime_ms=round((time.perf_counter() - self._start) * 1000, 2),
        )

    def tend_sdk(self, sdk_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Caretakers watch over SDK internals — called by MAESIClient on run."""
        vitals = self.pulse()
        theorem = f"SDK knowledge load: {sdk_stats.get('technical_concepts', 0)} technical, {sdk_stats.get('research_entries', 0)} research"
        logic = self.logic.caretaker_run("prove", theorem=theorem)
        values = [
            float(sdk_stats.get("physical_laws", 0)),
            float(sdk_stats.get("chemical_elements", 0)),
            float(sdk_stats.get("technical_concepts", 0)),
            float(sdk_stats.get("research_entries", 0)),
            float(sdk_stats.get("speedup_ratio", 1.0)),
        ]
        pattern = self.pattern.caretaker_run("xray", values=values)
        return {
            "organism": vitals.organism,
            "sdk_health": vitals.sdk_health,
            "sovereign": True,
            "brand": SOLUS_BRAND,
            "heartbeat_ms": HEARTBEAT_MS,
            "logic_caretaker": {"ok": logic.ok, "brain": logic.brain, "heart": logic.heart},
            "pattern_caretaker": {"ok": pattern.ok, "brain": pattern.brain, "heart": pattern.heart, "signal_ratio": pattern.data.get("signal_ratio")},
        }

    def logic_action(self, action: str, **kwargs: Any) -> OrganismCaretakerResult:
        r = self.logic.caretaker_run(action, **kwargs)
        return OrganismCaretakerResult(self.logic.caretaker_id, action, r.ok, r.data, r.heart, r.brain)

    def pattern_action(self, action: str, **kwargs: Any) -> OrganismCaretakerResult:
        r = self.pattern.caretaker_run(action, **kwargs)
        return OrganismCaretakerResult(self.pattern.caretaker_id, action, r.ok, r.data, r.heart, r.brain)

    def analyze_spectrum(self, frequencies: Sequence[float], amplitudes: Sequence[float]) -> Dict[str, Any]:
        """Bridge caretakers to MESIE spectral records for external SDK users."""
        xray = self.pattern.caretaker_run("xray", values=list(amplitudes))
        spectral = self.pattern.caretaker_run("spectral", values=list(amplitudes))
        complexity = self.logic.caretaker_run("complexity", problem=f"spectrum_{len(amplitudes)}_points")
        return {
            "xray": xray.data,
            "spectral": spectral.data,
            "complexity": complexity.data,
            "caretakers": self.caretaker_names,
            "engine": LOCAL_ENGINE,
        }