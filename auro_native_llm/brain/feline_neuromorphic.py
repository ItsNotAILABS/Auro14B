"""Feline-inspired event-driven neural dynamics for HIM.

This module borrows *computational principles* associated with mammalian/feline
sensory systems: hierarchical feed-forward drive, recurrent context, fast
orienting responses, sparse event-driven firing, inhibitory competition,
adaptive thresholds, synaptic traces, and homeostatic energy regulation.
It is not a biological cat-brain simulation and does not imply neuroscience
or consciousness equivalence.

All energy values are normalized compute-energy units (CEU), not joules. A
hardware backend may later calibrate CEU against measured wall-power telemetry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class NeuromorphicConfig:
    leak: float = 0.82
    recurrent_gain: float = 0.22
    feedforward_gain: float = 0.52
    inhibitory_gain: float = 0.20
    base_threshold: float = 0.58
    threshold_adaptation: float = 0.08
    threshold_recovery: float = 0.94
    refractory_cycles: int = 1
    trace_decay: float = 0.78
    plasticity_rate: float = 0.025
    plasticity_decay: float = 0.997
    target_spike_rate: float = 0.16
    spike_energy_ceu: float = 1.0
    integration_energy_ceu: float = 0.08
    synaptic_event_energy_ceu: float = 0.035
    plasticity_energy_ceu: float = 0.12
    idle_energy_ceu: float = 0.01
    energy_budget_ceu: float = 12.0
    orienting_gain: float = 0.18
    orienting_regions: tuple[str, ...] = ("SC", "THL_L", "THL_R", "LC", "V1")


@dataclass(frozen=True)
class Synapse:
    source: str
    target: str
    weight: float
    kind: str = "excitatory"
    plastic: bool = True
    pathway: str = "association"


@dataclass(frozen=True)
class SpikeRegionState:
    membrane: float
    threshold: float
    trace: float
    spike: int
    refractory: int
    synaptic_gain: float
    energy_ceu: float


@dataclass(frozen=True)
class NeuromorphicCycle:
    cycle: int
    spike_count: int
    spike_rate: float
    sparsity: float
    inhibitory_tone: float
    synaptic_events: int
    excitatory_current: float
    inhibitory_current: float
    energy_ceu: float
    energy_budget_ceu: float
    energy_pressure: float
    orienting_burst: bool
    active_regions: tuple[str, ...]
    receipt_hash: str


def _bound(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return low


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def default_feline_synapses(region_ids: Iterable[str]) -> tuple[Synapse, ...]:
    """Return a bounded engineering connectome over available HIM region names.

    The graph emphasizes feline-inspired sensory hierarchy and rapid orienting
    motifs while preserving the repository's established 44-region namespace.
    Missing regions are simply omitted, making the graph usable in focused tests.
    """
    available = set(region_ids)
    candidates = (
        # visual hierarchy / identity path
        Synapse("V1", "V2V3", 0.78, pathway="visual_feedforward"),
        Synapse("V2V3", "ITG_L", 0.62, pathway="visual_feedforward"),
        Synapse("V2V3", "FFA", 0.58, pathway="visual_feedforward"),
        Synapse("ITG_L", "PPC_L", 0.38, pathway="visual_association"),
        Synapse("FFA", "PPC_R", 0.34, pathway="visual_association"),
        # fast orienting / thalamic relay
        Synapse("SC", "THL_L", 0.72, pathway="orienting"),
        Synapse("SC", "THL_R", 0.72, pathway="orienting"),
        Synapse("THL_L", "V1", 0.64, pathway="orienting"),
        Synapse("THL_R", "V1", 0.64, pathway="orienting"),
        Synapse("SC", "LC", 0.55, pathway="orienting"),
        Synapse("LC", "V1", 0.34, pathway="arousal_gain"),
        Synapse("LC", "DLPFC_R", 0.30, pathway="arousal_gain"),
        # sensory -> executive integration
        Synapse("PPC_L", "DLPFC_L", 0.46, pathway="executive_integration"),
        Synapse("PPC_R", "DLPFC_R", 0.46, pathway="executive_integration"),
        Synapse("STG_L", "DLPFC_L", 0.40, pathway="language_integration"),
        Synapse("WER", "DLPFC_L", 0.44, pathway="language_integration"),
        Synapse("DLPFC_L", "ACC", 0.42, pathway="executive_recurrent"),
        Synapse("ACC", "DLPFC_L", 0.36, pathway="executive_recurrent"),
        Synapse("DLPFC_R", "PPC_R", 0.31, pathway="attention_feedback"),
        # memory / salience recurrence
        Synapse("HPC_L", "PCC", 0.38, pathway="memory_context"),
        Synapse("HPC_R", "DLPFC_L", 0.34, pathway="memory_context"),
        Synapse("AMY_R", "ACC", 0.35, pathway="salience"),
        Synapse("INS_L", "ACC", 0.32, pathway="interoceptive"),
        # action-selection path
        Synapse("DLPFC_L", "CAU", 0.36, pathway="action_selection"),
        Synapse("CAU", "GP", 0.42, pathway="action_selection"),
        Synapse("GP", "M1_L", 0.30, pathway="action_selection"),
        Synapse("SMA", "M1_L", 0.46, pathway="motor"),
        # explicit inhibitory competition / conflict damping
        Synapse("ACC", "M1_L", 0.28, kind="inhibitory", pathway="conflict_inhibition"),
        Synapse("GP", "SMA", 0.25, kind="inhibitory", pathway="action_inhibition"),
        Synapse("AMY_L", "DLPFC_L", 0.20, kind="inhibitory", pathway="threat_brake"),
    )
    return tuple(s for s in candidates if s.source in available and s.target in available)


class FelineNeuromorphicEngine:
    """Deterministic sparse-spiking controller over named brain regions.

    The engine is deliberately small and inspectable. It consumes normalized
    region drives and emits sparse spike events plus synaptic, energy, and
    plasticity state. It never mutates model weights and therefore cannot be used
    as evidence that a language-model checkpoint learned these dynamics.
    """

    schema = "him.feline-neuromorphic.v2"

    def __init__(
        self,
        region_ids: Iterable[str],
        config: NeuromorphicConfig | None = None,
        synapses: Iterable[Synapse] | None = None,
    ):
        self.config = config or NeuromorphicConfig()
        self.region_ids = tuple(dict.fromkeys(str(value) for value in region_ids))
        if not self.region_ids:
            raise ValueError("at least one region is required")
        self.synapses = tuple(synapses) if synapses is not None else default_feline_synapses(self.region_ids)
        known = set(self.region_ids)
        for synapse in self.synapses:
            if synapse.source not in known or synapse.target not in known:
                raise ValueError(f"synapse references unknown region: {synapse}")
            if synapse.kind not in {"excitatory", "inhibitory"}:
                raise ValueError(f"unsupported synapse kind: {synapse.kind}")
            if synapse.weight < 0:
                raise ValueError("synapse weights must be non-negative; use kind for inhibition")
        self.cycle_number = 0
        self.membrane = {region: 0.0 for region in self.region_ids}
        self.threshold = {region: self.config.base_threshold for region in self.region_ids}
        self.trace = {region: 0.0 for region in self.region_ids}
        self.refractory = {region: 0 for region in self.region_ids}
        self.synaptic_gain = {region: 1.0 for region in self.region_ids}
        self.edge_gain = {(s.source, s.target, s.kind, s.pathway): 1.0 for s in self.synapses}
        self.previous_hash = "0" * 64
        self.total_energy_ceu = 0.0

    def _synaptic_currents(self) -> tuple[dict[str, float], dict[str, float], int]:
        excitatory = {region: 0.0 for region in self.region_ids}
        inhibitory = {region: 0.0 for region in self.region_ids}
        events = 0
        for synapse in self.synapses:
            source_trace = self.trace[synapse.source]
            if source_trace <= 1e-9:
                continue
            key = (synapse.source, synapse.target, synapse.kind, synapse.pathway)
            current = source_trace * synapse.weight * self.edge_gain[key]
            if synapse.kind == "excitatory":
                excitatory[synapse.target] += current
            else:
                inhibitory[synapse.target] += current
            events += 1
        return excitatory, inhibitory, events

    def cycle(
        self,
        drives: Mapping[str, float],
        *,
        salience: float = 0.5,
        novelty: float = 0.0,
        energy_budget_ceu: float | None = None,
    ) -> NeuromorphicCycle:
        cfg = self.config
        budget = max(0.001, float(energy_budget_ceu if energy_budget_ceu is not None else cfg.energy_budget_ceu))
        salience = _bound(salience)
        novelty = _bound(novelty)
        population_trace = sum(self.trace.values()) / len(self.region_ids)
        inhibitory_tone = _bound(
            cfg.inhibitory_gain * population_trace
            + max(0.0, population_trace - cfg.target_spike_rate) * 0.8
        )
        orienting_burst = novelty >= 0.62 or (novelty >= 0.45 and salience >= 0.72)
        excitatory_current, inhibitory_current, synaptic_events = self._synaptic_currents()

        spikes: list[str] = []
        cycle_energy = cfg.idle_energy_ceu * len(self.region_ids) + synaptic_events * cfg.synaptic_event_energy_ceu
        region_plastic_updates = 0

        for region in self.region_ids:
            drive = _bound(drives.get(region, 0.0))
            if orienting_burst and region in cfg.orienting_regions:
                drive = _bound(drive + cfg.orienting_gain * (0.5 + novelty * 0.5))

            if self.refractory[region] > 0:
                self.refractory[region] -= 1
                self.membrane[region] *= cfg.leak
                self.trace[region] *= cfg.trace_decay
                self.threshold[region] = cfg.base_threshold + (self.threshold[region] - cfg.base_threshold) * cfg.threshold_recovery
                cycle_energy += cfg.integration_energy_ceu * 0.25
                continue

            recurrent = cfg.recurrent_gain * self.trace[region]
            local_inhibition = inhibitory_tone + inhibitory_current[region]
            energy_pressure = _bound(cycle_energy / budget)
            energy_gate = 1.0 - 0.45 * energy_pressure
            integrated = (
                self.membrane[region] * cfg.leak
                + cfg.feedforward_gain * drive * self.synaptic_gain[region] * energy_gate
                + recurrent
                + 0.16 * excitatory_current[region]
                + 0.07 * salience
                - local_inhibition
            )
            self.membrane[region] = max(0.0, integrated)
            cycle_energy += cfg.integration_energy_ceu

            dynamic_threshold = self.threshold[region] + 0.12 * energy_pressure
            fired = self.membrane[region] >= dynamic_threshold
            if fired:
                spikes.append(region)
                self.membrane[region] = 0.0
                self.trace[region] = min(1.0, self.trace[region] * cfg.trace_decay + 1.0)
                self.threshold[region] = min(1.25, dynamic_threshold + cfg.threshold_adaptation)
                self.refractory[region] = max(0, int(cfg.refractory_cycles))
                cycle_energy += cfg.spike_energy_ceu
            else:
                self.trace[region] *= cfg.trace_decay
                self.threshold[region] = cfg.base_threshold + (self.threshold[region] - cfg.base_threshold) * cfg.threshold_recovery

            before = self.synaptic_gain[region]
            if fired and drive >= 0.45:
                self.synaptic_gain[region] = min(1.35, before + cfg.plasticity_rate * salience * drive)
            else:
                self.synaptic_gain[region] = 1.0 + (before - 1.0) * cfg.plasticity_decay
            if abs(self.synaptic_gain[region] - before) > 1e-12:
                region_plastic_updates += 1

        # Event-gated edge plasticity: active source traces reinforce plastic
        # excitatory edges into a concurrently active target and decay otherwise.
        edge_plastic_updates = 0
        active = set(spikes)
        for synapse in self.synapses:
            if not synapse.plastic:
                continue
            key = (synapse.source, synapse.target, synapse.kind, synapse.pathway)
            before = self.edge_gain[key]
            coincident = self.trace[synapse.source] >= 0.5 and synapse.target in active
            if coincident and synapse.kind == "excitatory":
                self.edge_gain[key] = min(1.25, before + cfg.plasticity_rate * salience)
            else:
                self.edge_gain[key] = 1.0 + (before - 1.0) * cfg.plasticity_decay
            if abs(self.edge_gain[key] - before) > 1e-12:
                edge_plastic_updates += 1

        cycle_energy += (region_plastic_updates + edge_plastic_updates) * cfg.plasticity_energy_ceu
        self.total_energy_ceu += cycle_energy
        self.cycle_number += 1
        rate = len(spikes) / len(self.region_ids)
        total_exc = sum(excitatory_current.values())
        total_inh = sum(inhibitory_current.values())
        payload = {
            "schema": self.schema,
            "cycle": self.cycle_number,
            "spikes": spikes,
            "spike_rate": rate,
            "synaptic_events": synaptic_events,
            "excitatory_current": total_exc,
            "inhibitory_current": total_inh,
            "inhibitory_tone": inhibitory_tone,
            "energy_ceu": cycle_energy,
            "budget_ceu": budget,
            "orienting_burst": orienting_burst,
            "previous": self.previous_hash,
        }
        receipt = hashlib.sha256(_canonical(payload)).hexdigest()
        self.previous_hash = receipt
        return NeuromorphicCycle(
            cycle=self.cycle_number,
            spike_count=len(spikes),
            spike_rate=round(rate, 8),
            sparsity=round(1.0 - rate, 8),
            inhibitory_tone=round(inhibitory_tone, 8),
            synaptic_events=synaptic_events,
            excitatory_current=round(total_exc, 8),
            inhibitory_current=round(total_inh, 8),
            energy_ceu=round(cycle_energy, 8),
            energy_budget_ceu=round(budget, 8),
            energy_pressure=round(cycle_energy / budget, 8),
            orienting_burst=orienting_burst,
            active_regions=tuple(spikes),
            receipt_hash=receipt,
        )

    def region_state(self, region: str) -> SpikeRegionState:
        if region not in self.membrane:
            raise KeyError(region)
        return SpikeRegionState(
            membrane=self.membrane[region],
            threshold=self.threshold[region],
            trace=self.trace[region],
            spike=int(self.trace[region] >= 0.5),
            refractory=self.refractory[region],
            synaptic_gain=self.synaptic_gain[region],
            energy_ceu=0.0,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cycle": self.cycle_number,
            "region_count": len(self.region_ids),
            "synapse_count": len(self.synapses),
            "config": asdict(self.config),
            "total_energy_ceu": round(self.total_energy_ceu, 8),
            "energy_unit": "normalized_compute_energy_unit_not_joule",
            "receipt_head": self.previous_hash,
            "synapses": [
                {
                    **asdict(synapse),
                    "gain": round(self.edge_gain[(synapse.source, synapse.target, synapse.kind, synapse.pathway)], 8),
                }
                for synapse in self.synapses
            ],
            "regions": {
                region: {
                    "membrane": round(self.membrane[region], 8),
                    "threshold": round(self.threshold[region], 8),
                    "trace": round(self.trace[region], 8),
                    "refractory": self.refractory[region],
                    "synaptic_gain": round(self.synaptic_gain[region], 8),
                }
                for region in self.region_ids
            },
            "claim_boundary": {
                "feline_inspired_computational_principles": True,
                "biological_cat_brain_simulation": False,
                "language_model_weights_changed": False,
                "consciousness_claim": False,
                "physical_energy_efficiency_verified": False,
            },
        }
