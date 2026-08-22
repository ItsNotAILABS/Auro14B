"""HIM brain controller with an attached feline-inspired spiking substrate."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .feline_neuromorphic import FelineNeuromorphicEngine, NeuromorphicCycle
from .fused import BrainRegion, HIMBrain as BaseHIMBrain
from .neuromorphic_state import NeuromorphicStateStore
from .timing_plasticity import TimingPlasticityController


@dataclass(frozen=True)
class BrainCycle:
    cycle: int
    salience: float
    coherence: float
    anomaly: float
    dominant_system: str
    route: str
    working_memory: tuple[str, ...]
    receipt_hash: str
    neuromorphic: dict[str, Any]


class HIMBrain(BaseHIMBrain):
    """Canonical cognitive controller plus persistent event-driven dynamics.

    The base controller remains authoritative for salience, routing, working
    memory, and receipt continuity. The neuromorphic substrate adds sparse event
    dynamics, explicit synapses, local plasticity, timing plasticity, and
    normalized compute-energy pressure. It never alters language-model weights.
    """

    schema = "him.brain.v2.neuromorphic"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        region_ids = tuple(region.abbreviation for region in self.regions)
        self.neuromorphic = FelineNeuromorphicEngine(region_ids)
        self.timing_plasticity = TimingPlasticityController(region_ids)
        self.last_neuromorphic: NeuromorphicCycle | None = None
        self.neuromorphic_store = NeuromorphicStateStore.for_brain_state(self.state_path) if self.state_path else None
        self.neuromorphic_state_restored = False
        if self.neuromorphic_store:
            self.neuromorphic_state_restored = self.neuromorphic_store.load(self.neuromorphic, self.timing_plasticity)

    def cycle(self, observation: str, *, importance: float = 0.5, execute_requested: bool = False) -> BrainCycle:
        before = dict(self.activations)
        base = super().cycle(observation, importance=importance, execute_requested=execute_requested)
        drives = {region: self.activations[region] for region in self.activations}
        mean_delta = sum(abs(self.activations[key] - before[key]) for key in self.activations) / max(1, len(self.activations))
        novelty = min(1.0, base.anomaly * 0.55 + mean_delta * 1.8)
        neuro = self.neuromorphic.cycle(drives, salience=base.salience, novelty=novelty)
        timing = self.timing_plasticity.apply(self.neuromorphic, neuro.active_regions, salience=base.salience)
        self.last_neuromorphic = neuro
        if self.neuromorphic_store:
            self.neuromorphic_store.save(self.neuromorphic, self.timing_plasticity)

        route = base.route
        if neuro.energy_pressure > 1.0 and route == "execute":
            route = "deliberate"

        neuro_payload = asdict(neuro)
        neuro_payload["timing_plasticity"] = asdict(timing)
        return BrainCycle(
            cycle=base.cycle,
            salience=base.salience,
            coherence=base.coherence,
            anomaly=base.anomaly,
            dominant_system=base.dominant_system,
            route=route,
            working_memory=base.working_memory,
            receipt_hash=base.receipt_hash,
            neuromorphic=neuro_payload,
        )

    def snapshot(self) -> dict[str, Any]:
        base = super().snapshot()
        base["schema"] = self.schema
        base["neuromorphic"] = self.neuromorphic.snapshot()
        base["timing_plasticity"] = self.timing_plasticity.snapshot()
        persistence_status = self.neuromorphic_store.status() if self.neuromorphic_store else {}
        base["neuromorphic_persistence"] = {
            "enabled": self.neuromorphic_store is not None,
            "restored_on_start": self.neuromorphic_state_restored,
            "path": str(self.neuromorphic_store.path) if self.neuromorphic_store else None,
            "timing_state_persisted": self.neuromorphic_store is not None,
            "degraded": bool(persistence_status.get("last_error")),
            "last_error": persistence_status.get("last_error"),
            "quarantined_path": persistence_status.get("quarantined_path"),
            "durable_atomic_write": persistence_status.get("durable_atomic_write", False),
            "transactional_load": persistence_status.get("transactional_load", False),
        }
        if self.last_neuromorphic:
            last = asdict(self.last_neuromorphic)
            if self.timing_plasticity.last_receipt:
                last["timing_plasticity"] = asdict(self.timing_plasticity.last_receipt)
            base["last_neuromorphic_cycle"] = last
        else:
            base["last_neuromorphic_cycle"] = None
        base["architecture_notes"] = {
            "hierarchical_recurrent_processing": True,
            "explicit_synaptic_graph": True,
            "sparse_event_driven_activation": True,
            "inhibitory_balance": True,
            "adaptive_spike_thresholds": True,
            "short_synaptic_traces": True,
            "bounded_local_and_edge_plasticity": True,
            "timing_order_plasticity": True,
            "orienting_burst_path": ["SC", "THL_L", "THL_R", "LC", "V1"],
            "persistent_neuromorphic_state": self.neuromorphic_store is not None,
            "energy_homeostasis": "normalized CEU budget; hardware calibration pending",
            "biological_equivalence_claim": False,
            "biological_stdp_equivalence_claim": False,
        }
        return base

    state = snapshot


__all__ = ["HIMBrain", "BrainCycle", "BrainRegion"]
