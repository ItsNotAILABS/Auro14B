"""Bounded STDP-inspired timing refinement for HIM neuromorphic synapses.

This is an engineering timing rule, not a claim of biological STDP fidelity.
It operates on runtime edge gains only and never changes AURO checkpoint weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TimingPlasticityConfig:
    window_cycles: int = 3
    potentiation_rate: float = 0.012
    depression_rate: float = 0.008
    minimum_edge_gain: float = 0.75
    maximum_edge_gain: float = 1.25
    neutral_decay: float = 0.998


@dataclass(frozen=True)
class TimingPlasticityReceipt:
    cycle: int
    potentiated_edges: int
    depressed_edges: int
    decayed_edges: int
    active_regions: tuple[str, ...]
    biological_stdp_equivalence_claim: bool = False
    checkpoint_weights_changed: bool = False


class TimingPlasticityController:
    """Apply pre/post spike ordering to plastic runtime synaptic gains."""

    def __init__(self, region_ids: Iterable[str], config: TimingPlasticityConfig | None = None):
        self.config = config or TimingPlasticityConfig()
        if self.config.window_cycles < 1:
            raise ValueError("window_cycles must be >= 1")
        if not 0 < self.config.minimum_edge_gain <= 1.0 <= self.config.maximum_edge_gain:
            raise ValueError("edge gain bounds must straddle neutral gain 1.0")
        self.last_spike_cycle = {str(region): -1 for region in region_ids}
        self.last_receipt: TimingPlasticityReceipt | None = None

    def apply(self, engine: Any, active_regions: Iterable[str], *, salience: float = 0.5) -> TimingPlasticityReceipt:
        active = {str(region) for region in active_regions}
        cycle = int(getattr(engine, "cycle_number", 0))
        salience = max(0.0, min(1.0, float(salience)))
        potentiated = 0
        depressed = 0
        decayed = 0

        for synapse in getattr(engine, "synapses", ()):
            if not getattr(synapse, "plastic", False):
                continue
            key = (synapse.source, synapse.target, synapse.kind, synapse.pathway)
            if key not in engine.edge_gain:
                continue
            before = float(engine.edge_gain[key])
            after = before
            pre_last = int(self.last_spike_cycle.get(synapse.source, -1))
            post_last = int(self.last_spike_cycle.get(synapse.target, -1))
            pre_now = synapse.source in active
            post_now = synapse.target in active

            # Pre-before-post within the timing window potentiates excitatory
            # transmission; post-before-pre depresses it. Inhibitory edges keep
            # their sign and use the same bounded gain mechanics.
            pre_age = cycle - pre_last if pre_last >= 0 else self.config.window_cycles + 1
            post_age = cycle - post_last if post_last >= 0 else self.config.window_cycles + 1
            if post_now and 1 <= pre_age <= self.config.window_cycles:
                after = min(
                    self.config.maximum_edge_gain,
                    before + self.config.potentiation_rate * salience / pre_age,
                )
                potentiated += int(after != before)
            elif pre_now and 1 <= post_age <= self.config.window_cycles:
                after = max(
                    self.config.minimum_edge_gain,
                    before - self.config.depression_rate * salience / post_age,
                )
                depressed += int(after != before)
            else:
                after = 1.0 + (before - 1.0) * self.config.neutral_decay
                decayed += int(abs(after - before) > 1e-12)
            engine.edge_gain[key] = after

        for region in active:
            if region in self.last_spike_cycle:
                self.last_spike_cycle[region] = cycle

        receipt = TimingPlasticityReceipt(
            cycle=cycle,
            potentiated_edges=potentiated,
            depressed_edges=depressed,
            decayed_edges=decayed,
            active_regions=tuple(sorted(active)),
        )
        self.last_receipt = receipt
        return receipt

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "him.timing-plasticity.v1",
            "config": asdict(self.config),
            "last_spike_cycle": dict(self.last_spike_cycle),
            "last_receipt": asdict(self.last_receipt) if self.last_receipt else None,
            "claim_boundary": {
                "stdp_inspired_timing_rule": True,
                "biological_stdp_equivalence": False,
                "checkpoint_weights_changed": False,
            },
        }
