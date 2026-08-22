from auro_native_llm.brain import FelineNeuromorphicEngine, HIMBrain, NeuromorphicConfig


def test_spiking_engine_is_sparse_and_energy_accounted():
    engine = FelineNeuromorphicEngine(("V1", "V2V3", "SC", "THL_L", "LC", "DLPFC_L"))
    cycle = engine.cycle({"V1": 1.0, "SC": 0.9}, salience=0.8, novelty=0.7)
    assert 0 <= cycle.spike_rate <= 1
    assert cycle.sparsity == round(1.0 - cycle.spike_rate, 8)
    assert cycle.energy_ceu > 0
    assert cycle.energy_budget_ceu > 0
    assert cycle.orienting_burst is True
    assert len(cycle.receipt_hash) == 64


def test_refractory_and_adaptive_thresholds_suppress_immediate_refire():
    cfg = NeuromorphicConfig(base_threshold=0.30, refractory_cycles=1)
    engine = FelineNeuromorphicEngine(("V1",), cfg)
    first = engine.cycle({"V1": 1.0}, salience=1.0, novelty=1.0)
    second = engine.cycle({"V1": 1.0}, salience=1.0, novelty=1.0)
    assert first.spike_count == 1
    assert second.spike_count == 0
    assert engine.region_state("V1").threshold >= cfg.base_threshold


def test_inhibitory_tone_rises_after_dense_recent_activity():
    cfg = NeuromorphicConfig(base_threshold=0.2, target_spike_rate=0.05)
    regions = tuple(f"R{i}" for i in range(12))
    engine = FelineNeuromorphicEngine(regions, cfg)
    first = engine.cycle({region: 1.0 for region in regions}, salience=1.0)
    engine.cycle({region: 0.0 for region in regions}, salience=0.1)
    third = engine.cycle({region: 0.5 for region in regions}, salience=0.5)
    assert first.spike_rate > 0
    assert third.inhibitory_tone > 0


def test_visual_hierarchy_and_orienting_path_are_real_synapses():
    regions = ("SC", "THL_L", "THL_R", "LC", "V1", "V2V3", "ITG_L", "FFA", "DLPFC_R")
    engine = FelineNeuromorphicEngine(regions)
    snapshot = engine.snapshot()
    edges = {(item["source"], item["target"], item["kind"], item["pathway"]) for item in snapshot["synapses"]}
    assert ("V1", "V2V3", "excitatory", "visual_feedforward") in edges
    assert ("V2V3", "ITG_L", "excitatory", "visual_feedforward") in edges
    assert ("SC", "THL_L", "excitatory", "orienting") in edges
    assert ("THL_L", "V1", "excitatory", "orienting") in edges
    assert snapshot["synapse_count"] >= 6


def test_synaptic_events_consume_energy_and_propagate_on_later_cycle():
    cfg = NeuromorphicConfig(base_threshold=0.25, refractory_cycles=0)
    engine = FelineNeuromorphicEngine(("V1", "V2V3"), cfg)
    first = engine.cycle({"V1": 1.0}, salience=1.0)
    second = engine.cycle({"V1": 0.0, "V2V3": 0.0}, salience=0.4)
    assert first.spike_count >= 1
    assert second.synaptic_events >= 1
    assert second.excitatory_current > 0
    assert second.energy_ceu > cfg.idle_energy_ceu * 2


def test_energy_pressure_can_throttle_spiking_without_claiming_physical_joules():
    cfg = NeuromorphicConfig(base_threshold=0.2, energy_budget_ceu=0.3)
    engine = FelineNeuromorphicEngine(("V1", "SC", "LC"), cfg)
    cycle = engine.cycle({"V1": 1.0, "SC": 1.0, "LC": 1.0}, salience=1.0, novelty=1.0)
    snap = engine.snapshot()
    assert cycle.energy_pressure > 1.0
    assert snap["energy_unit"] == "normalized_compute_energy_unit_not_joule"
    assert snap["claim_boundary"]["biological_cat_brain_simulation"] is False
    assert snap["claim_boundary"]["physical_energy_efficiency_verified"] is False


def test_canonical_him_brain_exposes_neuromorphic_state():
    brain = HIMBrain()
    cycle = brain.cycle("urgent visual browser signal verify and compare", importance=0.9)
    snapshot = brain.snapshot()
    assert snapshot["schema"] == "him.brain.v2.neuromorphic"
    assert snapshot["neuromorphic"]["region_count"] == 44
    assert snapshot["neuromorphic"]["synapse_count"] > 0
    assert cycle.neuromorphic["energy_ceu"] > 0
    assert "orienting_burst" in cycle.neuromorphic
    assert snapshot["architecture_notes"]["biological_equivalence_claim"] is False


def test_neuromorphic_layer_never_upgrades_execution_authority():
    brain = HIMBrain()
    cycle = brain.cycle("answer this visual observation", importance=0.2, execute_requested=False)
    assert cycle.route != "execute"
