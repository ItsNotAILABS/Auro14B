from types import SimpleNamespace

from auro_native_llm.brain import TimingPlasticityController


def fake_engine():
    synapse = SimpleNamespace(
        source="PRE",
        target="POST",
        kind="excitatory",
        pathway="test",
        plastic=True,
    )
    key = ("PRE", "POST", "excitatory", "test")
    return SimpleNamespace(synapses=(synapse,), edge_gain={key: 1.0}, cycle_number=0), key


def test_pre_before_post_potentiates_runtime_edge_gain():
    engine, key = fake_engine()
    controller = TimingPlasticityController(("PRE", "POST"))

    engine.cycle_number = 1
    controller.apply(engine, ("PRE",), salience=1.0)
    engine.cycle_number = 2
    receipt = controller.apply(engine, ("POST",), salience=1.0)

    assert receipt.potentiated_edges == 1
    assert receipt.depressed_edges == 0
    assert engine.edge_gain[key] > 1.0
    assert engine.edge_gain[key] <= controller.config.maximum_edge_gain


def test_post_before_pre_depresses_runtime_edge_gain():
    engine, key = fake_engine()
    controller = TimingPlasticityController(("PRE", "POST"))

    engine.cycle_number = 1
    controller.apply(engine, ("POST",), salience=1.0)
    engine.cycle_number = 2
    receipt = controller.apply(engine, ("PRE",), salience=1.0)

    assert receipt.depressed_edges == 1
    assert receipt.potentiated_edges == 0
    assert engine.edge_gain[key] < 1.0
    assert engine.edge_gain[key] >= controller.config.minimum_edge_gain


def test_spikes_outside_timing_window_do_not_potentiate_or_depress():
    engine, key = fake_engine()
    controller = TimingPlasticityController(("PRE", "POST"))
    engine.edge_gain[key] = 1.2

    engine.cycle_number = 1
    controller.apply(engine, ("PRE",), salience=1.0)
    engine.cycle_number = 8
    receipt = controller.apply(engine, ("POST",), salience=1.0)

    assert receipt.potentiated_edges == 0
    assert receipt.depressed_edges == 0
    assert engine.edge_gain[key] < 1.2
    assert engine.edge_gain[key] > 1.0


def test_timing_plasticity_is_runtime_only_and_not_biological_equivalence():
    engine, _ = fake_engine()
    controller = TimingPlasticityController(("PRE", "POST"))
    engine.cycle_number = 1
    receipt = controller.apply(engine, ("PRE",), salience=0.5)
    snapshot = controller.snapshot()

    assert receipt.checkpoint_weights_changed is False
    assert receipt.biological_stdp_equivalence_claim is False
    assert snapshot["claim_boundary"]["stdp_inspired_timing_rule"] is True
    assert snapshot["claim_boundary"]["biological_stdp_equivalence"] is False
