from types import SimpleNamespace

from auro_native_llm.model.train import TrainConfig, _neuromorphic_training_control


def fake_model(receipt=None):
    gate = SimpleNamespace(last_receipt=receipt) if receipt is not None else None
    bridge = SimpleNamespace(spiking_gate=gate)
    return SimpleNamespace(_neuro=bridge)


def test_no_gate_receipt_leaves_lr_unchanged():
    metrics, multiplier = _neuromorphic_training_control(fake_model(), TrainConfig())
    assert metrics == {}
    assert multiplier == 1.0


def test_neuromorphic_regularizer_reduces_next_lr_with_hard_bound():
    receipt = SimpleNamespace(
        activity_rate=0.6,
        sparsity=0.4,
        inhibitory_tone=0.5,
        energy_proxy=3.0,
        regularizer=1.0,
    )
    cfg = TrainConfig(neuromorphic_max_lr_reduction=0.25, neuromorphic_penalty_gain=5.0)
    metrics, multiplier = _neuromorphic_training_control(fake_model(receipt), cfg)
    assert multiplier == 0.75
    assert metrics["neuro_lr_multiplier"] == 0.75
    assert metrics["neuro_energy_proxy"] == 3.0


def test_lr_control_can_be_disabled_without_disabling_forward_gate():
    receipt = SimpleNamespace(
        activity_rate=0.3,
        sparsity=0.7,
        inhibitory_tone=0.2,
        energy_proxy=1.4,
        regularizer=0.2,
    )
    cfg = TrainConfig(neuromorphic_lr_control=False)
    metrics, multiplier = _neuromorphic_training_control(fake_model(receipt), cfg)
    assert multiplier == 1.0
    assert metrics["neuro_control_regularizer"] == 0.2
    assert metrics["neuro_lr_multiplier"] == 1.0
