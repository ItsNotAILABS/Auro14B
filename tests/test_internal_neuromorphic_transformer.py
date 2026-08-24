import numpy as np

from auro_native_llm.model.auro_lm import AuroLanguageModel
from auro_native_llm.neuro.internal_modulation import InternalNeuromorphicModulator
from auro_native_llm.model.working_memory import clear_compute_pressure, publish_compute_pressure


def test_internal_modulator_changes_block_residual_not_base_shape():
    modulator = InternalNeuromorphicModulator(4)
    x = np.zeros((1, 3, 4), dtype=np.float64)
    proposed = np.asarray(
        [[[1.0, 0.1, 0.01, 0.001], [0.8, 0.1, 0.02, 0.001], [1.2, 0.1, 0.03, 0.001]]]
    )
    modulated, receipt = modulator.modulate(x, proposed, layer_idx=0)
    assert modulated.shape == proposed.shape
    assert receipt.changed_hidden_stream is True
    assert receipt.layer_idx == 0
    assert 0.0 <= receipt.activity_rate <= 1.0
    assert receipt.gate_mean < 1.0
    assert not np.allclose(modulated, proposed)


def test_working_memory_pressure_recruits_internal_spiking_activity():
    clear_compute_pressure()
    low = InternalNeuromorphicModulator(8)
    x = np.zeros((1, 2, 8), dtype=np.float64)
    proposed = np.linspace(0.1, 1.0, 16, dtype=np.float64).reshape(1, 2, 8)
    _, low_receipt = low.modulate(x, proposed, layer_idx=0)

    publish_compute_pressure(1.0, source="test")
    high = InternalNeuromorphicModulator(8)
    _, high_receipt = high.modulate(x, proposed, layer_idx=0)
    assert high_receipt.working_memory_pressure == 1.0
    assert high_receipt.effective_threshold <= low_receipt.effective_threshold
    assert high_receipt.activity_rate >= low_receipt.activity_rate
    clear_compute_pressure()


def test_neuro_bridge_wraps_transformer_blocks_before_moe():
    model = AuroLanguageModel.build("Auro-2B", mode="dev")
    bridge = model._neuro
    assert bridge is not None
    assert bridge.internal_modulator is not None
    assert bridge.info()["internal_position"] == "after_transformer_block_before_moe_and_next_layer"

    blocks = [layer["transformer"] for layer in model.core.layers]
    assert blocks
    assert all(getattr(block, "_auro_neuromorphic_wrapped", False) for block in blocks)
    assert all(callable(getattr(block, "_auro_original_forward", None)) for block in blocks)


def test_internal_receipts_are_emitted_during_model_forward():
    model = AuroLanguageModel.build("Auro-2B", mode="dev")
    ids = np.asarray([[1, 7, 8, 9]], dtype=np.int64)
    out = model.forward_ids(ids)
    assert out["internal_neuromorphic_transformer_enabled"] is True
    internal = out["internal_neuromorphic_transformer"]
    assert internal["acts_between_transformer_layers"] is True
    assert internal["affects_pre_moe_hidden_state"] is True
    assert internal["total_calls"] >= len(model.core.layers)
    assert internal["layers_seen"]
