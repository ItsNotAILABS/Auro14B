import numpy as np

from auro_native_llm.model.auro_lm import AuroLanguageModel
from auro_native_llm.model.checkpoint import load_checkpoint, save_checkpoint


def test_working_memory_parameters_survive_checkpoint_round_trip(tmp_path):
    model = AuroLanguageModel.build("Auro-2B", mode="dev")
    memory = model.delta_attention.working_memory
    memory.input_gate[:] = np.linspace(0.7, 1.3, memory.hidden_dim)
    memory.fast_gate[:] = 1.11
    memory.slow_gate[:] = 0.91
    memory.read_gate[:] = 1.07

    # Create transient state too; it must not be restored as session state.
    memory.step(np.ones(memory.hidden_dim, dtype=np.float64))
    assert memory.tokens_seen == 1

    meta = save_checkpoint(model, tmp_path)
    assert meta["schema"] == "auro.lm.checkpoint.v4"
    assert meta["working_memory_parameters"] is True
    assert (tmp_path / "working_memory.npz").exists()
    assert (tmp_path / "working_memory.json").exists()

    restored = load_checkpoint(tmp_path, allow_quarantined=True)
    loaded = restored.delta_attention.working_memory
    np.testing.assert_allclose(loaded.input_gate, memory.input_gate)
    np.testing.assert_allclose(loaded.fast_gate, memory.fast_gate)
    np.testing.assert_allclose(loaded.slow_gate, memory.slow_gate)
    np.testing.assert_allclose(loaded.read_gate, memory.read_gate)

    # Durable plastic parameters survive; transient live-session state does not.
    assert loaded.tokens_seen == 0
    assert loaded.consolidations == 0
    assert loaded.compute_pressure == 0.0
    assert np.linalg.norm(loaded.fast) == 0.0
    assert np.linalg.norm(loaded.slow) == 0.0
