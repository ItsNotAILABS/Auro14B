import numpy as np

from auro_native_llm.model.delta_attention import DeltaAttentionEngine
from auro_native_llm.model.recurrent_memory import RecurrentMemoryConfig, RecurrentSurpriseMemory


def test_surprise_memory_writes_and_reads_informative_states():
    memory = RecurrentSurpriseMemory(
        4,
        RecurrentMemoryConfig(max_slots=8, surprise_threshold=0.05, read_strength=0.25, top_k=2),
    )
    hidden = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.1, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    fused, receipt = memory.fuse(hidden)
    assert fused.shape == hidden.shape
    assert receipt["writes"] >= 1
    assert receipt["slots"] >= 1
    assert receipt["last_surprise"] >= 0.0
    recalled, gate = memory.read(np.asarray([0.0, 1.0, 0.0, 0.0]))
    assert recalled.shape == (4,)
    assert 0.0 <= gate <= 0.25


def test_recurrent_memory_persists_across_single_stream_calls_until_reset():
    memory = RecurrentSurpriseMemory(3, RecurrentMemoryConfig(max_slots=4, surprise_threshold=0.01))
    memory.fuse(np.asarray([[1.0, 0.0, 0.0]]))
    first_slots = memory.snapshot()["slots"]
    memory.fuse(np.asarray([[0.0, 1.0, 0.0]]))
    second_slots = memory.snapshot()["slots"]
    assert first_slots >= 1
    assert second_slots >= first_slots
    memory.reset()
    assert memory.snapshot()["slots"] == 0


def test_salience_retention_prefers_strong_memory_over_weak_old_slot():
    memory = RecurrentSurpriseMemory(
        2,
        RecurrentMemoryConfig(
            max_slots=2,
            surprise_threshold=0.0,
            retention_age_penalty=0.01,
            retention_strength_weight=1.0,
        ),
    )
    memory.fuse(np.asarray([[10.0, 0.0], [0.0, 0.01]]))
    assert memory.snapshot()["slots"] == 2
    before_evictions = memory.receipt.evictions
    memory.fuse(np.asarray([[0.0, 10.0]]))
    assert memory.snapshot()["slots"] == 2
    assert memory.receipt.evictions >= before_evictions


def test_multi_sample_batch_uses_isolated_ephemeral_memory():
    memory = RecurrentSurpriseMemory(3, RecurrentMemoryConfig(max_slots=8, surprise_threshold=0.01))
    batch = np.asarray([
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
        [[0.0, 1.0, 0.0], [0.1, 0.9, 0.0]],
    ])
    fused, receipt = memory.fuse(batch)
    assert fused.shape == batch.shape
    assert receipt["batch_isolation"] is True
    assert receipt["batch_size"] == 2
    assert receipt["persistent_runtime_state_used"] is False
    assert memory.snapshot()["slots"] == 0


def test_delta_attention_hybrid_memory_is_persistent_for_single_stream():
    engine = DeltaAttentionEngine(4, max_slots=8, novelty_threshold=0.01, blend=0.05)
    hidden = np.asarray([[[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]])
    fused, receipt = engine.fuse(hidden)
    assert fused.shape == hidden.shape
    assert receipt["schema"] == "auro.delta-attention.hybrid-memory.v2"
    assert receipt["surprise_memory"]["slots"] >= 1
    before = engine.snapshot()["surprise"]["slots"]
    engine.fuse(np.asarray([[[0.0, 1.0, 0.0, 0.0]]]))
    assert engine.snapshot()["surprise"]["slots"] >= before


def test_delta_attention_isolates_multi_sample_batches():
    engine = DeltaAttentionEngine(3, max_slots=4, novelty_threshold=0.01)
    batch = np.asarray([
        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
    ])
    fused, receipt = engine.fuse(batch)
    assert fused.shape == batch.shape
    assert receipt["batch_isolation"] is True
    assert receipt["persistent_runtime_state_used"] is False
    assert engine.snapshot()["delta_slots"] == 0
    assert engine.snapshot()["surprise"]["slots"] == 0
