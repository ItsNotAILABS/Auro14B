import numpy as np

from auro_native_llm.model.working_memory import (
    ActiveWorkingMemory,
    clear_compute_pressure,
    current_compute_pressure,
    current_compute_source,
    publish_compute_pressure,
)
from auro_native_llm.model.delta_attention import DeltaAttentionEngine
from mesie.foundation.models.mixture_of_experts import MixtureOfExperts


def test_fast_slow_memory_updates_and_publishes_pressure():
    clear_compute_pressure()
    memory = ActiveWorkingMemory(4)
    hidden = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    fused, receipt = memory.fuse(hidden)
    assert fused.shape == hidden.shape
    assert receipt["tokens_seen"] == 2
    assert receipt["plasticity_updates"] == 2
    assert receipt["consolidations"] >= 1
    assert receipt["fast_norm"] > 0.0
    assert receipt["slow_norm"] > 0.0
    assert current_compute_pressure() > 0.0
    assert current_compute_source() == "active_working_memory"


def test_incremental_working_memory_does_not_relearn_dense_prefix():
    memory = ActiveWorkingMemory(4)
    first = np.eye(4, dtype=np.float64)
    memory.fuse(first, incremental=True)
    seen_after_seed = memory.tokens_seen
    assert seen_after_seed == 4

    longer = np.concatenate([first, np.asarray([[1.0, 1.0, 0.0, 0.0]])], axis=0)
    memory.fuse(longer, incremental=True)
    assert memory.tokens_seen == seen_after_seed + 1


def test_recurrent_pressure_changes_real_moe_specialist_budget():
    clear_compute_pressure()
    moe = MixtureOfExperts(
        hidden_dim=4,
        num_experts=4,
        top_k=3,
        expert_dim=8,
        modality_aware=False,
        noise_std=0.0,
        adaptive_compute=True,
        min_k=1,
    )
    # Make router intrinsically highly confident in expert zero.
    moe.router.router_weights[:] = 0.0
    moe.router.router_bias[:] = np.asarray([12.0, 0.0, 0.0, 0.0])
    x = np.ones((1, 3, 4), dtype=np.float64)

    _, low = moe.forward(x, training=False)
    low_k = low["active_k_mean"]
    assert low["difficulty_source"] == "router_intrinsic"

    publish_compute_pressure(1.0, source="test_working_memory")
    _, high = moe.forward(x, training=False)
    high_k = high["active_k_mean"]
    assert high["difficulty_source"] == "test_working_memory"
    assert high["working_memory_coupled"] is True
    assert high["inherited_working_memory_pressure"] == 1.0
    assert high_k > low_k


def test_delta_surprise_working_memory_publishes_next_cycle_pressure():
    clear_compute_pressure()
    engine = DeltaAttentionEngine(
        4,
        max_slots=8,
        novelty_threshold=0.05,
        blend=0.05,
        surprise_max_slots=8,
        surprise_threshold=0.05,
        surprise_blend=0.1,
        surprise_top_k=2,
    )
    hidden = np.asarray([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    fused, receipt = engine.fuse(hidden)
    assert fused.shape == hidden.shape
    assert receipt["working_memory"]["active"] is True
    assert receipt["compute_pressure_controls_next_moe_cycle"] is True
    assert receipt["hybrid_compute_pressure"] > 0.0
    assert current_compute_source() == "hybrid_surprise_working_memory"


def test_multi_sample_batch_clears_persistent_pressure():
    publish_compute_pressure(1.0, source="preexisting")
    engine = DeltaAttentionEngine(4, max_slots=8)
    hidden = np.asarray(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ]
    )
    _, receipt = engine.fuse(hidden)
    assert receipt["batch_isolation"] is True
    assert receipt["persistent_runtime_state_used"] is False
    assert receipt["working_memory_pressure_cleared"] is True
    assert current_compute_pressure() == 0.0
