"""Tests for intelligence AI protocols."""

import numpy as np
import pytest

from mesie.ai.intelligence_protocols import (
    AttentionFocusModule,
    IntelligenceConfig,
    IntelligenceLevel,
    IntelligenceProtocol,
    ReasoningResult,
    ReasoningStrategy,
    SpectralMemoryBuffer,
)


class TestSpectralMemoryBuffer:
    """Tests for SpectralMemoryBuffer."""

    def test_store_and_retrieve(self):
        memory = SpectralMemoryBuffer(capacity=10)
        obs = np.random.randn(64)
        memory.store(obs, context={"label": "test"})
        assert memory.size == 1

        results = memory.retrieve(obs, top_k=1)
        assert len(results) == 1
        assert results[0]["context"]["label"] == "test"

    def test_capacity_eviction(self):
        memory = SpectralMemoryBuffer(capacity=3)
        for i in range(5):
            memory.store(np.random.randn(32), importance=float(i))
        assert memory.size == 3

    def test_clear(self):
        memory = SpectralMemoryBuffer(capacity=10)
        memory.store(np.random.randn(32))
        memory.store(np.random.randn(32))
        memory.clear()
        assert memory.size == 0

    def test_retrieve_empty(self):
        memory = SpectralMemoryBuffer(capacity=10)
        results = memory.retrieve(np.random.randn(32))
        assert results == []


class TestAttentionFocusModule:
    """Tests for AttentionFocusModule."""

    def test_compute_focus(self):
        attn = AttentionFocusModule(n_frequency_bins=64, n_attention_heads=2)
        spectrum = np.random.randn(64)
        focus = attn.compute_focus(spectrum)
        assert focus.shape == (64,)
        assert np.all(focus >= 0)
        assert abs(np.sum(focus) - 1.0) < 1e-5

    def test_compute_focus_different_length(self):
        attn = AttentionFocusModule(n_frequency_bins=64, n_attention_heads=2)
        spectrum = np.random.randn(128)
        focus = attn.compute_focus(spectrum)
        assert focus.shape == (64,)

    def test_update_attention(self):
        attn = AttentionFocusModule(n_frequency_bins=32, n_attention_heads=2)
        feedback = np.random.randn(32)
        attn.update_attention(feedback, learning_rate=0.1)
        # Weights should still sum to ~1 per head
        for h in range(2):
            assert abs(np.sum(attn._attention_weights[h]) - 1.0) < 1e-5

    def test_current_focus_none_initially(self):
        attn = AttentionFocusModule()
        assert attn.current_focus is None

    def test_current_focus_after_compute(self):
        attn = AttentionFocusModule(n_frequency_bins=32)
        attn.compute_focus(np.ones(32))
        assert attn.current_focus is not None


class TestIntelligenceProtocol:
    """Tests for IntelligenceProtocol."""

    def test_default_config(self):
        protocol = IntelligenceProtocol()
        assert protocol.config.level == IntelligenceLevel.ADAPTIVE
        assert protocol.config.strategy == ReasoningStrategy.ENSEMBLE
        assert protocol.observation_count == 0

    def test_observe(self):
        protocol = IntelligenceProtocol()
        protocol.observe(np.random.randn(64))
        assert protocol.observation_count == 1
        assert protocol.memory_utilization > 0

    def test_reason_normal(self):
        protocol = IntelligenceProtocol()
        # Normal signal with moderate std
        spectrum = np.random.randn(64) * 0.5 + 5.0
        result = protocol.reason(spectrum)
        assert isinstance(result, ReasoningResult)
        assert result.confidence > 0
        assert result.confidence <= 1.0
        assert len(result.metadata) > 0

    def test_reason_anomaly(self):
        protocol = IntelligenceProtocol()
        # High variability signal (std >> mean)
        spectrum = np.random.randn(64) * 100
        result = protocol.reason(spectrum)
        assert isinstance(result, ReasoningResult)

    def test_reason_low_signal(self):
        protocol = IntelligenceProtocol()
        # Very low energy
        spectrum = np.ones(64) * 1e-8
        result = protocol.reason(spectrum)
        assert result.conclusion == "low_signal"

    def test_adapt(self):
        config = IntelligenceConfig(level=IntelligenceLevel.ADAPTIVE)
        protocol = IntelligenceProtocol(config=config)
        protocol.observe(np.random.randn(128))
        protocol.adapt(np.random.randn(128))
        # Should not raise

    def test_reasoning_history(self):
        protocol = IntelligenceProtocol()
        protocol.reason(np.random.randn(64))
        protocol.reason(np.random.randn(64))
        assert len(protocol.reasoning_history) == 2

    def test_memory_utilization(self):
        config = IntelligenceConfig(memory_window=5)
        protocol = IntelligenceProtocol(config=config)
        for _ in range(5):
            protocol.observe(np.random.randn(32))
        assert protocol.memory_utilization == 1.0

    def test_disabled_memory_and_attention(self):
        config = IntelligenceConfig(enable_memory=False, enable_attention=False)
        protocol = IntelligenceProtocol(config=config)
        protocol.observe(np.random.randn(64))
        result = protocol.reason(np.random.randn(64))
        assert isinstance(result, ReasoningResult)
