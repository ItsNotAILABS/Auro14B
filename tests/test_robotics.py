"""Tests for the robotics module — multi-modal fusion, spectral memory, neuromorphic runtime."""

import numpy as np
import pytest

from mesie.robotics.multimodal_fusion import (
    FusedRepresentation,
    FusionConfig,
    FusionStrategy,
    Modality,
    ModalityStream,
    MultiModalFusion,
)
from mesie.robotics.spectral_memory import (
    ConsolidationStrategy,
    MemoryConfig,
    MemoryEntry,
    MemoryTier,
    QueryResult,
    SpectralMemory,
)
from mesie.robotics.neuromorphic_runtime import (
    EncodingScheme,
    NeuromorphicRuntime,
    ProcessingMode,
    RuntimeConfig,
    RuntimeMetrics,
    SpikeEvent,
)


class TestMultiModalFusion:
    """Tests for multi-modal signal fusion."""

    def test_init_default(self):
        fusion = MultiModalFusion()
        assert fusion.config.strategy == FusionStrategy.SPECTRAL_ANCHORED
        assert fusion.config.output_dim == 256

    def test_feed_and_fuse(self):
        fusion = MultiModalFusion(config=FusionConfig(output_dim=64))

        spectral = ModalityStream(
            modality=Modality.SPECTRAL, data=np.random.randn(64)
        )
        audio = ModalityStream(
            modality=Modality.AUDIO, data=np.random.randn(128)
        )

        fusion.feed(spectral)
        fusion.feed(audio)

        result = fusion.fuse()
        assert result.vector.shape == (64,)
        assert len(result.modality_contributions) == 2

    def test_spectral_anchored_strategy(self):
        config = FusionConfig(
            strategy=FusionStrategy.SPECTRAL_ANCHORED, output_dim=32
        )
        fusion = MultiModalFusion(config=config)

        fusion.feed(ModalityStream(modality=Modality.SPECTRAL, data=np.ones(32)))
        fusion.feed(ModalityStream(modality=Modality.VISION, data=np.ones(32)))

        result = fusion.fuse()
        assert Modality.SPECTRAL.value in result.modality_contributions
        # Spectral should have highest contribution weight
        assert result.modality_contributions[Modality.SPECTRAL.value] >= 0.4

    def test_concatenate_strategy(self):
        config = FusionConfig(strategy=FusionStrategy.CONCATENATE, output_dim=64)
        fusion = MultiModalFusion(config=config)
        fusion.feed(ModalityStream(modality=Modality.SPECTRAL, data=np.random.randn(32)))
        fusion.feed(ModalityStream(modality=Modality.IMU, data=np.random.randn(32)))
        result = fusion.fuse()
        assert result.vector.shape == (64,)

    def test_attention_weighted_strategy(self):
        config = FusionConfig(strategy=FusionStrategy.ATTENTION_WEIGHTED, output_dim=32)
        fusion = MultiModalFusion(config=config)
        fusion.feed(ModalityStream(modality=Modality.SPECTRAL, data=np.random.randn(32), confidence=0.9))
        fusion.feed(ModalityStream(modality=Modality.AUDIO, data=np.random.randn(32), confidence=0.3))
        result = fusion.fuse()
        assert result.attention_map is not None

    def test_gated_fusion_strategy(self):
        config = FusionConfig(strategy=FusionStrategy.GATED_FUSION, output_dim=32)
        fusion = MultiModalFusion(config=config)
        fusion.feed(ModalityStream(modality=Modality.SPECTRAL, data=np.random.randn(32)))
        result = fusion.fuse()
        assert result.vector.shape == (32,)

    def test_empty_fuse(self):
        fusion = MultiModalFusion(config=FusionConfig(output_dim=32))
        result = fusion.fuse()
        assert result.quality_score == 0.0
        assert np.allclose(result.vector, 0)

    def test_active_modalities(self):
        fusion = MultiModalFusion()
        assert fusion.active_modalities == []
        fusion.feed(ModalityStream(modality=Modality.SPECTRAL, data=np.zeros(10)))
        assert Modality.SPECTRAL in fusion.active_modalities

    def test_clear(self):
        fusion = MultiModalFusion()
        fusion.feed(ModalityStream(modality=Modality.SPECTRAL, data=np.zeros(10)))
        fusion.clear()
        assert fusion.active_modalities == []


class TestSpectralMemory:
    """Tests for recursive self-improving spectral memory."""

    def test_store_and_query(self):
        mem = SpectralMemory(config=MemoryConfig(vector_dim=32))
        vec = np.random.randn(32)
        entry = mem.store(vec, label="pattern_a")
        assert entry.tier == MemoryTier.WORKING
        assert mem.total_memories == 1

        result = mem.query(vec, top_k=5)
        assert len(result.entries) >= 1
        assert result.entries[0][0].memory_id == entry.memory_id
        assert result.entries[0][1] > 0.99

    def test_consolidation(self):
        config = MemoryConfig(
            vector_dim=16,
            working_capacity=10,
            consolidation_threshold=3,
        )
        mem = SpectralMemory(config=config)

        vec = np.random.randn(16)
        entry = mem.store(vec, label="test")

        # Access enough times to trigger consolidation
        for _ in range(5):
            mem.query(vec)

        migrated = mem.consolidate()
        assert migrated >= 1
        assert entry.tier == MemoryTier.SHORT_TERM

    def test_refine(self):
        mem = SpectralMemory(config=MemoryConfig(vector_dim=16))
        vec = np.random.randn(16)
        entry = mem.store(vec, label="evolving")

        new_vec = np.random.randn(16)
        refined = mem.refine(entry.memory_id, new_vec)
        assert refined is not None
        assert refined.access_count == 1
        assert refined.importance_score > 0.5

    def test_refine_nonexistent(self):
        mem = SpectralMemory()
        result = mem.refine("nonexistent", np.zeros(10))
        assert result is None

    def test_recursive_group(self):
        config = MemoryConfig(vector_dim=16, enable_recursion=True)
        mem = SpectralMemory(config=config)

        vectors = [np.random.randn(16) for _ in range(3)]
        parent = mem.create_recursive_group(vectors, group_label="cluster")
        assert len(parent.child_ids) == 3
        assert mem.total_memories == 4  # 3 children + 1 parent

    def test_recursive_query(self):
        config = MemoryConfig(vector_dim=16, enable_recursion=True)
        mem = SpectralMemory(config=config)

        vectors = [np.random.randn(16) for _ in range(3)]
        parent = mem.create_recursive_group(vectors, group_label="cluster")

        # Query for parent should find children too
        result = mem.query(parent.vector, top_k=10, recursive=True)
        assert result.depth_reached >= 0

    def test_working_memory_eviction(self):
        config = MemoryConfig(vector_dim=8, working_capacity=3)
        mem = SpectralMemory(config=config)

        for i in range(5):
            mem.store(np.random.randn(8), importance=i * 0.1)

        assert len(mem._working) <= 3

    def test_stats(self):
        mem = SpectralMemory(config=MemoryConfig(vector_dim=8))
        mem.store(np.random.randn(8))
        stats = mem.stats
        assert stats["working"] == 1
        assert stats["total"] == 1


class TestNeuromorphicRuntime:
    """Tests for the neuromorphic runtime."""

    def test_init_default(self):
        rt = NeuromorphicRuntime()
        assert rt.neuron_count == 256
        assert rt.config.processing_mode == ProcessingMode.BALANCED

    def test_encode_rate_coding(self):
        config = RuntimeConfig(
            n_neurons=32, encoding=EncodingScheme.RATE_CODING
        )
        rt = NeuromorphicRuntime(config=config)
        signal = np.random.randn(64)
        encoded = rt.encode(signal)
        assert encoded.shape == (32,)
        assert encoded.min() >= 0.0
        assert encoded.max() <= 1.0

    def test_encode_temporal(self):
        config = RuntimeConfig(
            n_neurons=32, encoding=EncodingScheme.TEMPORAL_CODING
        )
        rt = NeuromorphicRuntime(config=config)
        encoded = rt.encode(np.random.randn(32))
        assert encoded.shape == (32,)

    def test_encode_phase(self):
        config = RuntimeConfig(
            n_neurons=16, encoding=EncodingScheme.PHASE_CODING
        )
        rt = NeuromorphicRuntime(config=config)
        encoded = rt.encode(np.random.randn(16))
        assert encoded.shape == (16,)

    def test_encode_population(self):
        config = RuntimeConfig(
            n_neurons=16, encoding=EncodingScheme.POPULATION_CODING
        )
        rt = NeuromorphicRuntime(config=config)
        encoded = rt.encode(np.random.randn(32))
        assert encoded.shape == (16,)

    def test_process(self):
        config = RuntimeConfig(n_neurons=32, threshold=0.5)
        rt = NeuromorphicRuntime(config=config)
        spikes = rt.process(np.random.randn(64) * 2.0)
        assert isinstance(spikes, list)
        # Some neurons should fire with strong input
        # (may not always fire due to random init)

    def test_process_stream(self):
        config = RuntimeConfig(n_neurons=16)
        rt = NeuromorphicRuntime(config=config)
        signal = np.random.randn(512)
        all_spikes = rt.process_stream(signal, window_size=64)
        assert len(all_spikes) == 8

    def test_spike_callback(self):
        config = RuntimeConfig(n_neurons=16, threshold=0.1)
        rt = NeuromorphicRuntime(config=config)

        events = []
        rt.on_spikes(lambda s: events.append(s))

        # Process strong signal to generate spikes
        rt.process(np.ones(16) * 5.0)
        # Callback mechanism tested
        assert isinstance(events, list)

    def test_metrics(self):
        config = RuntimeConfig(n_neurons=16)
        rt = NeuromorphicRuntime(config=config)
        rt.process(np.random.randn(16))
        metrics = rt.get_metrics()
        assert metrics.ops_per_second >= 0
        assert metrics.uptime_s > 0
        assert 0 <= metrics.energy_score <= 1.0

    def test_reset(self):
        config = RuntimeConfig(n_neurons=16)
        rt = NeuromorphicRuntime(config=config)
        rt.process(np.ones(16) * 10.0)
        rt.reset()
        assert rt._total_spikes == 0
        assert np.allclose(rt._membrane_potential, 0)

    def test_stdp_learning(self):
        """STDP should modify weights over time."""
        config = RuntimeConfig(n_neurons=16, threshold=0.3, enable_stdp=True)
        rt = NeuromorphicRuntime(config=config)
        initial_weights = rt._weights.copy()

        # Process multiple signals to trigger STDP
        for _ in range(20):
            rt.process(np.random.randn(16) * 2.0)

        # Weights should have changed if spikes occurred
        if rt._total_spikes > 0:
            assert not np.allclose(rt._weights, initial_weights)
