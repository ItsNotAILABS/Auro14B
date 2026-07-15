"""Tests for TAURUS memory system and NeuroCores."""

import time

import numpy as np
import pytest

from mesie.cognitive.taurus_memory import (
    MemoryTrace,
    RetrievalResult,
    TaurusMemoryStore,
    TaurusWorkingMemory,
)
from mesie.cognitive.neurocores import (
    CoreProcessingResult,
    NeuroCoreCluster,
    NeuroCoreConfig,
    SpectralNeuroCore,
)


class TestMemoryTrace:
    """Tests for MemoryTrace."""

    def test_effective_strength_fresh(self):
        trace = MemoryTrace(embedding=np.ones(10), importance=1.0)
        strength = trace.effective_strength()
        assert strength > 0.0

    def test_effective_strength_increases_with_access(self):
        trace = MemoryTrace(embedding=np.ones(10), importance=1.0)
        s1 = trace.effective_strength()
        trace.access_count = 5
        s2 = trace.effective_strength()
        assert s2 > s1

    def test_effective_strength_scales_with_importance(self):
        trace_low = MemoryTrace(embedding=np.ones(10), importance=0.1)
        trace_high = MemoryTrace(embedding=np.ones(10), importance=2.0)
        assert trace_high.effective_strength() > trace_low.effective_strength()


class TestTaurusMemoryStore:
    """Tests for TaurusMemoryStore."""

    def test_store_and_retrieve(self):
        store = TaurusMemoryStore(capacity=100)
        emb = np.random.randn(64)
        store.store(embedding=emb, context={"test": True}, importance=1.0)
        assert store.size == 1

        results = store.retrieve(query=emb, top_k=1)
        assert len(results) == 1
        assert results[0].similarity > 0.9

    def test_capacity_enforcement(self):
        store = TaurusMemoryStore(capacity=5)
        for i in range(10):
            store.store(embedding=np.random.randn(32), importance=float(i))
        assert store.size <= 5

    def test_semantic_filter(self):
        store = TaurusMemoryStore(capacity=100)
        store.store(embedding=np.ones(16), semantic_tag="alpha")
        store.store(embedding=np.ones(16) * 2, semantic_tag="beta")

        results = store.retrieve(np.ones(16), top_k=10, semantic_filter="alpha")
        assert all(r.trace.semantic_tag == "alpha" for r in results)

    def test_retrieve_by_attention(self):
        store = TaurusMemoryStore(capacity=100)
        emb = np.random.randn(32)
        store.store(embedding=emb, importance=1.0)

        attention = np.ones(32)
        results = store.retrieve_by_attention(emb, attention, top_k=1)
        assert len(results) >= 1

    def test_get_attention_analysis_empty(self):
        store = TaurusMemoryStore(capacity=10)
        analysis = store.get_attention_analysis()
        assert analysis["total_traces"] == 0
        assert analysis["attention_entropy"] == 0.0

    def test_get_attention_analysis_populated(self):
        store = TaurusMemoryStore(capacity=100)
        for _ in range(5):
            store.store(embedding=np.random.randn(16), importance=np.random.rand())
        analysis = store.get_attention_analysis()
        assert analysis["total_traces"] == 5
        assert analysis["attention_entropy"] > 0.0

    def test_clear(self):
        store = TaurusMemoryStore(capacity=100)
        store.store(embedding=np.ones(8))
        store.clear()
        assert store.size == 0

    def test_store_count(self):
        store = TaurusMemoryStore(capacity=100)
        store.store(embedding=np.ones(8))
        store.store(embedding=np.ones(8))
        assert store.store_count == 2


class TestTaurusWorkingMemory:
    """Tests for TaurusWorkingMemory."""

    def test_hold_and_scan(self):
        wm = TaurusWorkingMemory(capacity=5)
        emb = np.random.randn(32)
        wm.hold(embedding=emb)
        result = wm.scan(emb)
        assert result is not None
        assert result.similarity > 0.9

    def test_capacity_eviction(self):
        wm = TaurusWorkingMemory(capacity=3)
        for i in range(5):
            wm.hold(embedding=np.random.randn(16))
        assert wm.size <= 3

    def test_promotion_to_long_term(self):
        ltm = TaurusMemoryStore(capacity=100)
        wm = TaurusWorkingMemory(capacity=2, promotion_threshold=2, long_term_store=ltm)

        # Hold an item and access it multiple times
        emb = np.random.randn(16)
        trace = wm.hold(embedding=emb)
        trace.access_count = 3  # Simulate repeated access

        # Fill working memory to trigger eviction
        wm.hold(embedding=np.random.randn(16))
        wm.hold(embedding=np.random.randn(16))

        # The promoted item should be in long-term memory
        assert ltm.size >= 1

    def test_scan_empty(self):
        wm = TaurusWorkingMemory(capacity=5)
        result = wm.scan(np.ones(16))
        assert result is None


class TestSpectralNeuroCore:
    """Tests for SpectralNeuroCore."""

    def test_process_basic(self):
        config = NeuroCoreConfig(d_model=32, n_attention_heads=4)
        core = SpectralNeuroCore(config)
        result = core.process(np.random.randn(64))

        assert isinstance(result, CoreProcessingResult)
        assert result.embedding.shape == (32,)
        assert result.attention_map.shape == (32,)
        assert len(result.multi_scale_features) == config.multi_scale_levels

    def test_attention_analysis(self):
        config = NeuroCoreConfig(d_model=16, n_attention_heads=2)
        core = SpectralNeuroCore(config)
        core.process(np.random.randn(32))
        core.process(np.random.randn(32))

        analysis = core.get_attention_analysis()
        assert analysis["n_processed"] == 2
        assert "mean_entropy" in analysis
        assert "mean_max_attention" in analysis
        assert "mean_sparsity" in analysis
        assert "memory_analysis" in analysis

    def test_cross_band_detection(self):
        config = NeuroCoreConfig(d_model=64, enable_cross_band=True)
        core = SpectralNeuroCore(config)
        result = core.process(np.random.randn(128))
        assert result.cross_band_scores is not None
        assert result.cross_band_scores.shape[0] == result.cross_band_scores.shape[1]

    def test_harmonic_detection(self):
        config = NeuroCoreConfig(d_model=32, enable_harmonics=True)
        core = SpectralNeuroCore(config)
        # Create spectrum with clear harmonic peaks
        spectrum = np.zeros(128)
        spectrum[10] = 5.0  # Fundamental
        spectrum[20] = 3.0  # 2nd harmonic
        spectrum[30] = 2.0  # 3rd harmonic
        result = core.process(spectrum)
        assert result.harmonic_peaks is not None

    def test_memory_integration(self):
        config = NeuroCoreConfig(d_model=16, memory_capacity=50)
        core = SpectralNeuroCore(config)
        core.process(np.random.randn(32))
        core.process(np.random.randn(32))
        assert core.memory_store.size > 0
        assert core.processing_count == 2

    def test_no_memory_storage(self):
        config = NeuroCoreConfig(d_model=16, memory_capacity=50)
        core = SpectralNeuroCore(config)
        core.process(np.random.randn(32), store_in_memory=False)
        assert core.memory_store.size == 0


class TestNeuroCoreCluster:
    """Tests for NeuroCoreCluster."""

    def test_distributed_processing(self):
        cluster = NeuroCoreCluster(n_cores=3, config=NeuroCoreConfig(d_model=16))
        results = cluster.process_distributed(np.random.randn(32))
        assert len(results) == 3
        for r in results:
            assert r.embedding.shape == (16,)

    def test_ensemble_embedding(self):
        cluster = NeuroCoreCluster(n_cores=4, config=NeuroCoreConfig(d_model=32))
        embedding = cluster.get_ensemble_embedding(np.random.randn(64))
        assert embedding.shape == (32,)

    def test_cluster_attention_analysis(self):
        cluster = NeuroCoreCluster(n_cores=2, config=NeuroCoreConfig(d_model=16))
        cluster.process_distributed(np.random.randn(32))
        analysis = cluster.get_cluster_attention_analysis()
        assert analysis["n_cores"] == 2
        assert analysis["total_processed"] == 2
        assert "per_core_analysis" in analysis

    def test_total_memory_size(self):
        cluster = NeuroCoreCluster(n_cores=2, config=NeuroCoreConfig(d_model=16, memory_capacity=20))
        cluster.process_distributed(np.random.randn(32))
        assert cluster.total_memory_size > 0
