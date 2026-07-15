"""Tests for the miniverse nesting module — recursive containment, scale-bridging, and downward attention."""

import time

import numpy as np
import pytest

from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.cognitive.miniverse import (
    ContainedEngine,
    DownwardAttention,
    RecursiveMemoryContainer,
    ScaleBridge,
    ScaleBridgeConfig,
)
from mesie.matching.matcher import MatchResult, SpectralMatcher
from mesie.pretraining.spectral_memory import MemoryEntry, SpectralMemoryStore


def _make_record(record_id: str = "rec1", freq_shift: float = 0.0) -> MultiElementRecord:
    """Create a simple test record."""
    freq = np.linspace(0.1, 10.0, 64)
    amp = np.sin(freq + freq_shift) + 1.5
    comp = SpectralComponent(
        name="test_component",
        frequency=freq,
        amplitude=amp,
    )
    return MultiElementRecord(
        record_id=record_id,
        components=[comp],
    )


def _make_match_result(score: float = 0.85) -> MatchResult:
    """Create a test MatchResult."""
    return MatchResult(
        reference_id="ref1",
        candidate_id="cand1",
        score=score,
        metrics={"cosine_similarity": score, "rmse": 0.1},
        component_scores={"test_component": score},
    )


# ---------------------------------------------------------------------------
# Scale-Bridging Protocol Tests
# ---------------------------------------------------------------------------


class TestScaleBridge:
    """Tests for MatchResult → MemoryEntry promotion."""

    def test_promote_creates_memory_entry(self):
        bridge = ScaleBridge()
        result = _make_match_result(score=0.8)
        record = _make_record("candidate_1")

        entry = bridge.promote(result, record, timestamp=100.0)

        assert isinstance(entry, MemoryEntry)
        assert entry.timestamp == 100.0
        assert entry.importance > 0
        assert entry.metadata["match_score"] == 0.8
        assert entry.metadata["source_reference_id"] == "ref1"
        assert entry.metadata["source_candidate_id"] == "cand1"

    def test_promote_importance_increases_with_resonance(self):
        bridge = ScaleBridge()
        # Higher amplitude peaks → higher resonance → higher importance
        freq = np.linspace(0.1, 10.0, 64)
        low_res = MultiElementRecord(
            record_id="low",
            components=[SpectralComponent(
                name="c", frequency=freq, amplitude=np.ones(64),
            )],
        )
        high_res = MultiElementRecord(
            record_id="high",
            components=[SpectralComponent(
                name="c", frequency=freq,
                amplitude=np.ones(64) + 5.0 * (np.arange(64) == 32),
            )],
        )

        result = _make_match_result(score=0.5)
        entry_low = bridge.promote(result, low_res, timestamp=1.0)
        entry_high = bridge.promote(result, high_res, timestamp=2.0)

        assert entry_high.importance > entry_low.importance

    def test_promote_event_type_from_score(self):
        bridge = ScaleBridge()
        record = _make_record()

        high_result = _make_match_result(score=0.95)
        entry = bridge.promote(high_result, record)
        assert entry.event_type == "anomaly"

        low_result = _make_match_result(score=0.3)
        entry_low = bridge.promote(low_result, record)
        assert entry_low.event_type == "normal"

    def test_batch_promote(self):
        bridge = ScaleBridge()
        results = [_make_match_result(0.7), _make_match_result(0.9)]
        records = [_make_record("r1"), _make_record("r2")]

        entries = bridge.batch_promote(results, records, base_timestamp=50.0)

        assert len(entries) == 2
        assert entries[0].timestamp == 50.0
        assert entries[1].timestamp == 51.0

    def test_min_importance_floor(self):
        config = ScaleBridgeConfig(min_importance=0.5)
        bridge = ScaleBridge(config=config)
        # Very low score should still get min_importance
        result = _make_match_result(score=0.0)
        record = _make_record()

        entry = bridge.promote(result, record)
        assert entry.importance >= 0.5


# ---------------------------------------------------------------------------
# Recursive Containment Tests
# ---------------------------------------------------------------------------


class TestRecursiveMemoryContainer:
    """Tests for recursive containment of inner MESIE engines."""

    def test_contain_creates_engine(self):
        container = RecursiveMemoryContainer()
        refs = [_make_record("ref1"), _make_record("ref2", freq_shift=0.5)]

        contained = container.contain(refs)

        assert isinstance(contained, ContainedEngine)
        assert len(container.contained_engines) == 1
        assert contained.activation_count == 0

    def test_query_and_resonate_returns_results(self):
        container = RecursiveMemoryContainer()
        refs = [_make_record("ref1"), _make_record("ref2", freq_shift=0.1)]
        container.contain(refs, activation_threshold=0.0)

        candidate = _make_record("candidate", freq_shift=0.05)
        results = container.query_and_resonate(candidate)

        # Should get at least one result since threshold is 0
        assert len(results) >= 1
        assert all(isinstance(r, MatchResult) for r in results)

    def test_query_and_resonate_empty_container(self):
        container = RecursiveMemoryContainer()
        candidate = _make_record("candidate")

        results = container.query_and_resonate(candidate)
        assert results == []

    def test_activation_count_increments(self):
        container = RecursiveMemoryContainer()
        refs = [_make_record("ref1")]
        contained = container.contain(refs, activation_threshold=0.0)

        candidate = _make_record("candidate")
        container.query_and_resonate(candidate)

        assert contained.activation_count >= 1

    def test_query_promote_and_store(self):
        container = RecursiveMemoryContainer()
        refs = [_make_record("ref1")]
        container.contain(refs, activation_threshold=0.0)

        candidate = _make_record("candidate")
        promoted = container.query_promote_and_store(candidate, timestamp=200.0)

        assert len(promoted) >= 1
        assert all(isinstance(e, MemoryEntry) for e in promoted)
        # Memory store should have grown (initial contain + promoted)
        assert container.memory_store.size >= 2

    def test_multiple_contained_engines(self):
        container = RecursiveMemoryContainer()
        container.contain([_make_record("a1")], activation_threshold=0.0)
        container.contain([_make_record("b1", freq_shift=2.0)], activation_threshold=0.0)

        assert len(container.contained_engines) == 2

    def test_high_threshold_filters_weak_matches(self):
        container = RecursiveMemoryContainer()
        refs = [_make_record("ref1", freq_shift=5.0)]
        container.contain(refs, activation_threshold=0.99)

        # Candidate very different from reference
        candidate = _make_record("candidate", freq_shift=-5.0)
        results = container.query_and_resonate(candidate)

        # High threshold should filter out weak matches
        # (may or may not produce results depending on spectral overlap)
        assert all(r.score >= 0.99 for r in results)


# ---------------------------------------------------------------------------
# Downward Attention Tests
# ---------------------------------------------------------------------------


class TestDownwardAttention:
    """Tests for outer-layer attention over inner engines."""

    def test_compute_attention_empty(self):
        da = DownwardAttention()
        weights = da.compute_attention_over_engines()
        assert weights == {}

    def test_compute_attention_weights_sum_to_one(self):
        container = RecursiveMemoryContainer()
        container.contain([_make_record("a")])
        container.contain([_make_record("b", freq_shift=1.0)])
        container.contain([_make_record("c", freq_shift=2.0)])

        da = DownwardAttention(container=container)
        weights = da.compute_attention_over_engines()

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_compute_attention_with_query(self):
        container = RecursiveMemoryContainer()
        container.contain([_make_record("a")])
        container.contain([_make_record("b", freq_shift=3.0)])

        da = DownwardAttention(container=container)
        query = _make_record("q")  # Similar to "a"
        weights = da.compute_attention_over_engines(query=query)

        assert len(weights) == 2
        # Engine 0 (similar to query) should have higher weight
        assert weights[0] > weights[1]

    def test_amplify_boosts_importance(self):
        container = RecursiveMemoryContainer()
        container.contain([_make_record("a")])
        container.contain([_make_record("b", freq_shift=1.0)])

        da = DownwardAttention(container=container, amplification_factor=3.0)
        original_importances = [
            e.memory_entry.importance for e in container.contained_engines
        ]

        amplified = da.amplify()

        assert len(amplified) >= 1
        # At least one engine should have boosted importance
        new_importances = [
            e.memory_entry.importance for e in container.contained_engines
        ]
        assert any(n > o for n, o in zip(new_importances, original_importances))

    def test_focused_resonate(self):
        container = RecursiveMemoryContainer()
        container.contain([_make_record("a")], activation_threshold=0.0)
        container.contain([_make_record("b", freq_shift=3.0)], activation_threshold=0.0)

        da = DownwardAttention(container=container)
        candidate = _make_record("q")

        results = da.focused_resonate(
            candidate, attention_threshold=0.0, top_k=2
        )

        assert all(isinstance(r, MatchResult) for r in results)

    def test_focused_resonate_with_context(self):
        container = RecursiveMemoryContainer()
        container.contain([_make_record("a")], activation_threshold=0.0)
        container.contain([_make_record("b", freq_shift=3.0)], activation_threshold=0.0)

        da = DownwardAttention(container=container)
        candidate = _make_record("q")
        context = _make_record("ctx")  # Context similar to "a"

        results = da.focused_resonate(
            candidate, context_query=context, attention_threshold=0.0
        )

        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Integration Test
# ---------------------------------------------------------------------------


class TestMiniverseIntegration:
    """End-to-end integration test of the miniverse nesting architecture."""

    def test_full_pipeline(self):
        """Test: contain → query → resonate → promote → store → attend."""
        # 1. Create container with scale bridge
        bridge = ScaleBridge()
        store = SpectralMemoryStore(capacity=100)
        container = RecursiveMemoryContainer(
            memory_store=store,
            scale_bridge=bridge,
            activation_threshold=0.0,
        )

        # 2. Contain two pattern sets (inner engines)
        pattern_a = [_make_record("a1"), _make_record("a2", freq_shift=0.1)]
        pattern_b = [_make_record("b1", freq_shift=2.0), _make_record("b2", freq_shift=2.1)]
        container.contain(pattern_a)
        container.contain(pattern_b)

        # 3. Query with a candidate similar to pattern_a
        candidate = _make_record("query", freq_shift=0.05)
        results = container.query_and_resonate(candidate)
        assert len(results) >= 1

        # 4. Promote results to outer memory
        promoted = container.query_promote_and_store(candidate, timestamp=1000.0)
        assert len(promoted) >= 1

        # 5. Apply downward attention
        da = DownwardAttention(container=container, amplification_factor=2.0)
        weights = da.compute_attention_over_engines(query=candidate)
        assert len(weights) == 2

        # 6. Engine similar to candidate should get more attention
        amplified = da.amplify(query=candidate)
        assert len(amplified) >= 1

        # 7. Verify memory store has accumulated entries
        assert store.size >= 3  # 2 containments + at least 1 promotion
