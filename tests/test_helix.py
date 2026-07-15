"""Tests for the Vector Helix module."""

import numpy as np
import pytest

from mesie.helix import (
    VectorHelix,
    HelixConfig,
    HelixNode,
    HelixTraversalResult,
    HelixEncoder,
    HelixProjection,
    HelixRetriever,
    HelixSearchResult,
)
from mesie.core.records import MultiElementRecord, SpectralComponent


def _make_record(record_id: str = "test-001", seed: int = 42) -> MultiElementRecord:
    """Create a test record."""
    rng = np.random.RandomState(seed)
    freq = np.linspace(0.1, 10.0, 64)
    amp = np.abs(rng.randn(64)) + 0.5
    comp = SpectralComponent(
        name="E1",
        frequency=freq,
        amplitude=amp,
    )
    return MultiElementRecord(record_id=record_id, components=[comp])


class TestVectorHelix:
    def test_create_default(self):
        helix = VectorHelix()
        assert helix.size == 0
        assert helix.progression == 0.0

    def test_create_with_config(self):
        config = HelixConfig(pitch=2.0, base_radius=0.5, phase_resolution=32)
        helix = VectorHelix(config=config)
        assert helix.config.pitch == 2.0
        assert helix.config.base_radius == 0.5

    def test_insert_single(self):
        helix = VectorHelix()
        record = _make_record()
        node = helix.insert(record)

        assert isinstance(node, HelixNode)
        assert node.record_id == "test-001"
        assert isinstance(node.embedding, np.ndarray)
        assert isinstance(node.helix_position, np.ndarray)
        assert len(node.helix_position) == 3
        assert 0.0 <= node.phase < 2 * np.pi
        assert node.radius > 0
        assert helix.size == 1

    def test_insert_batch(self):
        helix = VectorHelix()
        records = [_make_record(f"rec-{i}", seed=i) for i in range(5)]
        nodes = helix.insert_batch(records)
        assert len(nodes) == 5
        assert helix.size == 5
        assert helix.progression == 5.0

    def test_progression_increments(self):
        helix = VectorHelix()
        for i in range(3):
            node = helix.insert(_make_record(f"rec-{i}", seed=i))
            assert node.progression == float(i)

    def test_helix_position_varies(self):
        helix = VectorHelix()
        records = [_make_record(f"rec-{i}", seed=i * 10) for i in range(3)]
        nodes = helix.insert_batch(records)
        # Positions should differ
        positions = [n.helix_position for n in nodes]
        assert not np.allclose(positions[0], positions[1])

    def test_traverse_full(self):
        helix = VectorHelix()
        records = [_make_record(f"rec-{i}", seed=i) for i in range(5)]
        helix.insert_batch(records)

        result = helix.traverse()
        assert isinstance(result, HelixTraversalResult)
        assert len(result.path) == 5
        assert result.total_arc_length > 0
        assert 0.0 <= result.phase_continuity <= 1.0
        assert result.coherence_integral > 0
        assert len(result.unwound_vector) > 0

    def test_traverse_range(self):
        helix = VectorHelix()
        records = [_make_record(f"rec-{i}", seed=i) for i in range(10)]
        helix.insert_batch(records)

        result = helix.traverse(start_progression=2.0, end_progression=5.0)
        assert all(2.0 <= n.progression <= 5.0 for n in result.path)

    def test_traverse_with_phase_window(self):
        helix = VectorHelix()
        records = [_make_record(f"rec-{i}", seed=i) for i in range(20)]
        helix.insert_batch(records)

        result = helix.traverse(phase_window=(0.0, np.pi))
        assert all(0.0 <= n.phase <= np.pi for n in result.path)

    def test_query_by_phase(self):
        helix = VectorHelix()
        records = [_make_record(f"rec-{i}", seed=i) for i in range(20)]
        helix.insert_batch(records)

        results = helix.query_by_phase(np.pi, tolerance=0.5)
        for node in results:
            diff = abs(node.phase - np.pi)
            diff = min(diff, 2 * np.pi - diff)
            assert diff <= 0.5

    def test_query_by_coherence(self):
        helix = VectorHelix()
        records = [_make_record(f"rec-{i}", seed=i) for i in range(10)]
        helix.insert_batch(records)

        results = helix.query_by_coherence(min_coherence=0.0)
        assert len(results) == 10  # All should have coherence >= 0

    def test_compute_helix_distance(self):
        helix = VectorHelix()
        node_a = helix.insert(_make_record("a", seed=1))
        node_b = helix.insert(_make_record("b", seed=100))
        dist = helix.compute_helix_distance(node_a, node_b)
        assert dist > 0

    def test_get_statistics(self):
        helix = VectorHelix()
        records = [_make_record(f"rec-{i}", seed=i) for i in range(10)]
        helix.insert_batch(records)

        stats = helix.get_helix_statistics()
        assert stats["node_count"] == 10
        assert stats["mean_coherence"] > 0
        assert stats["phase_coverage"] > 0
        assert stats["progression_range"] == 9.0
        assert stats["mean_radius"] > 0

    def test_empty_statistics(self):
        helix = VectorHelix()
        stats = helix.get_helix_statistics()
        assert stats["node_count"] == 0

    def test_max_nodes_eviction(self):
        config = HelixConfig(max_nodes=5)
        helix = VectorHelix(config=config)
        records = [_make_record(f"rec-{i}", seed=i) for i in range(10)]
        helix.insert_batch(records)
        assert helix.size <= 5


class TestHelixEncoder:
    def test_encode_single(self):
        encoder = HelixEncoder()
        record = _make_record()
        proj = encoder.encode(record)

        assert isinstance(proj, HelixProjection)
        assert proj.record_id == "test-001"
        assert 0.0 <= proj.phase < 2 * np.pi
        assert proj.radius > 0
        assert proj.elevation == 0.0
        assert len(proj.cartesian) == 3
        assert encoder.encode_count == 1

    def test_encode_batch(self):
        encoder = HelixEncoder()
        records = [_make_record(f"rec-{i}", seed=i) for i in range(5)]
        projections = encoder.encode_batch(records)
        assert len(projections) == 5
        assert encoder.encode_count == 5

    def test_explicit_elevation(self):
        encoder = HelixEncoder()
        proj = encoder.encode(_make_record(), elevation=5.0)
        assert proj.elevation == 5.0

    def test_phase_distance(self):
        encoder = HelixEncoder()
        proj_a = encoder.encode(_make_record("a", seed=1))
        proj_b = encoder.encode(_make_record("b", seed=100))
        dist = encoder.compute_phase_distance(proj_a, proj_b)
        assert 0.0 <= dist <= np.pi

    def test_helix_distance(self):
        encoder = HelixEncoder()
        proj_a = encoder.encode(_make_record("a", seed=1))
        proj_b = encoder.encode(_make_record("b", seed=100))
        dist = encoder.compute_helix_distance(proj_a, proj_b)
        assert dist > 0

    def test_phase_offset(self):
        encoder = HelixEncoder(phase_offset=np.pi)
        proj = encoder.encode(_make_record())
        assert 0.0 <= proj.phase < 2 * np.pi


class TestHelixRetriever:
    def test_create_default(self):
        retriever = HelixRetriever()
        assert retriever.index_size == 0

    def test_index_records(self):
        retriever = HelixRetriever()
        records = [_make_record(f"rec-{i}", seed=i) for i in range(5)]
        count = retriever.index(records)
        assert count == 5
        assert retriever.index_size == 5

    def test_search(self):
        retriever = HelixRetriever()
        records = [_make_record(f"rec-{i}", seed=i) for i in range(10)]
        retriever.index(records)

        query = _make_record("query", seed=0)
        results = retriever.search(query, top_k=3)

        assert len(results) <= 3
        assert all(isinstance(r, HelixSearchResult) for r in results)
        # Results should be sorted by distance
        for i in range(len(results) - 1):
            assert results[i].distance <= results[i + 1].distance

    def test_search_with_phase_tolerance(self):
        retriever = HelixRetriever()
        records = [_make_record(f"rec-{i}", seed=i) for i in range(20)]
        retriever.index(records)

        query = _make_record("query", seed=5)
        results = retriever.search(query, top_k=5, phase_tolerance=0.5)
        # All results should be within phase tolerance
        assert all(isinstance(r, HelixSearchResult) for r in results)

    def test_search_with_coherence_filter(self):
        retriever = HelixRetriever()
        records = [_make_record(f"rec-{i}", seed=i) for i in range(10)]
        retriever.index(records)

        results = retriever.search(_make_record("q", seed=0), min_coherence=0.0)
        assert len(results) > 0

    def test_search_by_phase_range(self):
        retriever = HelixRetriever()
        records = [_make_record(f"rec-{i}", seed=i) for i in range(20)]
        retriever.index(records)

        results = retriever.search_by_phase_range(0.0, np.pi)
        # Results sorted by coherence descending
        for i in range(len(results) - 1):
            assert results[i].coherence_score >= results[i + 1].coherence_score

    def test_empty_search(self):
        retriever = HelixRetriever()
        results = retriever.search(_make_record("q"), top_k=5)
        assert results == []
