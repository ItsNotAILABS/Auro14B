"""Tests for the integration module — AI connector, library bridge, and orchestrator."""

import numpy as np
import pytest

from mesie.integration import (
    AISystemConnector,
    ConnectorConfig,
    LibraryBridge,
    BridgeState,
    PipelineOrchestrator,
    OrchestratorConfig,
)
from mesie.core.records import MultiElementRecord, SpectralComponent


def _make_record(record_id: str = "test-001") -> MultiElementRecord:
    """Create a test record."""
    freq = np.linspace(0.1, 10.0, 64)
    amp = np.sin(freq) + 0.5
    comp = SpectralComponent(
        name="E1",
        frequency=freq,
        amplitude=amp,
    )
    return MultiElementRecord(record_id=record_id, components=[comp])


class TestAISystemConnector:
    def test_create_default(self):
        connector = AISystemConnector()
        assert connector.processed_count == 0
        assert connector.connected_systems == []

    def test_connect_systems(self):
        connector = AISystemConnector()
        connector.connect("embeddings")
        connector.connect("reasoning")
        assert "embeddings" in connector.connected_systems
        assert "reasoning" in connector.connected_systems

    def test_connect_no_duplicates(self):
        connector = AISystemConnector()
        connector.connect("embeddings")
        connector.connect("embeddings")
        assert len(connector.connected_systems) == 1

    def test_ingest_record(self):
        connector = AISystemConnector()
        record = _make_record()
        result = connector.ingest(record)

        assert result["record_id"] == "test-001"
        assert "embedding" in result
        assert isinstance(result["embedding"], np.ndarray)
        assert "spectral_signature" in result
        assert "memory_object" in result
        assert "state_vector" in result
        assert "anomaly_score" in result
        assert "lineage" in result
        assert connector.processed_count == 1

    def test_ingest_with_topology(self):
        config = ConnectorConfig(enable_topology=True)
        node_graph = {"N1": ["N2"], "N2": ["N1"]}
        connector = AISystemConnector(config=config, node_graph=node_graph)
        record = _make_record()
        result = connector.ingest(record)
        assert "topology_weights" in result

    def test_reason(self):
        connector = AISystemConnector()
        record = _make_record()
        result = connector.reason(record)
        assert result.conclusion in ("normal_operation", "anomaly_detected", "low_signal")
        assert 0.0 <= result.confidence <= 1.0

    def test_compute_attention(self):
        connector = AISystemConnector()
        records = [_make_record(f"rec-{i}") for i in range(3)]
        weights = connector.compute_attention(records)
        assert len(weights) == 3
        assert abs(np.sum(weights) - 1.0) < 1e-6

    def test_index_and_retrieve(self):
        connector = AISystemConnector()
        records = [_make_record(f"rec-{i}") for i in range(5)]
        connector.index_records(records)
        results = connector.retrieve_similar(records[0], top_k=3)
        assert len(results) <= 3

    def test_fit_anomaly_baseline(self):
        connector = AISystemConnector()
        records = [_make_record(f"rec-{i}") for i in range(5)]
        connector.fit_anomaly_baseline(records)
        result = connector.ingest(_make_record("new"))
        assert "anomaly_score" in result


class TestLibraryBridge:
    def test_initial_state(self):
        bridge = LibraryBridge()
        assert bridge.state == BridgeState.IDLE
        assert bridge.cache_size == 0

    def test_activate(self):
        bridge = LibraryBridge()
        bridge.activate()
        assert bridge.state == BridgeState.CONNECTED

    def test_process_record(self):
        bridge = LibraryBridge()
        bridge.activate()
        record = _make_record()
        result = bridge.process_record(record)

        assert result["record_id"] == "test-001"
        assert "embedding" in result
        assert "features" in result
        assert "signature" in result
        assert bridge.state == BridgeState.CONNECTED
        assert bridge.cache_size == 1

    def test_event_hooks(self):
        bridge = LibraryBridge()
        bridge.activate()
        events_received = []

        def on_processed(**kwargs):
            events_received.append(kwargs)

        bridge.register_hook("record_processed", on_processed)
        bridge.process_record(_make_record())
        assert len(events_received) == 1
        assert "record_id" in events_received[0]

    def test_shared_state(self):
        bridge = LibraryBridge()
        bridge.set_shared("threshold", 0.8)
        assert bridge.get_shared("threshold") == 0.8
        assert bridge.get_shared("missing", "default") == "default"

    def test_match_records(self):
        bridge = LibraryBridge()
        bridge.activate()
        ref = _make_record("ref")
        candidates = [_make_record(f"cand-{i}") for i in range(3)]
        results = bridge.match_records(ref, candidates)
        assert len(results) == 3


class TestPipelineOrchestrator:
    def test_create_default(self):
        orch = PipelineOrchestrator()
        assert orch.config.stages == ["embed", "extract_features", "reason", "route"]

    def test_run_pipeline(self):
        orch = PipelineOrchestrator()
        record = _make_record()
        result = orch.run(record)

        assert "record_id" in result
        assert "stages_completed" in result
        assert "embed" in result["stages_completed"]
        assert "extract_features" in result["stages_completed"]
        assert "reason" in result["stages_completed"]
        assert "route" in result["stages_completed"]

    def test_pipeline_caching(self):
        orch = PipelineOrchestrator()
        record = _make_record()
        result1 = orch.run(record)
        result2 = orch.run(record)
        assert result1 is result2  # Same cached object

    def test_clear_cache(self):
        orch = PipelineOrchestrator()
        record = _make_record()
        orch.run(record)
        orch.clear_cache()
        result = orch.run(record)
        assert "stages_completed" in result

    def test_custom_stage(self):
        orch = PipelineOrchestrator(config=OrchestratorConfig(
            stages=["embed", "custom"],
            enable_caching=False,
        ))

        def custom_handler(context):
            return {"custom_output": True}

        orch.register_stage("custom", custom_handler)
        result = orch.run(_make_record())
        assert "custom" in result["stages_completed"]
        assert result["stage_results"]["custom"] == {"custom_output": True}

    def test_run_batch(self):
        orch = PipelineOrchestrator(config=OrchestratorConfig(enable_caching=False))
        records = [_make_record(f"rec-{i}") for i in range(3)]
        results = orch.run_batch(records)
        assert len(results) == 3

    def test_routing_anomaly(self):
        orch = PipelineOrchestrator(config=OrchestratorConfig(enable_caching=False))
        # Create a record with high variability to trigger anomaly
        freq = np.linspace(0.1, 10.0, 64)
        amp = np.random.RandomState(42).exponential(scale=10.0, size=64)
        comp = SpectralComponent(
            name="E1", frequency=freq, amplitude=amp
        )
        record = MultiElementRecord(record_id="anomaly", components=[comp])
        result = orch.run(record)
        assert "route" in result["stage_results"]
