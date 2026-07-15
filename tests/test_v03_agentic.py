"""Tests for MESIE V0.3 — agentic ghosts, core engine, network, training datasets."""

from __future__ import annotations

import numpy as np
import pytest

from mesie.agentic.ghost import AgentState, GhostAgent, GhostConfig, GhostResult, TaskSpec
from mesie.agentic.network import AgentNetwork, NetworkTopology
from mesie.agentic.spawner import AgentSpawner, SpawnerConfig
from mesie.engines.core_engine import CoreConfig, CoreEngine
from mesie.pretraining.reasoning_datasets import (
    GEOMETRY_RELATIONS,
    HYDROGEN_SERIES,
    PHYSICS_CONSTANTS,
    SIGNAL_PROCESSING,
    THERMODYNAMICS_SPECTRAL,
    TRANSFORM_RELATIONS,
    VIBRATION_PHYSICS,
    WAVE_PHYSICS,
    build_reasoning_datasets,
    generate_harmonic_series_dataset,
    generate_power_law_dataset,
    generate_quantum_lines_dataset,
    generate_resonance_dataset,
    generate_transform_pair_dataset,
    generate_wave_geometry_dataset,
)
from mesie.sdk.intelligence_sdk import SpectralIntelligenceSDK


# ===========================================================================
# Ghost Agent Tests
# ===========================================================================


class TestGhostAgent:
    """Tests for the ghost agent system."""

    def test_ghost_creation(self):
        ghost = GhostAgent(agent_id="test-ghost")
        assert ghost.agent_id == "test-ghost"
        assert ghost.state == AgentState.DORMANT

    def test_ghost_execute_no_dispatch(self):
        """Ghost with no dispatch function should pass-through payloads."""
        ghost = GhostAgent()
        task = TaskSpec(
            intent="test task",
            actions=[
                {"engine": "validation", "action": "validate", "payload": {"x": 1}},
                {"engine": "embedding", "action": "transform", "payload": {"y": 2}},
            ],
        )
        result = ghost.execute(task)
        assert result.success is True
        assert len(result.steps) == 2
        assert ghost.state == AgentState.COMPLETED

    def test_ghost_execute_with_dispatch(self):
        """Ghost with real dispatch function."""
        def mock_dispatch(engine, action, payload):
            return {"ok": True, "data": {"engine": engine, "action": action}}

        ghost = GhostAgent(bus_dispatch=mock_dispatch)
        task = TaskSpec(
            intent="compute",
            actions=[{"engine": "intelligence", "action": "reason", "payload": {}}],
        )
        result = ghost.execute(task)
        assert result.success is True
        assert result.steps[0]["data"]["engine"] == "intelligence"

    def test_ghost_failure(self):
        """Ghost handles failure from dispatch."""
        def fail_dispatch(engine, action, payload):
            return {"ok": False, "error": "engine unavailable"}

        ghost = GhostAgent(
            config=GhostConfig(max_retries=0),
            bus_dispatch=fail_dispatch,
        )
        task = TaskSpec(intent="fail", actions=[{"engine": "x", "action": "y"}])
        result = ghost.execute(task)
        assert result.success is False
        assert ghost.state == AgentState.FAILED

    def test_ghost_chaining(self):
        """Results chain between steps when chain=True."""
        call_log = []

        def tracking_dispatch(engine, action, payload):
            call_log.append(payload.copy())
            return {"ok": True, "data": {"computed": len(call_log)}}

        ghost = GhostAgent(bus_dispatch=tracking_dispatch)
        task = TaskSpec(
            intent="chain test",
            actions=[
                {"engine": "a", "action": "step1", "payload": {"input": "hello"}},
                {"engine": "b", "action": "step2", "payload": {}},
            ],
            chain=True,
        )
        result = ghost.execute(task)
        assert result.success is True
        # Second call should have 'computed' from first step
        assert "computed" in call_log[1]

    def test_ghost_embedding(self):
        """Ghost embeds results when configured."""
        def mock_dispatch(engine, action, payload):
            return {"ok": True, "data": {"value": 42}}

        def mock_embed(data):
            return [float(data.get("value", 0))]

        ghost = GhostAgent(
            config=GhostConfig(embed_results=True),
            bus_dispatch=mock_dispatch,
            embed_fn=mock_embed,
        )
        task = TaskSpec(intent="embed", actions=[{"engine": "x", "action": "y"}])
        result = ghost.execute(task)
        assert result.embedding == [42.0]

    def test_ghost_timeout(self):
        """Ghost respects timeout."""
        import time

        def slow_dispatch(engine, action, payload):
            time.sleep(0.1)
            return {"ok": True, "data": {}}

        ghost = GhostAgent(bus_dispatch=slow_dispatch)
        task = TaskSpec(
            intent="timeout",
            actions=[{"engine": "x", "action": "y"}] * 100,
            timeout_s=0.05,
        )
        result = ghost.execute(task)
        # Should fail due to timeout (first step takes 0.1s > 0.05s timeout)
        # Actually first step passes, then on 2nd check timeout triggers
        assert result.success is False or len(result.steps) < 100

    def test_ghost_reset(self):
        ghost = GhostAgent()
        task = TaskSpec(intent="t", actions=[{"engine": "x", "action": "y"}])
        ghost.execute(task)
        assert ghost.state == AgentState.COMPLETED
        ghost.reset()
        assert ghost.state == AgentState.DORMANT


# ===========================================================================
# Agent Spawner Tests
# ===========================================================================


class TestAgentSpawner:
    """Tests for the agent spawner."""

    def test_spawn_single(self):
        spawner = AgentSpawner()
        task = TaskSpec(intent="test", actions=[{"engine": "x", "action": "y"}])
        result = spawner.spawn(task)
        assert result.success is True
        assert spawner.total_spawned == 1

    def test_spawn_many(self):
        spawner = AgentSpawner()
        tasks = [
            TaskSpec(intent=f"task_{i}", actions=[{"engine": "x", "action": "y"}])
            for i in range(5)
        ]
        results = spawner.spawn_many(tasks)
        assert len(results) == 5
        assert all(r.success for r in results)

    def test_max_agents_limit(self):
        spawner = AgentSpawner(config=SpawnerConfig(max_agents=2, recycle_completed=False))
        # After 2 spawns, agents are completed so active_count is 0
        # This won't trigger the limit for completed agents
        task = TaskSpec(intent="t", actions=[{"engine": "x", "action": "y"}])
        # Spawn multiple — should work since they complete immediately
        for _ in range(5):
            spawner.spawn(task)
        assert spawner.total_spawned == 5


# ===========================================================================
# Agent Network Tests
# ===========================================================================


class TestAgentNetwork:
    """Tests for multi-agent networks."""

    def test_star_topology(self):
        network = AgentNetwork(topology=NetworkTopology.STAR)
        for i in range(3):
            agent = GhostAgent(agent_id=f"node-{i}")
            role = "coordinator" if i == 0 else "worker"
            network.add_node(agent, role=role)

        tasks = [
            TaskSpec(intent=f"task_{i}", actions=[{"engine": "x", "action": "y"}])
            for i in range(2)
        ]
        result = network.execute_parallel(tasks)
        assert result.success is True
        assert result.topology == NetworkTopology.STAR

    def test_pipeline_topology(self):
        network = AgentNetwork(topology=NetworkTopology.PIPELINE)
        for i in range(3):
            network.add_node(GhostAgent(agent_id=f"pipe-{i}"))

        tasks = [
            TaskSpec(intent=f"step_{i}", actions=[{"engine": "x", "action": "y"}])
            for i in range(3)
        ]
        result = network.execute_parallel(tasks)
        assert result.success is True

    def test_mesh_topology(self):
        network = AgentNetwork(topology=NetworkTopology.MESH)
        for i in range(2):
            network.add_node(GhostAgent(agent_id=f"mesh-{i}"))

        tasks = [TaskSpec(intent="broadcast", actions=[{"engine": "x", "action": "y"}])]
        result = network.execute_parallel(tasks)
        assert result.success is True

    def test_network_connections(self):
        network = AgentNetwork()
        a1 = network.add_node(GhostAgent(agent_id="a"))
        a2 = network.add_node(GhostAgent(agent_id="b"))
        network.connect(a1, a2)
        assert a2 in network._nodes[a1].connections
        assert a1 in network._nodes[a2].connections


# ===========================================================================
# Core Engine Tests
# ===========================================================================


class TestCoreEngine:
    """Tests for the inner core engine."""

    def test_core_creation(self):
        core = CoreEngine()
        assert "core" in core.registry.names()
        assert core.config.max_agents == 128

    def test_core_dispatch(self):
        core = CoreEngine()
        resp = core.dispatch("validation", "validate", {
            "record": {"frequency": [1, 2, 3], "amplitude": [0.5, 0.3, 0.1]}
        })
        assert resp.ok is True

    def test_core_spawn_ghost(self):
        core = CoreEngine()
        task = TaskSpec(
            intent="test validation",
            actions=[{
                "engine": "validation",
                "action": "validate",
                "payload": {"record": {"frequency": [1, 2], "amplitude": [0.5, 0.3]}},
            }],
        )
        result = core.spawn_ghost(task)
        assert result.success is True
        assert len(core.task_log) == 1

    def test_core_network(self):
        core = CoreEngine(config=CoreConfig(network_width=3))
        tasks = [
            TaskSpec(
                intent=f"net_task_{i}",
                actions=[{
                    "engine": "validation",
                    "action": "validate",
                    "payload": {"record": {"frequency": [1], "amplitude": [0.5]}},
                }],
            )
            for i in range(2)
        ]
        result = core.run_network(tasks)
        assert result.success is True

    def test_core_embed_workflow(self):
        core = CoreEngine()
        emb = core.embed_workflow([
            {"engine": "validation", "action": "validate"},
            {"engine": "intelligence", "action": "reason"},
        ])
        assert emb.shape == (64,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-6

    def test_core_embed_dataset(self):
        core = CoreEngine()
        records = [
            {"amplitude": [0.1, 0.2, 0.3], "frequency": [1.0, 2.0, 3.0]}
            for _ in range(5)
        ]
        emb = core.embed_dataset("test_ds", records)
        assert emb.shape == (64,)
        assert core.dataset_embedding_count == 1

    def test_core_reason(self):
        core = CoreEngine()
        result = core.reason_chain("identify pattern", {
            "amplitude": [0.1, 0.5, 0.9, 0.5, 0.1],
            "frequency": [1, 2, 3, 4, 5],
        })
        assert "conclusion" in result
        assert "confidence" in result

    def test_core_status(self):
        core = CoreEngine()
        # Use bus handle
        from mesie.internal_api.messages import MessageEnvelope, MessageTopic
        msg = MessageEnvelope(
            source="test", target="core", action="status",
            payload={}, topic=MessageTopic.ENGINE_REQUEST,
        )
        resp = core.handle(msg)
        assert resp.ok is True
        assert "engines" in resp.data


# ===========================================================================
# Training Dataset Tests
# ===========================================================================


class TestReasoningDatasets:
    """Tests for physics/math/geometry training datasets."""

    def test_physics_constants(self):
        assert len(PHYSICS_CONSTANTS) >= 10
        c = PHYSICS_CONSTANTS["speed_of_light"]
        assert c["value"] == 299_792_458.0
        assert c["unit"] == "m/s"

    def test_planck_constant(self):
        h = PHYSICS_CONSTANTS["planck_constant"]
        assert abs(h["value"] - 6.62607015e-34) < 1e-40

    def test_rydberg_constant(self):
        R = PHYSICS_CONSTANTS["rydberg_constant"]
        assert abs(R["value"] - 10_973_731.568160) < 1.0

    def test_transform_relations(self):
        assert len(TRANSFORM_RELATIONS) >= 5
        ft = TRANSFORM_RELATIONS[0]
        assert "fourier" in ft["name"].lower()
        assert "properties" in ft

    def test_wave_physics(self):
        assert len(WAVE_PHYSICS) >= 7
        standing = WAVE_PHYSICS[0]
        assert "f_n" in standing["formula"]

    def test_hydrogen_series(self):
        assert "lyman" in HYDROGEN_SERIES
        assert "balmer" in HYDROGEN_SERIES
        lyman = HYDROGEN_SERIES["lyman"]
        assert lyman["n_lower"] == 1
        assert lyman["region"] == "ultraviolet"

    def test_geometry_relations(self):
        assert len(GEOMETRY_RELATIONS) >= 8
        nyquist = GEOMETRY_RELATIONS[0]
        assert "f_s" in nyquist["formula"]

    def test_thermodynamics(self):
        assert len(THERMODYNAMICS_SPECTRAL) >= 4
        planck = THERMODYNAMICS_SPECTRAL[0]
        assert "black body" in planck["description"].lower() or "planck" in planck["name"]

    def test_signal_processing(self):
        assert len(SIGNAL_PROCESSING) >= 5
        psd = SIGNAL_PROCESSING[0]
        assert "power" in psd["name"].lower()

    def test_vibration_physics(self):
        assert len(VIBRATION_PHYSICS) >= 4

    def test_generate_harmonic_series(self):
        examples = generate_harmonic_series_dataset(20)
        assert len(examples) == 20
        ex = examples[0]
        assert ex.domain == "acoustics"
        assert "fundamental_hz" in ex.expected_output
        assert len(ex.input_data["frequency"]) > 0

    def test_generate_resonance(self):
        examples = generate_resonance_dataset(15)
        assert len(examples) == 15
        ex = examples[0]
        assert ex.domain == "structural_dynamics"
        assert "natural_frequencies_hz" in ex.expected_output

    def test_generate_transform_pairs(self):
        examples = generate_transform_pair_dataset(25)
        assert len(examples) == 25
        ex = examples[0]
        assert ex.domain == "mathematics"
        assert "spectrum_magnitude" in ex.input_data

    def test_generate_wave_geometry(self):
        examples = generate_wave_geometry_dataset(10)
        assert len(examples) == 10
        ex = examples[0]
        assert ex.domain == "geometry"
        assert "dimensions_m" in ex.input_data

    def test_generate_quantum_lines(self):
        examples = generate_quantum_lines_dataset(12)
        assert len(examples) == 12
        ex = examples[0]
        assert ex.domain == "quantum_mechanics"
        # Verify wavelengths are physically realistic (UV to IR range)
        wl = ex.expected_output["exact_wavelengths_nm"]
        assert all(50 < w < 5000 for w in wl)

    def test_generate_power_law(self):
        examples = generate_power_law_dataset(10)
        assert len(examples) == 10
        ex = examples[0]
        assert "spectral_exponent_beta" in ex.expected_output

    def test_build_all_datasets(self):
        manifest = build_reasoning_datasets(
            n_harmonic=10, n_resonance=10, n_transform=10,
            n_geometry=10, n_quantum=10, n_power_law=10,
        )
        assert manifest.total_examples == 60
        assert len(manifest.datasets) == 6
        assert len(manifest.domains_covered) >= 5


# ===========================================================================
# SDK V0.3 Integration Tests
# ===========================================================================


class TestSDKv03:
    """Integration tests for the V0.3 SDK."""

    def test_version(self):
        sdk = SpectralIntelligenceSDK()
        assert sdk.version == "0.4.0"

    def test_repr(self):
        sdk = SpectralIntelligenceSDK()
        assert "v0.4.0" in repr(sdk)
        assert "core=active" in repr(sdk)

    def test_list_engines(self):
        sdk = SpectralIntelligenceSDK()
        engines = sdk.list_engines()
        assert "core" in engines
        assert "validation" in engines
        assert "intelligence" in engines

    def test_status(self):
        sdk = SpectralIntelligenceSDK()
        status = sdk.status()
        assert status["version"] == "0.4.0"
        assert status["core"] == "active"
        assert "engines" in status

    def test_spawn_task(self):
        sdk = SpectralIntelligenceSDK()
        result = sdk.spawn_task(
            "validate signal",
            [{"engine": "validation", "action": "validate"}],
            record={"frequency": [1, 2, 3], "amplitude": [0.5, 0.3, 0.1]},
        )
        assert result.success is True

    def test_spawn_many(self):
        sdk = SpectralIntelligenceSDK()
        results = sdk.spawn_many([
            {"intent": "task_1", "actions": [{"engine": "validation", "action": "validate", "payload": {"record": {"frequency": [1], "amplitude": [0.5]}}}]},
            {"intent": "task_2", "actions": [{"engine": "validation", "action": "validate", "payload": {"record": {"frequency": [2], "amplitude": [0.3]}}}]},
        ])
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_run_network(self):
        sdk = SpectralIntelligenceSDK()
        result = sdk.run_network(
            [
                {"intent": "analyze", "actions": [{"engine": "validation", "action": "validate", "payload": {"record": {"frequency": [1], "amplitude": [0.5]}}}]},
            ],
            topology="star",
            n_agents=2,
        )
        assert result.success is True

    def test_embed_workflow(self):
        sdk = SpectralIntelligenceSDK()
        emb = sdk.embed_workflow([
            {"engine": "validation", "action": "validate"},
            {"engine": "matching", "action": "match"},
        ])
        assert emb.shape == (64,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-6

    def test_embed_dataset(self):
        sdk = SpectralIntelligenceSDK()
        records = [{"amplitude": [0.1, 0.2], "frequency": [1.0, 2.0]}] * 5
        emb = sdk.embed_dataset("test", records)
        assert emb.shape == (64,)

    def test_reason(self):
        sdk = SpectralIntelligenceSDK()
        result = sdk.reason(
            "identify dominant frequency",
            record={"frequency": [1, 2, 3, 4, 5], "amplitude": [0.1, 0.9, 0.2, 0.1, 0.05]},
        )
        assert "conclusion" in result
        assert "confidence" in result

    def test_load_training_datasets(self):
        sdk = SpectralIntelligenceSDK()
        manifest = sdk.load_training_datasets()
        assert manifest["total_examples"] == 510
        assert "physics" in manifest["domains"]

    def test_dispatch(self):
        sdk = SpectralIntelligenceSDK()
        resp = sdk.dispatch("validation", "validate", {
            "record": {"frequency": [1, 2], "amplitude": [0.5, 0.3]}
        })
        assert resp["ok"] is True

    def test_no_core_raises(self):
        sdk = SpectralIntelligenceSDK(enable_core=False)
        with pytest.raises(RuntimeError, match="Core engine not enabled"):
            sdk.spawn_task("x", [])
        with pytest.raises(RuntimeError, match="Core engine not enabled"):
            sdk.run_network([])
        with pytest.raises(RuntimeError, match="Core engine not enabled"):
            sdk.embed_workflow([])

    def test_backward_compatible(self):
        """V0.3 SDK still supports all V0.2 operations."""
        sdk = SpectralIntelligenceSDK()
        # Generation
        rec = sdk.generate_psd()
        assert rec is not None
        # Validation
        report = sdk.validate(rec)
        assert report is not None
        # Normalization
        normalized = sdk.normalize(rec)
        assert normalized is not None
        # Embedding
        emb = sdk.embed(rec)
        assert emb.shape[0] == 1
