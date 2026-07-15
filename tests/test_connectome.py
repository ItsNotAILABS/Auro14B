"""Tests for the connectome module — 3D brain intelligence backend."""

import numpy as np
import pytest

from mesie.connectome import (
    ActivationState,
    BrainRegion,
    BrainSystem,
    ConnectomeEnvironment3D,
    ConnectomeGraph,
    Connection,
    SignalPacket,
    build_default_connectome,
    get_default_regions,
    get_region_positions,
    get_regions_by_system,
)


# === Brain Regions Tests ===


class TestBrainRegions:
    def test_default_regions_not_empty(self):
        regions = get_default_regions()
        assert len(regions) > 30  # We defined 44 regions

    def test_all_regions_have_3d_positions(self):
        for region in get_default_regions():
            assert len(region.position_3d) == 3
            assert all(isinstance(c, float) for c in region.position_3d)

    def test_all_regions_have_system(self):
        for region in get_default_regions():
            assert isinstance(region.system, BrainSystem)

    def test_all_systems_represented(self):
        regions = get_default_regions()
        systems = {r.system for r in regions}
        for s in BrainSystem:
            assert s in systems

    def test_get_regions_by_system(self):
        prefrontal = get_regions_by_system(BrainSystem.PREFRONTAL)
        assert len(prefrontal) >= 4
        assert all(r.system == BrainSystem.PREFRONTAL for r in prefrontal)

    def test_get_region_positions_shape(self):
        regions = get_default_regions()
        positions = get_region_positions(regions)
        assert positions.shape == (len(regions), 3)
        assert positions.dtype == np.float64

    def test_position_array_property(self):
        region = BrainRegion(
            name="Test",
            abbreviation="TST",
            system=BrainSystem.PREFRONTAL,
            position_3d=(10.0, 20.0, 30.0),
        )
        arr = region.position_array
        np.testing.assert_array_equal(arr, [10.0, 20.0, 30.0])


# === Connectome Graph Tests ===


class TestConnectomeGraph:
    def test_default_construction(self):
        cg = ConnectomeGraph()
        assert cg.num_regions > 30
        assert cg.num_connections == 0  # No connections by default

    def test_add_connection(self):
        cg = ConnectomeGraph()
        conn = cg.add_connection("DLPFC_L", "PPC_L", weight=0.8)
        assert conn.source == "DLPFC_L"
        assert conn.target == "PPC_L"
        assert conn.weight == 0.8
        assert conn.distance_mm > 0
        assert conn.delay_ms > 0

    def test_add_connection_invalid_region(self):
        cg = ConnectomeGraph()
        with pytest.raises(KeyError):
            cg.add_connection("FAKE_REGION", "DLPFC_L")

    def test_connectivity_matrix(self):
        cg = ConnectomeGraph()
        cg.add_connection("DLPFC_L", "PPC_L", weight=0.8)
        matrix = cg.build_connectivity_matrix()
        assert matrix.shape[0] == cg.num_regions
        assert matrix.shape[1] == cg.num_regions
        # Should be symmetric
        np.testing.assert_array_equal(matrix, matrix.T)

    def test_distance_matrix(self):
        cg = ConnectomeGraph()
        dist = cg.build_distance_matrix()
        assert dist.shape == (cg.num_regions, cg.num_regions)
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(dist), 0.0)
        # All distances non-negative
        assert np.all(dist >= 0)

    def test_build_default_connectome(self):
        cg = build_default_connectome()
        assert cg.num_regions > 30
        assert cg.num_connections > 50
        # Check adjacency populated
        assert len(cg.adjacency) > 0


# === Environment Tests ===


class TestConnectomeEnvironment3D:
    def test_initialization(self):
        env = ConnectomeEnvironment3D()
        assert env.num_regions > 30
        assert env.time_ms == 0.0

    def test_all_activations_start_zero(self):
        env = ConnectomeEnvironment3D()
        for val in env.get_all_activations().values():
            assert val == 0.0

    def test_inject_stimulus(self):
        env = ConnectomeEnvironment3D(noise_level=0.0)
        env.inject_stimulus("V1", amplitude=0.5)
        assert env.get_activation("V1") == 0.5

    def test_inject_stimulus_invalid_region(self):
        env = ConnectomeEnvironment3D()
        with pytest.raises(KeyError):
            env.inject_stimulus("NONEXISTENT")

    def test_step_advances_time(self):
        env = ConnectomeEnvironment3D()
        env.step(5)
        assert env.time_ms == 5.0

    def test_signal_propagation(self):
        env = ConnectomeEnvironment3D(
            noise_level=0.0, decay_rate=0.0, propagation_gain=1.0
        )
        env.inject_stimulus("V1", amplitude=0.9)
        # Run enough steps for signals to propagate
        env.run(duration_ms=30.0)
        # V2V3 is directly connected to V1 — should have activation
        assert env.get_activation("V2V3") > 0.0

    def test_decay(self):
        env = ConnectomeEnvironment3D(
            noise_level=0.0, decay_rate=0.1, propagation_gain=0.0
        )
        env.inject_stimulus("V1", amplitude=1.0)
        initial = env.get_activation("V1")
        env.step(1)
        assert env.get_activation("V1") < initial

    def test_run_returns_states(self):
        env = ConnectomeEnvironment3D()
        states = env.run(duration_ms=10.0)
        assert len(states) == 10
        assert all(isinstance(s, ActivationState) for s in states)

    def test_get_system_activation(self):
        env = ConnectomeEnvironment3D(noise_level=0.0)
        env.inject_stimulus("V1", amplitude=1.0)
        act = env.get_system_activation(BrainSystem.OCCIPITAL)
        assert act > 0.0

    def test_get_dominant_system(self):
        env = ConnectomeEnvironment3D(noise_level=0.0)
        env.inject_stimulus("V1", amplitude=1.0)
        env.inject_stimulus("V2V3", amplitude=1.0)
        env.inject_stimulus("FFA", amplitude=1.0)
        dominant = env.get_dominant_system()
        assert dominant == BrainSystem.OCCIPITAL

    def test_global_coherence_zero_when_inactive(self):
        env = ConnectomeEnvironment3D(noise_level=0.0)
        assert env.compute_global_coherence() == 0.0

    def test_get_3d_state(self):
        env = ConnectomeEnvironment3D()
        env.inject_stimulus("DLPFC_L", amplitude=0.5)
        state = env.get_3d_state()
        assert "positions" in state
        assert "activations" in state
        assert "edges" in state
        assert "labels" in state
        assert state["positions"].shape[0] == env.num_regions
        assert state["positions"].shape[1] == 3
        assert state["num_regions"] == env.num_regions

    def test_reset(self):
        env = ConnectomeEnvironment3D()
        env.inject_stimulus("V1", amplitude=1.0)
        env.run(duration_ms=5.0)
        env.reset()
        assert env.time_ms == 0.0
        assert all(v == 0.0 for v in env.get_all_activations().values())

    def test_history(self):
        env = ConnectomeEnvironment3D()
        env.inject_stimulus("V1", amplitude=0.5)
        env.run(duration_ms=5.0)
        history = env.get_history()
        assert len(history) == 5

    def test_activation_vector(self):
        env = ConnectomeEnvironment3D(noise_level=0.0)
        vec = env.get_activation_vector()
        assert vec.shape == (env.num_regions,)
        assert np.all(vec == 0.0)
