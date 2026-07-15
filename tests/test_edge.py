"""Tests for the edge communication module (Hz-ladder, satellite nodes, protocol)."""

import json
import math
import pathlib

import numpy as np
import pytest

from mesie.edge.hz_ladder import (
    SPEED_OF_LIGHT_M_S,
    STANDARD_TIERS,
    FrequencyTier,
    HzLadder,
    LadderLink,
    compute_doppler_shift,
    compute_free_space_loss,
    compute_link_budget_dB,
)
from mesie.edge.satellite_nodes import (
    ECO_HZ_REFERENCES,
    ORBITAL_TIERS,
    EcoHzReference,
    OrbitalTier,
    SatelliteEdgeNode,
    VirtualNodeNetwork,
    compute_contact_duration,
    compute_orbital_frequency,
)
from mesie.edge.edge_protocol import (
    EdgeMessage,
    EdgeMessageType,
    EdgeRoute,
    EdgeSpectralProtocol,
    SpectralHandshake,
)


DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "spectral_library"


class TestHzLadder:
    """Tests for Hz-ladder frequency tier math."""

    def test_standard_tiers_exist(self):
        assert len(STANDARD_TIERS) == 7
        assert STANDARD_TIERS[0].name == "ELF/Schumann"
        assert STANDARD_TIERS[-1].name == "Optical/Laser"

    def test_tier_frequencies_increase(self):
        for i in range(len(STANDARD_TIERS) - 1):
            assert STANDARD_TIERS[i].center_frequency_Hz < STANDARD_TIERS[i + 1].center_frequency_Hz

    def test_ladder_construction(self):
        ladder = HzLadder()
        assert len(ladder.tiers) == 7
        assert ladder.get_tier(0) is not None
        assert ladder.get_tier(6) is not None
        assert ladder.get_tier(99) is None

    def test_frequency_to_tier(self):
        ladder = HzLadder()
        tier = ladder.frequency_to_tier(7.83)
        assert tier is not None
        assert tier.tier_id == 0

        tier = ladder.frequency_to_tier(14e9)
        assert tier is not None
        assert tier.tier_id == 4

    def test_route_vertical(self):
        ladder = HzLadder()
        route = ladder.route_vertical(0, 4)
        assert len(route) == 4
        assert route[0].source_tier == 0
        assert route[-1].dest_tier == 4

    def test_route_downward(self):
        ladder = HzLadder()
        route = ladder.route_vertical(5, 2)
        assert len(route) == 3

    def test_ladder_spectrum(self):
        ladder = HzLadder()
        spectrum = ladder.ladder_spectrum()
        assert len(spectrum) == 7
        assert spectrum[0] == 7.83

    def test_shannon_capacity(self):
        tier = STANDARD_TIERS[4]  # SHF
        cap = tier.shannon_capacity_bps
        assert cap > 0
        assert cap > tier.bandwidth_Hz  # SNR > 0 dB means capacity > bandwidth

    def test_free_space_loss(self):
        # GPS L1 at 20,200 km
        loss = compute_free_space_loss(1.5754e9, 20_200_000.0)
        assert 180.0 < loss < 190.0  # Known to be ~182.5 dB

    def test_free_space_loss_edge_cases(self):
        assert compute_free_space_loss(0, 1000) == 0.0
        assert compute_free_space_loss(1e9, 0) == 0.0

    def test_doppler_shift(self):
        # LEO at 7.5 km/s, Ku-band
        shifted = compute_doppler_shift(14e9, 7500.0)
        doppler = shifted - 14e9
        assert 300_000 < doppler < 400_000  # ~350 kHz

    def test_link_budget(self):
        received = compute_link_budget_dB(
            tx_power_dBW=10.0,
            tx_gain_dBi=35.0,
            rx_gain_dBi=40.0,
            frequency_Hz=14e9,
            distance_m=550_000.0,
        )
        # Should be negative (weak signal) but reasonable
        assert -120.0 < received < 0.0


class TestOrbitalTiers:
    """Tests for orbital tier physics."""

    def test_orbital_tiers_computed(self):
        for tier in ORBITAL_TIERS:
            assert tier.orbital_frequency_Hz > 0
            assert tier.orbital_period_s > 0
            assert tier.velocity_m_s > 0

    def test_leo_faster_than_geo(self):
        leo = ORBITAL_TIERS[0]  # LEO 300 km
        geo = ORBITAL_TIERS[-1]  # GEO
        assert leo.velocity_m_s > geo.velocity_m_s
        assert leo.orbital_frequency_Hz > geo.orbital_frequency_Hz

    def test_compute_orbital_frequency(self):
        # GEO should be ~1/86164 Hz
        geo_freq = compute_orbital_frequency(35786)
        assert abs(geo_freq - 1.16e-5) < 1e-7

    def test_contact_duration(self):
        duration = compute_contact_duration(550, min_elevation_deg=10.0)
        assert 200 < duration < 800  # LEO pass is typically 5-10 minutes

    def test_eco_hz_references(self):
        assert len(ECO_HZ_REFERENCES) > 0
        schumann = ECO_HZ_REFERENCES[0]
        assert schumann.frequency_Hz == 7.83
        assert "schumann" in schumann.name


class TestSatelliteEdgeNode:
    """Tests for satellite edge nodes."""

    def test_node_creation(self):
        tier = OrbitalTier(name="LEO_550", altitude_km=550)
        node = SatelliteEdgeNode(
            node_id="test-001",
            orbital_tier=tier,
        )
        assert node.node_id == "test-001"
        assert node.active

    def test_doppler_at_max_rate(self):
        tier = OrbitalTier(name="LEO_550", altitude_km=550)
        node = SatelliteEdgeNode(node_id="test-001", orbital_tier=tier)
        doppler = node.doppler_at_max_rate()
        assert doppler > 0
        assert doppler < 1_000_000  # Less than 1 MHz shift

    def test_path_loss_to_ground(self):
        tier = OrbitalTier(name="LEO_550", altitude_km=550)
        node = SatelliteEdgeNode(node_id="test-001", orbital_tier=tier)
        loss = node.path_loss_to_ground_dB()
        assert 155 < loss < 170  # Ku-band at 550 km

    def test_data_volume_per_pass(self):
        tier = OrbitalTier(name="LEO_550", altitude_km=550)
        node = SatelliteEdgeNode(node_id="test-001", orbital_tier=tier)
        volume = node.data_volume_per_pass_bits()
        assert volume > 0


class TestVirtualNodeNetwork:
    """Tests for virtual node network."""

    def test_default_constellation(self):
        network = VirtualNodeNetwork()
        network.create_default_constellation()
        assert len(network.nodes) == 7  # 4 LEO + 2 MEO + 1 GEO

    def test_get_nodes_at_tier(self):
        network = VirtualNodeNetwork()
        network.create_default_constellation()
        leo_nodes = network.get_nodes_at_tier("LEO_550")
        assert len(leo_nodes) == 4

    def test_compute_route(self):
        network = VirtualNodeNetwork()
        network.create_default_constellation()
        route = network.compute_route("leo-edge-000", "geo-backbone-000")
        assert "error" not in route
        assert route["total_latency_ms"] > 0

    def test_network_orbital_frequencies(self):
        network = VirtualNodeNetwork()
        network.create_default_constellation()
        freqs = network.network_orbital_frequencies()
        assert len(freqs) > 0
        # Should have distinct frequencies for LEO, MEO, GEO
        assert len(set(freqs.tolist())) >= 3

    def test_total_compute_capacity(self):
        network = VirtualNodeNetwork()
        network.create_default_constellation()
        total = network.total_compute_capacity()
        assert total > 0


class TestEdgeProtocol:
    """Tests for edge spectral protocol."""

    @pytest.fixture
    def network_with_nodes(self):
        network = VirtualNodeNetwork()
        network.create_default_constellation()
        return network

    def test_handshake(self, network_with_nodes):
        proto = EdgeSpectralProtocol(network_with_nodes)
        hs = proto.initiate_handshake("leo-edge-000", "leo-edge-001")
        assert hs.established
        assert hs.agreed_frequency_Hz > 0
        assert hs.eco_hz_sync > 0

    def test_send_spectral_data(self, network_with_nodes):
        proto = EdgeSpectralProtocol(network_with_nodes)
        freqs = np.linspace(1e9, 2e9, 100)
        amps = np.random.rand(100)
        msg = proto.send_spectral_data("leo-edge-000", "geo-backbone-000", freqs, amps)
        assert msg.message_type == EdgeMessageType.SPECTRAL_RECORD
        assert msg.frequency_Hz > 0
        assert proto.pending_messages == 1

    def test_send_beacon(self, network_with_nodes):
        proto = EdgeSpectralProtocol(network_with_nodes)
        msg = proto.send_beacon("leo-edge-000")
        assert msg.message_type == EdgeMessageType.BEACON
        assert msg.payload["eco_hz"] == 7.83

    def test_process_queue(self, network_with_nodes):
        proto = EdgeSpectralProtocol(network_with_nodes)
        proto.send_beacon("leo-edge-000")
        delivered = proto.process_queue()
        assert len(delivered) == 1
        assert proto.pending_messages == 0
        assert proto.delivered_count == 1

    def test_compute_route(self, network_with_nodes):
        proto = EdgeSpectralProtocol(network_with_nodes)
        route = proto.compute_route("leo-edge-000", "geo-backbone-000")
        assert route.source_id == "leo-edge-000"
        assert route.dest_id == "geo-backbone-000"
        assert route.total_latency_ms > 0


class TestSpectralLibraryData:
    """Tests that spectral library JSON files contain real, valid data."""

    def test_hydrogen_spectrum_exists(self):
        path = DATA_DIR / "hydrogen_spectrum.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["element"] == "hydrogen"
        # Verify real H-alpha line
        balmer = data["series"]["balmer"]["lines"]
        h_alpha = balmer[0]
        assert abs(h_alpha["wavelength_nm"] - 656.281) < 1.0
        assert abs(h_alpha["frequency_Hz"] - 4.568e14) < 1e12

    def test_hydrogen_21cm(self):
        path = DATA_DIR / "hydrogen_spectrum.json"
        data = json.loads(path.read_text())
        h21 = data["hydrogen_21cm"]
        assert abs(h21["frequency_MHz"] - 1420.405752) < 0.001

    def test_electromagnetic_bands_exists(self):
        path = DATA_DIR / "electromagnetic_bands.json"
        assert path.exists()
        data = json.loads(path.read_text())
        bands = data["radio_bands"]
        assert "ELF" in bands
        assert "EHF" in bands
        assert bands["ELF"]["frequency_low_Hz"] == 3.0

    def test_satellite_frequencies_exists(self):
        path = DATA_DIR / "satellite_frequencies.json"
        assert path.exists()
        data = json.loads(path.read_text())
        # Verify real GPS L1 frequency
        gps_l1 = data["constellations"]["GPS"]["signals"]["L1"]
        assert gps_l1["center_frequency_Hz"] == 1575420000

    def test_atmospheric_absorption_exists(self):
        path = DATA_DIR / "atmospheric_absorption.json"
        assert path.exists()
        data = json.loads(path.read_text())
        windows = data["transmission_windows"]["windows"]
        assert len(windows) > 5

    def test_schumann_resonances_exists(self):
        path = DATA_DIR / "schumann_resonances.json"
        assert path.exists()
        data = json.loads(path.read_text())
        modes = data["schumann_resonances"]["modes"]
        assert modes[0]["frequency_Hz"] == 7.83
        assert modes[1]["frequency_Hz"] == 14.3
