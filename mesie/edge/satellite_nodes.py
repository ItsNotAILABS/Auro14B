"""Satellite edge virtual nodes using eco-Hz and orbital-Hz.

Implements virtual compute/relay nodes positioned at satellite orbital
tiers. Each node is characterized by its orbital frequency (derived from
Kepler's laws) and communicates using the Hz-ladder vertical backbone.

Eco-Hz: Natural Earth frequencies (Schumann resonances, geophysical
oscillations) used as timing references and low-power sync beacons.

Orbital-Hz: The orbital frequency of a satellite (revolutions per second)
which determines Doppler shift, contact windows, and handover timing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mesie.edge.hz_ladder import (
    EARTH_RADIUS_M,
    MU_EARTH,
    SPEED_OF_LIGHT_M_S,
    FrequencyTier,
    HzLadder,
    compute_doppler_shift,
    compute_free_space_loss,
)


@dataclass
class OrbitalTier:
    """An orbital altitude tier with derived physical parameters.

    All parameters are computed from altitude using Keplerian mechanics.

    Attributes:
        name: Tier name (e.g., 'LEO_550').
        altitude_km: Orbital altitude in km.
        orbital_frequency_Hz: Revolutions per second.
        orbital_period_s: Time for one complete orbit.
        velocity_m_s: Orbital velocity.
        max_elevation_window_s: Maximum ground contact time per pass.
        slant_range_max_m: Maximum slant range to ground station.
    """

    name: str
    altitude_km: float
    orbital_frequency_Hz: float = 0.0
    orbital_period_s: float = 0.0
    velocity_m_s: float = 0.0
    max_elevation_window_s: float = 0.0
    slant_range_max_m: float = 0.0

    def __post_init__(self) -> None:
        r = EARTH_RADIUS_M + self.altitude_km * 1000.0
        self.orbital_period_s = 2.0 * math.pi * math.sqrt(r**3 / MU_EARTH)
        self.orbital_frequency_Hz = 1.0 / self.orbital_period_s
        self.velocity_m_s = math.sqrt(MU_EARTH / r)
        # Approximate max contact window (10° elevation mask)
        half_angle = math.acos(EARTH_RADIUS_M / r)
        arc_fraction = (2 * half_angle) / (2 * math.pi)
        self.max_elevation_window_s = arc_fraction * self.orbital_period_s
        # Max slant range at horizon
        self.slant_range_max_m = math.sqrt(r**2 - EARTH_RADIUS_M**2)


# Real orbital tiers with data from orbital mechanics
ORBITAL_TIERS: List[OrbitalTier] = [
    OrbitalTier(name="LEO_300", altitude_km=300),
    OrbitalTier(name="LEO_550", altitude_km=550),   # Starlink shell 1
    OrbitalTier(name="LEO_780", altitude_km=780),   # Iridium
    OrbitalTier(name="LEO_1200", altitude_km=1200), # OneWeb
    OrbitalTier(name="MEO_2000", altitude_km=2000),
    OrbitalTier(name="MEO_20200", altitude_km=20200),  # GPS
    OrbitalTier(name="GEO", altitude_km=35786),
]


@dataclass
class EcoHzReference:
    """Natural Earth frequency reference for eco-aware timing and sync.

    Uses real Schumann resonances and geophysical oscillations as
    ultra-low-power synchronization beacons for edge networks.

    Attributes:
        name: Reference signal name.
        frequency_Hz: Frequency in Hz.
        source: Physical origin of the frequency.
        stability: Fractional frequency stability.
        applications: Use cases in edge network.
    """

    name: str
    frequency_Hz: float
    source: str
    stability: float = 1e-4
    applications: List[str] = field(default_factory=list)


# Real eco-Hz references from Schumann resonances and geophysics
ECO_HZ_REFERENCES: List[EcoHzReference] = [
    EcoHzReference(
        name="schumann_1",
        frequency_Hz=7.83,
        source="earth_ionosphere_cavity_mode_1",
        stability=5e-3,
        applications=["global_timing_beacon", "ionosphere_health_monitor"],
    ),
    EcoHzReference(
        name="schumann_2",
        frequency_Hz=14.3,
        source="earth_ionosphere_cavity_mode_2",
        stability=5e-3,
        applications=["timing_harmonic_check"],
    ),
    EcoHzReference(
        name="schumann_3",
        frequency_Hz=20.8,
        source="earth_ionosphere_cavity_mode_3",
        stability=5e-3,
        applications=["timing_harmonic_check"],
    ),
    EcoHzReference(
        name="earth_rotation",
        frequency_Hz=1.1574e-5,
        source="diurnal_rotation",
        stability=1e-8,
        applications=["day_boundary_sync", "solar_power_scheduling"],
    ),
    EcoHzReference(
        name="lunar_tide_M2",
        frequency_Hz=2.2362e-5,
        source="lunar_gravitational_tide",
        stability=1e-10,
        applications=["ocean_monitoring", "long_period_calibration"],
    ),
    EcoHzReference(
        name="circadian_fundamental",
        frequency_Hz=1.1574e-5,
        source="biological_24h_rhythm",
        stability=1e-3,
        applications=["human_interface_timing", "power_management"],
    ),
]


def compute_orbital_frequency(altitude_km: float) -> float:
    """Compute orbital frequency from altitude using Kepler's third law.

    Args:
        altitude_km: Orbital altitude above Earth's surface in km.

    Returns:
        Orbital frequency in Hz (revolutions per second).
    """
    r = EARTH_RADIUS_M + altitude_km * 1000.0
    period_s = 2.0 * math.pi * math.sqrt(r**3 / MU_EARTH)
    return 1.0 / period_s


def compute_contact_duration(altitude_km: float, min_elevation_deg: float = 10.0) -> float:
    """Compute maximum contact duration for a LEO satellite pass.

    Args:
        altitude_km: Satellite altitude in km.
        min_elevation_deg: Minimum elevation angle in degrees.

    Returns:
        Maximum contact time in seconds.
    """
    r = EARTH_RADIUS_M + altitude_km * 1000.0
    elev_rad = math.radians(min_elevation_deg)
    # Earth central angle from satellite to horizon
    cos_angle = EARTH_RADIUS_M * math.cos(elev_rad) / r
    if abs(cos_angle) > 1.0:
        return 0.0
    central_angle = math.acos(cos_angle) - elev_rad
    arc_fraction = (2.0 * central_angle) / (2.0 * math.pi)
    period_s = 2.0 * math.pi * math.sqrt(r**3 / MU_EARTH)
    return arc_fraction * period_s


@dataclass
class SatelliteEdgeNode:
    """A virtual compute node at a satellite edge position.

    Represents a processing/relay node in the orbital network that
    communicates via the Hz-ladder backbone. Each node has:
    - An orbital tier determining its physical parameters
    - Communication frequencies for up/down links
    - Edge compute capacity for data processing
    - Eco-Hz references for timing synchronization

    Attributes:
        node_id: Unique node identifier.
        orbital_tier: Orbital position parameters.
        comm_frequency_Hz: Primary communication frequency.
        uplink_frequency_Hz: Ground-to-satellite frequency.
        downlink_frequency_Hz: Satellite-to-ground frequency.
        crosslink_frequency_Hz: Inter-satellite link frequency.
        eco_hz_refs: Eco-Hz timing references.
        compute_capacity_flops: Edge compute in FLOPS.
        storage_bytes: Available storage.
        active: Whether node is currently active.
        metadata: Additional node properties.
    """

    node_id: str
    orbital_tier: OrbitalTier
    comm_frequency_Hz: float = 14_000_000_000.0  # Ku-band default
    uplink_frequency_Hz: float = 14_000_000_000.0
    downlink_frequency_Hz: float = 11_700_000_000.0
    crosslink_frequency_Hz: float = 60_000_000_000.0  # V-band ISL
    eco_hz_refs: List[EcoHzReference] = field(default_factory=list)
    compute_capacity_flops: float = 1e12  # 1 TFLOPS default
    storage_bytes: int = 1_000_000_000_000  # 1 TB
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def doppler_at_max_rate(self) -> float:
        """Maximum Doppler shift at full orbital velocity (worst case)."""
        return compute_doppler_shift(
            self.comm_frequency_Hz, self.orbital_tier.velocity_m_s
        ) - self.comm_frequency_Hz

    def path_loss_to_ground_dB(self) -> float:
        """Free-space path loss to ground station at nadir."""
        distance_m = self.orbital_tier.altitude_km * 1000.0
        return compute_free_space_loss(self.downlink_frequency_Hz, distance_m)

    def path_loss_crosslink_dB(self, distance_km: float = 1000.0) -> float:
        """Path loss to neighboring satellite via crosslink."""
        return compute_free_space_loss(
            self.crosslink_frequency_Hz, distance_km * 1000.0
        )

    def contact_window_s(self) -> float:
        """Maximum contact window with a ground station."""
        return compute_contact_duration(self.orbital_tier.altitude_km)

    def data_volume_per_pass_bits(self, data_rate_bps: float = 1_000_000_000.0) -> float:
        """Maximum data volume transferable in one ground contact pass.

        Args:
            data_rate_bps: Downlink data rate in bits per second.

        Returns:
            Total data volume in bits.
        """
        return data_rate_bps * self.contact_window_s()


class VirtualNodeNetwork:
    """Network of virtual satellite edge nodes across orbital tiers.

    Manages a constellation of edge nodes, computes inter-node routes
    using the Hz-ladder, and provides network-wide spectral scheduling.

    Args:
        hz_ladder: The Hz-ladder backbone for vertical communication.
        nodes: Initial set of edge nodes.
    """

    def __init__(
        self,
        hz_ladder: Optional[HzLadder] = None,
        nodes: Optional[Sequence[SatelliteEdgeNode]] = None,
    ) -> None:
        self.hz_ladder = hz_ladder or HzLadder()
        self.nodes: Dict[str, SatelliteEdgeNode] = {}
        if nodes:
            for node in nodes:
                self.nodes[node.node_id] = node

    def add_node(self, node: SatelliteEdgeNode) -> None:
        """Add a satellite edge node to the network."""
        self.nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the network."""
        self.nodes.pop(node_id, None)

    def get_nodes_at_tier(self, tier_name: str) -> List[SatelliteEdgeNode]:
        """Get all nodes at a specific orbital tier."""
        return [n for n in self.nodes.values() if n.orbital_tier.name == tier_name]

    def compute_route(
        self, source_id: str, dest_id: str
    ) -> Dict[str, Any]:
        """Compute communication route between two nodes.

        Uses the Hz-ladder to find the vertical path and computes
        total latency and path loss.

        Args:
            source_id: Source node ID.
            dest_id: Destination node ID.

        Returns:
            Route dictionary with path, latency, and loss info.
        """
        source = self.nodes.get(source_id)
        dest = self.nodes.get(dest_id)
        if source is None or dest is None:
            return {"error": "Node not found", "path": []}

        # Determine Hz-ladder tiers for each node
        src_tier = self.hz_ladder.frequency_to_tier(source.comm_frequency_Hz)
        dst_tier = self.hz_ladder.frequency_to_tier(dest.comm_frequency_Hz)

        if src_tier is None or dst_tier is None:
            return {"error": "No matching ladder tier", "path": []}

        ladder_path = self.hz_ladder.route_vertical(src_tier.tier_id, dst_tier.tier_id)

        # Compute total latency: propagation + ladder transitions
        propagation_latency_ms = (
            source.orbital_tier.altitude_km * 1000.0 / SPEED_OF_LIGHT_M_S * 1000.0
            + dest.orbital_tier.altitude_km * 1000.0 / SPEED_OF_LIGHT_M_S * 1000.0
        )
        ladder_latency_ms = sum(link.latency_ms for link in ladder_path)

        return {
            "source": source_id,
            "destination": dest_id,
            "ladder_tiers": [link.source_tier for link in ladder_path] + (
                [ladder_path[-1].dest_tier] if ladder_path else []
            ),
            "total_latency_ms": propagation_latency_ms + ladder_latency_ms,
            "propagation_latency_ms": propagation_latency_ms,
            "ladder_latency_ms": ladder_latency_ms,
            "total_path_loss_dB": sum(link.path_loss_dB for link in ladder_path),
            "hops": len(ladder_path),
        }

    def network_orbital_frequencies(self) -> np.ndarray:
        """Get all orbital frequencies in the network.

        Returns:
            Array of orbital frequencies in Hz for all active nodes.
        """
        freqs = [
            n.orbital_tier.orbital_frequency_Hz
            for n in self.nodes.values()
            if n.active
        ]
        return np.array(sorted(freqs)) if freqs else np.array([])

    def network_comm_spectrum(self) -> np.ndarray:
        """Get all communication frequencies in use.

        Returns:
            Array of unique communication frequencies in Hz.
        """
        freqs = set()
        for n in self.nodes.values():
            if n.active:
                freqs.add(n.uplink_frequency_Hz)
                freqs.add(n.downlink_frequency_Hz)
                freqs.add(n.crosslink_frequency_Hz)
        return np.array(sorted(freqs)) if freqs else np.array([])

    def total_compute_capacity(self) -> float:
        """Total edge compute capacity across all active nodes (FLOPS)."""
        return sum(n.compute_capacity_flops for n in self.nodes.values() if n.active)

    def create_default_constellation(self) -> None:
        """Create a default multi-tier constellation for demonstration.

        Populates the network with nodes at LEO, MEO, and GEO tiers
        using real Starlink/GPS/GEO frequency allocations.
        """
        leo_tier = OrbitalTier(name="LEO_550", altitude_km=550)
        meo_tier = OrbitalTier(name="MEO_20200", altitude_km=20200)
        geo_tier = OrbitalTier(name="GEO", altitude_km=35786)

        # LEO edge nodes (Starlink-like)
        for i in range(4):
            self.add_node(SatelliteEdgeNode(
                node_id=f"leo-edge-{i:03d}",
                orbital_tier=leo_tier,
                uplink_frequency_Hz=14_000_000_000.0,
                downlink_frequency_Hz=11_700_000_000.0,
                crosslink_frequency_Hz=60_000_000_000.0,
                eco_hz_refs=ECO_HZ_REFERENCES[:3],
                compute_capacity_flops=2e12,
            ))

        # MEO relay nodes (GPS-like)
        for i in range(2):
            self.add_node(SatelliteEdgeNode(
                node_id=f"meo-relay-{i:03d}",
                orbital_tier=meo_tier,
                comm_frequency_Hz=1_575_420_000.0,
                uplink_frequency_Hz=1_575_420_000.0,
                downlink_frequency_Hz=1_227_600_000.0,
                crosslink_frequency_Hz=23_180_000_000.0,
                eco_hz_refs=ECO_HZ_REFERENCES[:2],
                compute_capacity_flops=500e9,
            ))

        # GEO backbone node
        self.add_node(SatelliteEdgeNode(
            node_id="geo-backbone-000",
            orbital_tier=geo_tier,
            uplink_frequency_Hz=14_250_000_000.0,
            downlink_frequency_Hz=12_000_000_000.0,
            crosslink_frequency_Hz=60_000_000_000.0,
            eco_hz_refs=ECO_HZ_REFERENCES,
            compute_capacity_flops=5e12,
            storage_bytes=10_000_000_000_000,
        ))
