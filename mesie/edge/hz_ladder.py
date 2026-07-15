"""Hz-Ladder: Vertical frequency-tier backend math and physics.

Implements a frequency-based communication ladder where each rung corresponds
to a real electromagnetic frequency tier. Data propagates vertically through
tiers following physics-based link budgets and propagation models.

The ladder maps directly to real frequency allocations:
  - Tier 0 (ELF/Schumann): 3-30 Hz — ground truth / eco timing
  - Tier 1 (VLF): 3-30 kHz — submarine / deep penetration
  - Tier 2 (HF): 3-30 MHz — skywave / ionospheric reflection
  - Tier 3 (VHF/UHF): 30 MHz - 3 GHz — terrestrial / line-of-sight
  - Tier 4 (SHF/Ku/Ka): 3-40 GHz — satellite uplink/downlink
  - Tier 5 (V/W/EHF): 40-300 GHz — inter-satellite crosslinks
  - Tier 6 (Optical): 193 THz — laser inter-satellite links
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# Physical constants
SPEED_OF_LIGHT_M_S = 299_792_458.0
BOLTZMANN_K = 1.380649e-23  # J/K
EARTH_RADIUS_M = 6_371_000.0
MU_EARTH = 3.986004418e14  # m^3/s^2 gravitational parameter


@dataclass
class FrequencyTier:
    """A single rung on the Hz ladder representing a frequency communication tier.

    Attributes:
        tier_id: Integer tier level (0 = lowest frequency).
        name: Human-readable tier name.
        frequency_low_Hz: Lower bound frequency in Hz.
        frequency_high_Hz: Upper bound frequency in Hz.
        center_frequency_Hz: Representative center frequency.
        propagation_model: Type of radio propagation.
        max_data_rate_bps: Theoretical max data rate for this tier.
        typical_latency_ms: Expected one-way latency.
        penetration_depth_m: How far signal penetrates ground/water.
        applications: List of real-world uses.
    """

    tier_id: int
    name: str
    frequency_low_Hz: float
    frequency_high_Hz: float
    center_frequency_Hz: float
    propagation_model: str = "free_space"
    max_data_rate_bps: float = 0.0
    typical_latency_ms: float = 0.0
    penetration_depth_m: float = float("inf")
    applications: List[str] = field(default_factory=list)

    @property
    def bandwidth_Hz(self) -> float:
        """Available bandwidth in Hz."""
        return self.frequency_high_Hz - self.frequency_low_Hz

    @property
    def wavelength_m(self) -> float:
        """Wavelength at center frequency in meters."""
        return SPEED_OF_LIGHT_M_S / self.center_frequency_Hz

    @property
    def shannon_capacity_bps(self) -> float:
        """Shannon capacity assuming 20 dB SNR (100:1)."""
        snr_linear = 100.0  # 20 dB
        return self.bandwidth_Hz * math.log2(1 + snr_linear)


@dataclass
class LadderLink:
    """A connection between two frequency tiers in the ladder.

    Attributes:
        source_tier: Origin tier ID.
        dest_tier: Destination tier ID.
        path_loss_dB: Signal path loss in dB.
        frequency_translation_ratio: Ratio between center frequencies.
        latency_ms: Added latency for tier transition.
        reliability: Link reliability 0-1.
    """

    source_tier: int
    dest_tier: int
    path_loss_dB: float
    frequency_translation_ratio: float
    latency_ms: float
    reliability: float = 0.99


# Real frequency tier definitions
STANDARD_TIERS: List[FrequencyTier] = [
    FrequencyTier(
        tier_id=0,
        name="ELF/Schumann",
        frequency_low_Hz=3.0,
        frequency_high_Hz=30.0,
        center_frequency_Hz=7.83,  # Schumann fundamental
        propagation_model="earth_ionosphere_waveguide",
        max_data_rate_bps=1.0,
        typical_latency_ms=133.0,  # Earth circumference / c
        penetration_depth_m=10000.0,
        applications=["timing_reference", "submarine_sync", "eco_resonance"],
    ),
    FrequencyTier(
        tier_id=1,
        name="VLF",
        frequency_low_Hz=3_000.0,
        frequency_high_Hz=30_000.0,
        center_frequency_Hz=15_000.0,
        propagation_model="ground_wave_waveguide",
        max_data_rate_bps=300.0,
        typical_latency_ms=50.0,
        penetration_depth_m=20.0,
        applications=["submarine_communication", "navigation", "time_distribution"],
    ),
    FrequencyTier(
        tier_id=2,
        name="HF",
        frequency_low_Hz=3_000_000.0,
        frequency_high_Hz=30_000_000.0,
        center_frequency_Hz=14_000_000.0,
        propagation_model="skywave_ionospheric",
        max_data_rate_bps=64_000.0,
        typical_latency_ms=10.0,
        penetration_depth_m=0.01,
        applications=["long_range_communication", "OTH_radar", "amateur_radio"],
    ),
    FrequencyTier(
        tier_id=3,
        name="UHF/Terrestrial",
        frequency_low_Hz=300_000_000.0,
        frequency_high_Hz=3_000_000_000.0,
        center_frequency_Hz=1_575_420_000.0,  # GPS L1
        propagation_model="line_of_sight",
        max_data_rate_bps=100_000_000.0,
        typical_latency_ms=1.0,
        penetration_depth_m=0.001,
        applications=["GPS", "cellular", "WiFi", "ground_station_uplink"],
    ),
    FrequencyTier(
        tier_id=4,
        name="SHF/Satellite",
        frequency_low_Hz=3_000_000_000.0,
        frequency_high_Hz=40_000_000_000.0,
        center_frequency_Hz=14_000_000_000.0,  # Ku-band uplink
        propagation_model="line_of_sight_satellite",
        max_data_rate_bps=1_000_000_000.0,
        typical_latency_ms=5.0,  # LEO
        penetration_depth_m=0.0,
        applications=["satellite_broadband", "earth_observation", "VSAT"],
    ),
    FrequencyTier(
        tier_id=5,
        name="EHF/Crosslink",
        frequency_low_Hz=40_000_000_000.0,
        frequency_high_Hz=300_000_000_000.0,
        center_frequency_Hz=60_000_000_000.0,  # V-band ISL
        propagation_model="free_space_vacuum",
        max_data_rate_bps=10_000_000_000.0,
        typical_latency_ms=3.7,  # ~1100 km ISL
        penetration_depth_m=0.0,
        applications=["inter_satellite_link", "mesh_backbone", "5G_mmWave"],
    ),
    FrequencyTier(
        tier_id=6,
        name="Optical/Laser",
        frequency_low_Hz=1.5e14,
        frequency_high_Hz=3.0e14,
        center_frequency_Hz=1.934e14,  # 1550 nm telecom window
        propagation_model="free_space_optical",
        max_data_rate_bps=100_000_000_000.0,
        typical_latency_ms=3.7,
        penetration_depth_m=0.0,
        applications=["optical_ISL", "deep_space_comm", "quantum_key_distribution"],
    ),
]


def compute_free_space_loss(frequency_Hz: float, distance_m: float) -> float:
    """Compute free-space path loss (Friis equation).

    Args:
        frequency_Hz: Carrier frequency in Hz.
        distance_m: Distance between transmitter and receiver in meters.

    Returns:
        Path loss in dB (positive value = loss).
    """
    if frequency_Hz <= 0 or distance_m <= 0:
        return 0.0
    wavelength = SPEED_OF_LIGHT_M_S / frequency_Hz
    loss = (4.0 * math.pi * distance_m / wavelength) ** 2
    return 10.0 * math.log10(loss)


def compute_doppler_shift(
    frequency_Hz: float, relative_velocity_m_s: float
) -> float:
    """Compute Doppler frequency shift for a moving satellite.

    Args:
        frequency_Hz: Transmitted frequency in Hz.
        relative_velocity_m_s: Radial velocity component (positive = approaching).

    Returns:
        Doppler-shifted frequency in Hz.
    """
    return frequency_Hz * (1.0 + relative_velocity_m_s / SPEED_OF_LIGHT_M_S)


def compute_link_budget_dB(
    tx_power_dBW: float,
    tx_gain_dBi: float,
    rx_gain_dBi: float,
    frequency_Hz: float,
    distance_m: float,
    atmospheric_loss_dB: float = 0.0,
    rain_loss_dB: float = 0.0,
) -> float:
    """Compute received signal power using link budget equation.

    Args:
        tx_power_dBW: Transmit power in dBW.
        tx_gain_dBi: Transmit antenna gain in dBi.
        rx_gain_dBi: Receive antenna gain in dBi.
        frequency_Hz: Carrier frequency in Hz.
        distance_m: Path distance in meters.
        atmospheric_loss_dB: Atmospheric absorption loss in dB.
        rain_loss_dB: Rain fade loss in dB.

    Returns:
        Received power in dBW.
    """
    fspl = compute_free_space_loss(frequency_Hz, distance_m)
    return tx_power_dBW + tx_gain_dBi + rx_gain_dBi - fspl - atmospheric_loss_dB - rain_loss_dB


class HzLadder:
    """Frequency-ladder communication backbone.

    Organizes frequency tiers into a vertical ladder where data can
    propagate up/down through tiers. Each tier has physics-based
    properties determining capacity, latency, and reach.

    The ladder is the mathematical backbone for routing spectral
    data between ground stations, satellites, and edge nodes.

    Args:
        tiers: Optional custom tier list. Defaults to STANDARD_TIERS.
    """

    def __init__(self, tiers: Optional[Sequence[FrequencyTier]] = None) -> None:
        self.tiers: List[FrequencyTier] = list(tiers or STANDARD_TIERS)
        self.tiers.sort(key=lambda t: t.tier_id)
        self._links: List[LadderLink] = []
        self._build_default_links()

    def _build_default_links(self) -> None:
        """Build default inter-tier links based on physics."""
        for i in range(len(self.tiers) - 1):
            src = self.tiers[i]
            dst = self.tiers[i + 1]
            ratio = dst.center_frequency_Hz / src.center_frequency_Hz
            # Estimate path loss for a typical transition distance
            loss = compute_free_space_loss(dst.center_frequency_Hz, 1000.0)
            latency = src.typical_latency_ms + dst.typical_latency_ms
            self._links.append(
                LadderLink(
                    source_tier=src.tier_id,
                    dest_tier=dst.tier_id,
                    path_loss_dB=loss,
                    frequency_translation_ratio=ratio,
                    latency_ms=latency,
                )
            )

    def get_tier(self, tier_id: int) -> Optional[FrequencyTier]:
        """Get a tier by ID."""
        for t in self.tiers:
            if t.tier_id == tier_id:
                return t
        return None

    def frequency_to_tier(self, frequency_Hz: float) -> Optional[FrequencyTier]:
        """Find which tier a frequency belongs to.

        Args:
            frequency_Hz: Frequency in Hz.

        Returns:
            The matching FrequencyTier or None.
        """
        for t in self.tiers:
            if t.frequency_low_Hz <= frequency_Hz <= t.frequency_high_Hz:
                return t
        return None

    def route_vertical(
        self, source_tier_id: int, dest_tier_id: int
    ) -> List[LadderLink]:
        """Compute the vertical route between two tiers.

        Args:
            source_tier_id: Starting tier.
            dest_tier_id: Destination tier.

        Returns:
            Ordered list of LadderLinks forming the path.
        """
        if source_tier_id == dest_tier_id:
            return []

        path = []
        step = 1 if dest_tier_id > source_tier_id else -1
        current = source_tier_id

        while current != dest_tier_id:
            next_tier = current + step
            link = self._find_link(current, next_tier)
            if link is None:
                # Create reverse link
                fwd = self._find_link(next_tier, current)
                if fwd:
                    link = LadderLink(
                        source_tier=current,
                        dest_tier=next_tier,
                        path_loss_dB=fwd.path_loss_dB,
                        frequency_translation_ratio=1.0 / fwd.frequency_translation_ratio,
                        latency_ms=fwd.latency_ms,
                    )
                else:
                    break
            path.append(link)
            current = next_tier

        return path

    def total_path_loss(self, source_tier_id: int, dest_tier_id: int) -> float:
        """Total path loss in dB for a vertical route."""
        route = self.route_vertical(source_tier_id, dest_tier_id)
        return sum(link.path_loss_dB for link in route)

    def total_latency_ms(self, source_tier_id: int, dest_tier_id: int) -> float:
        """Total latency in ms for a vertical route."""
        route = self.route_vertical(source_tier_id, dest_tier_id)
        return sum(link.latency_ms for link in route)

    def ladder_spectrum(self) -> np.ndarray:
        """Return array of all tier center frequencies (the ladder rungs).

        Returns:
            NumPy array of center frequencies in Hz.
        """
        return np.array([t.center_frequency_Hz for t in self.tiers])

    def tier_capacities(self) -> Dict[int, float]:
        """Shannon capacity for each tier.

        Returns:
            Dict mapping tier_id to capacity in bps.
        """
        return {t.tier_id: t.shannon_capacity_bps for t in self.tiers}

    def _find_link(self, src: int, dst: int) -> Optional[LadderLink]:
        """Find a link between two tier IDs."""
        for link in self._links:
            if link.source_tier == src and link.dest_tier == dst:
                return link
        return None
