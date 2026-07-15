"""Edge communication module using frequency-ladder (Hz) backend physics.

Provides vertical-tier frequency-based communication layers, satellite edge
virtual nodes, and spectral edge protocols using real electromagnetic data.
"""

from mesie.edge.hz_ladder import (
    HzLadder,
    FrequencyTier,
    LadderLink,
    compute_free_space_loss,
    compute_doppler_shift,
)
from mesie.edge.satellite_nodes import (
    SatelliteEdgeNode,
    OrbitalTier,
    EcoHzReference,
    VirtualNodeNetwork,
    compute_orbital_frequency,
)
from mesie.edge.edge_protocol import (
    EdgeSpectralProtocol,
    EdgeMessage,
    EdgeRoute,
    SpectralHandshake,
)

__all__ = [
    "HzLadder",
    "FrequencyTier",
    "LadderLink",
    "compute_free_space_loss",
    "compute_doppler_shift",
    "SatelliteEdgeNode",
    "OrbitalTier",
    "EcoHzReference",
    "VirtualNodeNetwork",
    "compute_orbital_frequency",
    "EdgeSpectralProtocol",
    "EdgeMessage",
    "EdgeRoute",
    "SpectralHandshake",
]
