"""MESIE Connectome — 3D Brain Intelligence Backend.

The connectome module provides a simulated 3D neural environment where
the brain IS the backend, IS the AI, IS the intelligence. Brain regions
are real anatomical structures placed in 3D space, connected by
biologically-inspired white-matter tracts.

Modules:
    brain_regions: Anatomical brain region definitions with 3D coordinates.
    connectome_graph: Structural/functional connectivity graph.
    environment: 3D simulation engine for neural dynamics.
"""

from mesie.connectome.brain_regions import (
    BrainRegion,
    BrainSystem,
    get_default_regions,
    get_region_positions,
    get_regions_by_system,
)
from mesie.connectome.connectome_graph import (
    ConnectomeGraph,
    Connection,
    build_default_connectome,
)
from mesie.connectome.environment import (
    ActivationState,
    ConnectomeEnvironment3D,
    SignalPacket,
)

__all__ = [
    "ActivationState",
    "BrainRegion",
    "BrainSystem",
    "ConnectomeEnvironment3D",
    "ConnectomeGraph",
    "Connection",
    "SignalPacket",
    "build_default_connectome",
    "get_default_regions",
    "get_region_positions",
    "get_regions_by_system",
]
