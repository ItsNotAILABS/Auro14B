"""3D Connectome graph — neural connectivity between brain regions.

Builds and manages the structural/functional connectivity graph that
serves as the AI backend intelligence network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mesie.connectome.brain_regions import (
    BrainRegion,
    BrainSystem,
    get_default_regions,
    get_region_positions,
)

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    nx = None
    HAS_NETWORKX = False


@dataclass
class Connection:
    """A directed or undirected neural connection between two regions.

    Attributes:
        source: Source region abbreviation.
        target: Target region abbreviation.
        weight: Connection strength [0, 1].
        tract_type: Type of tract (structural, functional, effective).
        distance_mm: Euclidean distance between region centroids.
        delay_ms: Simulated conduction delay.
    """

    source: str
    target: str
    weight: float = 1.0
    tract_type: str = "structural"
    distance_mm: float = 0.0
    delay_ms: float = 0.0


@dataclass
class ConnectomeGraph:
    """3D neural connectome graph representing the AI brain backend.

    The connectome is the structural and functional wiring diagram of
    the brain, here serving as the intelligence backbone for the MESIE
    spectral engine.

    Attributes:
        regions: List of brain regions (nodes).
        connections: List of connections (edges).
        adjacency: Adjacency dict mapping abbreviation -> list of connected abbreviations.
    """

    regions: List[BrainRegion] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.regions:
            self.regions = get_default_regions()
        self._region_map: Dict[str, BrainRegion] = {
            r.abbreviation: r for r in self.regions
        }
        if not self.adjacency:
            self.adjacency = {r.abbreviation: [] for r in self.regions}

    @property
    def num_regions(self) -> int:
        """Number of brain regions (nodes) in the connectome."""
        return len(self.regions)

    @property
    def num_connections(self) -> int:
        """Number of connections (edges) in the connectome."""
        return len(self.connections)

    def get_region(self, abbreviation: str) -> Optional[BrainRegion]:
        """Lookup a region by abbreviation."""
        return self._region_map.get(abbreviation)

    def add_connection(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        tract_type: str = "structural",
    ) -> Connection:
        """Add a connection between two regions.

        Args:
            source: Source region abbreviation.
            target: Target region abbreviation.
            weight: Connection strength [0, 1].
            tract_type: Connection type.

        Returns:
            The created Connection object.

        Raises:
            KeyError: If source or target region not found.
        """
        if source not in self._region_map:
            raise KeyError(f"Source region '{source}' not found in connectome")
        if target not in self._region_map:
            raise KeyError(f"Target region '{target}' not found in connectome")

        src_pos = self._region_map[source].position_array
        tgt_pos = self._region_map[target].position_array
        distance = float(np.linalg.norm(src_pos - tgt_pos))
        # Conduction velocity ~6 m/s for myelinated fibers
        delay = distance / 6.0

        conn = Connection(
            source=source,
            target=target,
            weight=weight,
            tract_type=tract_type,
            distance_mm=distance,
            delay_ms=delay,
        )
        self.connections.append(conn)
        self.adjacency.setdefault(source, []).append(target)
        self.adjacency.setdefault(target, []).append(source)
        return conn

    def build_connectivity_matrix(self) -> np.ndarray:
        """Build a weighted adjacency/connectivity matrix.

        Returns:
            NxN numpy array where entry (i,j) is the connection weight.
        """
        n = self.num_regions
        matrix = np.zeros((n, n), dtype=np.float64)
        abbrev_to_idx = {r.abbreviation: i for i, r in enumerate(self.regions)}

        for conn in self.connections:
            i = abbrev_to_idx[conn.source]
            j = abbrev_to_idx[conn.target]
            matrix[i, j] = conn.weight
            matrix[j, i] = conn.weight  # Symmetric

        return matrix

    def build_distance_matrix(self) -> np.ndarray:
        """Build a pairwise Euclidean distance matrix between all regions.

        Returns:
            NxN numpy array of inter-region distances in mm.
        """
        positions = get_region_positions(self.regions)
        # Pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

    def to_networkx(self):
        """Convert to a NetworkX graph for analysis and visualization.

        Returns:
            nx.Graph with region attributes and edge weights.

        Raises:
            ImportError: If networkx is not installed.
        """
        if not HAS_NETWORKX:
            raise ImportError(
                "NetworkX is required for graph export. "
                "Install with: pip install networkx"
            )

        g = nx.Graph()
        for region in self.regions:
            g.add_node(
                region.abbreviation,
                name=region.name,
                system=region.system.value,
                position_3d=region.position_3d,
                volume=region.volume_mm3,
                role=region.role,
            )

        for conn in self.connections:
            g.add_edge(
                conn.source,
                conn.target,
                weight=conn.weight,
                tract_type=conn.tract_type,
                distance_mm=conn.distance_mm,
                delay_ms=conn.delay_ms,
            )

        return g


def build_default_connectome() -> ConnectomeGraph:
    """Build a biologically-inspired default connectome with key pathways.

    Creates connections based on known major white-matter tracts and
    functional connectivity patterns in the human brain.

    Returns:
        A ConnectomeGraph with default regions and anatomically-inspired
        connections.
    """
    cg = ConnectomeGraph()

    # === Intra-hemispheric connections ===
    # Prefrontal interconnections
    _connect_system(cg, BrainSystem.PREFRONTAL, weight=0.8)

    # Fronto-parietal network (executive/attention)
    cg.add_connection("DLPFC_L", "PPC_L", weight=0.75, tract_type="functional")
    cg.add_connection("DLPFC_R", "PPC_R", weight=0.75, tract_type="functional")
    cg.add_connection("ACC", "DLPFC_L", weight=0.7, tract_type="functional")
    cg.add_connection("ACC", "DLPFC_R", weight=0.7, tract_type="functional")

    # Language network (arcuate fasciculus)
    cg.add_connection("BRO", "WER", weight=0.9, tract_type="structural")
    cg.add_connection("BRO", "DLPFC_L", weight=0.6, tract_type="structural")
    cg.add_connection("WER", "AG", weight=0.7, tract_type="structural")
    cg.add_connection("STG_L", "WER", weight=0.8, tract_type="structural")
    cg.add_connection("STG_L", "BRO", weight=0.6, tract_type="functional")

    # Visual hierarchy
    cg.add_connection("V1", "V2V3", weight=0.95, tract_type="structural")
    cg.add_connection("V2V3", "FFA", weight=0.7, tract_type="structural")
    cg.add_connection("V2V3", "PPC_L", weight=0.6, tract_type="structural")
    cg.add_connection("V2V3", "PPC_R", weight=0.6, tract_type="structural")
    cg.add_connection("V1", "SC", weight=0.5, tract_type="structural")

    # Motor/Sensory pathways
    cg.add_connection("M1_L", "S1_L", weight=0.85, tract_type="structural")
    cg.add_connection("M1_R", "S1_R", weight=0.85, tract_type="structural")
    cg.add_connection("SMA", "M1_L", weight=0.8, tract_type="structural")
    cg.add_connection("SMA", "M1_R", weight=0.8, tract_type="structural")
    cg.add_connection("PMC", "M1_L", weight=0.75, tract_type="structural")
    cg.add_connection("PMC", "SMA", weight=0.7, tract_type="structural")

    # Limbic circuit (Papez circuit + extensions)
    cg.add_connection("HPC_L", "HPC_R", weight=0.8, tract_type="structural")
    cg.add_connection("HPC_L", "AMY_L", weight=0.7, tract_type="structural")
    cg.add_connection("HPC_R", "AMY_R", weight=0.7, tract_type="structural")
    cg.add_connection("AMY_L", "vmPFC", weight=0.65, tract_type="structural")
    cg.add_connection("AMY_R", "vmPFC", weight=0.65, tract_type="structural")
    cg.add_connection("HPC_L", "PCC", weight=0.6, tract_type="functional")
    cg.add_connection("PCC", "vmPFC", weight=0.7, tract_type="functional")
    cg.add_connection("INS_L", "AMY_L", weight=0.6, tract_type="structural")
    cg.add_connection("INS_L", "ACC", weight=0.65, tract_type="functional")

    # Thalamo-cortical relay
    cg.add_connection("THL_L", "V1", weight=0.9, tract_type="structural")
    cg.add_connection("THL_R", "V1", weight=0.9, tract_type="structural")
    cg.add_connection("THL_L", "S1_L", weight=0.85, tract_type="structural")
    cg.add_connection("THL_R", "S1_R", weight=0.85, tract_type="structural")
    cg.add_connection("THL_L", "DLPFC_L", weight=0.6, tract_type="structural")
    cg.add_connection("THL_R", "DLPFC_R", weight=0.6, tract_type="structural")

    # Basal ganglia loops
    cg.add_connection("CAU", "PUT", weight=0.8, tract_type="structural")
    cg.add_connection("PUT", "GP", weight=0.75, tract_type="structural")
    cg.add_connection("GP", "THL_L", weight=0.7, tract_type="structural")
    cg.add_connection("NAc", "VTA", weight=0.8, tract_type="structural")
    cg.add_connection("VTA", "DLPFC_L", weight=0.5, tract_type="functional")
    cg.add_connection("VTA", "NAc", weight=0.8, tract_type="structural")
    cg.add_connection("CAU", "DLPFC_L", weight=0.55, tract_type="functional")

    # Cerebellar loops
    cg.add_connection("CBV", "CBH_L", weight=0.85, tract_type="structural")
    cg.add_connection("CBV", "CBH_R", weight=0.85, tract_type="structural")
    cg.add_connection("CBH_L", "M1_L", weight=0.6, tract_type="functional")
    cg.add_connection("CBH_R", "M1_R", weight=0.6, tract_type="functional")
    cg.add_connection("CBV", "THL_L", weight=0.5, tract_type="structural")

    # Brainstem arousal/regulation
    cg.add_connection("LC", "DLPFC_L", weight=0.4, tract_type="functional")
    cg.add_connection("LC", "DLPFC_R", weight=0.4, tract_type="functional")
    cg.add_connection("LC", "ACC", weight=0.5, tract_type="functional")
    cg.add_connection("PON", "CBV", weight=0.6, tract_type="structural")
    cg.add_connection("MED", "PON", weight=0.7, tract_type="structural")
    cg.add_connection("SC", "THL_L", weight=0.5, tract_type="structural")
    cg.add_connection("HYP", "AMY_L", weight=0.5, tract_type="structural")
    cg.add_connection("HYP", "VTA", weight=0.4, tract_type="functional")

    # Default mode network
    cg.add_connection("PCC", "AG", weight=0.7, tract_type="functional")
    cg.add_connection("PCC", "HPC_L", weight=0.6, tract_type="functional")
    cg.add_connection("vmPFC", "AG", weight=0.5, tract_type="functional")

    return cg


def _connect_system(
    cg: ConnectomeGraph,
    system: BrainSystem,
    weight: float = 0.7,
) -> None:
    """Connect all regions within a system to each other."""
    system_regions = [r for r in cg.regions if r.system == system]
    for i, r1 in enumerate(system_regions):
        for r2 in system_regions[i + 1:]:
            cg.add_connection(
                r1.abbreviation,
                r2.abbreviation,
                weight=weight,
                tract_type="intra_system",
            )
