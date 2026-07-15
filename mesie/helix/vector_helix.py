"""Core Vector Helix implementation.

The Vector Helix arranges spectral embedding vectors along a 3D helical
manifold parameterized by:
- t (progression): Linear advancement along the helix axis
- θ (phase): Angular position on the helix cross-section
- r (radius): Distance from the helix axis (embedding magnitude)

This structure encodes both sequential relationships (via progression)
and spectral similarity (via phase proximity on the helix).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mesie.core.records import MultiElementRecord
from mesie.io.loaders import RecordInput, load_record
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.features.electro_spectral import ElectroSpectralLayer


@dataclass
class HelixConfig:
    """Configuration for the Vector Helix.

    Args:
        pitch: Vertical distance per full revolution (helix pitch).
        base_radius: Base radius of the helix.
        phase_resolution: Number of discrete phase bins per revolution.
        max_nodes: Maximum nodes on the helix.
        coherence_decay: Decay rate for coherence over helix distance.
        embedding_bands: Number of frequency bands for vectorization.
    """

    pitch: float = 1.0
    base_radius: float = 1.0
    phase_resolution: int = 64
    max_nodes: int = 10000
    coherence_decay: float = 0.1
    embedding_bands: int = 8


@dataclass
class HelixNode:
    """A node positioned on the Vector Helix.

    Args:
        record_id: Identifier of the source record.
        embedding: Raw embedding vector.
        helix_position: 3D position on the helix (x, y, z).
        progression: Linear progression parameter t.
        phase: Angular phase θ in radians.
        radius: Radial distance from helix axis.
        coherence: Coherence score of this node.
        metadata: Additional node metadata.
    """

    record_id: str
    embedding: np.ndarray
    helix_position: np.ndarray
    progression: float
    phase: float
    radius: float
    coherence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HelixTraversalResult:
    """Result from a helix traversal operation.

    Args:
        path: Ordered list of traversed nodes.
        total_arc_length: Total arc length of the traversal path.
        phase_continuity: Measure of phase smoothness along path.
        coherence_integral: Integrated coherence along the path.
        unwound_vector: The helix path unwound into a linear vector.
    """

    path: List[HelixNode]
    total_arc_length: float
    phase_continuity: float
    coherence_integral: float
    unwound_vector: np.ndarray


class VectorHelix:
    """The full Vector Helix — a helical manifold for spectral embeddings.

    Places spectral embeddings onto a 3D helix where:
    - Vertical position encodes sequential/temporal ordering
    - Angular position encodes spectral phase relationships
    - Radius encodes embedding magnitude/energy

    This enables phase-aware retrieval, coherent traversal, and
    rotational analysis of spectral intelligence vectors.

    Args:
        config: Helix configuration.
        vectorizer: SpectralVectorizer for embedding computation.
    """

    def __init__(
        self,
        config: Optional[HelixConfig] = None,
        vectorizer: Optional[SpectralVectorizer] = None,
    ) -> None:
        self.config = config or HelixConfig()
        self._vectorizer = vectorizer or SpectralVectorizer(n_bands=self.config.embedding_bands)
        self._electro = ElectroSpectralLayer()
        self._nodes: List[HelixNode] = []
        self._progression_counter: float = 0.0

    def _compute_phase(self, embedding: np.ndarray) -> float:
        """Compute helical phase from an embedding vector.

        Uses the angular position in the first two principal
        dimensions of the embedding as the phase angle.

        Args:
            embedding: Input embedding vector.

        Returns:
            Phase angle in [0, 2π).
        """
        if len(embedding) < 2:
            return 0.0
        angle = float(np.arctan2(embedding[1], embedding[0]))
        # Normalize to [0, 2π)
        return angle % (2.0 * np.pi)

    def _compute_radius(self, embedding: np.ndarray) -> float:
        """Compute helical radius from embedding magnitude.

        Args:
            embedding: Input embedding vector.

        Returns:
            Radius scaled by base_radius configuration.
        """
        magnitude = float(np.linalg.norm(embedding))
        return self.config.base_radius * (1.0 + np.log1p(magnitude))

    def _to_helix_position(
        self, progression: float, phase: float, radius: float
    ) -> np.ndarray:
        """Convert helix parameters to 3D Cartesian position.

        Args:
            progression: Linear progression parameter.
            phase: Angular phase.
            radius: Radial distance.

        Returns:
            3D position array [x, y, z].
        """
        x = radius * np.cos(phase)
        y = radius * np.sin(phase)
        z = progression * self.config.pitch
        return np.array([x, y, z])

    def insert(self, record: RecordInput, metadata: Optional[Dict[str, Any]] = None) -> HelixNode:
        """Insert a spectral record onto the Vector Helix.

        Args:
            record: Input spectral record.
            metadata: Optional metadata to attach.

        Returns:
            The created HelixNode.
        """
        rec = load_record(record)
        embedding = self._vectorizer.transform(rec)
        sig = self._electro.compute_signature(rec)

        phase = self._compute_phase(embedding)
        radius = self._compute_radius(embedding)
        progression = self._progression_counter
        self._progression_counter += 1.0

        helix_pos = self._to_helix_position(progression, phase, radius)

        node = HelixNode(
            record_id=rec.record_id,
            embedding=embedding,
            helix_position=helix_pos,
            progression=progression,
            phase=phase,
            radius=radius,
            coherence=sig.coherence_signature,
            metadata=metadata or {},
        )

        if len(self._nodes) < self.config.max_nodes:
            self._nodes.append(node)
        else:
            # Replace node with lowest coherence
            min_idx = int(np.argmin([n.coherence for n in self._nodes]))
            if node.coherence > self._nodes[min_idx].coherence:
                self._nodes[min_idx] = node

        return node

    def insert_batch(self, records: Sequence[RecordInput]) -> List[HelixNode]:
        """Insert multiple records onto the helix.

        Args:
            records: Sequence of input records.

        Returns:
            List of created HelixNodes.
        """
        return [self.insert(r) for r in records]

    def traverse(
        self,
        start_progression: float = 0.0,
        end_progression: Optional[float] = None,
        phase_window: Optional[Tuple[float, float]] = None,
    ) -> HelixTraversalResult:
        """Traverse the helix along a progression range.

        Args:
            start_progression: Starting progression value.
            end_progression: Ending progression value (default: max).
            phase_window: Optional (min_phase, max_phase) filter in radians.

        Returns:
            HelixTraversalResult with path and metrics.
        """
        if end_progression is None:
            end_progression = self._progression_counter

        # Filter nodes in range
        path_nodes = [
            n for n in self._nodes
            if start_progression <= n.progression <= end_progression
        ]

        # Apply phase window filter
        if phase_window is not None:
            min_p, max_p = phase_window
            if min_p <= max_p:
                path_nodes = [n for n in path_nodes if min_p <= n.phase <= max_p]
            else:
                # Wrapping case (e.g., 5.5 to 1.0 wraps around)
                path_nodes = [n for n in path_nodes if n.phase >= min_p or n.phase <= max_p]

        # Sort by progression
        path_nodes.sort(key=lambda n: n.progression)

        # Compute metrics
        total_arc = 0.0
        phase_diffs: List[float] = []
        coherence_sum = 0.0

        for i in range(len(path_nodes)):
            coherence_sum += path_nodes[i].coherence
            if i > 0:
                diff = np.linalg.norm(
                    path_nodes[i].helix_position - path_nodes[i - 1].helix_position
                )
                total_arc += float(diff)
                # Phase continuity measure
                pd = abs(path_nodes[i].phase - path_nodes[i - 1].phase)
                pd = min(pd, 2 * np.pi - pd)  # Wrap-aware difference
                phase_diffs.append(pd)

        phase_continuity = 1.0 - (float(np.mean(phase_diffs)) / np.pi) if phase_diffs else 1.0
        coherence_integral = coherence_sum / max(len(path_nodes), 1)

        # Unwound vector: concatenate embeddings along path
        if path_nodes:
            unwound = np.concatenate([n.embedding for n in path_nodes])
        else:
            unwound = np.array([])

        return HelixTraversalResult(
            path=path_nodes,
            total_arc_length=total_arc,
            phase_continuity=phase_continuity,
            coherence_integral=coherence_integral,
            unwound_vector=unwound,
        )

    def query_by_phase(self, target_phase: float, tolerance: float = 0.2) -> List[HelixNode]:
        """Find nodes at a specific phase angle.

        Args:
            target_phase: Target phase angle in radians.
            tolerance: Angular tolerance in radians.

        Returns:
            Nodes within tolerance of target phase.
        """
        results = []
        for node in self._nodes:
            diff = abs(node.phase - target_phase)
            diff = min(diff, 2 * np.pi - diff)
            if diff <= tolerance:
                results.append(node)
        return results

    def query_by_coherence(self, min_coherence: float = 0.5) -> List[HelixNode]:
        """Find nodes above a coherence threshold.

        Args:
            min_coherence: Minimum coherence value.

        Returns:
            Nodes with coherence >= min_coherence.
        """
        return [n for n in self._nodes if n.coherence >= min_coherence]

    def compute_helix_distance(self, node_a: HelixNode, node_b: HelixNode) -> float:
        """Compute geodesic distance along the helix between two nodes.

        Args:
            node_a: First node.
            node_b: Second node.

        Returns:
            Approximate geodesic distance along the helix.
        """
        # Euclidean distance in helix space
        euclidean = float(np.linalg.norm(node_a.helix_position - node_b.helix_position))

        # Arc correction for helical geometry
        delta_phase = abs(node_a.phase - node_b.phase)
        delta_phase = min(delta_phase, 2 * np.pi - delta_phase)
        avg_radius = (node_a.radius + node_b.radius) / 2.0
        arc_component = avg_radius * delta_phase

        delta_z = abs(node_a.progression - node_b.progression) * self.config.pitch
        return float(np.sqrt(arc_component**2 + delta_z**2 + euclidean**2) / np.sqrt(2))

    def get_helix_statistics(self) -> Dict[str, Any]:
        """Compute statistics about the current helix state.

        Returns:
            Dictionary of helix metrics.
        """
        if not self._nodes:
            return {
                "node_count": 0,
                "mean_coherence": 0.0,
                "phase_coverage": 0.0,
                "progression_range": 0.0,
                "mean_radius": 0.0,
            }

        coherences = np.array([n.coherence for n in self._nodes])
        phases = np.array([n.phase for n in self._nodes])
        radii = np.array([n.radius for n in self._nodes])
        progressions = np.array([n.progression for n in self._nodes])

        # Phase coverage: fraction of phase bins occupied
        bins = np.linspace(0, 2 * np.pi, self.config.phase_resolution + 1)
        occupied = len(set(np.digitize(phases, bins))) / self.config.phase_resolution

        return {
            "node_count": len(self._nodes),
            "mean_coherence": float(np.mean(coherences)),
            "std_coherence": float(np.std(coherences)),
            "phase_coverage": min(occupied, 1.0),
            "progression_range": float(np.max(progressions) - np.min(progressions)),
            "mean_radius": float(np.mean(radii)),
            "total_helix_length": float(self._progression_counter * self.config.pitch),
        }

    @property
    def nodes(self) -> List[HelixNode]:
        """All nodes on the helix."""
        return self._nodes

    @property
    def size(self) -> int:
        """Number of nodes on the helix."""
        return len(self._nodes)

    @property
    def progression(self) -> float:
        """Current progression counter."""
        return self._progression_counter
