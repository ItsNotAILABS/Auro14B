"""Helix encoder — projects spectral embeddings into helical coordinates.

Provides encoding utilities that transform flat embeddings into
helix-native representations with phase, radius, and progression components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from mesie.core.records import MultiElementRecord
from mesie.io.loaders import RecordInput, load_record
from mesie.embeddings.vectorizers import SpectralVectorizer
from mesie.features.electro_spectral import ElectroSpectralLayer


@dataclass
class HelixProjection:
    """A spectral embedding projected into helical coordinates.

    Args:
        record_id: Source record identifier.
        phase: Angular phase θ in [0, 2π).
        radius: Radial magnitude.
        elevation: Vertical/progression component.
        cartesian: 3D Cartesian position on helix.
        flat_embedding: Original flat embedding vector.
        coherence: Coherence metric for this projection.
    """

    record_id: str
    phase: float
    radius: float
    elevation: float
    cartesian: np.ndarray
    flat_embedding: np.ndarray
    coherence: float


class HelixEncoder:
    """Encodes spectral records into helical vector projections.

    Transforms embeddings from flat vector space into a helical
    coordinate system that captures phase relationships, radial
    energy, and sequential progression.

    Args:
        vectorizer: SpectralVectorizer for embedding computation.
        pitch: Helix pitch (vertical distance per revolution).
        base_radius: Base radius multiplier.
        phase_offset: Global phase offset in radians.
    """

    def __init__(
        self,
        vectorizer: Optional[SpectralVectorizer] = None,
        pitch: float = 1.0,
        base_radius: float = 1.0,
        phase_offset: float = 0.0,
    ) -> None:
        self._vectorizer = vectorizer or SpectralVectorizer()
        self._electro = ElectroSpectralLayer()
        self.pitch = pitch
        self.base_radius = base_radius
        self.phase_offset = phase_offset
        self._encode_counter: int = 0

    def _extract_phase(self, embedding: np.ndarray) -> float:
        """Extract phase angle from embedding vector."""
        if len(embedding) < 2:
            return self.phase_offset
        raw_phase = float(np.arctan2(embedding[1], embedding[0]))
        return (raw_phase + self.phase_offset) % (2.0 * np.pi)

    def _extract_radius(self, embedding: np.ndarray) -> float:
        """Extract radius from embedding magnitude."""
        magnitude = float(np.linalg.norm(embedding))
        return self.base_radius * (1.0 + np.log1p(magnitude))

    def encode(self, record: RecordInput, elevation: Optional[float] = None) -> HelixProjection:
        """Encode a spectral record into a helix projection.

        Args:
            record: Input spectral record.
            elevation: Optional explicit elevation. If None, auto-incremented.

        Returns:
            HelixProjection with helical coordinates.
        """
        rec = load_record(record)
        embedding = self._vectorizer.transform(rec)
        sig = self._electro.compute_signature(rec)

        phase = self._extract_phase(embedding)
        radius = self._extract_radius(embedding)

        if elevation is None:
            elevation = float(self._encode_counter) * self.pitch
            self._encode_counter += 1

        cartesian = np.array([
            radius * np.cos(phase),
            radius * np.sin(phase),
            elevation,
        ])

        return HelixProjection(
            record_id=rec.record_id,
            phase=phase,
            radius=radius,
            elevation=elevation,
            cartesian=cartesian,
            flat_embedding=embedding,
            coherence=sig.coherence_signature,
        )

    def encode_batch(
        self,
        records: Sequence[RecordInput],
        elevations: Optional[Sequence[float]] = None,
    ) -> list[HelixProjection]:
        """Encode multiple records into helix projections.

        Args:
            records: Sequence of input records.
            elevations: Optional explicit elevations for each record.

        Returns:
            List of HelixProjection objects.
        """
        if elevations is None:
            return [self.encode(r) for r in records]
        return [self.encode(r, e) for r, e in zip(records, elevations)]

    def compute_phase_distance(self, proj_a: HelixProjection, proj_b: HelixProjection) -> float:
        """Compute angular distance between two projections.

        Args:
            proj_a: First projection.
            proj_b: Second projection.

        Returns:
            Minimum angular distance in [0, π].
        """
        diff = abs(proj_a.phase - proj_b.phase)
        return min(diff, 2 * np.pi - diff)

    def compute_helix_distance(self, proj_a: HelixProjection, proj_b: HelixProjection) -> float:
        """Compute 3D distance between two helix projections.

        Args:
            proj_a: First projection.
            proj_b: Second projection.

        Returns:
            Euclidean distance in helix space.
        """
        return float(np.linalg.norm(proj_a.cartesian - proj_b.cartesian))

    @property
    def encode_count(self) -> int:
        """Number of records encoded."""
        return self._encode_counter
