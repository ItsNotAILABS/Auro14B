"""Extended metadata handling for spectral records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ElectroSpectralSignature:
    """Computed electro-spectral feature signature for a record.

    Attributes:
        spectral_centroid: Weighted mean frequency.
        spectral_spread: Frequency spread around centroid.
        band_energy: Energy in each frequency band.
        frequency_resonance: Peak-to-mean ratio.
        coherence_signature: Inter-component coherence.
        harmonic_alignment: Harmonic structure alignment score.
        metadata: Additional signature metadata.
    """

    spectral_centroid: float
    spectral_spread: float
    band_energy: Dict[str, float]
    frequency_resonance: float
    coherence_signature: float
    harmonic_alignment: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeTopologyMap:
    """Node topology mapping for graph-based spectral operations.

    Attributes:
        node_id: Unique node identifier.
        lineage_tags: Tags for lineage grouping.
        weight: Node importance weight.
        resonance_group: Optional resonance group assignment.
        neighbors: Connected node identifiers.
        embedding: Optional pre-computed node embedding.
    """

    node_id: str
    lineage_tags: List[str] = field(default_factory=list)
    weight: float = 1.0
    resonance_group: Optional[str] = None
    neighbors: List[str] = field(default_factory=list)
    embedding: Optional[Any] = None
