"""Core spectral record data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SpectralComponent:
    """A single spectral component with frequency, amplitude, and metadata.

    Attributes:
        name: Component identifier.
        frequency: Frequency values array (Hz).
        amplitude: Amplitude values array.
        phase: Optional phase values array (radians).
        domain: Signal domain ('frequency' or 'time').
        units: Amplitude units ('linear', 'psd', 'fas', etc.).
        element_weight: Weighting factor for multi-component records.
        node_id: Optional node topology identifier.
        metadata: Additional key-value metadata.
    """

    name: str
    frequency: np.ndarray
    amplitude: np.ndarray
    phase: Optional[np.ndarray] = None
    domain: str = "frequency"
    units: str = "linear"
    element_weight: float = 1.0
    node_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpectralMetadata:
    """Metadata associated with a spectral record.

    Attributes:
        source: Data source identifier.
        units: Default units for the record.
        domain: Default domain for the record.
        sampling_rate: Optional sampling rate (Hz).
        duration: Optional signal duration (seconds).
        station: Optional station/sensor identifier.
        event_id: Optional event identifier.
        custom: Additional custom metadata.
    """

    source: str = ""
    units: str = "linear"
    domain: str = "frequency"
    sampling_rate: Optional[float] = None
    duration: Optional[float] = None
    station: Optional[str] = None
    event_id: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiElementRecord:
    """Multi-component spectral record containing one or more spectral components.

    This is the primary data object in MESIE. It represents a structured
    spectral record with components, metadata, lineage, and topology information.

    Attributes:
        record_id: Unique record identifier.
        components: List of spectral components.
        metadata: Record-level metadata.
        lineage: Provenance/lineage tags.
        representation: Record type ('single', 'multi', 'psd', 'fas', 'rotdnn').
        node_tags: Node topology tags for graph-based operations.
        electro_metadata: Electro-spectral feature metadata.
        validation_status: Current validation state.
    """

    record_id: str
    components: List[SpectralComponent]
    metadata: SpectralMetadata = field(default_factory=SpectralMetadata)
    lineage: List[str] = field(default_factory=list)
    representation: str = "single"
    node_tags: Dict[str, Any] = field(default_factory=dict)
    electro_metadata: Dict[str, Any] = field(default_factory=dict)
    validation_status: Optional[str] = None
