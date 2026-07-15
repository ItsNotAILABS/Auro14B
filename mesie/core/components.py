"""Frequency grid and record lineage structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FrequencyGrid:
    """A reusable frequency grid definition.

    Attributes:
        values: Frequency values in Hz.
        spacing: Grid spacing type ('linear', 'log', 'irregular').
        units: Frequency units (default 'Hz').
    """

    values: np.ndarray
    spacing: str = "linear"
    units: str = "Hz"

    @classmethod
    def linear(cls, start: float, stop: float, num: int) -> "FrequencyGrid":
        """Create a linearly-spaced frequency grid."""
        return cls(values=np.linspace(start, stop, num), spacing="linear")

    @classmethod
    def logarithmic(cls, start: float, stop: float, num: int) -> "FrequencyGrid":
        """Create a logarithmically-spaced frequency grid."""
        return cls(values=np.logspace(np.log10(start), np.log10(stop), num), spacing="log")


@dataclass
class RecordLineage:
    """Lineage and provenance tracking for spectral records.

    Attributes:
        source_id: Original source record identifier.
        operations: List of operations applied to produce this record.
        parent_ids: Parent record identifiers.
        tags: Descriptive lineage tags.
        custom: Additional lineage metadata.
    """

    source_id: Optional[str] = None
    operations: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)
