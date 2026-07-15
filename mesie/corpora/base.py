"""Base corpus abstraction for multi-domain spectral data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

from mesie.core.records import MultiElementRecord, SpectralComponent, SpectralMetadata
from mesie.corpora.domains import SpectralDomain, DOMAIN_REGISTRY


@dataclass
class CorpusRecord:
    """A record from a spectral corpus with domain annotation.

    Attributes:
        record: The spectral record.
        domain_key: Domain identifier (e.g., 'seismic', 'eeg').
        split: Dataset split ('train', 'val', 'test').
        corpus_metadata: Additional corpus-level metadata.
    """

    record: MultiElementRecord
    domain_key: str
    split: str = "train"
    corpus_metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def domain(self) -> Optional[SpectralDomain]:
        """Return the SpectralDomain for this record."""
        return DOMAIN_REGISTRY.get(self.domain_key)


class SpectralCorpus:
    """Base class for multi-domain spectral corpora.

    Subclass this to implement domain-specific loaders that produce
    standardized MultiElementRecords from raw domain data.

    Args:
        domain_key: Domain identifier from DOMAIN_REGISTRY.
        canonical_n_points: Number of frequency points in canonical grid.
    """

    def __init__(self, domain_key: str, canonical_n_points: int = 256) -> None:
        if domain_key not in DOMAIN_REGISTRY:
            raise ValueError(
                f"Unknown domain '{domain_key}'. "
                f"Available: {list(DOMAIN_REGISTRY.keys())}"
            )
        self.domain_key = domain_key
        self.domain = DOMAIN_REGISTRY[domain_key]
        self.canonical_n_points = canonical_n_points
        self._records: List[CorpusRecord] = []

    def to_canonical_grid(self, frequency: np.ndarray, amplitude: np.ndarray) -> tuple:
        """Resample to canonical log-spaced frequency grid.

        Args:
            frequency: Original frequency array.
            amplitude: Original amplitude array.

        Returns:
            Tuple of (canonical_frequency, resampled_amplitude).
        """
        f_min, f_max = self.domain.frequency_range
        # Use log-spaced canonical grid for spectral data
        canonical_freq = np.logspace(
            np.log10(max(f_min, 1e-12)),
            np.log10(f_max),
            self.canonical_n_points,
        )
        # Linear interpolation to canonical grid
        resampled = np.interp(canonical_freq, frequency, amplitude, left=0.0, right=0.0)
        return canonical_freq, resampled

    def normalize_to_unit_energy(self, amplitude: np.ndarray) -> np.ndarray:
        """Normalize amplitude to unit energy (domain-invariant).

        Args:
            amplitude: Input amplitude array.

        Returns:
            Normalized amplitude array.
        """
        energy = np.sum(amplitude ** 2)
        if energy < 1e-30:
            return amplitude
        return amplitude / np.sqrt(energy)

    def add_record(
        self,
        record_id: str,
        frequency: np.ndarray,
        amplitude: np.ndarray,
        split: str = "train",
        metadata: Optional[Dict[str, object]] = None,
    ) -> CorpusRecord:
        """Add a record to the corpus after canonical normalization.

        Args:
            record_id: Unique identifier for the record.
            frequency: Raw frequency array.
            amplitude: Raw amplitude array.
            split: Dataset split.
            metadata: Optional additional metadata.

        Returns:
            The created CorpusRecord.
        """
        canon_freq, canon_amp = self.to_canonical_grid(frequency, amplitude)
        norm_amp = self.normalize_to_unit_energy(canon_amp)

        component = SpectralComponent(
            name=f"{self.domain_key}_component",
            frequency=canon_freq,
            amplitude=norm_amp,
            domain="frequency",
            units="normalized",
            metadata={"original_units": self.domain.amplitude_units},
        )

        record = MultiElementRecord(
            record_id=record_id,
            components=[component],
            metadata=SpectralMetadata(
                source=self.domain_key,
                units="normalized",
                domain="frequency",
                custom={"spectral_domain": self.domain_key},
            ),
            lineage=["corpus_ingested", "canonical_grid", "unit_energy_normalized"],
            representation="single",
        )

        corpus_record = CorpusRecord(
            record=record,
            domain_key=self.domain_key,
            split=split,
            corpus_metadata=metadata or {},
        )
        self._records.append(corpus_record)
        return corpus_record

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[CorpusRecord]:
        return iter(self._records)

    def get_split(self, split: str) -> List[CorpusRecord]:
        """Return records for a given split.

        Args:
            split: Dataset split ('train', 'val', 'test').

        Returns:
            List of CorpusRecords in the specified split.
        """
        return [r for r in self._records if r.split == split]

    def get_embedding_matrix(self) -> np.ndarray:
        """Return stacked amplitude arrays as a matrix.

        Returns:
            2D array of shape (n_records, canonical_n_points).
        """
        if not self._records:
            return np.empty((0, self.canonical_n_points))
        return np.vstack([
            r.record.components[0].amplitude for r in self._records
        ])
