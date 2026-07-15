"""Electro-spectral feature layer for computing spectral signatures."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.core.metadata import ElectroSpectralSignature


class ElectroSpectralLayer:
    """Electro-spectral feature computation and distance metrics.

    Computes spectral signatures including centroid, spread, band energy,
    resonance, coherence, and harmonic alignment.

    Args:
        band_edges: Frequency band definitions as (low, high, name) tuples.
    """

    def __init__(self, band_edges: Optional[Sequence[Tuple[float, float, str]]] = None) -> None:
        self.band_edges: List[Tuple[float, float, str]] = list(
            band_edges or [
                (0.0, 1.0, "band_low"),
                (1.0, 10.0, "band_mid"),
                (10.0, 100.0, "band_high"),
                (100.0, np.inf, "band_ultra"),
            ]
        )

    def _aggregate(self, record: MultiElementRecord) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate components into a single amplitude representation."""
        if not record.components:
            return np.array([], dtype=float), np.array([], dtype=float)

        base_grid = record.components[0].frequency
        amp = np.zeros_like(base_grid, dtype=float)
        for c in record.components:
            from mesie.processing.interpolate import interpolate_component
            ci = interpolate_component(c, base_grid)
            amp += np.abs(ci.amplitude) * max(c.element_weight, 0.0)
        return base_grid, amp

    def compute_signature(self, record: MultiElementRecord) -> ElectroSpectralSignature:
        """Compute electro-spectral signature for a record.

        Args:
            record: Input multi-element record.

        Returns:
            ElectroSpectralSignature with computed features.
        """
        freq, amp = self._aggregate(record)
        if len(freq) == 0:
            return ElectroSpectralSignature(0.0, 0.0, {}, 0.0, 0.0, 0.0)

        total = max(float(np.sum(amp)), 1e-12)
        centroid = float(np.sum(freq * amp) / total)
        spread = float(np.sqrt(np.sum(((freq - centroid) ** 2) * amp) / total))

        band_energy: Dict[str, float] = {}
        for low, high, name in self.band_edges:
            mask = (freq >= low) & (freq < high)
            band_energy[name] = float(np.sum(amp[mask]))

        resonance = float(np.max(amp) / max(float(np.mean(amp)), 1e-12))

        if len(record.components) > 1:
            from mesie.processing.interpolate import interpolate_component
            stack = np.vstack([
                interpolate_component(c, freq).amplitude for c in record.components
            ])
            coherence = float(1.0 / (1.0 + np.mean(np.std(stack, axis=0))))
        else:
            coherence = 1.0

        peak_idx = int(np.argmax(amp))
        f_peak = max(float(freq[peak_idx]), 1e-12)
        base = max(float(np.min(freq[freq > 0])) if np.any(freq > 0) else f_peak, 1e-12)
        harmonic_ratio = f_peak / base
        harmonic_alignment = float(1.0 / (1.0 + abs(harmonic_ratio - round(harmonic_ratio))))

        return ElectroSpectralSignature(
            spectral_centroid=centroid,
            spectral_spread=spread,
            band_energy=band_energy,
            frequency_resonance=resonance,
            coherence_signature=coherence,
            harmonic_alignment=harmonic_alignment,
        )

    def electro_distance(self, reference: MultiElementRecord, candidate: MultiElementRecord) -> float:
        """Compute electro-spectral distance between two records.

        Args:
            reference: Reference record.
            candidate: Candidate record.

        Returns:
            Euclidean distance in electro-spectral feature space.
        """
        rs = self.compute_signature(reference)
        cs = self.compute_signature(candidate)

        keys = sorted(set(rs.band_energy) | set(cs.band_energy))
        band_delta = sum(
            (rs.band_energy.get(k, 0.0) - cs.band_energy.get(k, 0.0)) ** 2 for k in keys
        )
        v = np.array([
            rs.spectral_centroid - cs.spectral_centroid,
            rs.spectral_spread - cs.spectral_spread,
            rs.frequency_resonance - cs.frequency_resonance,
            rs.coherence_signature - cs.coherence_signature,
            rs.harmonic_alignment - cs.harmonic_alignment,
            np.sqrt(band_delta),
        ], dtype=float)
        return float(np.linalg.norm(v))
