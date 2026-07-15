"""Band energy computation for spectral records."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mesie.core.records import MultiElementRecord, SpectralComponent


def compute_band_energy(
    component: SpectralComponent,
    bands: Optional[Sequence[Tuple[float, float, str]]] = None,
) -> Dict[str, float]:
    """Compute energy in each frequency band for a component.

    Args:
        component: Input spectral component.
        bands: Band definitions as (low_freq, high_freq, name) tuples.
            Defaults to standard low/mid/high/ultra bands.

    Returns:
        Dictionary mapping band names to energy values.
    """
    if bands is None:
        bands = [
            (0.0, 1.0, "low"),
            (1.0, 10.0, "mid"),
            (10.0, 100.0, "high"),
            (100.0, np.inf, "ultra"),
        ]

    freq = component.frequency
    amp = np.abs(component.amplitude)
    result: Dict[str, float] = {}
    for low, high, name in bands:
        mask = (freq >= low) & (freq < high)
        result[name] = float(np.sum(amp[mask] ** 2))
    return result


def compute_total_energy(component: SpectralComponent) -> float:
    """Compute total spectral energy of a component.

    Args:
        component: Input spectral component.

    Returns:
        Total energy (sum of squared amplitudes).
    """
    return float(np.sum(component.amplitude ** 2))
