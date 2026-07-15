"""Resonance analysis for spectral records."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from mesie.core.records import MultiElementRecord, SpectralComponent


def compute_resonance_score(component: SpectralComponent) -> float:
    """Compute resonance score (peak-to-mean ratio) for a component.

    Args:
        component: Input spectral component.

    Returns:
        Resonance score (higher = more resonant peak).
    """
    amp = np.abs(component.amplitude)
    mean_amp = max(float(np.mean(amp)), 1e-12)
    return float(np.max(amp) / mean_amp)


def find_resonance_peaks(
    component: SpectralComponent,
    threshold: float = 2.0,
) -> List[Tuple[float, float]]:
    """Find frequency peaks exceeding a resonance threshold.

    Args:
        component: Input spectral component.
        threshold: Minimum peak-to-mean ratio to consider resonant.

    Returns:
        List of (frequency, amplitude) tuples at resonance peaks.
    """
    amp = np.abs(component.amplitude)
    mean_amp = max(float(np.mean(amp)), 1e-12)

    peaks = []
    for i in range(1, len(amp) - 1):
        if amp[i] > amp[i - 1] and amp[i] > amp[i + 1]:
            if amp[i] / mean_amp >= threshold:
                peaks.append((float(component.frequency[i]), float(amp[i])))
    return peaks


def compute_resonance_bandwidth(
    component: SpectralComponent,
    peak_fraction: float = 0.707,
) -> float:
    """Compute bandwidth at a fraction of peak amplitude (e.g., -3dB bandwidth).

    Args:
        component: Input spectral component.
        peak_fraction: Fraction of peak amplitude for bandwidth calculation.

    Returns:
        Bandwidth in frequency units.
    """
    amp = np.abs(component.amplitude)
    peak_val = np.max(amp)
    threshold = peak_val * peak_fraction

    above = np.where(amp >= threshold)[0]
    if len(above) < 2:
        return 0.0
    return float(component.frequency[above[-1]] - component.frequency[above[0]])
