"""Spectral smoothing operations."""

from __future__ import annotations

import numpy as np

from mesie.core.records import SpectralComponent

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    savgol_filter = None
    HAS_SCIPY = False


def smooth_component(
    component: SpectralComponent,
    window_length: int = 9,
    polyorder: int = 2,
) -> SpectralComponent:
    """Smooth component amplitude using Savitzky-Golay filter.

    Falls back to moving average if scipy is not available.

    Args:
        component: Input spectral component.
        window_length: Smoothing window length (must be odd).
        polyorder: Polynomial order for Savitzky-Golay filter.

    Returns:
        New SpectralComponent with smoothed amplitude.
    """
    amp = component.amplitude.astype(float)
    if len(amp) < 3:
        return component

    wl = max(3, (int(window_length) // 2) * 2 + 1)
    if wl > len(amp):
        wl = len(amp) if len(amp) % 2 == 1 else len(amp) - 1
    if wl < 3:
        return component

    if HAS_SCIPY and savgol_filter is not None and wl > polyorder:
        smoothed = savgol_filter(amp, window_length=wl, polyorder=min(polyorder, wl - 1))
    else:
        kernel = np.ones(wl, dtype=float) / wl
        smoothed = np.convolve(amp, kernel, mode="same")

    return SpectralComponent(**{**component.__dict__, "amplitude": smoothed})
