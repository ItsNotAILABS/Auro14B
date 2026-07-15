"""Spectral transform operations."""

from __future__ import annotations

import numpy as np

from mesie.core.records import SpectralComponent


def log_transform(component: SpectralComponent, base: float = 10.0) -> SpectralComponent:
    """Apply log transform to component amplitude.

    Args:
        component: Input spectral component.
        base: Logarithm base (default 10).

    Returns:
        New SpectralComponent with log-transformed amplitude.
    """
    amp = np.maximum(component.amplitude, 1e-12)
    if base == 10.0:
        transformed = np.log10(amp)
    elif base == np.e:
        transformed = np.log(amp)
    else:
        transformed = np.log(amp) / np.log(base)
    return SpectralComponent(**{**component.__dict__, "amplitude": transformed})


def amplitude_scale(component: SpectralComponent, scale: float) -> SpectralComponent:
    """Scale component amplitude by a constant factor.

    Args:
        component: Input spectral component.
        scale: Scaling factor.

    Returns:
        New SpectralComponent with scaled amplitude.
    """
    return SpectralComponent(**{**component.__dict__, "amplitude": component.amplitude * float(scale)})
