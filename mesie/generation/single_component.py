"""Single-component spectral generation."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.core.config import GenerationConfig


def _default_frequency_grid() -> np.ndarray:
    """Generate a default logarithmic frequency grid."""
    return np.logspace(-1, 2, 256)


def _shape_amplitude(freq: np.ndarray, shape: str) -> np.ndarray:
    """Generate amplitude array from shape template.

    Args:
        freq: Frequency grid.
        shape: Shape name ('flat', 'power_law', 'gaussian', 'broadband', 'pulse').

    Returns:
        Amplitude array.

    Raises:
        ValueError: If shape is not supported.
    """
    if shape == "flat":
        return np.ones_like(freq)
    if shape == "power_law":
        return 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    if shape in ("gaussian", "pulse"):
        center = np.exp(np.mean(np.log(np.maximum(freq, 1e-6))))
        sigma = center / 2.5
        return np.exp(-0.5 * ((freq - center) / max(sigma, 1e-6)) ** 2)
    if shape == "broadband":
        return np.ones_like(freq) * 0.5 + 0.5 / np.sqrt(np.maximum(freq, 1e-6))
    raise ValueError(f"Unsupported amplitude_shape '{shape}'.")


def generate_single(config: GenerationConfig) -> MultiElementRecord:
    """Generate a single-component spectral record.

    Args:
        config: Generation configuration.

    Returns:
        Generated MultiElementRecord.
    """
    rng = np.random.default_rng(config.seed)

    if config.target_frequency is not None:
        freq = np.asarray(config.target_frequency, dtype=float)
    else:
        freq = _default_frequency_grid()

    amp = _shape_amplitude(freq, config.amplitude_shape)

    if config.stochastic_perturbation > 0:
        noise = rng.normal(0.0, float(config.stochastic_perturbation), size=len(freq))
        amp = amp * np.exp(noise)

    if config.electro_modulation != 0.0:
        modulation = 1.0 + float(config.electro_modulation) * np.sin(
            2 * np.pi * np.log10(np.maximum(freq, 1e-6))
        )
        amp = amp * np.maximum(modulation, 1e-6)

    amp = np.clip(amp, config.physical_min_amplitude, config.physical_max_amplitude)

    component = SpectralComponent(
        name="component_0",
        frequency=freq,
        amplitude=amp,
        units="linear",
    )

    return MultiElementRecord(
        record_id="generated_record",
        components=[component],
        representation="single",
        lineage=["generated", "synthetic"],
    )
