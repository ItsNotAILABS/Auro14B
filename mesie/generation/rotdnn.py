"""RotDnn-compatible spectral generation."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from mesie.core.records import MultiElementRecord, SpectralComponent
from mesie.core.config import GenerationConfig
from mesie.generation.single_component import _default_frequency_grid, _shape_amplitude


def generate_rotdnn(config: GenerationConfig, **kwargs) -> MultiElementRecord:
    """Generate a RotDnn-compatible multi-component spectral record.

    Generates rotational-dependent spectra (e.g., RotD50, RotD100) as
    separate components within a single record.

    Args:
        config: Generation configuration.
        **kwargs: Additional arguments (reserved for future use).

    Returns:
        MultiElementRecord with RotDnn representation and multiple components.
    """
    rng = np.random.default_rng(config.seed)

    if config.target_frequency is not None:
        freq = np.asarray(config.target_frequency, dtype=float)
    else:
        freq = _default_frequency_grid()

    base_amp = _shape_amplitude(freq, config.amplitude_shape)

    if config.stochastic_perturbation > 0:
        noise = rng.normal(0.0, float(config.stochastic_perturbation), size=len(freq))
        base_amp = base_amp * np.exp(noise)

    # Default RotDnn blending: RotD50 and RotD100
    blend = config.multi_element_blending or {"RotD50": 0.5, "RotD100": 0.5}

    components = []
    for name, weight in blend.items():
        # Each RotDnn component gets slightly different amplitude
        comp_noise = rng.normal(0.0, 0.05, size=len(freq))
        amp = base_amp * float(weight) * np.exp(comp_noise)
        amp = np.clip(amp, config.physical_min_amplitude, config.physical_max_amplitude)

        components.append(SpectralComponent(
            name=name,
            frequency=freq,
            amplitude=amp,
            units="linear",
        ))

    return MultiElementRecord(
        record_id="generated_record",
        components=components,
        representation="rotdnn",
        lineage=["generated", "synthetic", "rotdnn"],
    )
