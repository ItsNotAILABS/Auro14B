"""Configuration objects for MESIE operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class GenerationConfig:
    """Configuration for spectral generation.

    Attributes:
        target_frequency: Target frequency grid for output.
        amplitude_shape: Shape template ('flat', 'power_law', 'gaussian', 'broadband', 'pulse').
        stochastic_perturbation: Noise level for stochastic generation.
        seed: Random seed for reproducibility.
        multi_element_blending: Component name to weight mapping.
        node_influence: Node-based weighting factors.
        electro_modulation: Electro-spectral modulation strength.
        physical_min_amplitude: Minimum physical amplitude constraint.
        physical_max_amplitude: Maximum physical amplitude constraint.
        output_format: Output representation type.
        band_weighting: Optional band-specific weighting.
        smoothing_constraints: Optional smoothing parameters.
    """

    target_frequency: Optional[np.ndarray] = None
    amplitude_shape: str = "flat"
    stochastic_perturbation: float = 0.0
    seed: Optional[int] = None
    multi_element_blending: Dict[str, float] = field(default_factory=dict)
    node_influence: Dict[str, float] = field(default_factory=dict)
    electro_modulation: float = 0.0
    physical_min_amplitude: float = 1e-12
    physical_max_amplitude: float = 1e6
    output_format: str = "single"
    band_weighting: Optional[Dict[str, float]] = None
    smoothing_constraints: Optional[Dict[str, Any]] = None
