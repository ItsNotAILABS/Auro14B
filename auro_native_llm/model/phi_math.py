"""φ-mathematics for Auro weight geometry (SOLUS / MESIE lineage).

Five core constants used across Medina/MESIE builds:
  φ (golden ratio), φ², 1/φ, golden angle, φ-harmonic series.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# Canonical constants (match mesie.sdk.solus.constants)
PHI = 1.618033988749895
PHI_INV = PHI - 1.0  # 1/φ = φ - 1
PHI_SQ = PHI * PHI
GOLDEN_ANGLE_DEG = 137.5077640500378
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE_DEG)
FIVE_MATH = {
    "phi": PHI,
    "phi_inverse": PHI_INV,
    "phi_squared": PHI_SQ,
    "golden_angle_rad": GOLDEN_ANGLE_RAD,
    "fibonacci_seed": (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144),
}


def phi_scale(n: int) -> float:
    """φ-scaled gain for layer/index n."""
    return PHI ** (-(1.0 + (n % 8) / 8.0))


def phi_init(shape: Tuple[int, ...], seed: int = 42, layer: int = 0) -> np.ndarray:
    """φ-harmonic weight initialization (replaces pure Gaussian).

    Combines fan-in scaling with a golden-angle phase lattice so early
    spectral structure is present before any training step.
    """
    rng = np.random.default_rng(seed + layer * 17)
    flat = int(np.prod(shape))
    fan_in = shape[0] if len(shape) >= 1 else flat
    base = rng.standard_normal(flat).astype(np.float64)
    idx = np.arange(flat, dtype=np.float64)
    phase = np.sin(idx * GOLDEN_ANGLE_RAD + layer * PHI_INV)
    harmonic = np.cos(idx / max(PHI, 1e-9) + layer * PHI)
    scale = (2.0 / max(fan_in, 1)) ** 0.5 * phi_scale(layer)
    out = (base * 0.7 + phase * 0.2 + harmonic * 0.1) * scale
    return out.reshape(shape)


def nearest_phi_multiple(n: int, base: int = 64) -> int:
    """Round channel count toward a φ-friendly multiple of base."""
    if n <= base:
        return base
    k = max(1, int(round(n / (base * PHI))))
    return int(k * base)


def fibonacci_up_to(limit: int) -> list[int]:
    a, b = 1, 1
    out = [a, b]
    while b < limit:
        a, b = b, a + b
        out.append(b)
    return out
