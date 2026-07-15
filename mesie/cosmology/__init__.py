"""Cosmology module — Aztec-Maya layered spectral decomposition and token governance.

Maps Mesoamerican cosmological structures onto multi-resolution spectral
processing: 13 heavens (upper frequency strata) + 9 underworlds (lower
frequency strata) as a unified 22-layer decomposition framework.

Calendrical token flows implement bounded compute governance where spectral
matching must converge within token-cost constraints — sacrifice as bounded proof.
"""

from mesie.cosmology.layers import (
    CosmicLayer,
    CosmicSpectralDecomposer,
    LayerDomain,
)
from mesie.cosmology.token_governor import (
    CalendricalTokenGovernor,
    TokenBudget,
    TokenExpenditure,
)
from mesie.cosmology.teotl_flow import TeotlEnergyFlow

__all__ = [
    "CalendricalTokenGovernor",
    "CosmicLayer",
    "CosmicSpectralDecomposer",
    "LayerDomain",
    "TeotlEnergyFlow",
    "TokenBudget",
    "TokenExpenditure",
]
