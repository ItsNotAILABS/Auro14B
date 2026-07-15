"""Pretraining objectives for spectral foundation models.

Implements multiple self-supervised objectives optimized
for learning universal spectral representations.
"""

from mesie.foundation.objectives.losses import (
    SpectralReconstructionLoss,
    FrequencyBandLoss,
    MultiScaleLoss,
    PerceptualSpectralLoss,
)
from mesie.foundation.objectives.masked_spectral import (
    MaskedSpectralModeling,
    BandMasking,
    StructuredMasking,
    HierarchicalMasking,
)
from mesie.foundation.objectives.contrastive import (
    SpectralInfoNCE,
    BarlowTwins,
    VICReg,
    DINO,
)
from mesie.foundation.objectives.physics_informed import (
    PhysicsInformedLoss,
    ConservationLoss,
    CausalityLoss,
    SymmetryLoss,
    SpectralConsistencyLoss,
)

__all__ = [
    "SpectralReconstructionLoss",
    "FrequencyBandLoss",
    "MultiScaleLoss",
    "PerceptualSpectralLoss",
    "MaskedSpectralModeling",
    "BandMasking",
    "StructuredMasking",
    "HierarchicalMasking",
    "SpectralInfoNCE",
    "BarlowTwins",
    "VICReg",
    "DINO",
    "PhysicsInformedLoss",
    "ConservationLoss",
    "CausalityLoss",
    "SymmetryLoss",
    "SpectralConsistencyLoss",
]
