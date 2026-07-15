"""Universal spectral latent space module.

Provides the shared latent representation that unifies
all spectral modalities into a common embedding space.
"""

from mesie.foundation.latent.universal_latent_space import (
    UniversalSpectralLatentSpace,
    LatentSpaceConfig,
    ModalityProjector,
    MomentumEncoder,
)
from mesie.foundation.latent.alignment import (
    CrossModalAligner,
    ContrastiveAligner,
    DistillationAligner,
    OptimalTransportAligner,
)
from mesie.foundation.latent.projections import (
    LinearProjection,
    MLPProjection,
    GatedProjection,
    ModalityAdaptiveProjection,
    ProjectionFactory,
)

__all__ = [
    "UniversalSpectralLatentSpace",
    "LatentSpaceConfig",
    "ModalityProjector",
    "MomentumEncoder",
    "CrossModalAligner",
    "ContrastiveAligner",
    "DistillationAligner",
    "OptimalTransportAligner",
    "LinearProjection",
    "MLPProjection",
    "GatedProjection",
    "ModalityAdaptiveProjection",
    "ProjectionFactory",
]
