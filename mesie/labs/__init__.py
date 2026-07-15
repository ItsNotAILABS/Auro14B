"""Domain-specific lab environments for multi-disciplinary research."""

from mesie.labs.base_lab import BaseLab, LabConfig, LabResult, LabRegistry
from mesie.labs.spectral_lab import SpectralLab
from mesie.labs.chemistry_lab import ChemistryLab
from mesie.labs.physics_lab import PhysicsLab
from mesie.labs.bio_lab import BioLab
from mesie.labs.earth_lab import EarthLab

__all__ = [
    "BaseLab",
    "BioLab",
    "ChemistryLab",
    "EarthLab",
    "LabConfig",
    "LabRegistry",
    "LabResult",
    "PhysicsLab",
    "SpectralLab",
]
