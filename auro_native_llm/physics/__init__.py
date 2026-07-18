"""Real physics AI formulas for Auro/MESIE — not scaffolds.

Novel spectral–field equations used in train, embed, fuse, and generate:
  - Nonlinear wave dispersion & action density
  - Spectral coherence (Wiener) & resonance
  - φ-potential Schrödinger residual
  - Kuramoto multi-mode synchronization
  - Landau order-parameter free energy
  - Fisher-natural gradient correction
  - Physics-regularized language loss
"""

from auro_native_llm.physics.formulas import (
    PHI,
    PHI_INV,
    GOLDEN_ANGLE,
    EQUATIONS,
    dispersion_omega,
    spectral_action_density,
    wiener_coherence,
    resonance_score,
    phi_schrodinger_step,
    kuramoto_order,
    kuramoto_step,
    landau_free_energy,
    landau_field_force,
    fisher_natural_grad,
    physics_regularized_loss,
    spectral_force_on_hidden,
    physics_lr_schedule,
    PhysicsState,
    PhysicsReport,
)
from auro_native_llm.physics.engine import PhysicsAIEngine, get_physics_engine

__all__ = [
    "PHI",
    "PHI_INV",
    "GOLDEN_ANGLE",
    "EQUATIONS",
    "dispersion_omega",
    "spectral_action_density",
    "wiener_coherence",
    "resonance_score",
    "phi_schrodinger_step",
    "kuramoto_order",
    "kuramoto_step",
    "landau_free_energy",
    "landau_field_force",
    "fisher_natural_grad",
    "physics_regularized_loss",
    "spectral_force_on_hidden",
    "physics_lr_schedule",
    "PhysicsState",
    "PhysicsReport",
    "PhysicsAIEngine",
    "get_physics_engine",
]
