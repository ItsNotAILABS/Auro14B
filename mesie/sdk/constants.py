"""Universal physical constants encoded as spectral entities.

Every fundamental constant of nature is represented as a spectral object
within the MAESI/NeuroAIX framework. Each constant carries:

- Its SI value and units
- A spectral fingerprint (characteristic frequency signature)
- A connectome binding role (which brain system processes it)
- A cognitive weight (importance in reasoning)

This enables the NeuroAIX connectome to reason about physics natively
through spectral propagation rather than symbolic manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class UniversalSpectralConstant:
    """A physical constant encoded as a spectral entity.

    Attributes
    ----------
    name : str
        Official IUPAC/CODATA name.
    symbol : str
        Standard symbol (e.g., 'ℏ', 'k_B', 'c').
    value : float
        SI value.
    units : str
        SI units string.
    spectral_frequency_hz : float
        Characteristic frequency when this constant manifests spectrally.
    spectral_fingerprint : np.ndarray
        16-dimensional spectral signature vector.
    cognitive_role : str
        Role in the NeuroAIX connectome reasoning.
    domain : str
        Physics domain (quantum, thermal, electromagnetic, gravitational).
    uncertainty : float
        Relative standard uncertainty.
    """

    name: str
    symbol: str
    value: float
    units: str
    spectral_frequency_hz: float
    spectral_fingerprint: np.ndarray = field(default_factory=lambda: np.zeros(16))
    cognitive_role: str = "universal_constraint"
    domain: str = "fundamental"
    uncertainty: float = 0.0

    def to_embedding(self) -> np.ndarray:
        """Convert constant to a 32-dim embedding for connectome injection."""
        value_features = np.array([
            np.log10(abs(self.value) + 1e-300),
            self.spectral_frequency_hz / 1e15,
            self.uncertainty,
            hash(self.domain) % 1000 / 1000.0,
        ])
        padding = np.zeros(12)
        return np.concatenate([value_features, self.spectral_fingerprint, padding])


def _fingerprint(seed: int) -> np.ndarray:
    """Generate deterministic spectral fingerprint from seed."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, 16).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL CONSTANTS — CODATA 2018 Values
# ═══════════════════════════════════════════════════════════════════════════════

PLANCK_SPECTRAL = UniversalSpectralConstant(
    name="Planck Constant",
    symbol="h",
    value=6.62607015e-34,
    units="J·s",
    spectral_frequency_hz=1.5e33,  # Planck frequency
    spectral_fingerprint=_fingerprint(6626),
    cognitive_role="quantum_discretization",
    domain="quantum",
    uncertainty=0.0,  # Exact by definition since 2019
)

REDUCED_PLANCK_SPECTRAL = UniversalSpectralConstant(
    name="Reduced Planck Constant (Dirac Constant)",
    symbol="ℏ",
    value=1.054571817e-34,
    units="J·s",
    spectral_frequency_hz=1.5e33,
    spectral_fingerprint=_fingerprint(1054),
    cognitive_role="angular_momentum_quantum",
    domain="quantum",
    uncertainty=0.0,
)

BOLTZMANN_SPECTRAL = UniversalSpectralConstant(
    name="Boltzmann Constant",
    symbol="k_B",
    value=1.380649e-23,
    units="J/K",
    spectral_frequency_hz=6.25e12,  # Thermal frequency at 300K
    spectral_fingerprint=_fingerprint(1380),
    cognitive_role="thermal_energy_scale",
    domain="thermal",
    uncertainty=0.0,
)

SPEED_OF_LIGHT_SPECTRAL = UniversalSpectralConstant(
    name="Speed of Light in Vacuum",
    symbol="c",
    value=299792458.0,
    units="m/s",
    spectral_frequency_hz=5.0e14,  # Visible light center
    spectral_fingerprint=_fingerprint(2997),
    cognitive_role="causal_propagation_limit",
    domain="electromagnetic",
    uncertainty=0.0,
)

GRAVITATIONAL_SPECTRAL = UniversalSpectralConstant(
    name="Newtonian Constant of Gravitation",
    symbol="G",
    value=6.67430e-11,
    units="m³/(kg·s²)",
    spectral_frequency_hz=1.0e-4,  # Gravitational wave band
    spectral_fingerprint=_fingerprint(6674),
    cognitive_role="spacetime_curvature",
    domain="gravitational",
    uncertainty=2.2e-5,
)

AVOGADRO_SPECTRAL = UniversalSpectralConstant(
    name="Avogadro Constant",
    symbol="N_A",
    value=6.02214076e23,
    units="mol⁻¹",
    spectral_frequency_hz=1.0e13,  # Molecular vibration band
    spectral_fingerprint=_fingerprint(6022),
    cognitive_role="micro_macro_bridge",
    domain="chemical",
    uncertainty=0.0,
)

FINE_STRUCTURE_SPECTRAL = UniversalSpectralConstant(
    name="Fine-Structure Constant",
    symbol="α",
    value=7.2973525693e-3,
    units="dimensionless",
    spectral_frequency_hz=6.58e15,  # Hydrogen Lyman-alpha
    spectral_fingerprint=_fingerprint(7297),
    cognitive_role="electromagnetic_coupling_strength",
    domain="quantum_electrodynamics",
    uncertainty=1.5e-10,
)

ELEMENTARY_CHARGE_SPECTRAL = UniversalSpectralConstant(
    name="Elementary Charge",
    symbol="e",
    value=1.602176634e-19,
    units="C",
    spectral_frequency_hz=2.42e14,  # Compton frequency
    spectral_fingerprint=_fingerprint(1602),
    cognitive_role="charge_quantization",
    domain="electromagnetic",
    uncertainty=0.0,
)

ELECTRON_MASS_SPECTRAL = UniversalSpectralConstant(
    name="Electron Mass",
    symbol="m_e",
    value=9.1093837015e-31,
    units="kg",
    spectral_frequency_hz=1.24e20,  # Compton frequency of electron
    spectral_fingerprint=_fingerprint(9109),
    cognitive_role="lepton_mass_scale",
    domain="particle",
    uncertainty=3.0e-10,
)

PROTON_MASS_SPECTRAL = UniversalSpectralConstant(
    name="Proton Mass",
    symbol="m_p",
    value=1.67262192369e-27,
    units="kg",
    spectral_frequency_hz=2.27e23,  # Proton Compton frequency
    spectral_fingerprint=_fingerprint(1672),
    cognitive_role="nuclear_mass_scale",
    domain="particle",
    uncertainty=3.1e-10,
)

VACUUM_PERMITTIVITY_SPECTRAL = UniversalSpectralConstant(
    name="Vacuum Electric Permittivity",
    symbol="ε₀",
    value=8.8541878128e-12,
    units="F/m",
    spectral_frequency_hz=5.0e14,
    spectral_fingerprint=_fingerprint(8854),
    cognitive_role="electric_field_medium",
    domain="electromagnetic",
    uncertainty=1.5e-10,
)

VACUUM_PERMEABILITY_SPECTRAL = UniversalSpectralConstant(
    name="Vacuum Magnetic Permeability",
    symbol="μ₀",
    value=1.25663706212e-6,
    units="N/A²",
    spectral_frequency_hz=5.0e14,
    spectral_fingerprint=_fingerprint(1256),
    cognitive_role="magnetic_field_medium",
    domain="electromagnetic",
    uncertainty=1.5e-10,
)

STEFAN_BOLTZMANN_SPECTRAL = UniversalSpectralConstant(
    name="Stefan-Boltzmann Constant",
    symbol="σ",
    value=5.670374419e-8,
    units="W/(m²·K⁴)",
    spectral_frequency_hz=1.0e14,  # Thermal radiation peak
    spectral_fingerprint=_fingerprint(5670),
    cognitive_role="thermal_radiation_scaling",
    domain="thermal",
    uncertainty=0.0,
)

GAS_CONSTANT_SPECTRAL = UniversalSpectralConstant(
    name="Molar Gas Constant",
    symbol="R",
    value=8.314462618,
    units="J/(mol·K)",
    spectral_frequency_hz=6.25e12,
    spectral_fingerprint=_fingerprint(8314),
    cognitive_role="thermodynamic_state_bridge",
    domain="thermal",
    uncertainty=0.0,
)

HUBBLE_SPECTRAL = UniversalSpectralConstant(
    name="Hubble Constant",
    symbol="H₀",
    value=67.4e3,  # m/s/Mpc → converted to SI-like
    units="km/(s·Mpc)",
    spectral_frequency_hz=2.18e-18,  # Hubble frequency
    spectral_fingerprint=_fingerprint(6740),
    cognitive_role="cosmic_expansion_rate",
    domain="cosmological",
    uncertainty=0.01,
)

COSMOLOGICAL_CONSTANT_SPECTRAL = UniversalSpectralConstant(
    name="Cosmological Constant",
    symbol="Λ",
    value=1.1056e-52,
    units="m⁻²",
    spectral_frequency_hz=1.0e-35,  # Deep cosmic frequency
    spectral_fingerprint=_fingerprint(1105),
    cognitive_role="vacuum_energy_density",
    domain="cosmological",
    uncertainty=0.02,
)


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

ALL_CONSTANTS: Dict[str, UniversalSpectralConstant] = {
    "h": PLANCK_SPECTRAL,
    "hbar": REDUCED_PLANCK_SPECTRAL,
    "k_B": BOLTZMANN_SPECTRAL,
    "c": SPEED_OF_LIGHT_SPECTRAL,
    "G": GRAVITATIONAL_SPECTRAL,
    "N_A": AVOGADRO_SPECTRAL,
    "alpha": FINE_STRUCTURE_SPECTRAL,
    "e": ELEMENTARY_CHARGE_SPECTRAL,
    "m_e": ELECTRON_MASS_SPECTRAL,
    "m_p": PROTON_MASS_SPECTRAL,
    "epsilon_0": VACUUM_PERMITTIVITY_SPECTRAL,
    "mu_0": VACUUM_PERMEABILITY_SPECTRAL,
    "sigma": STEFAN_BOLTZMANN_SPECTRAL,
    "R": GAS_CONSTANT_SPECTRAL,
    "H_0": HUBBLE_SPECTRAL,
    "Lambda": COSMOLOGICAL_CONSTANT_SPECTRAL,
}


def get_constant(symbol: str) -> Optional[UniversalSpectralConstant]:
    """Retrieve a constant by its standard symbol."""
    return ALL_CONSTANTS.get(symbol)


def get_all_embeddings() -> np.ndarray:
    """Return (N, 32) matrix of all constant embeddings."""
    return np.stack([c.to_embedding() for c in ALL_CONSTANTS.values()])
