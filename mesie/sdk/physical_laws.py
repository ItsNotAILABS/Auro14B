"""Physical laws encoded as spectral entities for the MAESI/NeuroAIX framework.

Every fundamental law of physics is represented as a first-class spectral
object that the NeuroAIX connectome can reason about through propagation
and integration. Laws carry:

- Official name and mathematical form
- Domain classification
- Spectral resonance signature (how the law manifests in frequency space)
- Connectome binding (which brain regions process this law)
- Constraint relationships (which other laws it depends on or constrains)

Named Laws Registry (Official IUPAC/ISO/SI Nomenclature)
---------------------------------------------------------
Newton's Laws of Motion | Coulomb's Law | Maxwell's Equations |
Schrödinger Equation | Einstein Field Equations | Boltzmann Distribution |
Navier-Stokes Equations | Heisenberg Uncertainty Principle |
Pauli Exclusion Principle | Conservation Laws | Thermodynamic Laws |
Planck-Einstein Relation | de Broglie Relation | Dirac Equation |
Standard Model Lagrangian | Hubble's Law | Kepler's Laws |
Ohm's Law | Faraday's Law | Gauss's Law
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class PhysicsDomain(Enum):
    """Domains of physical law."""

    CLASSICAL_MECHANICS = "classical_mechanics"
    ELECTROMAGNETISM = "electromagnetism"
    QUANTUM_MECHANICS = "quantum_mechanics"
    THERMODYNAMICS = "thermodynamics"
    STATISTICAL_MECHANICS = "statistical_mechanics"
    GENERAL_RELATIVITY = "general_relativity"
    SPECIAL_RELATIVITY = "special_relativity"
    FLUID_DYNAMICS = "fluid_dynamics"
    PARTICLE_PHYSICS = "particle_physics"
    COSMOLOGY = "cosmology"
    OPTICS = "optics"
    NUCLEAR_PHYSICS = "nuclear_physics"
    CONDENSED_MATTER = "condensed_matter"
    QUANTUM_FIELD_THEORY = "quantum_field_theory"


@dataclass
class PhysicalLaw:
    """A physical law encoded as a spectral entity.

    Attributes
    ----------
    name : str
        Official name of the law.
    latex_form : str
        Mathematical expression in LaTeX notation.
    domain : PhysicsDomain
        Branch of physics.
    description : str
        Plain-language description of what the law states.
    constants_involved : List[str]
        Symbols of constants used in this law.
    spectral_signature : np.ndarray
        32-dimensional spectral resonance vector.
    connectome_targets : List[str]
        Brain region abbreviations that process this law.
    constraint_type : str
        Nature of constraint: 'conservation', 'symmetry', 'dynamics', 'equilibrium'.
    energy_scale_ev : float
        Characteristic energy scale in electron volts.
    frequency_regime_hz : Tuple[float, float]
        Frequency band where this law dominates.
    """

    name: str
    latex_form: str
    domain: PhysicsDomain
    description: str
    constants_involved: List[str] = field(default_factory=list)
    spectral_signature: np.ndarray = field(default_factory=lambda: np.zeros(32))
    connectome_targets: List[str] = field(default_factory=list)
    constraint_type: str = "dynamics"
    energy_scale_ev: float = 1.0
    frequency_regime_hz: Tuple[float, float] = (0.0, 1e20)

    def to_embedding(self) -> np.ndarray:
        """Produce 64-dim embedding for connectome injection."""
        domain_vec = np.zeros(14)
        domain_vec[list(PhysicsDomain).index(self.domain)] = 1.0
        meta = np.array([
            np.log10(self.energy_scale_ev + 1e-30),
            np.log10(self.frequency_regime_hz[0] + 1),
            np.log10(self.frequency_regime_hz[1] + 1),
            len(self.constants_involved) / 10.0,
        ])
        padding = np.zeros(14)
        return np.concatenate([self.spectral_signature, domain_vec, meta, padding])


def _sig(seed: int) -> np.ndarray:
    """Deterministic spectral signature from seed."""
    return np.random.default_rng(seed).uniform(0, 1, 32).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNDAMENTAL LAWS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

_LAWS: List[PhysicalLaw] = [
    # --- CLASSICAL MECHANICS ---
    PhysicalLaw(
        name="Newton's First Law (Law of Inertia)",
        latex_form=r"\sum \vec{F} = 0 \implies \frac{d\vec{v}}{dt} = 0",
        domain=PhysicsDomain.CLASSICAL_MECHANICS,
        description="A body remains at rest or in uniform motion unless acted upon by a net force.",
        constants_involved=[],
        spectral_signature=_sig(1001),
        connectome_targets=["M1", "SMA", "PMC"],
        constraint_type="symmetry",
        energy_scale_ev=1e-2,
        frequency_regime_hz=(0.0, 1e6),
    ),
    PhysicalLaw(
        name="Newton's Second Law",
        latex_form=r"\vec{F} = m\vec{a}",
        domain=PhysicsDomain.CLASSICAL_MECHANICS,
        description="Net force equals mass times acceleration.",
        constants_involved=[],
        spectral_signature=_sig(1002),
        connectome_targets=["M1", "PMC", "SPL"],
        constraint_type="dynamics",
        energy_scale_ev=1e-2,
        frequency_regime_hz=(0.0, 1e6),
    ),
    PhysicalLaw(
        name="Newton's Third Law",
        latex_form=r"\vec{F}_{12} = -\vec{F}_{21}",
        domain=PhysicsDomain.CLASSICAL_MECHANICS,
        description="Every action has an equal and opposite reaction.",
        constants_involved=[],
        spectral_signature=_sig(1003),
        connectome_targets=["M1", "S1", "PMC"],
        constraint_type="symmetry",
        energy_scale_ev=1e-2,
        frequency_regime_hz=(0.0, 1e6),
    ),
    PhysicalLaw(
        name="Newton's Law of Universal Gravitation",
        latex_form=r"F = G\frac{m_1 m_2}{r^2}",
        domain=PhysicsDomain.CLASSICAL_MECHANICS,
        description="Every mass attracts every other mass with force proportional to their product and inversely proportional to distance squared.",
        constants_involved=["G"],
        spectral_signature=_sig(1004),
        connectome_targets=["SPL", "IPL", "DLPFC"],
        constraint_type="dynamics",
        energy_scale_ev=1e-10,
        frequency_regime_hz=(1e-4, 1e4),
    ),
    PhysicalLaw(
        name="Kepler's First Law (Law of Ellipses)",
        latex_form=r"r = \frac{a(1-e^2)}{1 + e\cos\theta}",
        domain=PhysicsDomain.CLASSICAL_MECHANICS,
        description="Planetary orbits are ellipses with the Sun at one focus.",
        constants_involved=["G"],
        spectral_signature=_sig(1005),
        connectome_targets=["SPL", "V1", "V2"],
        constraint_type="dynamics",
        energy_scale_ev=1e-10,
        frequency_regime_hz=(1e-8, 1e-3),
    ),
    PhysicalLaw(
        name="Kepler's Third Law",
        latex_form=r"T^2 = \frac{4\pi^2}{GM} a^3",
        domain=PhysicsDomain.CLASSICAL_MECHANICS,
        description="The square of orbital period is proportional to the cube of semi-major axis.",
        constants_involved=["G"],
        spectral_signature=_sig(1006),
        connectome_targets=["SPL", "DLPFC"],
        constraint_type="dynamics",
        energy_scale_ev=1e-10,
        frequency_regime_hz=(1e-8, 1e-3),
    ),
    # --- ELECTROMAGNETISM ---
    PhysicalLaw(
        name="Coulomb's Law",
        latex_form=r"F = \frac{1}{4\pi\epsilon_0}\frac{q_1 q_2}{r^2}",
        domain=PhysicsDomain.ELECTROMAGNETISM,
        description="Electrostatic force between two charges is proportional to their product and inversely proportional to distance squared.",
        constants_involved=["epsilon_0", "e"],
        spectral_signature=_sig(2001),
        connectome_targets=["DLPFC", "SPL"],
        constraint_type="dynamics",
        energy_scale_ev=13.6,
        frequency_regime_hz=(0, 1e18),
    ),
    PhysicalLaw(
        name="Gauss's Law for Electricity",
        latex_form=r"\nabla \cdot \vec{E} = \frac{\rho}{\epsilon_0}",
        domain=PhysicsDomain.ELECTROMAGNETISM,
        description="Electric flux through a closed surface is proportional to enclosed charge.",
        constants_involved=["epsilon_0"],
        spectral_signature=_sig(2002),
        connectome_targets=["DLPFC", "IPL"],
        constraint_type="conservation",
        energy_scale_ev=1.0,
        frequency_regime_hz=(0, 1e18),
    ),
    PhysicalLaw(
        name="Gauss's Law for Magnetism",
        latex_form=r"\nabla \cdot \vec{B} = 0",
        domain=PhysicsDomain.ELECTROMAGNETISM,
        description="No magnetic monopoles exist; magnetic field lines form closed loops.",
        constants_involved=[],
        spectral_signature=_sig(2003),
        connectome_targets=["DLPFC", "IPL"],
        constraint_type="conservation",
        energy_scale_ev=1.0,
        frequency_regime_hz=(0, 1e18),
    ),
    PhysicalLaw(
        name="Faraday's Law of Induction",
        latex_form=r"\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}",
        domain=PhysicsDomain.ELECTROMAGNETISM,
        description="A time-varying magnetic field induces an electric field.",
        constants_involved=[],
        spectral_signature=_sig(2004),
        connectome_targets=["DLPFC", "PMC", "SPL"],
        constraint_type="dynamics",
        energy_scale_ev=1.0,
        frequency_regime_hz=(1e3, 1e18),
    ),
    PhysicalLaw(
        name="Ampère-Maxwell Law",
        latex_form=r"\nabla \times \vec{B} = \mu_0\vec{J} + \mu_0\epsilon_0\frac{\partial \vec{E}}{\partial t}",
        domain=PhysicsDomain.ELECTROMAGNETISM,
        description="Magnetic fields are generated by currents and changing electric fields.",
        constants_involved=["mu_0", "epsilon_0"],
        spectral_signature=_sig(2005),
        connectome_targets=["DLPFC", "PMC"],
        constraint_type="dynamics",
        energy_scale_ev=1.0,
        frequency_regime_hz=(1e3, 1e18),
    ),
    PhysicalLaw(
        name="Ohm's Law",
        latex_form=r"V = IR",
        domain=PhysicsDomain.ELECTROMAGNETISM,
        description="Voltage across a conductor equals current times resistance.",
        constants_involved=["e"],
        spectral_signature=_sig(2006),
        connectome_targets=["PMC", "M1"],
        constraint_type="equilibrium",
        energy_scale_ev=1e-2,
        frequency_regime_hz=(0, 1e12),
    ),
    # --- QUANTUM MECHANICS ---
    PhysicalLaw(
        name="Schrödinger Equation (Time-Dependent)",
        latex_form=r"i\hbar\frac{\partial}{\partial t}|\psi\rangle = \hat{H}|\psi\rangle",
        domain=PhysicsDomain.QUANTUM_MECHANICS,
        description="The fundamental equation of motion for quantum states.",
        constants_involved=["hbar"],
        spectral_signature=_sig(3001),
        connectome_targets=["DLPFC", "ACC", "OFC"],
        constraint_type="dynamics",
        energy_scale_ev=13.6,
        frequency_regime_hz=(1e12, 1e20),
    ),
    PhysicalLaw(
        name="Heisenberg Uncertainty Principle",
        latex_form=r"\Delta x \cdot \Delta p \geq \frac{\hbar}{2}",
        domain=PhysicsDomain.QUANTUM_MECHANICS,
        description="Position and momentum cannot both be precisely known simultaneously.",
        constants_involved=["hbar"],
        spectral_signature=_sig(3002),
        connectome_targets=["DLPFC", "ACC"],
        constraint_type="symmetry",
        energy_scale_ev=1.0,
        frequency_regime_hz=(1e12, 1e20),
    ),
    PhysicalLaw(
        name="Pauli Exclusion Principle",
        latex_form=r"\psi(x_1, x_2) = -\psi(x_2, x_1)",
        domain=PhysicsDomain.QUANTUM_MECHANICS,
        description="No two identical fermions can occupy the same quantum state.",
        constants_involved=["hbar"],
        spectral_signature=_sig(3003),
        connectome_targets=["DLPFC", "OFC"],
        constraint_type="symmetry",
        energy_scale_ev=1.0,
        frequency_regime_hz=(1e14, 1e20),
    ),
    PhysicalLaw(
        name="Planck-Einstein Relation",
        latex_form=r"E = h\nu",
        domain=PhysicsDomain.QUANTUM_MECHANICS,
        description="Energy of a photon is proportional to its frequency.",
        constants_involved=["h"],
        spectral_signature=_sig(3004),
        connectome_targets=["V1", "V2", "DLPFC"],
        constraint_type="conservation",
        energy_scale_ev=2.0,
        frequency_regime_hz=(1e12, 1e20),
    ),
    PhysicalLaw(
        name="de Broglie Relation",
        latex_form=r"\lambda = \frac{h}{p}",
        domain=PhysicsDomain.QUANTUM_MECHANICS,
        description="Every particle has an associated wavelength inversely proportional to momentum.",
        constants_involved=["h"],
        spectral_signature=_sig(3005),
        connectome_targets=["SPL", "DLPFC"],
        constraint_type="symmetry",
        energy_scale_ev=1.0,
        frequency_regime_hz=(1e12, 1e20),
    ),
    # --- THERMODYNAMICS ---
    PhysicalLaw(
        name="First Law of Thermodynamics",
        latex_form=r"\Delta U = Q - W",
        domain=PhysicsDomain.THERMODYNAMICS,
        description="Energy is conserved: internal energy change equals heat added minus work done.",
        constants_involved=["k_B"],
        spectral_signature=_sig(4001),
        connectome_targets=["DLPFC", "INS", "ACC"],
        constraint_type="conservation",
        energy_scale_ev=0.025,
        frequency_regime_hz=(0, 1e14),
    ),
    PhysicalLaw(
        name="Second Law of Thermodynamics",
        latex_form=r"\Delta S_{universe} \geq 0",
        domain=PhysicsDomain.THERMODYNAMICS,
        description="Total entropy of an isolated system never decreases.",
        constants_involved=["k_B"],
        spectral_signature=_sig(4002),
        connectome_targets=["DLPFC", "ACC", "OFC"],
        constraint_type="symmetry",
        energy_scale_ev=0.025,
        frequency_regime_hz=(0, 1e14),
    ),
    PhysicalLaw(
        name="Third Law of Thermodynamics (Nernst's Theorem)",
        latex_form=r"\lim_{T \to 0} S = 0",
        domain=PhysicsDomain.THERMODYNAMICS,
        description="Entropy approaches zero as temperature approaches absolute zero.",
        constants_involved=["k_B"],
        spectral_signature=_sig(4003),
        connectome_targets=["DLPFC", "ACC"],
        constraint_type="equilibrium",
        energy_scale_ev=1e-5,
        frequency_regime_hz=(0, 1e10),
    ),
    PhysicalLaw(
        name="Boltzmann Distribution",
        latex_form=r"P(E) \propto e^{-E/k_B T}",
        domain=PhysicsDomain.STATISTICAL_MECHANICS,
        description="Probability of a state decreases exponentially with its energy.",
        constants_involved=["k_B"],
        spectral_signature=_sig(4004),
        connectome_targets=["DLPFC", "HPC", "ACC"],
        constraint_type="equilibrium",
        energy_scale_ev=0.025,
        frequency_regime_hz=(1e9, 1e14),
    ),
    PhysicalLaw(
        name="Stefan-Boltzmann Law",
        latex_form=r"j = \sigma T^4",
        domain=PhysicsDomain.THERMODYNAMICS,
        description="Total radiant power of a blackbody is proportional to the fourth power of temperature.",
        constants_involved=["sigma", "k_B"],
        spectral_signature=_sig(4005),
        connectome_targets=["V1", "SPL", "DLPFC"],
        constraint_type="dynamics",
        energy_scale_ev=0.5,
        frequency_regime_hz=(1e11, 1e15),
    ),
    # --- GENERAL & SPECIAL RELATIVITY ---
    PhysicalLaw(
        name="Einstein Field Equations",
        latex_form=r"G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}",
        domain=PhysicsDomain.GENERAL_RELATIVITY,
        description="Spacetime geometry (curvature) is determined by energy-momentum distribution.",
        constants_involved=["G", "c", "Lambda"],
        spectral_signature=_sig(5001),
        connectome_targets=["DLPFC", "SPL", "IPL", "ACC"],
        constraint_type="dynamics",
        energy_scale_ev=1e19,
        frequency_regime_hz=(1e-4, 1e20),
    ),
    PhysicalLaw(
        name="Mass-Energy Equivalence",
        latex_form=r"E = mc^2",
        domain=PhysicsDomain.SPECIAL_RELATIVITY,
        description="Energy and mass are interchangeable; rest energy equals mass times speed of light squared.",
        constants_involved=["c"],
        spectral_signature=_sig(5002),
        connectome_targets=["DLPFC", "SPL"],
        constraint_type="conservation",
        energy_scale_ev=5.11e5,  # Electron rest energy
        frequency_regime_hz=(1e18, 1e23),
    ),
    PhysicalLaw(
        name="Lorentz Transformation",
        latex_form=r"x' = \gamma(x - vt), \quad t' = \gamma(t - vx/c^2)",
        domain=PhysicsDomain.SPECIAL_RELATIVITY,
        description="Spacetime coordinates transform between inertial frames via Lorentz boost.",
        constants_involved=["c"],
        spectral_signature=_sig(5003),
        connectome_targets=["SPL", "IPL", "DLPFC"],
        constraint_type="symmetry",
        energy_scale_ev=1e6,
        frequency_regime_hz=(0, 1e20),
    ),
    # --- FLUID DYNAMICS ---
    PhysicalLaw(
        name="Navier-Stokes Equations",
        latex_form=r"\rho\left(\frac{\partial \vec{v}}{\partial t} + \vec{v}\cdot\nabla\vec{v}\right) = -\nabla p + \mu\nabla^2\vec{v} + \vec{f}",
        domain=PhysicsDomain.FLUID_DYNAMICS,
        description="Momentum conservation for viscous incompressible fluid flow.",
        constants_involved=[],
        spectral_signature=_sig(6001),
        connectome_targets=["PMC", "SMA", "SPL"],
        constraint_type="dynamics",
        energy_scale_ev=1e-4,
        frequency_regime_hz=(0, 1e9),
    ),
    PhysicalLaw(
        name="Bernoulli's Principle",
        latex_form=r"P + \frac{1}{2}\rho v^2 + \rho gh = \text{const}",
        domain=PhysicsDomain.FLUID_DYNAMICS,
        description="In steady flow, pressure plus kinetic plus potential energy density is constant along a streamline.",
        constants_involved=[],
        spectral_signature=_sig(6002),
        connectome_targets=["PMC", "SPL"],
        constraint_type="conservation",
        energy_scale_ev=1e-4,
        frequency_regime_hz=(0, 1e6),
    ),
    # --- PARTICLE PHYSICS ---
    PhysicalLaw(
        name="Dirac Equation",
        latex_form=r"(i\gamma^\mu \partial_\mu - m)\psi = 0",
        domain=PhysicsDomain.PARTICLE_PHYSICS,
        description="Relativistic quantum equation for spin-1/2 fermions, predicting antimatter.",
        constants_involved=["hbar", "c", "m_e"],
        spectral_signature=_sig(7001),
        connectome_targets=["DLPFC", "ACC", "OFC"],
        constraint_type="dynamics",
        energy_scale_ev=5.11e5,
        frequency_regime_hz=(1e18, 1e23),
    ),
    PhysicalLaw(
        name="Standard Model Lagrangian",
        latex_form=r"\mathcal{L}_{SM} = \mathcal{L}_{gauge} + \mathcal{L}_{fermion} + \mathcal{L}_{Higgs} + \mathcal{L}_{Yukawa}",
        domain=PhysicsDomain.QUANTUM_FIELD_THEORY,
        description="Complete quantum field theory of electromagnetic, weak, and strong interactions.",
        constants_involved=["hbar", "c", "e", "alpha"],
        spectral_signature=_sig(7002),
        connectome_targets=["DLPFC", "ACC", "OFC", "SPL"],
        constraint_type="symmetry",
        energy_scale_ev=1.25e11,  # Electroweak scale
        frequency_regime_hz=(1e20, 1e26),
    ),
    # --- COSMOLOGY ---
    PhysicalLaw(
        name="Hubble's Law",
        latex_form=r"v = H_0 d",
        domain=PhysicsDomain.COSMOLOGY,
        description="Recessional velocity of galaxies is proportional to their distance.",
        constants_involved=["H_0"],
        spectral_signature=_sig(8001),
        connectome_targets=["SPL", "IPL", "V1"],
        constraint_type="dynamics",
        energy_scale_ev=1e-4,
        frequency_regime_hz=(1e-18, 1e9),
    ),
    PhysicalLaw(
        name="Friedmann Equations",
        latex_form=r"H^2 = \frac{8\pi G}{3}\rho - \frac{k}{a^2} + \frac{\Lambda}{3}",
        domain=PhysicsDomain.COSMOLOGY,
        description="Govern the expansion dynamics of a homogeneous isotropic universe.",
        constants_involved=["G", "Lambda", "c"],
        spectral_signature=_sig(8002),
        connectome_targets=["DLPFC", "SPL", "IPL"],
        constraint_type="dynamics",
        energy_scale_ev=1e19,
        frequency_regime_hz=(1e-18, 1e-4),
    ),
    # --- OPTICS ---
    PhysicalLaw(
        name="Snell's Law (Law of Refraction)",
        latex_form=r"n_1 \sin\theta_1 = n_2 \sin\theta_2",
        domain=PhysicsDomain.OPTICS,
        description="Ratio of sines of incidence and refraction angles equals ratio of refractive indices.",
        constants_involved=["c"],
        spectral_signature=_sig(9001),
        connectome_targets=["V1", "V2", "V3"],
        constraint_type="symmetry",
        energy_scale_ev=2.0,
        frequency_regime_hz=(4e14, 8e14),
    ),
    # --- NUCLEAR PHYSICS ---
    PhysicalLaw(
        name="Radioactive Decay Law",
        latex_form=r"N(t) = N_0 e^{-\lambda t}",
        domain=PhysicsDomain.NUCLEAR_PHYSICS,
        description="Number of undecayed nuclei decreases exponentially with time.",
        constants_involved=["hbar"],
        spectral_signature=_sig(10001),
        connectome_targets=["DLPFC", "HPC"],
        constraint_type="dynamics",
        energy_scale_ev=1e6,
        frequency_regime_hz=(1e-10, 1e20),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY & API
# ═══════════════════════════════════════════════════════════════════════════════


class SpectralLawRegistry:
    """Registry of all physical laws as spectral entities."""

    def __init__(self, laws: Optional[List[PhysicalLaw]] = None):
        self.laws = laws if laws is not None else list(_LAWS)
        self._index: Dict[str, PhysicalLaw] = {law.name: law for law in self.laws}

    def get(self, name: str) -> Optional[PhysicalLaw]:
        """Look up a law by exact name."""
        return self._index.get(name)

    def search(self, query: str) -> List[PhysicalLaw]:
        """Search laws by substring match in name or description."""
        q = query.lower()
        return [l for l in self.laws if q in l.name.lower() or q in l.description.lower()]

    def by_domain(self, domain: PhysicsDomain) -> List[PhysicalLaw]:
        """Filter laws by physics domain."""
        return [l for l in self.laws if l.domain == domain]

    def get_embedding_matrix(self) -> np.ndarray:
        """Return (N, 64) embedding matrix of all laws."""
        return np.stack([law.to_embedding() for law in self.laws])

    def inject_into_connectome(self, target_regions: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Prepare law embeddings grouped by connectome target region."""
        region_map: Dict[str, List[np.ndarray]] = {}
        for law in self.laws:
            targets = target_regions or law.connectome_targets
            for region in targets:
                if region not in region_map:
                    region_map[region] = []
                region_map[region].append(law.to_embedding())
        return {k: np.mean(v, axis=0) for k, v in region_map.items()}

    def __len__(self) -> int:
        return len(self.laws)

    def __iter__(self):
        return iter(self.laws)


def get_fundamental_laws() -> List[PhysicalLaw]:
    """Return all fundamental physical laws."""
    return list(_LAWS)


def get_law_by_name(name: str) -> Optional[PhysicalLaw]:
    """Look up a law by its official name."""
    for law in _LAWS:
        if law.name == name:
            return law
    return None
