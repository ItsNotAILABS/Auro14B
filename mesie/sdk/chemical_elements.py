"""Chemical elements encoded as spectral entities for MAESI/NeuroAIX.

The periodic table of elements is embedded into the MESIE framework as
first-class spectral citizens. Each element carries:

- Official IUPAC name and symbol
- Atomic number and mass
- Characteristic emission/absorption spectral lines
- Electron configuration spectral fingerprint
- Connectome binding (sensory modality associations)
- Chemical group and period classification

This enables the NeuroAIX connectome to reason about chemistry through
spectral patterns — recognizing elements by their emission signatures
just as a spectrometer does, but with cognitive integration.

Nomenclature follows IUPAC 2021 recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class ElementGroup(Enum):
    """Chemical element group classification."""

    ALKALI_METAL = "alkali_metal"
    ALKALINE_EARTH = "alkaline_earth_metal"
    TRANSITION_METAL = "transition_metal"
    POST_TRANSITION_METAL = "post_transition_metal"
    METALLOID = "metalloid"
    NONMETAL = "reactive_nonmetal"
    HALOGEN = "halogen"
    NOBLE_GAS = "noble_gas"
    LANTHANIDE = "lanthanide"
    ACTINIDE = "actinide"


@dataclass
class SpectralElement:
    """A chemical element encoded as a spectral entity.

    Attributes
    ----------
    name : str
        Official IUPAC name.
    symbol : str
        1-2 letter chemical symbol.
    atomic_number : int
        Number of protons (Z).
    atomic_mass : float
        Standard atomic weight in unified atomic mass units (Da).
    group : ElementGroup
        Chemical group classification.
    period : int
        Period (row) in the periodic table.
    electron_config : str
        Abbreviated electron configuration.
    emission_lines_nm : List[float]
        Characteristic emission wavelengths in nanometers.
    ionization_energy_ev : float
        First ionization energy in electron volts.
    spectral_fingerprint : np.ndarray
        24-dimensional spectral encoding vector.
    cognitive_role : str
        Role in NeuroAIX reasoning.
    """

    name: str
    symbol: str
    atomic_number: int
    atomic_mass: float
    group: ElementGroup
    period: int
    electron_config: str
    emission_lines_nm: List[float] = field(default_factory=list)
    ionization_energy_ev: float = 0.0
    spectral_fingerprint: np.ndarray = field(default_factory=lambda: np.zeros(24))
    cognitive_role: str = "elemental_identity"

    def to_embedding(self) -> np.ndarray:
        """Produce 48-dim embedding for connectome injection."""
        atomic_features = np.array([
            self.atomic_number / 118.0,
            self.atomic_mass / 294.0,
            self.period / 7.0,
            self.ionization_energy_ev / 25.0,
            len(self.emission_lines_nm) / 20.0,
            np.mean(self.emission_lines_nm) / 800.0 if self.emission_lines_nm else 0.0,
        ])
        group_vec = np.zeros(10)
        group_vec[list(ElementGroup).index(self.group)] = 1.0
        padding = np.zeros(8)
        return np.concatenate([atomic_features, self.spectral_fingerprint, group_vec, padding])

    def emission_frequencies_hz(self) -> List[float]:
        """Convert emission wavelengths to frequencies."""
        c = 299792458.0  # m/s
        return [c / (wl * 1e-9) for wl in self.emission_lines_nm if wl > 0]


def _fp(z: int) -> np.ndarray:
    """Deterministic fingerprint from atomic number."""
    return np.random.default_rng(z * 137).uniform(0, 1, 24).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# PERIODIC TABLE — First 36 Elements + Key Heavy Elements
# Official IUPAC names, spectral lines from NIST ASD
# ═══════════════════════════════════════════════════════════════════════════════

_ELEMENTS: List[SpectralElement] = [
    SpectralElement("Hydrogen", "H", 1, 1.008, ElementGroup.NONMETAL, 1,
                    "[1s¹]", [656.3, 486.1, 434.0, 410.2], 13.598, _fp(1), "fundamental_building_block"),
    SpectralElement("Helium", "He", 2, 4.003, ElementGroup.NOBLE_GAS, 1,
                    "[1s²]", [587.6, 501.6, 471.3, 447.1], 24.587, _fp(2), "inert_reference"),
    SpectralElement("Lithium", "Li", 3, 6.941, ElementGroup.ALKALI_METAL, 2,
                    "[He]2s¹", [670.8, 610.4, 460.3], 5.392, _fp(3), "electrochemical_potential"),
    SpectralElement("Beryllium", "Be", 4, 9.012, ElementGroup.ALKALINE_EARTH, 2,
                    "[He]2s²", [313.0, 332.1, 234.9], 9.323, _fp(4), "structural_lightness"),
    SpectralElement("Boron", "B", 5, 10.81, ElementGroup.METALLOID, 2,
                    "[He]2s²2p¹", [249.7, 249.8, 208.9], 8.298, _fp(5), "network_former"),
    SpectralElement("Carbon", "C", 6, 12.011, ElementGroup.NONMETAL, 2,
                    "[He]2s²2p²", [247.9, 193.1, 165.7], 11.260, _fp(6), "organic_backbone"),
    SpectralElement("Nitrogen", "N", 7, 14.007, ElementGroup.NONMETAL, 2,
                    "[He]2s²2p³", [746.8, 744.2, 742.4, 568.0], 14.534, _fp(7), "biological_fixation"),
    SpectralElement("Oxygen", "O", 8, 15.999, ElementGroup.NONMETAL, 2,
                    "[He]2s²2p⁴", [777.4, 777.2, 844.6, 615.8], 13.618, _fp(8), "oxidation_energy"),
    SpectralElement("Fluorine", "F", 9, 18.998, ElementGroup.HALOGEN, 2,
                    "[He]2s²2p⁵", [685.6, 690.2, 703.7], 17.423, _fp(9), "electronegativity_maximum"),
    SpectralElement("Neon", "Ne", 10, 20.180, ElementGroup.NOBLE_GAS, 2,
                    "[He]2s²2p⁶", [640.2, 633.4, 585.2, 540.1], 21.565, _fp(10), "noble_stability"),
    SpectralElement("Sodium", "Na", 11, 22.990, ElementGroup.ALKALI_METAL, 3,
                    "[Ne]3s¹", [589.0, 589.6, 568.8, 498.3], 5.139, _fp(11), "ionic_signaling"),
    SpectralElement("Magnesium", "Mg", 12, 24.305, ElementGroup.ALKALINE_EARTH, 3,
                    "[Ne]3s²", [518.4, 517.3, 516.7, 285.2], 7.646, _fp(12), "enzymatic_cofactor"),
    SpectralElement("Aluminum", "Al", 13, 26.982, ElementGroup.POST_TRANSITION_METAL, 3,
                    "[Ne]3s²3p¹", [396.2, 394.4, 309.3], 5.986, _fp(13), "lightweight_structure"),
    SpectralElement("Silicon", "Si", 14, 28.086, ElementGroup.METALLOID, 3,
                    "[Ne]3s²3p²", [288.2, 251.6, 250.7], 8.152, _fp(14), "semiconductor_foundation"),
    SpectralElement("Phosphorus", "P", 15, 30.974, ElementGroup.NONMETAL, 3,
                    "[Ne]3s²3p³", [253.6, 255.3, 213.6], 10.487, _fp(15), "energy_currency_ATP"),
    SpectralElement("Sulfur", "S", 16, 32.065, ElementGroup.NONMETAL, 3,
                    "[Ne]3s²3p⁴", [921.3, 922.8, 469.4], 10.360, _fp(16), "disulfide_bonding"),
    SpectralElement("Chlorine", "Cl", 17, 35.453, ElementGroup.HALOGEN, 3,
                    "[Ne]3s²3p⁵", [725.7, 741.4, 837.6], 12.968, _fp(17), "ionic_balance"),
    SpectralElement("Argon", "Ar", 18, 39.948, ElementGroup.NOBLE_GAS, 3,
                    "[Ne]3s²3p⁶", [811.5, 763.5, 750.4, 696.5], 15.760, _fp(18), "atmospheric_inert"),
    SpectralElement("Potassium", "K", 19, 39.098, ElementGroup.ALKALI_METAL, 4,
                    "[Ar]4s¹", [766.5, 769.9, 404.7], 4.341, _fp(19), "neural_potential"),
    SpectralElement("Calcium", "Ca", 20, 40.078, ElementGroup.ALKALINE_EARTH, 4,
                    "[Ar]4s²", [422.7, 396.8, 393.4], 6.113, _fp(20), "structural_signaling"),
    SpectralElement("Titanium", "Ti", 22, 47.867, ElementGroup.TRANSITION_METAL, 4,
                    "[Ar]3d²4s²", [498.2, 499.1, 500.7], 6.828, _fp(22), "corrosion_resistance"),
    SpectralElement("Chromium", "Cr", 24, 51.996, ElementGroup.TRANSITION_METAL, 4,
                    "[Ar]3d⁵4s¹", [520.8, 425.4, 427.5, 428.9], 6.767, _fp(24), "catalytic_oxidation"),
    SpectralElement("Manganese", "Mn", 25, 54.938, ElementGroup.TRANSITION_METAL, 4,
                    "[Ar]3d⁵4s²", [403.1, 403.3, 403.4, 279.5], 7.434, _fp(25), "biological_redox"),
    SpectralElement("Iron", "Fe", 26, 55.845, ElementGroup.TRANSITION_METAL, 4,
                    "[Ar]3d⁶4s²", [438.4, 440.5, 516.7, 527.0], 7.902, _fp(26), "oxygen_transport"),
    SpectralElement("Cobalt", "Co", 27, 58.933, ElementGroup.TRANSITION_METAL, 4,
                    "[Ar]3d⁷4s²", [345.4, 350.2, 352.7], 7.881, _fp(27), "vitamin_B12_core"),
    SpectralElement("Nickel", "Ni", 28, 58.693, ElementGroup.TRANSITION_METAL, 4,
                    "[Ar]3d⁸4s²", [341.5, 349.3, 352.5], 7.640, _fp(28), "catalytic_hydrogenation"),
    SpectralElement("Copper", "Cu", 29, 63.546, ElementGroup.TRANSITION_METAL, 4,
                    "[Ar]3d¹⁰4s¹", [324.8, 327.4, 510.6, 515.3], 7.726, _fp(29), "electron_transport"),
    SpectralElement("Zinc", "Zn", 30, 65.380, ElementGroup.TRANSITION_METAL, 4,
                    "[Ar]3d¹⁰4s²", [213.9, 330.3, 334.5, 468.0], 9.394, _fp(30), "enzymatic_zinc_finger"),
    SpectralElement("Gallium", "Ga", 31, 69.723, ElementGroup.POST_TRANSITION_METAL, 4,
                    "[Ar]3d¹⁰4s²4p¹", [417.2, 403.3, 294.4], 5.999, _fp(31), "semiconductor_III_V"),
    SpectralElement("Germanium", "Ge", 32, 72.630, ElementGroup.METALLOID, 4,
                    "[Ar]3d¹⁰4s²4p²", [265.1, 270.9, 303.9], 7.900, _fp(32), "infrared_optics"),
    SpectralElement("Selenium", "Se", 34, 78.971, ElementGroup.NONMETAL, 4,
                    "[Ar]3d¹⁰4s²4p⁴", [196.1, 204.0, 206.3], 9.752, _fp(34), "antioxidant_biology"),
    SpectralElement("Bromine", "Br", 35, 79.904, ElementGroup.HALOGEN, 4,
                    "[Ar]3d¹⁰4s²4p⁵", [470.5, 478.6, 481.7], 11.814, _fp(35), "organic_synthesis"),
    SpectralElement("Krypton", "Kr", 36, 83.798, ElementGroup.NOBLE_GAS, 4,
                    "[Ar]3d¹⁰4s²4p⁶", [557.0, 587.1, 760.2], 14.000, _fp(36), "lighting_spectral_standard"),
    # --- KEY HEAVY ELEMENTS ---
    SpectralElement("Silver", "Ag", 47, 107.868, ElementGroup.TRANSITION_METAL, 5,
                    "[Kr]4d¹⁰5s¹", [328.1, 338.3, 520.9], 7.576, _fp(47), "plasmonic_resonance"),
    SpectralElement("Iodine", "I", 53, 126.904, ElementGroup.HALOGEN, 5,
                    "[Kr]4d¹⁰5s²5p⁵", [206.2, 516.1, 546.5], 10.451, _fp(53), "thyroid_metabolism"),
    SpectralElement("Gold", "Au", 79, 196.967, ElementGroup.TRANSITION_METAL, 6,
                    "[Xe]4f¹⁴5d¹⁰6s¹", [267.6, 312.3, 627.8], 9.226, _fp(79), "nanoparticle_sensing"),
    SpectralElement("Platinum", "Pt", 78, 195.084, ElementGroup.TRANSITION_METAL, 6,
                    "[Xe]4f¹⁴5d⁹6s¹", [265.9, 306.5, 340.8], 8.959, _fp(78), "catalytic_surface"),
    SpectralElement("Uranium", "U", 92, 238.029, ElementGroup.ACTINIDE, 7,
                    "[Rn]5f³6d¹7s²", [358.5, 385.9, 591.5], 6.194, _fp(92), "nuclear_fission_fuel"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRY & API
# ═══════════════════════════════════════════════════════════════════════════════

_ELEMENT_INDEX: Dict[str, SpectralElement] = {e.symbol: e for e in _ELEMENTS}
_ELEMENT_NAME_INDEX: Dict[str, SpectralElement] = {e.name.lower(): e for e in _ELEMENTS}


def get_periodic_table() -> List[SpectralElement]:
    """Return all registered spectral elements."""
    return list(_ELEMENTS)


def get_element_by_symbol(symbol: str) -> Optional[SpectralElement]:
    """Look up element by chemical symbol."""
    return _ELEMENT_INDEX.get(symbol)


def get_element_by_name(name: str) -> Optional[SpectralElement]:
    """Look up element by IUPAC name (case-insensitive)."""
    return _ELEMENT_NAME_INDEX.get(name.lower())


def get_elements_by_group(group: ElementGroup) -> List[SpectralElement]:
    """Filter elements by chemical group."""
    return [e for e in _ELEMENTS if e.group == group]


def get_element_embedding_matrix() -> np.ndarray:
    """Return (N, 48) embedding matrix of all elements."""
    return np.stack([e.to_embedding() for e in _ELEMENTS])


def identify_from_spectrum(emission_wavelengths_nm: List[float], tolerance_nm: float = 2.0) -> List[SpectralElement]:
    """Identify elements by matching emission line wavelengths.

    Parameters
    ----------
    emission_wavelengths_nm : List[float]
        Observed emission wavelengths.
    tolerance_nm : float
        Matching tolerance in nanometers.

    Returns
    -------
    List[SpectralElement]
        Elements whose emission lines match within tolerance.
    """
    matches = []
    for element in _ELEMENTS:
        for obs_wl in emission_wavelengths_nm:
            for ref_wl in element.emission_lines_nm:
                if abs(obs_wl - ref_wl) <= tolerance_nm:
                    matches.append(element)
                    break
            else:
                continue
            break
    return matches
