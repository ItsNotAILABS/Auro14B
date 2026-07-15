"""Cosmic Spectral Layers — 13 heavens + 9 underworlds frequency decomposition.

Mesoamerican cosmology modeled 22 vertical layers of reality:
- 9 underworlds (Mictlan descending): deep sub-bass to low-frequency strata
- Earth/middle plane: the transition point
- 13 heavens (Ilhuicatl ascending): mid to ultra-high frequency strata

This maps directly to multi-resolution spectral decomposition where each layer
processes a specific frequency band, with energy flowing between adjacent layers
following the Teotl principle of continuous transformation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mesie.core.records import MultiElementRecord, SpectralComponent


class LayerDomain(enum.Enum):
    """Cosmological domain classification for a spectral layer."""

    UNDERWORLD = "underworld"  # Low-frequency strata (Mictlan)
    MIDDLE = "middle"  # Earth/transition band
    HEAVEN = "heaven"  # High-frequency strata (Ilhuicatl)


# The 9 underworld layers (Mictlan) — named after Aztec descent stages
_UNDERWORLD_NAMES = [
    "Apanohuaia",       # Layer 1: crossing the river (deepest sub-bass)
    "Tepetl Monamictia",  # Layer 2: clashing mountains
    "Itztepetl",        # Layer 3: obsidian mountain
    "Itzehecayan",      # Layer 4: obsidian wind
    "Pancuecuetlacayan",  # Layer 5: place of flags
    "Temiminaloyan",    # Layer 6: place of arrows
    "Teyollocualoyan",  # Layer 7: heart devourer
    "Apanhuiayo",       # Layer 8: water of darkness
    "Chicunamictlan",   # Layer 9: place of the dead (threshold)
]

# The 13 heaven layers (Ilhuicatl) — Aztec celestial planes
_HEAVEN_NAMES = [
    "Ilhuicatl Meztli",      # Heaven 1: moon (first ascending)
    "Ilhuicatl Citlalco",    # Heaven 2: stars
    "Ilhuicatl Tonatiuh",    # Heaven 3: sun
    "Ilhuicatl Huitztlan",   # Heaven 4: Venus/salt
    "Ilhuicatl Mamalhuazocan",  # Heaven 5: comets/fire drill
    "Ilhuicatl Yayauhco",    # Heaven 6: dark/green
    "Ilhuicatl Xoxouhco",    # Heaven 7: blue/day
    "Ilhuicatl Nanatzcayan",  # Heaven 8: place of obsidian crackle
    "Ilhuicatl Teoiztac",    # Heaven 9: white/divine
    "Ilhuicatl Teocozauhco",  # Heaven 10: yellow/divine
    "Ilhuicatl Teotlatlauhco",  # Heaven 11: red/divine
    "Ilhuicatl Teteocalo",   # Heaven 12: place of gods
    "Omeyocan",              # Heaven 13: place of duality (highest)
]


@dataclass
class CosmicLayer:
    """A single cosmological spectral layer.

    Attributes:
        index: Layer position (0-21, underworlds 0-8, middle implied, heavens 9-21).
        name: Nahuatl name of the layer.
        domain: Whether this is underworld, middle, or heaven.
        freq_low: Lower frequency bound (Hz).
        freq_high: Upper frequency bound (Hz).
        energy: Total energy in this layer.
        amplitude_profile: Amplitude values within this band.
        frequency_grid: Frequency values within this band.
        resonance_state: Layer resonance metric (0-1).
    """

    index: int
    name: str
    domain: LayerDomain
    freq_low: float
    freq_high: float
    energy: float = 0.0
    amplitude_profile: Optional[np.ndarray] = None
    frequency_grid: Optional[np.ndarray] = None
    resonance_state: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize layer to dictionary."""
        return {
            "index": self.index,
            "name": self.name,
            "domain": self.domain.value,
            "freq_low": self.freq_low,
            "freq_high": self.freq_high,
            "energy": self.energy,
            "resonance_state": self.resonance_state,
        }


class CosmicSpectralDecomposer:
    """Decompose spectral signals into 22 cosmological layers.

    Maps frequency content onto the Aztec-Maya layered cosmology:
    9 underworld layers handle low-frequency content, 13 heavenly layers
    handle mid-to-high frequency content. Energy flows between adjacent
    layers following Teotl transformation principles.

    The decomposition uses logarithmic frequency partitioning to match
    human perception and natural spectral distributions.

    Args:
        freq_min: Minimum frequency for decomposition (Hz).
        freq_max: Maximum frequency for decomposition (Hz).
        log_scale: Use logarithmic frequency partitioning.
    """

    N_UNDERWORLDS = 9
    N_HEAVENS = 13
    N_LAYERS = 22  # 9 + 13

    def __init__(
        self,
        freq_min: float = 0.01,
        freq_max: float = 100.0,
        log_scale: bool = True,
    ) -> None:
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.log_scale = log_scale
        self._layers: List[CosmicLayer] = []
        self._build_layer_structure()

    def _build_layer_structure(self) -> None:
        """Construct the 22-layer frequency partition."""
        if self.log_scale:
            edges = np.logspace(
                np.log10(self.freq_min),
                np.log10(self.freq_max),
                self.N_LAYERS + 1,
            )
        else:
            edges = np.linspace(self.freq_min, self.freq_max, self.N_LAYERS + 1)

        self._layers = []
        for i in range(self.N_LAYERS):
            if i < self.N_UNDERWORLDS:
                domain = LayerDomain.UNDERWORLD
                name = _UNDERWORLD_NAMES[i]
            else:
                domain = LayerDomain.HEAVEN
                name = _HEAVEN_NAMES[i - self.N_UNDERWORLDS]

            self._layers.append(CosmicLayer(
                index=i,
                name=name,
                domain=domain,
                freq_low=float(edges[i]),
                freq_high=float(edges[i + 1]),
            ))

    @property
    def layers(self) -> List[CosmicLayer]:
        """Access the cosmic layer definitions."""
        return list(self._layers)

    @property
    def layer_edges(self) -> np.ndarray:
        """Frequency edges for all layers."""
        edges = [self._layers[0].freq_low]
        edges.extend(layer.freq_high for layer in self._layers)
        return np.array(edges)

    def decompose(self, component: SpectralComponent) -> List[CosmicLayer]:
        """Decompose a spectral component into cosmological layers.

        Distributes the signal's frequency content across the 22 layers,
        computing energy and amplitude profiles for each stratum.

        Args:
            component: Input spectral component.

        Returns:
            List of CosmicLayer objects with populated energy/amplitude data.
        """
        freq = component.frequency
        amp = np.abs(component.amplitude)

        result_layers = []
        for layer in self._layers:
            mask = (freq >= layer.freq_low) & (freq < layer.freq_high)
            layer_freq = freq[mask]
            layer_amp = amp[mask]
            energy = float(np.sum(layer_amp ** 2)) if len(layer_amp) > 0 else 0.0

            # Resonance: ratio of peak energy to mean (how concentrated)
            if len(layer_amp) > 0 and np.mean(layer_amp) > 0:
                resonance = float(np.max(layer_amp) / (np.mean(layer_amp) + 1e-12))
                resonance = min(resonance / 10.0, 1.0)  # normalize to [0,1]
            else:
                resonance = 0.0

            result_layers.append(CosmicLayer(
                index=layer.index,
                name=layer.name,
                domain=layer.domain,
                freq_low=layer.freq_low,
                freq_high=layer.freq_high,
                energy=energy,
                amplitude_profile=layer_amp if len(layer_amp) > 0 else None,
                frequency_grid=layer_freq if len(layer_freq) > 0 else None,
                resonance_state=resonance,
            ))

        return result_layers

    def decompose_record(self, record: MultiElementRecord) -> Dict[str, List[CosmicLayer]]:
        """Decompose all components in a record into cosmological layers.

        Args:
            record: Multi-element spectral record.

        Returns:
            Dictionary mapping component names to their layer decompositions.
        """
        results: Dict[str, List[CosmicLayer]] = {}
        for comp in record.components:
            results[comp.name] = self.decompose(comp)
        return results

    def energy_distribution(self, component: SpectralComponent) -> np.ndarray:
        """Get energy distribution across all 22 layers.

        Args:
            component: Input spectral component.

        Returns:
            Array of shape (22,) with energy values per layer.
        """
        layers = self.decompose(component)
        return np.array([layer.energy for layer in layers])

    def underworld_energy(self, component: SpectralComponent) -> float:
        """Total energy in the 9 underworld layers (low-frequency).

        Args:
            component: Input spectral component.

        Returns:
            Sum of energy across underworld layers.
        """
        dist = self.energy_distribution(component)
        return float(np.sum(dist[:self.N_UNDERWORLDS]))

    def heaven_energy(self, component: SpectralComponent) -> float:
        """Total energy in the 13 heaven layers (high-frequency).

        Args:
            component: Input spectral component.

        Returns:
            Sum of energy across heaven layers.
        """
        dist = self.energy_distribution(component)
        return float(np.sum(dist[self.N_UNDERWORLDS:]))

    def cosmic_balance(self, component: SpectralComponent) -> float:
        """Compute the cosmic balance ratio (heaven/underworld energy).

        A balanced signal has ratio near 1.0. Heavy underworld dominance
        (ratio < 1) means low-frequency dominated. Heavy heaven dominance
        (ratio > 1) means high-frequency dominated.

        Args:
            component: Input spectral component.

        Returns:
            Ratio of heaven energy to underworld energy.
        """
        uw = self.underworld_energy(component)
        hv = self.heaven_energy(component)
        return hv / max(uw, 1e-12)

    def layer_similarity(
        self,
        component_a: SpectralComponent,
        component_b: SpectralComponent,
    ) -> np.ndarray:
        """Compute per-layer similarity between two components.

        Args:
            component_a: First component.
            component_b: Second component.

        Returns:
            Array of shape (22,) with cosine similarity per layer.
        """
        layers_a = self.decompose(component_a)
        layers_b = self.decompose(component_b)

        similarities = np.zeros(self.N_LAYERS)
        for i, (la, lb) in enumerate(zip(layers_a, layers_b)):
            if la.amplitude_profile is not None and lb.amplitude_profile is not None:
                if len(la.amplitude_profile) > 0 and len(lb.amplitude_profile) > 0:
                    # Interpolate to common length
                    n = max(len(la.amplitude_profile), len(lb.amplitude_profile))
                    a_interp = np.interp(
                        np.linspace(0, 1, n),
                        np.linspace(0, 1, len(la.amplitude_profile)),
                        la.amplitude_profile,
                    )
                    b_interp = np.interp(
                        np.linspace(0, 1, n),
                        np.linspace(0, 1, len(lb.amplitude_profile)),
                        lb.amplitude_profile,
                    )
                    norm_a = np.linalg.norm(a_interp)
                    norm_b = np.linalg.norm(b_interp)
                    if norm_a > 0 and norm_b > 0:
                        similarities[i] = float(
                            np.dot(a_interp, b_interp) / (norm_a * norm_b)
                        )
        return similarities
